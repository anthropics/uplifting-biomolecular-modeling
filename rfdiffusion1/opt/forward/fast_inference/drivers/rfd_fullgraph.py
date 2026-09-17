"""
rfd_fullgraph — lever W1: CUDA-graph replay of the WHOLE RoseTTAFold forward of an RFdiffusion v1 denoising step
except the 4 data-dependent top-k refinement Str2Str calls (commit 86507b65; requires a torch 2.4 + dgl 2.4 stack and
the rfd_fastpath levers {'chain_breaks','full_graph','rbf','msa_index'}, whose capture-safe patches the graphs rely on).

Segments of RoseTTAFoldModule.forward(return_infer=True) per step:
   A  latent_emb + full_emb + recycle(zeros) + templ_emb + simulator prologue + 4 extra IterBlocks + 32 main IterBlocks + proj_state2
      -> ONE graph  (graphing the 36 blocks alone would leave ~2,000 eager launches/step in embeddings/template/glue)
   R  4 x str_refiner (make_topk_graph: torch.topk + torch.where -> data-dependent edge count)             -> eager (unchanged code)
   B  torch.stack(R/T/alpha) + aa_pred + lddt_pred + final-xyz einsum + pLDDT softmax                       -> ONE graph
Everything is the upstream code called through the module objects (so rfd_fastpath's patched functions are used); the graphed
callables `_seg_A` / `_seg_B` below are line-by-line transcriptions of RoseTTAFoldModule.forward / IterativeSimulator.forward
restricted to the return_infer=True path, with the python loop unrolled inside the capture.

Hardening:
 * value-keyed GENERATIONS: key = (idx values, B, N, L, motif_mask bytes, cyclic flag, t1d/t2d dims); one private memory pool per
   generation shared by its graphs (A, B), captured in execution order, replayed in that order; static input buffers refreshed by
   copy_ before every replay; outputs cloned before being returned; at most RFD_FG_MAX_GENERATIONS (default 2) resident;
   generation-atomic reset only (reset_all()).
 * eager-first: the call that triggers a capture returns the EAGER result computed before warm-up/capture; the graph serves from
   the next call on.
 * any capture error -> that generation falls back to eager permanently (counted, reported), never a silent mixed state.
 * non-infer calls (return_raw / return_full / use_checkpoint) and models with a timestep_embedder go to the upstream forward.

Use:  rfd_fastpath.apply({...4 levers...}); rfd_fullgraph.apply(model); ...; rfd_fullgraph.stats()
"""
import os, gc
import torch
from opt_core.oom import is_oom
import torch.nn as nn

_cfg = dict(max_generations=int(os.environ.get('RFD_FG_MAX_GENERATIONS', '2')),
            warmup=int(os.environ.get('RFD_FG_WARMUP', '2')),
            graph_heads=int(os.environ.get('RFD_FG_HEADS', '1')))
_state = dict(enabled=True, generations={}, n_replay=0, n_capture=0, n_generations_created=0, n_resets=0,
              individual_graph_pops=0, capture_errors=[], eager_fallback_calls=0,
              upstream_calls=0, capture_s=[])


# --------------------------------------------------------------------------------------------- graphed segments (upstream code)
def _seg_A(model, msa_latent, msa_full, seq, xyz, idx, t1d, t2d, xyz_t, alpha_t, motif_mask, cyclic_reses):
    """RoseTTAFoldModule.forward up to (and including) the 32 main blocks + proj_state2; returns what the refinement needs."""
    einsum = _upstream_einsum()
    B, N, L = msa_latent.shape[:3]
    msa_latent, pair, state = model.latent_emb(msa_latent, seq, idx, cyclic_reses)
    msa_full = model.full_emb(msa_full, seq, idx)
    msa_prev = torch.zeros_like(msa_latent[:, 0])
    pair_prev = torch.zeros_like(pair)
    state_prev = torch.zeros_like(state)
    msa_recycle, pair_recycle, state_recycle = model.recycle(seq, msa_prev, pair_prev, xyz, state_prev)
    msa_latent[:, 0] = msa_latent[:, 0] + msa_recycle.reshape(B, L, -1)
    pair = pair + pair_recycle
    state = state + state_recycle
    pair, state = model.templ_emb(t1d, t2d, alpha_t, xyz_t, pair, state, use_checkpoint=False)
    is_frozen_residue = motif_mask if model.freeze_track_motif else torch.zeros_like(motif_mask).bool()
    # ---- IterativeSimulator.forward (extra + main part) ----
    sim = model.simulator
    xyz_in = xyz[:, :, :3]
    mm = is_frozen_residue
    R_in = torch.eye(3, device=xyz_in.device).reshape(1, 1, 3, 3).expand(B, L, -1, -1)
    T_in = xyz_in[:, :, 1].clone()
    xyz_in = xyz_in - T_in.unsqueeze(-2)
    state = sim.proj_state(state)
    R_s, T_s, alpha_s = [], [], []
    for i_m in range(sim.n_extra_block):
        R_in = R_in.detach(); T_in = T_in.detach()
        xyz_c = einsum('bnij,bnaj->bnai', R_in, xyz_in) + T_in.unsqueeze(-2)
        msa_full, pair, R_in, T_in, state, alpha = sim.extra_block[i_m](msa_full, pair, R_in, T_in, xyz_c, state, idx,
                                                                          motif_mask=mm, use_checkpoint=False, cyclic_reses=cyclic_reses)
        R_s.append(R_in); T_s.append(T_in); alpha_s.append(alpha)
    msa = msa_latent
    for i_m in range(sim.n_main_block):
        R_in = R_in.detach(); T_in = T_in.detach()
        xyz_c = einsum('bnij,bnaj->bnai', R_in, xyz_in) + T_in.unsqueeze(-2)
        msa, pair, R_in, T_in, state, alpha = sim.main_block[i_m](msa, pair, R_in, T_in, xyz_c, state, idx,
                                                                   motif_mask=mm, use_checkpoint=False, cyclic_reses=cyclic_reses)
        R_s.append(R_in); T_s.append(T_in); alpha_s.append(alpha)
    state = sim.proj_state2(state)
    return (msa, pair, state, R_in, T_in, xyz_in, tuple(R_s), tuple(T_s), tuple(alpha_s))


def _seg_R(model, msa, pair, state, R_in, T_in, xyz_in, idx, motif_mask, cyclic_reses):
    """the 4 top-k refinement calls — eager, upstream code."""
    einsum = _upstream_einsum()
    sim = model.simulator
    mm = motif_mask if model.freeze_track_motif else torch.zeros_like(motif_mask).bool()
    R_r, T_r, A_r = [], [], []
    for i_m in range(sim.n_ref_block):
        R_in = R_in.detach(); T_in = T_in.detach()
        xyz_c = einsum('bnij,bnaj->bnai', R_in, xyz_in) + T_in.unsqueeze(-2)
        R_in, T_in, state, alpha = sim.str_refiner(msa, pair, R_in, T_in, xyz_c, state, idx, top_k=64, motif_mask=mm, cyclic_reses=cyclic_reses)
        R_r.append(R_in); T_r.append(T_in); A_r.append(alpha)
    return state, tuple(R_r), tuple(T_r), tuple(A_r)


def _seg_B(model, msa, pair, state, xyz, R_all, T_all, A_all):
    """heads: equal statements to the return_infer branch of RoseTTAFoldModule.forward."""
    einsum = _upstream_einsum()
    R = torch.stack(R_all, dim=0); T = torch.stack(T_all, dim=0); alpha_s = torch.stack(A_all, dim=0)
    logits_aa = model.aa_pred(msa)
    lddt = model.lddt_pred(state)
    xyz_out = einsum('bnij,bnaj->bnai', R[-1], xyz[:, :, :3] - xyz[:, :, 1].unsqueeze(-2)) + T[-1].unsqueeze(-2)
    nbin = lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=lddt.dtype, device=lddt.device)
    pred_lddt = nn.Softmax(dim=1)(lddt)
    pred_lddt = torch.sum(lddt_bins[None, :, None] * pred_lddt, dim=1)
    return (msa[:, 0], pair, xyz_out, state, alpha_s[-1], logits_aa.permute(0, 2, 1), pred_lddt)


def _upstream_einsum():
    """upstream binds `einsum` to opt_einsum.contract (RoseTTAFoldModel.py / Track_module.py via util_module import *), NOT torch.einsum;
    the two pick different contraction kernels -> last-bit differences that the 36-block trunk amplifies to 1e-1. Use the very same symbol."""
    import rfdiffusion.Track_module as TM
    return TM.einsum


def _flatten(x):
    out = []
    for o in x:
        if isinstance(o, (tuple, list)):
            out.extend(_flatten(o))
        else:
            out.append(o)
    return out


# --------------------------------------------------------------------------------------------- graph objects
class _Generation:
    def __init__(self, key):
        self.key = key
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream()
        self.gA = None; self.gB = None
        self.failed = False
        self.n_calls = 0


def _static_clone(a):
    if torch.is_tensor(a):
        return a.clone()
    if isinstance(a, (tuple, list)):
        return type(a)(_static_clone(x) for x in a)
    return a


def _copy_into(buf, a):
    if torch.is_tensor(buf):
        buf.copy_(a)
    elif isinstance(buf, (tuple, list)):
        assert len(buf) == len(a)
        for b_, a_ in zip(buf, a):
            _copy_into(b_, a_)


class _Captured:
    """captured replay of fn(*args): tensors (also inside tuples/lists) become static input buffers refreshed by copy_ before
    every replay; other python objects (the model) are baked in."""
    def __init__(self, fn, args, gen):
        dev = next(a for a in args if torch.is_tensor(a)).device
        self.in_args = [_static_clone(a) for a in args]
        s = gen.stream
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            for _ in range(_cfg['warmup']):
                _ = fn(*self.in_args)
        torch.cuda.current_stream(dev).wait_stream(s)
        torch.cuda.synchronize(dev)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=gen.pool, stream=s):
            self.out = fn(*self.in_args)
        torch.cuda.synchronize(dev)
        self.out_flat = _flatten(self.out) if isinstance(self.out, tuple) else [self.out]

    def replay(self, args):
        for buf, a in zip(self.in_args, args):
            _copy_into(buf, a)
        self.graph.replay()
        _state['n_replay'] += 1
        return self.out          # STATIC buffers — caller must clone what leaves the module


def _clone_struct(x):
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, tuple):
        return tuple(_clone_struct(o) for o in x)
    if isinstance(x, list):
        return [_clone_struct(o) for o in x]
    return x


def reset_all():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gens = _state['generations']
    _state['generations'] = {}
    for g in list(gens.values()):
        g.gA = None; g.gB = None
    del gens
    try:
        import rfd_fastpath
        rfd_fastpath.reset_caches()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache()
    _state['n_resets'] += 1


def stats():
    gens = _state['generations']
    return dict(n_generations=len(gens), graphs_per_generation=[int(g.gA is not None) + int(g.gB is not None) for g in gens.values()],
                n_replay=_state['n_replay'], n_capture=_state['n_capture'], n_generations_created=_state['n_generations_created'],
                n_resets=_state['n_resets'], individual_graph_pops=_state['individual_graph_pops'], eager_fallback_calls=_state['eager_fallback_calls'],
                upstream_calls=_state['upstream_calls'], capture_errors=_state['capture_errors'][-3:], capture_s=_state['capture_s'][-4:],
                pool_reserved_gb=round(torch.cuda.memory_reserved() / 2**30, 2) if torch.cuda.is_available() else None)


def _gen_key(idx, motif_mask, cyclic_reses, msa_latent, t1d, t2d):
    B, N, L = msa_latent.shape[:3]
    mk = tuple(motif_mask.reshape(-1).tolist()) if torch.is_tensor(motif_mask) else motif_mask
    cyc = bool(cyclic_reses.any()) if torch.is_tensor(cyclic_reses) else bool(cyclic_reses)
    return (tuple(idx.reshape(-1).tolist()), int(B), int(N), int(L), mk, cyc, tuple(t1d.shape), tuple(t2d.shape))


def supported():
    try:
        import dgl
        major = int(str(dgl.__version__).split('.')[0])
    except Exception as e:
        return False, f'dgl import failed: {e}'
    if major < 2:
        return False, f'dgl {dgl.__version__} < 2.0: CUDA-graph capture of DGL message passing unsupported (needs dgl >= 2.0, as installed in the kit image)'
    if not torch.cuda.is_available():
        return False, 'no CUDA'
    return True, 'ok'


def apply(model):
    if hasattr(model, '_fg_orig_forward'):
        return True
    import time
    model._fg_orig_forward = model.forward

    def forward(msa_latent, msa_full, seq, xyz, idx, t, t1d=None, t2d=None, xyz_t=None, alpha_t=None,
                msa_prev=None, pair_prev=None, state_prev=None, return_raw=False, return_full=False, return_infer=False,
                use_checkpoint=False, motif_mask=None, i_cycle=None, n_cycle=None, cyclic_reses=None):
        if (not _state['enabled']) or (not return_infer) or return_raw or use_checkpoint or msa_prev is not None \
                or hasattr(model, 'timestep_embedder') or motif_mask is None:
            _state['upstream_calls'] += 1
            return model._fg_orig_forward(msa_latent, msa_full, seq, xyz, idx, t, t1d=t1d, t2d=t2d, xyz_t=xyz_t, alpha_t=alpha_t,
                                          msa_prev=msa_prev, pair_prev=pair_prev, state_prev=state_prev, return_raw=return_raw,
                                          return_full=return_full, return_infer=return_infer, use_checkpoint=use_checkpoint,
                                          motif_mask=motif_mask, i_cycle=i_cycle, n_cycle=n_cycle, cyclic_reses=cyclic_reses)
        key = _gen_key(idx, motif_mask, cyclic_reses, msa_latent, t1d, t2d)
        gens = _state['generations']
        gen = gens.get(key)
        if gen is None:
            if len(gens) >= _cfg['max_generations']:
                reset_all(); gens = _state['generations']
            gen = _Generation(key); gens[key] = gen; _state['n_generations_created'] += 1
        gen.n_calls += 1
        if gen.failed:
            _state['eager_fallback_calls'] += 1
            return model._fg_orig_forward(msa_latent, msa_full, seq, xyz, idx, t, t1d=t1d, t2d=t2d, xyz_t=xyz_t, alpha_t=alpha_t,
                                          return_infer=True, motif_mask=motif_mask, cyclic_reses=cyclic_reses)
        argsA = (model, msa_latent, msa_full, seq, xyz, idx, t1d, t2d, xyz_t, alpha_t, motif_mask, cyclic_reses)
        if gen.gA is None:
            # eager-first: this call's result is the plain upstream forward (computed BEFORE warm-up/capture touch anything)
            eager_out = model._fg_orig_forward(*[a.clone() if torch.is_tensor(a) else a for a in (msa_latent, msa_full, seq, xyz, idx, t)],
                                               t1d=t1d.clone(), t2d=t2d.clone(), xyz_t=xyz_t.clone(), alpha_t=alpha_t.clone(),
                                               return_infer=True, motif_mask=motif_mask, cyclic_reses=cyclic_reses)
            eager_out = tuple(o.clone() for o in eager_out)
            torch.cuda.synchronize()
            t0 = time.time()
            try:
                gen.gA = _Captured(lambda *a: _seg_A(*a), argsA, gen)
                _state['n_capture'] += 1
                if _cfg['graph_heads']:
                    a_out = gen.gA.replay(argsA)             # run once to get correctly-shaped inputs for R and B
                    msa, pair, state, R_in, T_in, xyz_in, R_s, T_s, A_s = a_out
                    state_r, R_r, T_r, A_r = _seg_R(model, msa, pair, state, R_in, T_in, xyz_in, idx, motif_mask, cyclic_reses)
                    argsB = (model, msa, pair, state_r, xyz, R_s + R_r, T_s + T_r, A_s + A_r)
                    gen.gB = _Captured(lambda *a: _seg_B(*a), argsB, gen)
                    _state['n_capture'] += 1
            except Exception as e:
                if is_oom(e): raise
                _state['capture_errors'].append(f'{type(e).__name__}: {str(e)[:300]}')
                gen.failed = True; gen.gA = None; gen.gB = None
            torch.cuda.synchronize()
            _state['capture_s'].append(round(time.time() - t0, 2))
            return eager_out
        # ---- replay path ----
        a_out = gen.gA.replay(argsA)
        msa, pair, state, R_in, T_in, xyz_in, R_s, T_s, A_s = a_out
        # refinement (eager) reads graph-A's static output buffers; they are only overwritten by the next replay of graph A,
        # which is issued later on the same stream -> ordered.  It returns fresh tensors.
        state_r, R_r, T_r, A_r = _seg_R(model, msa, pair, state, R_in, T_in, xyz_in, idx, motif_mask, cyclic_reses)
        if gen.gB is not None:
            out = gen.gB.replay((model, msa, pair, state_r, xyz, R_s + R_r, T_s + T_r, A_s + A_r))
            out = tuple(o.clone() for o in out)
        else:
            out = _seg_B(model, msa, pair, state_r, xyz, R_s + R_r, T_s + T_r, A_s + A_r)
            out = tuple(o.clone() for o in out)          # msa[:,0] / pair are views of graph-A static buffers -> clone
        return out

    model.forward = forward
    return True
