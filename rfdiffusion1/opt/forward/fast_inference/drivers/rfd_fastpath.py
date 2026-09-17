"""
rfd_fastpath — code-level inference levers for RFdiffusion v1 (commit 86507b65), applied as runtime patches.

Every patch keeps the *same tensor operations on the same values* as upstream and only removes
(a) host<->device round trips whose result is a per-trajectory constant, and
(b) re-derivations of per-trajectory constants inside the 40 structure blocks x 50 steps.
Nothing here changes weights, inputs, step count, schedule, top-k, or the RNG stream.

Levers (each individually switchable so equality can be bisected):
  C1a cache_chain_breaks : PositionalEncoding2D.forward and util_module.get_seqsep call
                           find_breaks(idx.cpu().numpy()) -> numpy -> torch.from_numpy(...).to(device)
                           on EVERY call (37 + 40 calls per step). idx is constant for a trajectory, so the
                           chain-id tensor is memoised per (idx values, device). Exact: equal integer tensor.
  C1b cache_full_graph   : make_full_graph rebuilds the all-pairs (|i-j|>0) DGL graph 36x per step from
                           torch.where(sep.abs()>0) (device->host sync for the nnz count) + dgl.graph().
                           The topology depends only on idx (constant), so the graph structure and the (b,i,j)
                           index tensors are memoised; per call we only recompute the data that depends on xyz/pair:
                           G.edata['rel_pos'] = xyz[b,j]-xyz[b,i] and pair[b,i,j]. Exact: same index tensors,
                           same gather ops. (make_topk_graph in the 4 refinement blocks depends on xyz -> untouched.)
  C1c t_tensor_on_device : `t=torch.tensor(t)` is created on CPU each step and only used by the (absent)
                           timestep embedder; untouched (no cost). Listed for completeness.
  C1d rbf_linspace_cache : util_module.rbf builds torch.linspace(0,20,36).to(device) on every call (76/step);
                           memoise the constant per device. Exact: equal constant tensor.
  C1e cdist / misc       : untouched (real compute).

Use:  import rfd_fastpath; rfd_fastpath.apply(levers={'chain_breaks','full_graph','rbf'})
      rfd_fastpath.reset_caches()  # call between trajectories (automatically keyed anyway)
"""
import torch, numpy as np
import dgl

_APPLIED = set()
_caches = dict(chain=dict(), graph=dict(), rbf=dict())


def reset_caches():
    """drop all memoised per-trajectory constants. NOTE: when rfd_fullgraph has live captured graphs, call rfd_fullgraph.reset_all()
    instead (it drops the graphs first, then calls this) - captured graphs hold device pointers into these cached tensors."""
    for v in _caches.values():
        v.clear()
    _LAST_IDX[0] = None; _LAST_IDX[1] = None
    _CYC.clear()


_CYC = {}   # id(tensor) -> (tensor strong ref, bool)


_LAST_IDX = [None, None]   # [tensor object (strong ref), values key] - identity-keyed single slot (no address reuse possible while we hold the ref)


def _idx_key(idx):
    # idx: (B, L) long tensor on device -> hashable VALUES key. One D->H copy per new idx tensor OBJECT (upstream creates one
    # per denoising step); identity (not data_ptr) is used for the shortcut and the tensor is kept alive by this slot, so a
    # different tensor can never alias the cached key (address-reuse hazard of pointer-keyed caches).
    if _LAST_IDX[0] is idx:
        return _LAST_IDX[1]
    key = (tuple(idx.reshape(-1).tolist()), str(idx.device))
    _LAST_IDX[0] = idx; _LAST_IDX[1] = key
    return key


def apply(levers=('chain_breaks', 'full_graph', 'rbf')):
    import rfdiffusion.util_module as UM
    import rfdiffusion.Embeddings as EM
    import rfdiffusion.Track_module as TM
    levers = set(levers)

    # ---------------- C1a: chain-break / chain-id memoisation ----------------
    if 'chain_breaks' in levers and 'chain_breaks' not in _APPLIED:
        find_breaks = UM.find_breaks

        def chainids_for(idx, thresh=None):
            """chain ids (L,) from residue index; upstream calls find_breaks(idx.squeeze()) which assumes B==1.
            For B>1 (batched same-contig trajectories) all rows of idx are equal by construction (asserted once
            per new idx tensor), so row 0 is used — equal result to the B==1 path for every row."""
            key = ('cid', thresh) + _idx_key(idx)
            ent = _caches['chain'].get(key)
            if ent is None:
                if idx.dim() == 2 and idx.shape[0] > 1:
                    assert bool((idx == idx[:1]).all()), "batched fastpath requires identical idx rows"
                    ix = idx[0].cpu().numpy()
                else:
                    ix = idx.squeeze().cpu().numpy()
                breaks = find_breaks(ix, thresh=thresh) if thresh is not None else find_breaks(ix)
                chainids = np.zeros_like(ix)
                for i, b in enumerate(breaks):
                    chainids[b:] = i + 1
                ent = torch.from_numpy(chainids).to(device=idx.device)
                _caches['chain'][key] = ent
            return ent

        def _cyc_active(cyc):
            # upstream's cyclic loops are no-ops when no residue is cyclised; skipping them is exact and avoids
            # B==1-only indexing (and a torch.unique host sync per call). Identity-keyed with a strong ref (bounded).
            if cyc is None:
                return False
            ent = _CYC.get(id(cyc))
            if ent is not None and ent[0] is cyc:
                return ent[1]
            v = bool(cyc.any())
            if len(_CYC) > 64:
                _CYC.clear()
            _CYC[id(cyc)] = (cyc, v)
            return v

        # PositionalEncoding2D.forward — equal math, chainids from cache
        def pe_forward(self, x, idx, cyclize=None):
            bins = torch.arange(self.minpos, self.maxpos, device=x.device)
            seqsep = idx[:, None, :] - idx[:, :, None]  # (B, L, L)
            chainids = chainids_for(idx, thresh=35)
            if _cyc_active(cyclize):
                for chid in torch.unique(chainids):
                    is_chid = chainids == chid
                    cur_cyclize = cyclize * is_chid
                    cur_mask = cur_cyclize[:, None] * cur_cyclize[None, :]
                    cur_ncyc = torch.sum(cur_cyclize)
                    seqsep[:, cur_mask * (seqsep[0] > cur_ncyc // 2)] -= cur_ncyc
                    seqsep[:, cur_mask * (seqsep[0] < -cur_ncyc // 2)] += cur_ncyc
            ib = torch.bucketize(seqsep, bins).long()
            emb = self.emb(ib)
            x = x + emb
            return self.drop(x)
        EM.PositionalEncoding2D.forward = pe_forward

        def get_seqsep(idx, cyclic=None):
            seqsep = idx[:, None, :] - idx[:, :, None]
            sign = torch.sign(seqsep)
            neigh = torch.abs(seqsep)
            neigh[neigh > 1] = 0.0
            neigh = sign * neigh
            chainids = chainids_for(idx)
            if _cyc_active(cyclic):
                for chid in torch.unique(chainids):
                    is_chid = chainids == chid
                    cur_cyclic = cyclic * is_chid
                    cur_cres = cur_cyclic.nonzero()
                    if cur_cyclic.sum() >= 2:
                        neigh[:, cur_cres[-1], cur_cres[0]] = 1
                        neigh[:, cur_cres[0], cur_cres[-1]] = -1
            return neigh.unsqueeze(-1)
        UM.get_seqsep = get_seqsep
        TM.get_seqsep = get_seqsep
        _APPLIED.add('chain_breaks')

    # ---------------- C1b: constant full-graph topology ----------------
    if 'full_graph' in levers and 'full_graph' not in _APPLIED:
        def make_full_graph_cached(xyz, pair, idx, top_k=64, kmin=9):
            B, L = xyz.shape[:2]
            device = xyz.device
            key = ('full',) + _idx_key(idx) + (B, L)
            ent = _caches['graph'].get(key)
            if ent is None:
                sep = idx[:, None, :] - idx[:, :, None]
                b, i, j = torch.where(sep.abs() > 0)
                src = b * L + i
                tgt = b * L + j
                G0 = dgl.graph((src, tgt), num_nodes=B * L).to(device)
                ent = (G0, b, i, j)
                _caches['graph'][key] = ent
            G0, b, i, j = ent
            # The SE(3) layers only read G.edges() and G.edata['rel_pos'] synchronously inside this Str2Str call,
            # so the cached graph object is reused and rel_pos overwritten per call (same stream order as upstream).
            G = G0
            G.edata['rel_pos'] = (xyz[b, j, :] - xyz[b, i, :]).detach()
            return G, pair[b, i, j][..., None]
        UM.make_full_graph = make_full_graph_cached
        TM.make_full_graph = make_full_graph_cached
        _APPLIED.add('full_graph')

    # ---------------- C1d: rbf linspace constant ----------------
    if 'rbf' in levers and 'rbf' not in _APPLIED:
        def rbf(D):
            D_min, D_max, D_count = 0., 20., 36
            key = str(D.device)
            D_mu = _caches['rbf'].get(key)
            if D_mu is None:
                D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)[None, :]
                _caches['rbf'][key] = D_mu
            D_sigma = (D_max - D_min) / D_count
            D_expand = torch.unsqueeze(D, -1)
            RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
            return RBF
        UM.rbf = rbf
        TM.rbf = rbf
        EM.rbf = rbf
        _APPLIED.add('rbf')


    # ---------------- C1f: capture-safe MSAPairStr2MSA (needed for CUDA-graph capture, W1; also removes 36 tiny H2D copies/step) ----------------
    if 'msa_index' in levers and 'msa_index' not in _APPLIED:
        # upstream: msa = msa.index_add(1, torch.tensor([0,], device=state.device), state)
        # torch.tensor([0]) from a python list is a pageable host->device copy each call (illegal inside CUDA-graph capture).
        # We cache the constant index tensor per device. index_add with the same index tensor is the equal op.
        _idx0 = {}
        def msa2msa_forward(self, msa, pair, rbf_feat, state):
            B, N, L = msa.shape[:3]
            pair = self.norm_pair(pair)
            pair = torch.cat((pair, rbf_feat), dim=-1)
            pair = self.proj_pair(pair)
            state = self.norm_state(state)
            state = self.proj_state(state).reshape(B, 1, L, -1)
            key = str(state.device)
            if key not in _idx0:
                _idx0[key] = torch.tensor([0, ], device=state.device)
            msa = msa.index_add(1, _idx0[key], state)
            msa = msa + self.drop_row(self.row_attn(msa, pair))
            msa = msa + self.col_attn(msa)
            msa = msa + self.ff(msa)
            return msa
        TM.MSAPairStr2MSA.forward = msa2msa_forward
        _APPLIED.add('msa_index')

    return sorted(_APPLIED)
