"""forward_impl.py — the per-item forward composition, importable (forward.py runs as a script/__main__, so functions defined there can
never be wrapped by an import hook; anything a timing/instrumentation layer needs to patch by module:qualname lives here instead).
Moved verbatim from forward.py: same statements, same order, same tensors — forward.py now imports these names rather than defining them.
"""
import os
import sys
import time


def _load_rowpair_xfold():
    """The rowpair install module beside this file (rowpair_xfold.py), loaded by path like this script (the wrapper never imports it)."""
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rowpair_xfold.py")
    spec = importlib.util.spec_from_file_location("rowpair_xfold", p)
    m = importlib.util.module_from_spec(spec); sys.modules["rowpair_xfold"] = m; spec.loader.exec_module(m)
    return m

def _reset_item_state(model, torch):
    """Before an item: drop the diffusion head's static buffers and whole-step CUDA graph (xfold's DiffusionHead.clear_static) and the
    kernel kit's mask-term cache (af3_kernels.clear_caches). Returns whether a whole-step graph was dropped (0|1)."""
    dh = getattr(model, "diffusion_head", None)
    st = getattr(dh, "_static", None)
    had_graph = int(isinstance(st, dict) and st.get("graph") is not None)
    if dh is not None and hasattr(dh, "clear_static"):
        dh.clear_static()
    elif dh is not None:
        dh._static = None
    K = getattr(model, "_af3t_kernels", None)
    if K is not None and hasattr(K, "clear_caches"):
        K.clear_caches()
    return had_graph

def _forward_samples(A, model, batch, seed, torch, diff_free=False, item=None, prev_free=False, rpx=None):
    """AF3 ``Model.__call__``'s composition over ``S = model.num_samples`` diffusion samples, from the kit's own public steps: the trunk once,
    the sampler once (S samples, the torch RNG seeded), the confidence head ONCE PER SAMPLE (stacked to a leading sample axis [S, ...]),
    the distogram once. The kit's own ``forward`` runs the confidence head on sample 0 only (``af3_torch_api.py:219-226``, "1 diffusion
    sample" by its docstring) — with S > 1 postprocess would refuse its single-sample confidence arrays by name. At S = 1 the calls are
    the kit's own forward's, in its order, on the same tensors: byte-identical outputs."""
    phases = {}
    t_wall = time.time()                                                      # single GPU too: each stage's own peak allocation -> item['phases'] (a census: the peak counter is reset per stage, nothing computed changes)
    t_trunk = _clock(torch)                                                   # phase_s (the PHASE line): trunk = A.run_trunk's span (target-feat embedding + every Evoformer recycle) on every arm
    if rpx is None and not prev_free:
        emb = A.run_trunk(model, batch)                                       # n_gpu = 1 without prev_free: the kit api's own trunk, untouched
    elif rpx is None:
        emb = _run_trunk_prev_free(A, model, batch)                           # n_gpu = 1, prev_free: the same loop, each pass's prev released once embedded
        if item is not None:
            item["prev_free"] = 1
    else:                                                                     # n_gpu > 1: the trunk on this rank's rows; the pair STAYS this rank's row shard through every head
        if item is not None:                                                  # R1: the trunk's MSA row shuffle and the sampler draw from the torch RNG seeded per item above — the generators
            item.setdefault("guards", {})["rng_state"] = rpx.assert_rng_replicated(torch)   # must be bitwise identical on every rank (refused by name otherwise), or the ranks fold a chimera
        lay = rpx.layout()
        t_st = time.time()
        emb = rpx.run_trunk_sharded_xfold(A, model, batch, lay)              # {'pair': [R, N, 128] fp32 rows r0:r1, 'single', 'target_feat'} on every rank
        if item is not None and prev_free:
            item["prev_free"] = 1; item["prev_free_mode"] = "rowpair_carry"   # the driver carries z_prev as the shard and reads each prev block once (trunk.recycle_shard_)
        torch.cuda.synchronize(); rpx._mark("trunk_done"); phases["trunk"] = _stage(torch, t_st)      # per-stage {peak_gb (stage-local), wall_s} + the core census mark
        rpx.stage_boundary("trunk_done")                                      # every rank's stage work complete + the ranks meet (census boundary_sync=barrier)
        t_st = time.time()                                                    # the distogram head FIRST among the heads under n_gpu > 1: it is the trunk pair's last DENSE reader (row blocks
        dg = rpx.run_distogram_sharded(model, batch, emb, lay)               # AND column slabs of the device shard); after it the fp32 trunk shard is read row-block-wise only (roll-out
        torch.cuda.synchronize(); phases["distogram"] = _stage(torch, t_st)  # conditioning, per-sample confidence embeds) and lives parked on the host from the roll-out entry to its last
                                                                              # embed (rowpair_xfold: heads.ZTrunkPlan park / retire). A pure function of the trunk pair: same outputs.
    if rpx is None:
        t_wall = _census(torch, phases, "trunk", t_wall)
        if diff_free:                                                         # big's diff_free at the trunk boundary: the TriMul provider-face caches the trunk left (the native row's kept
            K = getattr(model, "_af3t_kernels", None)                         # small-N workspaces, descriptors, the LRU weight pack: 215 MiB at 448 tokens, transient above ~700) emptied --
            if K is not None and hasattr(K, "release_face_workspaces"):       # the sampler and the heads read none of it; frees only
                mb, n_geo, n_mod = K.release_face_workspaces(model)
                if item is not None:                                          # on the ITEM line as diff_freed_trimul_mib= and in forward.json
                    item["diff_freed_trimul_mib"] = round(mb, 1); item["diff_freed_trimul"] = {"geometries": n_geo, "modules": n_mod}
    t_sampler = _clock(torch)                                                 # trunk done; sampler = A.run_diffusion's span (every step of every sample)
    if rpx is None:
        xyz = A.run_diffusion(model, batch, emb, seed=seed)                   # [S, N, 24, 3]
    else:
        t_st = time.time()
        xyz = rpx.run_diffusion_sharded(model, batch, emb, lay, seed)        # [S, N, 24, 3] replicated (z_cond rows + DiT local query rows; positions proven identical across ranks)
        torch.cuda.synchronize(); rpx._mark("diffusion_done"); phases["diffusion"] = _stage(torch, t_st)
        rpx.stage_boundary("diffusion_done")                                  # the roll-out is complete on every rank and the ranks have met before the heads post their first
                                                                              # point-to-point group (census boundary_sync=barrier, boundaries=[trunk_done, diffusion_done])
    if rpx is None:
        t_wall = _census(torch, phases, "sampler", t_wall)
    t_sampled = _clock(torch)                                                 # sampler done (big's diff_free eviction below is in no phase)
    if diff_free:                                                             # big's diff_free: the diffusion statics (hoist caches, a whole-step graph, the mask-term cache) released
        freed_graph = _reset_item_state(model, torch)                         # before the heads, then the allocator's cached blocks returned; frees only, the heads read none of it
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        if item is not None:
            item["diff_free"] = 1; item["diff_freed"] = {"step_graph": freed_graph}
    t_conf = _clock(torch)                                                    # conf = the confidence head over every sample (A.run_confidence once per sample), summed; the distogram head after it is in no phase
    if rpx is None:
        confs = [A.run_confidence(model, batch, emb, xyz[s]) for s in range(int(xyz.shape[0]))]
    else:                                                                     # COLLECTIVE per sample on every rank; rank 0 receives the sample's dict, its [N, N] matrices already on the
        confs = []                                                            # HOST (conf_full_matrices_rank0: assembled there column block by column block); ranks > 0 receive None
        t_st = time.time()
        for s in range(int(xyz.shape[0])):
            c = rpx.run_confidence_sharded(model, batch, emb, lay, xyz[s])
            confs.append(None if c is None else {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in c.items()})
    t_heads = _clock(torch)
    if item is not None:                                                      # the item's phase walls (s), one record per (input, seed) on every arm and rank: report.PHASES / the PHASE line
        item["phase_s"] = {"lm": None, "trunk": round(t_sampler - t_trunk, 3), "sampler": round(t_sampled - t_sampler, 3), "conf": round(t_heads - t_conf, 3)}
    if rpx is None:
        dg = A.run_distogram(model, batch, emb)
        _census(torch, phases, "heads", t_wall)
        if phases:
            print("[forward] STAGES " + " ".join(f"{k}:peak={v['peak_gb']}GB,wall={v['wall_s']}s" for k, v in phases.items()), flush=True)
            if item is not None:
                item["phases"] = phases
    else:                                                                     # dg (rank 0: {'bin_edges', 'contact_probs' [N, N] on the host}; ranks > 0: None) was computed after the trunk
        torch.cuda.synchronize(); rpx._mark("heads_done"); phases["heads"] = _stage(torch, t_st)
        print("[forward] STAGES rank=%d %s boundary_sync=%s boundaries=%s" % (rpx.STATE["rank"], " ".join(f"{k}:peak={v['peak_gb']}GB,wall={v['wall_s']}s" for k, v in phases.items()),
                                                                                rpx.STATE["stats"].get("boundary_sync"), "+".join(rpx.STATE["stats"].get("boundaries") or [])), flush=True)
        if item is not None:
            item["phases"] = phases; item["rowpair_census"] = rpx.census()
        if rpx.STATE["rank"] != 0:                                            # ranks > 0 took part in every collective; rank 0 holds the outputs
            return None
        dg = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in dg.items()}
    conf = {k: (torch.stack([c[k] for c in confs]) if torch.is_tensor(confs[0][k]) else [c[k] for c in confs]) for k in confs[0]}
    return dict(embeddings=emb, atom_positions=xyz, confidence=conf, distogram=dg)

def _clock(torch):
    """A phase boundary of the item's ``phase_s`` record (the PHASE line): the device's queued work drained, then the wall clock —
    ``torch.cuda.synchronize()`` + ``time.perf_counter()``. A synchronize changes no computed value."""
    torch.cuda.synchronize()
    return time.perf_counter()

def _census(torch, phases, name, t_wall):
    """The single-GPU path's stage census: the stage's own peak allocation and wall into ``phases[name]`` (as the ranks record theirs);
    returns the next stage's start. A census only: the peak counter is reset per stage, nothing computed changes. A torch without the
    CUDA memory counters records nothing."""
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not hasattr(cuda, "max_memory_allocated") or not hasattr(cuda, "reset_peak_memory_stats"):
        return time.time()
    cuda.synchronize()
    phases[name] = _stage(torch, t_wall)
    return time.time()

def _stage(torch, t_start):
    """One stage's census under n_gpu > 1: the stage's OWN peak allocation on this rank (GiB; the peak counter is reset for the next stage)
    and the stage's wall (s). The item's ``peak_gb`` is the max over its stages."""
    rec = {"peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3), "wall_s": round(time.time() - t_start, 2)}
    torch.cuda.reset_peak_memory_stats()
    return rec

def _run_trunk_prev_free(A, model, batch):
    """``A.run_trunk`` (af3_torch_api.py run_trunk, the source the kit's tests lock) with big's prev_free: each recycle hands the Evoformer
    its prev embeddings through a consume-once mapping, so the fp32 prev pair / single are released as soon as they are embedded (the pass
    reads each exactly once: xfold/alphafold3.py Evoformer.forward). Same statements, same order, same tensors otherwise."""
    rpx = _load_rowpair_xfold() if "rowpair_xfold" not in sys.modules else sys.modules["rowpair_xfold"]
    b = A._B(model, batch)
    n_iter = model.num_recycles + 1
    tf = A.run_target_feat(model, b)
    N = tf.shape[0]
    import torch
    emb = {"pair": torch.zeros(N, N, model.evoformer.pair_channel, device=tf.device), "single": torch.zeros(N, model.evoformer.seq_channel, device=tf.device), "target_feat": tf}
    for _ in range(n_iter):
        e = model.evoformer(batch=b, prev=rpx.ConsumeOnce(emb), target_feat=tf)
        emb = {"pair": e["pair"].float(), "single": e["single"].float(), "target_feat": tf}
        del e
    return emb
