"""opt_core.mem.rowpair.confidence — K.T33: ``FastContextReducer`` == ``ContextReducer`` BITWISE, the ``ROWPAIR_CONF_REDUCER`` factory, the
label-vector context helpers, and the per-rank CPU-thread lever (``dist.cpu_threads`` / ``launch.rank_env(cpu_threads=)`` /
``ROWPAIR_RANK_THREADS``).

What is asserted (fp32 CPU; P in {2, 3} rank-THREADS on the ``threaded`` backend — :func:`opt_core.testing.run_ranks` — plus two multi-PROCESS
gloo launches through the launcher): for the OF3-kit context grammar (``key / mask / w / num / asym`` + the new ``chains``, contexts
FULL + per chain + per chain PAIR, chains in {1, 3, 7}, contiguous and interleaved chain order, masked / excluded tokens, uneven grid and
aligned layouts, S in {1, 2}) and for the core grammar (``ChainIndex.contexts`` / ``contexts_lite``), in the ``block``, ``sub`` and ``auto``
statement forms: ``n``, ``counts``, ``local_rows``, every block's ``pieces`` (rows and values), every gathered context matrix (values AND
strides) and the kit's finished pTM / ipTM per context (masked finish on the classic matrix vs index finish on the fast one) are IDENTICAL
(``torch.equal``); the fast reducer uses fewer collectives; ``both`` reports 0 mismatches and names ``conf_reducer_mismatch=0``; the factory
honours the lever and refuses unknown words / unstructured contexts / n_gpu=1 by name; ``ptm_contexts_from_labels`` reproduces the masked
statements' keys (types included), sizes, weights and weight sharing; ``rank_env`` is byte-identical to today unless the thread lever is set,
and a ``run_sharded`` worker applies and names the cap.

Run: ``python -m pytest tests/test_rowpair_confidence_fast_05213.py -q -rfE`` or ``python tests/test_rowpair_confidence_fast_05213.py``.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                      # the opt_core checkout under test
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from opt_core.mem.rowpair import RowpairRefused, launch  # noqa: E402
from opt_core.mem.rowpair.dist import Layout  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False

try:
    import pytest
    needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch not importable")
    parametrize = pytest.mark.parametrize
except Exception:  # noqa: BLE001
    def needs_torch(fn):
        return fn

    def parametrize(names, values):
        def deco(fn):
            fn.pytestmark = [type("M", (), {"name": "parametrize", "args": (names, values)})()]
            return fn
        return deco

BINS = 64
CFG_PTM = {"bin_min": 0.0, "bin_max": 32.0, "no_bins": BINS}


# ================================================================================================================ the KIT grammar (the OF3 kit's tp_rowpair/confidence.py statements, engine-free copies)
def get_bin_centers(bin_min, bin_max, no_bins, device, dtype):
    import torch
    width = (bin_max - bin_min) / no_bins
    return torch.linspace(bin_min, bin_max - width, no_bins, device=device, dtype=dtype) + 0.5 * width


class PTMContext(object):
    """The kit's context (``key mask w num asym``) + the NEW optional ``chains`` (label tuple; None = full)."""

    def __init__(self, key, mask, w, num, asym, chains=None):
        self.key, self.mask, self.w, self.num, self.asym, self.chains = key, mask, w, num, asym, chains


def _bin_weight(num_tokens_considered, device, dtype):
    import torch
    clipped = torch.maximum(num_tokens_considered, torch.tensor(19.0, device=device, dtype=dtype))
    d0 = 1.24 * (clipped - 15.0).clamp_min(0).pow(1.0 / 3.0) - 1.8
    bin_centers = get_bin_centers(CFG_PTM["bin_min"], CFG_PTM["bin_max"], CFG_PTM["no_bins"], device, dtype)
    return 1.0 / (1.0 + (bin_centers / d0) ** 2)


def ptm_contexts_masked(batch_b, device, dtype, want_chain_ptm=True, want_chain_pair=True):
    """The kit's ``ptm_contexts`` statements (one [N] bool mask + one .item() per context)."""
    import torch
    token_mask, asym_id = batch_b["token_mask"], batch_b["asym_id"]
    w_by_n = {}

    def make(key, mask_i, asym):
        mask_i = mask_i.to(device=device, dtype=torch.bool)
        num_tokens_considered = mask_i.sum().clamp_min(1).to(dtype)
        n = int(mask_i.sum().item())
        if n not in w_by_n:
            w_by_n[n] = _bin_weight(num_tokens_considered, device, dtype)
        return PTMContext(key, mask_i, w_by_n[n], num_tokens_considered, asym)

    ctxs = [make("full", token_mask.bool(), asym_id)]
    unique_chains = None
    if want_chain_ptm:
        for aid in asym_id.unique():
            ctxs.append(make(("chain", aid.item()), token_mask.bool() & (asym_id == aid), None))
    if want_chain_pair:
        asym_l = asym_id.long()
        unique_chains = torch.unique(asym_l).tolist()
        chain_masks = [(asym_l == aid) & token_mask.bool() for aid in unique_chains]
        for i in range(len(unique_chains)):
            for j in range(i + 1, len(unique_chains)):
                ctxs.append(make(("pair", i, j), chain_masks[i] | chain_masks[j], asym_l))
    return ctxs, unique_chains


def ptm_contexts_fast(batch_b, device, dtype, want_chain_ptm=True, want_chain_pair=True, masks=False, reducer=None):
    """The kit form over the CORE helper: the engine's factory keeps its own bin_weight / num statements; no per-context sync; per-context
    masks only when the reducer in force needs them (classic / both)."""
    import torch
    from opt_core.mem.rowpair.confidence import ptm_contexts_from_labels
    token_mask, asym_id = batch_b["token_mask"], batch_b["asym_id"]
    asym_l = asym_id.long()
    w_by_n = {}

    def make(key, n, chains, mask):
        num_tokens_considered = torch.tensor(int(n), dtype=torch.int64, device=device).clamp_min(1).to(dtype)
        if n not in w_by_n:
            w_by_n[n] = _bin_weight(num_tokens_considered, device, dtype)
        asym = asym_id if key == "full" else (asym_l if chains is not None and len(chains) == 2 else None)
        return PTMContext(key, mask, w_by_n[n], num_tokens_considered, asym, chains=chains)

    ctxs, unique_chains, labels, n_by_chain = ptm_contexts_from_labels(asym_id, token_mask, make, want_chain_ptm=want_chain_ptm, want_chain_pair=want_chain_pair,
                                                                        masks=masks, reducer=reducer)
    return ctxs, unique_chains, labels, n_by_chain


def finish_context_masked(ctx, ptm_ij, has_frame, interface, eps=1e-8):
    """The kit's ``finish_context``: two boolean-mask selections per context."""
    has_frame = has_frame[:, ctx.mask.to(has_frame.device)].bool()
    if interface:
        asym_id = ctx.asym.to(ptm_ij.device)[ctx.mask.to(ptm_ij.device)]
        pair_mask = asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)
        tm_i = (ptm_ij * pair_mask).sum(dim=-1) / pair_mask.sum(dim=-1).clamp_min(eps)
    else:
        tm_i = ptm_ij.sum(dim=-1) / ctx.num
    tm_i = tm_i.masked_fill(~has_frame.to(tm_i.device), 0.0)
    return tm_i.max(dim=-1).values


def finish_context_idx(ctx, ptm_ij, has_frame, interface, idx, asym_full, eps=1e-8):
    """The kit's NEW ``finish_context_idx``: the same statements over the context's ascending index (index_select: no sync)."""
    has_frame = has_frame.index_select(1, idx.to(has_frame.device)).bool()
    if interface:
        asym_id = asym_full.to(ptm_ij.device).index_select(0, idx.to(ptm_ij.device))
        pair_mask = asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)
        tm_i = (ptm_ij * pair_mask).sum(dim=-1) / pair_mask.sum(dim=-1).clamp_min(eps)
    else:
        tm_i = ptm_ij.sum(dim=-1) / ctx.num
    tm_i = tm_i.masked_fill(~has_frame.to(tm_i.device), 0.0)
    return tm_i.max(dim=-1).values


# ================================================================================================================ synthetic problems
def _problem(N, C, S, seed, order="blocks", holes=False, asym_dtype="float32"):
    """asym_id [N] (chain ids 1..C, uneven sizes; ``interleaved`` = chains cut in segments and alternated, so a row block holds several label
    runs), token_mask [N] float (``holes``: a few EXCLUDED tokens inside chains and at the end), has_frame [S, N] bool."""
    import torch
    g = torch.Generator().manual_seed(int(seed) + 31 * N + 7 * C)
    cuts = sorted(torch.randperm(N - 1, generator=g)[:C - 1].add(1).tolist()) if C > 1 else []
    sizes = [b - a for a, b in zip([0] + cuts, cuts + [N])]
    if order == "interleaved" and C > 1:
        segs = []
        for c, n in enumerate(sizes):
            k = max(1, n // 3)
            parts = [k, k, n - 2 * k] if n - 2 * k > 0 else [n]
            segs.extend((c + 1, p) for p in parts if p > 0)
        perm = torch.randperm(len(segs), generator=g).tolist()
        asym = torch.cat([torch.full((segs[i][1],), segs[i][0], dtype=torch.long) for i in perm])
    else:
        asym = torch.cat([torch.full((n,), c + 1, dtype=torch.long) for c, n in enumerate(sizes)])
    assert int(asym.numel()) == N
    token_mask = torch.ones(N, dtype=torch.float32)
    if holes:
        token_mask[-2:] = 0.0
        token_mask[N // 3] = 0.0
        token_mask[N // 2: N // 2 + 3] = 0.0
    has_frame = torch.rand((S, N), generator=g) > 0.2
    asym_t = asym.to(torch.float32) if asym_dtype == "float32" else asym
    return {"asym_id": asym_t, "token_mask": token_mask, "has_frame": has_frame, "sizes": sizes}


def _probs(S, rows, N, seed):
    import torch
    g = torch.Generator().manual_seed(int(seed))
    return torch.softmax(torch.randn((S, rows, N, BINS), generator=g) * 2.0, dim=-1)


def _layout(N, P, rank, kind):
    return Layout(N, P, rank, B=8, align=8) if kind == "aligned" else Layout(N, P, rank, B=16)


# ================================================================================================================ one rank: classic vs fast side by side
def _compare_rank(rank, P, N, C, S, rows, form, order, holes, layout_kind, grammar, seed=7):
    import torch
    from opt_core.mem.rowpair import confidence as CF
    from opt_core.mem.rowpair import shard
    from opt_core.mem.rowpair.dist import world
    if rank is None:
        P, rank = world()
    dev = torch.device("cpu")
    lay = _layout(N, P, rank, layout_kind)
    prob = _problem(N, C, S, seed, order, holes)
    batch_b = {"token_mask": prob["token_mask"], "asym_id": prob["asym_id"]}
    hf = prob["has_frame"]
    asym_l = prob["asym_id"].long()
    ck = {}
    if grammar == "kit":
        ctxs_old, uc = ptm_contexts_masked(batch_b, dev, torch.float32)
        ctxs_new, uc2, labels, n_by = ptm_contexts_fast(batch_b, dev, torch.float32)
        ck["unique_chains_same"] = uc == uc2
        ck["n_ctx"] = len(ctxs_old) == len(ctxs_new) == 1 + C + C * (C - 1) // 2
        ck["keys_same_typed"] = all(a.key == b.key and [type(x) for x in (a.key if isinstance(a.key, tuple) else (a.key,))] ==
                                    [type(x) for x in (b.key if isinstance(b.key, tuple) else (b.key,))] for a, b in zip(ctxs_old, ctxs_new))
        ck["w_num_same"] = all(torch.equal(a.w, b.w) and torch.equal(a.num, b.num) and a.num.dtype == b.num.dtype for a, b in zip(ctxs_old, ctxs_new))
        share_old = [[i for i, c in enumerate(ctxs_old) if c.w is d.w] for d in ctxs_old]
        share_new = [[i for i, c in enumerate(ctxs_new) if c.w is d.w] for d in ctxs_new]
        ck["w_sharing_same"] = share_old == share_new
    else:                                                                    # the core grammar (the RowBlockReducer-route kits)
        chains = CF.ChainIndex(prob["asym_id"].long(), hf[0])
        cen = CF.tm_bin_centers(0.0, 32.0, BINS)
        ctxs_old = chains.contexts(cen, dev)
        ctxs_new = chains.contexts_lite(cen, dev) if grammar == "core_lite" else ctxs_old
        labels = chains.asym
        ck["n_ctx"] = len(ctxs_old) == len(ctxs_new)
        ck["lite_same"] = all(a.kind == b.kind and a.a == b.a and a.b == b.b and a.N_d == b.N_d and torch.equal(a.w, b.w) for a, b in zip(ctxs_old, ctxs_new))
    red_old = CF.ContextReducer.for_layout(ctxs_old, lay, form=form)
    red_new = CF.FastContextReducer.for_layout(ctxs_new, lay, form=form, labels=labels, gather_bytes=64 << 10, cpu_threads=None)
    K = len(ctxs_old)
    ck["n_same"] = all(red_old.n(k) == red_new.n(k) for k in range(K))
    ck["counts_same"] = all(red_old.counts(k) == red_new.counts(k) for k in range(K))
    ck["local_rows_same"] = all(red_old.local_rows(k) == red_new.local_rows(k) for k in range(K))
    pieces_bit, order_same, nblocks = True, True, 0
    for bi, (i0, i1, g0, g1) in enumerate(shard.iter_row_blocks(lay, rows, unit=1)):
        probs = _probs(S, i1 - i0, N, seed * 13 + rank * 1000 + bi)
        po = [(k, r, T.clone()) for k, r, T in red_old.pieces(i0, i1, probs)]
        pn = [(k, r, T.clone()) for k, r, T in red_new.pieces(i0, i1, probs)]
        order_same = order_same and [k for k, _r, _t in po] == [k for k, _r, _t in pn]
        dn = {k: (r, T) for k, r, T in pn}
        pieces_bit = pieces_bit and len(po) == len(pn) and all(k in dn and dn[k][0] == r and tuple(dn[k][1].shape) == tuple(T.shape) and torch.equal(dn[k][1], T) for k, r, T in po)
        red_old.consume(i0, i1, probs)
        red_new.consume(i0, i1, probs)
        nblocks += 1
    ck["pieces_bitwise"], ck["pieces_order_same"], ck["blocks>=2"] = pieces_bit, order_same, nblocks >= 2 or lay.R <= rows
    got_old, got_new = {}, {}

    def fin_old(k, ctx, T):
        got_old[k] = {"T": T.contiguous().clone(), "strides": tuple(T.stride()), "shape": tuple(T.shape)}
        if grammar == "kit":
            interface = not (isinstance(ctx.key, tuple) and ctx.key[0] == "chain")
            got_old[k]["v"] = finish_context_masked(ctx, T, hf, interface)
            if ctx.key == "full":
                got_old[k]["ptm"] = finish_context_masked(ctx, T, hf, False)
        return None

    def fin_new(k, ctx, T):
        got_new[k] = {"T": T.contiguous().clone(), "strides": tuple(T.stride()), "shape": tuple(T.shape)}
        if grammar == "kit":
            interface = not (isinstance(ctx.key, tuple) and ctx.key[0] == "chain")
            idx = red_new.index(k, device=T.device)
            got_new[k]["v"] = finish_context_idx(ctx, T, hf, interface, idx, asym_l)
            if ctx.key == "full":
                got_new[k]["ptm"] = finish_context_idx(ctx, T, hf, False, idx, asym_l)
        return None

    out_old = red_old.finalize(finish=fin_old)
    out_new = red_new.finalize(finish=fin_new)
    if rank != 0:
        assert out_old is None and out_new is None
        return None
    ck["finalize_keys_same"] = list(out_old) == list(out_new) == list(range(K))
    ck["T_all"] = len(got_old) == K and len(got_new) == K
    ck["T_bitwise"] = all(got_old[k]["shape"] == got_new[k]["shape"] and torch.equal(got_old[k]["T"], got_new[k]["T"]) for k in range(K))
    ck["T_strides_same"] = all(got_old[k]["strides"] == got_new[k]["strides"] for k in range(K))
    if grammar == "kit":
        ck["ptm_iptm_bitwise"] = all(torch.equal(got_old[k]["v"], got_new[k]["v"]) for k in range(K)) and torch.equal(got_old[0]["ptm"], got_new[0]["ptm"])
        ck["finite"] = all(bool(torch.isfinite(got_new[k]["v"]).all()) for k in range(K) if red_new.n(k) > 0)
    ck["fewer_gathers"] = red_new.stats["gathers"] < K or K <= 2
    return {"checks": ck, "K": K, "gathers_new": int(red_new.stats["gathers"]), "R": [b[1] - b[0] for b in lay.bounds]}


CASES = [
    # P, N,   C, S, rows, form,   order,         holes, layout,    grammar
    (2, 203, 1, 1, 40, "block", "blocks",      False, "grid",    "kit"),
    (2, 203, 3, 2, 40, "block", "blocks",      True,  "grid",    "kit"),
    (3, 203, 7, 1, 24, "block", "interleaved", True,  "grid",    "kit"),
    (3, 200, 7, 1, 40, "block", "blocks",      False, "aligned", "kit"),
    (2, 203, 7, 1, 16, "sub",   "interleaved", False, "grid",    "kit"),
    (3, 176, 3, 2, 40, "sub",   "blocks",      True,  "aligned", "kit"),
    (2, 96,  3, 1, 16, "sub",   "interleaved", True,  "grid",    "kit"),
    (3, 203, 1, 1, 24, "auto",  "blocks",      True,  "grid",    "kit"),
    (2, 203, 7, 1, 40, "auto",  "interleaved", True,  "aligned", "kit"),
    (2, 160, 4, 1, 40, "sub",   "blocks",      False, "grid",    "core"),
    (3, 168, 3, 1, 16, "block", "interleaved", False, "aligned", "core"),
    (2, 160, 4, 1, 24, "sub",   "interleaved", False, "grid",    "core_lite"),
    (3, 203, 5, 2, 40, "block", "blocks",      False, "grid",    "core_lite"),
]


@needs_torch
@parametrize("P,N,C,S,rows,form,order,holes,layout_kind,grammar", CASES)
def test_fast_equals_classic_threaded(P, N, C, S, rows, form, order, holes, layout_kind, grammar):
    from opt_core.testing import run_ranks
    out = run_ranks(P, _compare_rank, N, C, S, rows, form, order, holes, layout_kind, grammar, timeout_s=300)
    rec = out[0]
    print("RESULT " + json.dumps({"case": f"P{P}_N{N}_C{C}_S{S}_r{rows}_{form}_{order}_{'holes' if holes else 'full'}_{layout_kind}_{grammar}", **rec}))
    bad = {k: v for k, v in rec["checks"].items() if v is not True}
    assert not bad, bad


# ================================================================================================================ multi-PROCESS (gloo) through the launcher, with the thread lever applied by the worker
def _gloo_entry(N, C, S, rows, form, order, holes, layout_kind, grammar):
    import torch
    from opt_core.mem.rowpair.evidence import schedule
    rec = _compare_rank(None, None, N, C, S, rows, form, order, holes, layout_kind, grammar)
    if rec is not None:
        rec["threads"] = int(torch.get_num_threads())
        rec["rank_threads_word"] = schedule().get("rank_threads")
        rec["rank_threads_source"] = schedule().get("rank_threads_source")
        rec["omp_env"] = os.environ.get("OMP_NUM_THREADS")
    return rec


@needs_torch
@parametrize("P,C,form,grammar", [(2, 3, "block", "kit"), (3, 7, "sub", "kit")])
def test_fast_equals_classic_gloo_processes(P, C, form, grammar):
    os.environ["ROWPAIR_RANK_THREADS"] = "2"
    try:
        rec = launch.run_sharded(P, _gloo_entry, 203, C, 1, 40, form, "interleaved", True, "grid", grammar, mode="big", backend="gloo",
                                 cpu_ok=True, nccl_timeout_s=180, run_timeout_s=600)
    finally:
        os.environ.pop("ROWPAIR_RANK_THREADS", None)
    print("RESULT " + json.dumps({"case": f"gloo_P{P}_C{C}_{form}", **rec}))
    bad = {k: v for k, v in rec["checks"].items() if v is not True}
    assert not bad, bad
    assert rec["threads"] == 2 and rec["rank_threads_word"] == 2 and rec["rank_threads_source"] == "given" and rec["omp_env"] == "2", rec


# ================================================================================================================ the factory + the `both` checker + refusals
def _factory_rank(rank, P, word):
    import torch
    from opt_core.mem.rowpair import confidence as CF
    from opt_core.mem.rowpair import shard
    from opt_core.mem.rowpair.evidence import reset_schedule, schedule
    N, C, S, rows = 144, 4, 1, 24
    lay = _layout(N, P, rank, "grid")
    prob = _problem(N, C, S, 11, "interleaved", True)
    batch_b = {"token_mask": prob["token_mask"], "asym_id": prob["asym_id"]}
    if word == "env:both":
        os.environ["ROWPAIR_CONF_REDUCER"] = "both"                          # the lever decides both the mask need and the reducer
        ctxs, uc, labels, n_by = ptm_contexts_fast(batch_b, torch.device("cpu"), torch.float32, masks=None)
        red = CF.context_reducer_for_layout(ctxs, lay, form="block", labels=labels)
    else:
        ctxs, uc, labels, n_by = ptm_contexts_fast(batch_b, torch.device("cpu"), torch.float32, masks=None, reducer=word)
        red = CF.context_reducer_for_layout(ctxs, lay, form="block", labels=labels, reducer=word)
    masked = [c.mask is not None for c in ctxs]
    kind = type(red).__name__
    for bi, (i0, i1, g0, g1) in enumerate(shard.iter_row_blocks(lay, rows, unit=1)):
        red.consume(i0, i1, _probs(S, i1 - i0, N, 5 + rank * 100 + bi))
    seen = {}
    out = red.finalize(finish=lambda k, ctx, T: seen.__setitem__(k, tuple(T.shape)))
    facts = schedule()
    rec = {"kind": kind, "K": len(ctxs), "finalized": sorted(out) == list(range(len(ctxs))) if rank == 0 else out is None,
           "facts": {k: facts.get(k) for k in ("conf_reducer", "conf_ctx", "conf_gathers", "conf_index_s", "conf_finalize_s", "conf_reducer_mismatch", "rank_threads")},
           "mismatch": getattr(red, "mismatch", None), "masks_built": all(masked), "no_masks": not any(masked)}
    idx_ok = True                                                             # every reducer offers .index(k): the kit's ONE finish form
    labs_l = prob["asym_id"].long()
    tmask = prob["token_mask"].bool()
    for k, c in enumerate(ctxs):
        want = (tmask if c.chains is None else (tmask & sum((labs_l == v) for v in c.chains).bool())).nonzero(as_tuple=True)[0]
        idx_ok = idx_ok and torch.equal(red.index(k, "cpu"), want)
    rec["index_ok"] = idx_ok
    return rec


@needs_torch
@parametrize("word,kind", [("classic", "ContextReducer"), ("fast", "FastContextReducer"), ("both", "BothContextReducer"), ("env:both", "BothContextReducer")])
def test_factory_lever_and_both_checker(word, kind):
    from opt_core.testing import run_ranks
    os.environ.pop("ROWPAIR_CONF_REDUCER", None)
    try:
        out = run_ranks(2, _factory_rank, word, timeout_s=300)
    finally:
        os.environ.pop("ROWPAIR_CONF_REDUCER", None)
    print("RESULT " + json.dumps({"case": f"factory_{word}", "ranks": out}))
    for r, rec in enumerate(out):
        assert rec["kind"] == kind and rec["finalized"] and rec["index_ok"], rec
        assert rec["no_masks"] if kind == "FastContextReducer" else rec["masks_built"], rec
        f = rec["facts"]
        assert f["conf_reducer"] == word.split(":")[-1] and f["conf_ctx"] == rec["K"] and isinstance(f["conf_index_s"], float), f
        assert isinstance(f["rank_threads"], int) and f["rank_threads"] >= 1, f
        if kind != "ContextReducer":
            assert isinstance(f["conf_gathers"], int) and isinstance(f["conf_finalize_s"], float), f
        if kind == "FastContextReducer":
            assert f["conf_gathers"] < rec["K"], f
        if kind == "BothContextReducer":
            assert f["conf_reducer_mismatch"] == 0 and sum(rec["mismatch"].values()) == 0, rec


@needs_torch
def test_refusals_by_name():
    import torch
    from opt_core.mem.rowpair import confidence as CF
    prob = _problem(64, 3, 1, 3, "blocks", False)
    batch_b = {"token_mask": prob["token_mask"], "asym_id": prob["asym_id"]}
    ctxs_m, _uc = ptm_contexts_masked(batch_b, torch.device("cpu"), torch.float32)
    ctxs_f, _uc2, labels, _n = ptm_contexts_fast(batch_b, torch.device("cpu"), torch.float32)
    lay = Layout(64, 1, 0)
    for ctor in (lambda: CF.FastContextReducer.for_layout(ctxs_f, lay, labels=labels),
                 lambda: CF.context_reducer_for_layout(ctxs_f, lay, labels=labels, reducer="fast"),
                 lambda: CF.context_reducer_for_layout(ctxs_m, lay, reducer="classic")):
        try:
            ctor()
        except RowpairRefused as e:
            assert "n_gpu=1" in str(e), str(e)
        else:
            raise AssertionError("a reducer at n_gpu=1 without allow_unsharded must be refused by name")
    try:
        CF.context_reducer_for_layout(ctxs_f, lay, labels=labels, reducer="turbo", allow_unsharded=True)
    except RowpairRefused as e:
        assert "ROWPAIR_CONF_REDUCER" in str(e), str(e)
    else:
        raise AssertionError("unknown reducer word must be refused by name")
    try:                                                                     # masked PAIR contexts carry no chain structure: fast refuses, naming the way out
        CF.FastContextReducer.for_layout(ctxs_m, lay, labels=labels, allow_unsharded=True)
    except RowpairRefused as e:
        assert "ptm_contexts_from_labels" in str(e) or "chain structure" in str(e), str(e)
    else:
        raise AssertionError("masked pair contexts without `chains` must be refused by name")
    red = CF.FastContextReducer.for_layout(ctxs_f, lay, labels=labels, allow_unsharded=True, form="block")   # the single-device opt-in works end to end
    red.consume(0, 64, _probs(1, 64, 64, 1))
    out = red.finalize()
    assert sorted(out) == list(range(len(ctxs_f))) and tuple(out[0].shape) == (1, 64, 64), {k: tuple(v.shape) for k, v in out.items()}
    dense = (_probs(1, 64, 64, 1) * ctxs_f[0].w).sum(-1)
    assert torch.equal(out[0], dense), "FULL context at P=1 must be the dense statement"
    k_pair = [i for i, c in enumerate(ctxs_f) if isinstance(c.key, tuple) and c.key[0] == "pair"][0]
    idx = red.index(k_pair)
    want = ((prob["asym_id"].long() == ctxs_f[k_pair].chains[0]) | (prob["asym_id"].long() == ctxs_f[k_pair].chains[1])).nonzero(as_tuple=True)[0]
    assert torch.equal(idx, want)
    dense_pair = (_probs(1, 64, 64, 1) * ctxs_f[k_pair].w).sum(-1)              # the pair context's own weight (d0 of n_a + n_b)
    assert torch.equal(out[k_pair], dense_pair.index_select(-2, idx).index_select(-1, idx))


@needs_torch
def test_label_helpers():
    import torch
    from opt_core.mem.rowpair import confidence as CF
    for asym_dtype in ("float32", "long"):
        prob = _problem(120, 5, 1, 5, "interleaved", True, asym_dtype=asym_dtype)
        batch_b = {"token_mask": prob["token_mask"], "asym_id": prob["asym_id"]}
        cm, uc = ptm_contexts_masked(batch_b, torch.device("cpu"), torch.float32)
        cf, uc2, labels, n_by = ptm_contexts_fast(batch_b, torch.device("cpu"), torch.float32)
        assert uc == uc2 and len(cm) == len(cf) == 1 + 5 + 10
        for a, b in zip(cm, cf):
            assert a.key == b.key and type(a.key) is type(b.key), (a.key, b.key)
            if isinstance(a.key, tuple):
                assert [type(x) for x in a.key] == [type(x) for x in b.key], (a.key, b.key)
            assert torch.equal(a.w, b.w) and torch.equal(a.num, b.num)
            want = a.mask.nonzero(as_tuple=True)[0]
            labs = CF.FastContextReducer.context_labels_of(b)
            got = (labels >= 0).nonzero(as_tuple=True)[0] if labs is None else torch.cat([(labels == v).nonzero(as_tuple=True)[0] for v in labs]).sort().values
            assert torch.equal(want, got), (a.key, labs)
        assert torch.equal(labels, CF.context_labels(prob["asym_id"], prob["token_mask"]))
        assert int((labels < 0).sum()) == int((prob["token_mask"] == 0).sum())
        if asym_dtype == "float32":
            assert all(isinstance(c.key[1], float) for c in cf if isinstance(c.key, tuple) and c.key[0] == "chain")
    chains = CF.ChainIndex(prob["asym_id"].long(), prob["has_frame"][0])
    cen = CF.tm_bin_centers(0.0, 32.0, BINS)
    full, lite = chains.contexts(cen, "cpu"), chains.contexts_lite(cen, "cpu")
    assert len(full) == len(lite) and all(a.kind == b.kind and a.a == b.a and a.b == b.b and a.N_d == b.N_d and torch.equal(a.w, b.w) and b.mask is None for a, b in zip(full, lite))
    assert len({id(c.w) for c in lite}) == len({c.N_d for c in lite}) <= len({id(c.w) for c in full})


# ================================================================================================================ the thread lever
@needs_torch
def test_cpu_threads_lever():
    import torch
    from opt_core.mem.rowpair import dist as RD
    assert RD.resolve_cpu_threads(None) == (None, "inherit") and RD.resolve_cpu_threads("inherit") == (None, "inherit") and RD.resolve_cpu_threads("") == (None, "inherit")
    assert RD.resolve_cpu_threads(0) == (None, "inherit") and RD.resolve_cpu_threads("0") == (None, "inherit")
    assert RD.resolve_cpu_threads("auto", local_world=8, cpus=64) == (8, "auto") and RD.resolve_cpu_threads("auto", local_world=8, cpus=6) == (1, "auto")
    assert RD.resolve_cpu_threads(3) == (3, "given") and RD.resolve_cpu_threads(" 5 ") == (5, "given")
    for bad in ("many", "-2", -1, True, "1.5"):
        try:
            RD.resolve_cpu_threads(bad)
        except ValueError as e:
            assert "ROWPAIR_RANK_THREADS" in str(e)
        else:
            raise AssertionError(f"{bad!r} must be refused")
    assert RD.rank_thread_env("inherit", 4) == {} and RD.rank_thread_env(None, 4) == {}
    d = RD.rank_thread_env("auto", 4, cpus=64)
    assert d == {"OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "16", "ROWPAIR_RANK_THREADS": "16"}, d
    prev = torch.get_num_threads()
    with RD.cpu_threads(2) as n:
        assert n == 2 and torch.get_num_threads() == 2
    assert torch.get_num_threads() == prev
    with RD.cpu_threads(None) as n:
        assert n is None and torch.get_num_threads() == prev
    with RD.cpu_threads("auto", local_world=RD.visible_cpus() * 4) as n:                  # more ranks than cpus: floor at 1
        assert n == 1 and torch.get_num_threads() == 1
    assert torch.get_num_threads() == prev


def test_rank_env_cpu_threads_kwarg():
    os.environ.pop("ROWPAIR_RANK_THREADS", None)
    base = {"HOME": "/x", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    today = launch.rank_env(1, 4, 29511, base=base, log_dir="/tmp/l", store="/tmp/s")
    assert not any(k in today for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ROWPAIR_RANK_THREADS")), today
    assert launch.rank_env(1, 4, 29511, base=base, log_dir="/tmp/l", store="/tmp/s", cpu_threads=None) == today
    assert launch.rank_env(1, 4, 29511, base=base, log_dir="/tmp/l", store="/tmp/s", cpu_threads="inherit") == today
    given = launch.rank_env(1, 4, 29511, base=base, cpu_threads=3)
    assert all(given[k] == "3" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ROWPAIR_RANK_THREADS")), given
    from opt_core.mem.rowpair import dist as RD
    auto = launch.rank_env(0, 4, 29511, base=base, cpu_threads="auto")
    assert auto["OMP_NUM_THREADS"] == str(max(1, RD.visible_cpus() // 4)) == auto["ROWPAIR_RANK_THREADS"], auto
    lever = launch.rank_env(2, 4, 29511, base=dict(base, ROWPAIR_RANK_THREADS="auto"))   # the lever in the launching environment, no kwarg
    assert lever["OMP_NUM_THREADS"] == str(max(1, RD.visible_cpus() // 4)) and lever["ROWPAIR_RANK_THREADS"] == lever["OMP_NUM_THREADS"], lever
    os.environ["ROWPAIR_RANK_THREADS"] = "6"                                # base None: read from this process's environment
    try:
        e = launch.rank_env(0, 2, 29511)
        assert e["OMP_NUM_THREADS"] == "6" and e["ROWPAIR_RANK_THREADS"] == "6", {k: e.get(k) for k in ("OMP_NUM_THREADS", "ROWPAIR_RANK_THREADS")}
        inh = launch.rank_env(0, 2, 29511, cpu_threads="inherit")            # an explicit inherit wins over the environment's word: the rank applies nothing
        assert inh.get("ROWPAIR_RANK_THREADS") == "inherit" and inh.get("OMP_NUM_THREADS") == os.environ.get("OMP_NUM_THREADS"), inh.get("ROWPAIR_RANK_THREADS")
    finally:
        os.environ.pop("ROWPAIR_RANK_THREADS", None)
    try:
        launch.rank_env(0, 2, 29511, cpu_threads="lots")
    except ValueError as e:
        assert "ROWPAIR_RANK_THREADS" in str(e)
    else:
        raise AssertionError("a bad thread word must be refused at launch")


if __name__ == "__main__":
    import inspect
    fails, ran = [], 0
    mod = sys.modules[__name__]
    t_all = time.monotonic()
    for name, fn in sorted(inspect.getmembers(mod, inspect.isfunction)):
        if not name.startswith("test_"):
            continue
        params = [m for m in getattr(fn, "pytestmark", []) if m.name == "parametrize"]
        arglists = [dict(zip(params[0].args[0].split(","), vals)) for vals in params[0].args[1]] if params else [{}]
        for kw in arglists:
            ran += 1
            t0 = time.monotonic()
            try:
                fn(**kw)
                print(f"PASS {name} {kw} ({time.monotonic() - t0:.1f}s)")
            except Exception as e:  # noqa: BLE001
                fails.append((name, kw, repr(e)))
                print(f"FAIL {name} {kw}: {e!r}")
    print(f"SUMMARY ran={ran} failed={len(fails)} wall={time.monotonic() - t_all:.1f}s")
    sys.exit(1 if fails else 0)
