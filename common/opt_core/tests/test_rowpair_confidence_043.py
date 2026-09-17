"""opt_core.mem.rowpair — the pair-conditioned heads / confidence reducer / atom-pair helpers under row sharding (``heads``,
``confidence.ContextReducer``, ``frames``): P in {2, 3, 4} multi-process gloo checks against an INDEPENDENT dense synthetic confidence
head (random weights, no engine; the launcher is the launcher, as in ``test_rowpair_logic``), the P = 1 refusals by name, and in-process
checks of the single-device opt-in and the atom helpers.

What is asserted (fp32 CPU): PAE rows, symmetrised PDE rows, distogram contact rows, every TM context matrix ``T_D`` gathered by the reducer,
and the finishing scalars (pTM / ipTM / chain pTM / chain-pair ipTM in the AF3-family engine statement form, with the sample dim S kept) equal the
dense head's within 1e-5 — bit-exact REPORTED per quantity; the symmetrisation's data movement (``z + z^T`` through the column-slab
exchange) bit-exact; no rank ever allocates more than the per-sample pair SHARD / one logits row block (numel guard: never an ``[S, N, N, C]`` pair tensor
or ``[N, N, bins]`` logits); uneven N;
the two statement forms of the reducer (``block`` / ``sub``) both equal the dense statement.

Run: ``python -m pytest tests/test_rowpair_confidence_043.py -q -rfE`` or ``python tests/test_rowpair_confidence_043.py`` (RESULT lines + SUMMARY).
"""
from __future__ import annotations

import json
import math
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

TOL = 1e-5
C_Z, C_S, BINS, DBINS, NB_DIST = 32, 6, 64, 64, 11      # pair channels, single channels, PAE/PDE bins, distogram bins, distance one-hot bins


# ================================================================================================================ the synthetic head
def _synthetic(N: int, S: int, nchains: int, seed: int = 7):
    """Seeded on CPU (identical on every rank): trunk pair rep, single rep, per-sample token coordinates, masks, chain ids, weights."""
    import torch
    g = torch.Generator().manual_seed(seed)
    rn = lambda *shape, scale=1.0: (torch.randn(shape, generator=g) * scale).float()  # noqa: E731
    d = {
        "z": rn(1, N, N, C_Z, scale=0.5), "s": rn(N, C_S), "x": rn(S, N, 3, scale=6.0),
        "W_i": rn(C_S, C_Z, scale=0.3), "W_j": rn(C_S, C_Z, scale=0.3), "W_d": rn(NB_DIST, C_Z, scale=0.3), "W_stack": rn(C_Z, C_Z, scale=0.2),
        "ln1_w": 1.0 + rn(C_Z, scale=0.1), "ln1_b": rn(C_Z, scale=0.1), "ln2_w": 1.0 + rn(C_Z, scale=0.1), "ln2_b": rn(C_Z, scale=0.1),
        "W_pae": rn(C_Z, BINS, scale=0.4), "W_pde": rn(C_Z, BINS, scale=0.4), "W_dg": rn(C_Z, DBINS, scale=0.4),
    }
    sizes = [N // nchains + (1 if i < N % nchains else 0) for i in range(nchains)]
    d["asym"] = torch.cat([torch.full((sz,), 3 + 2 * i, dtype=torch.long) for i, sz in enumerate(sizes)])
    tm = torch.ones(N)
    if nchains > 1:
        tm[-2:] = 0.0                                           # padded tokens: the OF3 contexts are masked by token_mask
    d["token_mask"] = tm
    d["has_frame"] = torch.rand((S, N), generator=g) > 0.3
    return d


def _bins2(device):
    import torch
    bins = torch.linspace(3.25, 50.75, NB_DIST, device=device)
    sq = bins ** 2
    upper = torch.cat([sq[1:], sq.new_tensor([1e8])], dim=-1)
    return sq, upper


def _embed_dense(d):
    """PairformerEmbedding.embed_zij-style statement, dense: [S, N, N, C]."""
    import torch
    z, s, x = d["z"], d["s"], d["x"]
    zij = z + (s @ d["W_i"])[None, :, None, :] + (s @ d["W_j"])[None, None, :, :]
    sq, upper = _bins2(z.device)
    d2 = torch.sum((x[:, :, None, :] - x[:, None, :, :]) ** 2, dim=-1, keepdim=True)
    oh = ((d2 > sq) * (d2 < upper)).to(x.dtype)
    return zij + oh @ d["W_d"]


def _embed_rows_fn(d):
    """The same statement for ROWS g0:g1 of the trunk shard block (the adapter's callable): zblk [1, w, N, C] -> [S, w, N, C]."""
    import torch

    def fn(zblk, g0, g1):
        s, x = d["s"], d["x"]
        zij = zblk + (s @ d["W_i"])[None, g0:g1, None, :] + (s @ d["W_j"])[None, None, :, :]
        sq, upper = _bins2(zblk.device)
        d2 = torch.sum((x[:, g0:g1, None, :] - x[:, None, :, :]) ** 2, dim=-1, keepdim=True)
        oh = ((d2 > sq) * (d2 < upper)).to(x.dtype)
        return zij + oh @ d["W_d"]
    return fn


def _stack_fn(d):
    """Row-local stand-in for the confidence pair stack (the real one is the pair-block driver's; here only the seam is exercised)."""
    import torch
    return lambda z: z + torch.tanh(z @ d["W_stack"])


def _heads(d):
    import torch
    F = torch.nn.functional
    pae_fn = lambda z: F.layer_norm(z, (C_Z,), d["ln1_w"], d["ln1_b"], 1e-5) @ d["W_pae"]  # noqa: E731
    pde_fn = lambda z: F.layer_norm(z, (C_Z,), d["ln2_w"], d["ln2_b"], 1e-5) @ d["W_pde"]  # noqa: E731
    dg_fn = lambda z: z @ d["W_dg"]  # noqa: E731
    return pae_fn, pde_fn, dg_fn


def _centers(device):
    from opt_core.mem.rowpair.confidence import tm_bin_centers
    return tm_bin_centers(0.0, 32.0, BINS).to(device)


def _of3_bin_weight_cache(device, dtype):
    """The AF3-family compute_ptm weight statement (fp32 TENSOR d0), cached per |D| so contexts of equal size share one tensor object."""
    import torch
    cache = {}
    centers = _centers(device).to(dtype)

    def w(n: int):
        if n not in cache:
            num = torch.tensor(float(n), device=device, dtype=dtype).clamp_min(1)
            clipped = torch.maximum(num, torch.tensor(19.0, device=device, dtype=dtype))
            d0 = 1.24 * (clipped - 15.0).clamp_min(0).pow(1.0 / 3.0) - 1.8
            cache[n] = 1.0 / (1.0 + (centers / d0) ** 2)
        return cache[n]
    return w


class Ctx(object):
    """A context as the adapter builds it: key, bool mask [N], weight tensor, chain ids for the interface finish."""
    def __init__(self, key, mask, w, asym):
        self.key, self.mask, self.w, self.asym, self.N_d = key, mask, w, asym, int(mask.sum())


def _contexts(d, device, dtype):
    import torch
    w_of = _of3_bin_weight_cache(device, dtype)
    tm = d["token_mask"].bool().to(device)
    asym = d["asym"].to(device)
    ctxs = [Ctx("full", tm, w_of(int(tm.sum())), asym)]
    uniq = torch.unique(asym).tolist()
    masks = [(asym == a) & tm for a in uniq]
    for a, m in zip(uniq, masks):
        ctxs.append(Ctx(("chain", a), m, w_of(int(m.sum())), asym))
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            m = masks[i] | masks[j]
            ctxs.append(Ctx(("pair", uniq[i], uniq[j]), m, w_of(int(m.sum())), asym))
    return ctxs


def _finish(ctx, T, has_frame, eps=1e-8):
    """The AF3-family compute_ptm lines after ptm_ij, on the context matrix T [S, n_D, n_D]: (ptm [S], iptm [S])."""
    hf = has_frame[:, ctx.mask].bool()
    num = ctx.mask.sum().clamp_min(1).to(T.dtype)
    ptm = (T.sum(dim=-1) / num).masked_fill(~hf, 0.0).max(dim=-1).values
    a = ctx.asym[ctx.mask]
    pair_mask = a.unsqueeze(-1) != a.unsqueeze(-2)
    itm = (T * pair_mask).sum(dim=-1) / pair_mask.sum(dim=-1).clamp_min(eps)
    iptm = itm.masked_fill(~hf, 0.0).max(dim=-1).values
    return ptm, iptm


def _dense_reference(d, ctxs):
    """Independent single-device statements on FULL tensors (the sharded path never builds these)."""
    import torch
    zij = _embed_dense(d)
    zij = _stack_fn(d)(zij)
    pae_fn, pde_fn, dg_fn = _heads(d)
    cen = _centers(zij.device)
    pae_logits = pae_fn(zij)                                                      # [S, N, N, b]
    pl = pde_fn(zij)
    pde_logits = pl + pl.transpose(-2, -3)
    dg = dg_fn(d["z"])
    dg = dg + dg.transpose(-2, -3)                                                # [1, N, N, b]
    k8 = torch.linspace(2.3125, 21.6875, DBINS + 1)[1:] <= 8.0
    ref = {"zij": zij}
    ref["contact"] = torch.sum(torch.softmax(dg, dim=-1)[..., k8], dim=-1)      # [1, N, N]
    pae_probs = torch.softmax(pae_logits, dim=-1)
    ref["pae"] = torch.sum(pae_probs * cen, dim=-1)                              # OF3 statement (not a GEMV)
    ref["pde"] = torch.sum(torch.softmax(pde_logits, dim=-1) * cen, dim=-1)
    ref["T"], ref["ptm"], ref["iptm"] = {}, {}, {}
    for k, ctx in enumerate(ctxs):
        sub = pae_probs[:, ctx.mask][:, :, ctx.mask]                               # [S, n, n, b]
        T = torch.sum(sub * ctx.w, dim=-1)
        ref["T"][k] = T
        ref["ptm"][k], ref["iptm"][k] = _finish(ctx, T, d["has_frame"])
    ref["zzT"] = d["z"][..., :5] + d["z"][..., :5].transpose(-2, -3)              # pure data-movement probe of the symmetrised exchange
    return ref


# ================================================================================================================ numel guard
def _numel_guard():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class Guard(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.max_numel, self.where = 0, None

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in tree_flatten(out)[0]:
                if isinstance(t, torch.Tensor) and t.numel() > self.max_numel:
                    self.max_numel, self.where = int(t.numel()), str(func)
            return out
    return Guard()


# ================================================================================================================ the sharded entry (every rank)
def _entry_conf043(N: int, S: int, nchains: int, B: int, rows, form: str):
    import torch
    import torch.distributed as tdist
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair import heads
    from opt_core.mem.rowpair.confidence import ContextReducer
    from opt_core.mem.rowpair.shard import iter_row_blocks, shard_rows, unshard_rows_to_rank0
    P, r = D.world()
    lay = D.Layout.auto(N, P, r) if not B else D.Layout.checked(N, P, r, B)
    dev = torch.device("cpu")
    d = _synthetic(N, S, nchains)
    ctxs = _contexts(d, dev, torch.float32)
    pae_fn, pde_fn, dg_fn = _heads(d)
    cen = _centers(dev)
    k8 = torch.linspace(2.3125, 21.6875, DBINS + 1)[1:] <= 8.0
    res = {"case": "conf043", "P": P, "N": N, "S": S, "B": lay.B, "bounds": lay.bounds, "chains": nchains, "rows": rows, "form": form, "checks": {}, "metrics": {}}
    zt_loc = shard_rows(d["z"], lay, dim=-3).clone()                            # the trunk shard with its OWN storage, as a rank holds it: [1, R, N, C]
    #   (the S == 1 in-place embed below consumes that storage — the FREE_ZTRUNK form; d["z"] stays whole for rank 0's dense reference)
    R = lay.R
    guard = _numel_guard()
    with guard:
        # -- distogram contact rows FIRST (reads the trunk shard before the S == 1 in-place embed consumes it)
        contact_loc = torch.empty((1, R, N))
        for i0, i1, lg in heads.sym_logit_rows(dg_fn, zt_loc, lay, rows=rows, bins=DBINS):
            contact_loc[:, i0:i1] = torch.sum(torch.softmax(lg, dim=-1)[..., k8], dim=-1)
        zzT_loc = torch.empty((1, R, N, 5))
        for i0, i1, lg in heads.sym_logit_rows(lambda z: z[..., :5], zt_loc, lay, rows=rows, bins=5):
            zzT_loc[:, i0:i1] = lg
        # -- the per-sample pair input, per row block, from the trunk shard (in place when S == 1)
        zij_loc = heads.embed_rows(_embed_rows_fn(d), zt_loc, lay, rows=rows, lead_out=(S,), inplace=(S == 1), bins=NB_DIST)
        res["checks"]["embed_shape"] = tuple(zij_loc.shape) == (S, R, N, C_Z)
        res["checks"]["embed_inplace_storage"] = (zij_loc.data_ptr() == zt_loc.data_ptr()) == (S == 1)
        # -- the pair stack (row-local stand-in for the driver callable), in place per row block
        stack = _stack_fn(d)
        rb, _ = heads.conf_rows(N, BINS, rows, S, record=False)
        for i0, i1, _g0, _g1 in iter_row_blocks(lay, rb, unit=1):
            zij_loc[:, i0:i1] = stack(zij_loc[:, i0:i1])
        # -- heads, row block by row block; the [S, w, N, 64] logits live one block at a time
        pde_loc, pae_loc = torch.empty((S, R, N)), torch.empty((S, R, N))
        red = ContextReducer.for_layout(ctxs, lay, form=form)
        for i0, i1, lg in heads.sym_logit_rows(pde_fn, zij_loc, lay, rows=rows, bins=BINS):
            pde_loc[:, i0:i1] = torch.sum(torch.softmax(lg, dim=-1) * cen, dim=-1)
        for i0, i1, lg in heads.logit_rows(pae_fn, zij_loc, lay, rows=rows, bins=BINS):
            probs = torch.softmax(lg, dim=-1)
            pae_loc[:, i0:i1] = torch.sum(probs * cen, dim=-1)
            red.consume(i0, i1, probs)
            del probs
    # -- the numel guard: no rank ever held a full pair representation (not even one sample's [N, N, C]) or an [N, N, bins] logits tensor
    allmax = [None] * P
    tdist.all_gather_object(allmax, (guard.max_numel, guard.where))
    rb_used, _ = heads.conf_rows(N, BINS, rows, S, record=False)
    expected_peak = max(S * lay.Rmax * N * C_Z,                                   # the per-sample pair SHARD (S/P pair-equivalents)
                        S * rb_used * N * max(BINS, DBINS),                       # one row block of logits / probs
                        N * rb_used * S * max(BINS, DBINS),                       # the column slabs of one exchange window
                        lay.Rmax * C_Z * max(BINS, DBINS, C_Z))                   # the head weight broadcast as a batched-matmul EXPAND view over
                                                                                  # R rows ([1, R, C, bins], no storage) — exceeds the shard only
                                                                                  # when N < bins (the ragged small-N cases)
    res["metrics"]["max_numel_per_rank"] = [m for m, _ in allmax]
    res["metrics"]["max_numel_where"] = allmax[max(range(P), key=lambda q: allmax[q][0])][1]
    res["metrics"]["expected_peak_numel"] = expected_peak
    res["metrics"]["full_pair_numel_per_sample"] = N * N * C_Z
    res["checks"]["peak_within_shard_schedule"] = max(m for m, _ in allmax) <= expected_peak
    res["checks"]["never_full_pair"] = expected_peak < S * N * N * C_Z and max(m for m, _ in allmax) < S * N * N * C_Z
    # -- output seams: per-row matrices to rank 0 (the writer's), context matrices to rank 0 (the finishing statements')
    pae_full = unshard_rows_to_rank0(pae_loc, lay, dim=-2)
    pde_full = unshard_rows_to_rank0(pde_loc, lay, dim=-2)
    contact_full = unshard_rows_to_rank0(contact_loc, lay, dim=-2)
    zzT_full = unshard_rows_to_rank0(zzT_loc, lay, dim=-3)
    Ts = red.finalize()
    if r != 0:
        res["ok"] = all(v is None for v in (pae_full, pde_full, contact_full, zzT_full, Ts)) and all(res["checks"].values())
        return res
    ref = _dense_reference(d, ctxs)

    def cmp(name, got, exp, tol=TOL):
        dmax = (got - exp).abs().max().item() if got.numel() else 0.0
        res["metrics"][name] = {"bitwise": bool(torch.equal(got, exp)), "maxabs": dmax, "shape": list(got.shape)}
        res["checks"][name] = bool(got.shape == exp.shape and dmax <= tol)

    cmp("pae_rows", pae_full, ref["pae"])
    cmp("pde_rows_sym", pde_full, ref["pde"])
    cmp("contact_rows_sym", contact_full, ref["contact"])
    cmp("zzT_data_movement", zzT_full, ref["zzT"], tol=0.0)                        # bit-exact: pure data movement + the same add
    hf = d["has_frame"]
    t_bit, s_bit, worst_T, worst_s = True, True, 0.0, 0.0
    for k, ctx in enumerate(ctxs):
        T = Ts[k]
        dT = (T - ref["T"][k]).abs().max().item()
        t_bit = t_bit and bool(torch.equal(T, ref["T"][k]))
        worst_T = max(worst_T, dT)
        ptm, iptm = _finish(ctx, T, hf)
        for got, exp in ((ptm, ref["ptm"][k]), (iptm, ref["iptm"][k])):
            s_bit = s_bit and bool(torch.equal(got, exp))
            worst_s = max(worst_s, (got - exp).abs().max().item())
        res["checks"][f"T_shape_{k}"] = tuple(T.shape) == (S, ctx.N_d, ctx.N_d)
    res["metrics"]["T_contexts"] = {"n_ctx": len(ctxs), "bitwise_all": t_bit, "maxabs": worst_T}
    res["metrics"]["ptm_iptm"] = {"bitwise_all": s_bit, "maxabs": worst_s, "ptm_full": [round(float(v), 6) for v in _finish(ctxs[0], Ts[0], hf)[0]]}
    res["checks"]["T_contexts"] = worst_T <= TOL
    res["checks"]["ptm_iptm"] = worst_s <= TOL
    res["ok"] = all(res["checks"].values())
    return res


# ================================================================================================================ harness (the launcher)
def _mp(P, entry, *args, **kw):
    prev = os.environ.get("ROWPAIR_TEST_DEVICE")
    os.environ["ROWPAIR_TEST_DEVICE"] = "cpu"
    try:
        return launch.run_sharded(P, entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=120, run_timeout_s=600, **kw)
    finally:
        if prev is None:
            os.environ.pop("ROWPAIR_TEST_DEVICE", None)
        else:
            os.environ["ROWPAIR_TEST_DEVICE"] = prev


def _report(res):
    print("RESULT " + json.dumps(res, sort_keys=True, default=str))
    assert res["ok"], res


import pytest  # noqa: E402

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch not importable")

# (P, N, S samples, chains, B (0 = Layout.auto: uneven N), rows (None = default int32 rule), reducer form)
CASES = [
    (2, 256, 1, 1, 128, 16, "block"),
    (2, 300, 2, 3, 0, 40, "sub"),
    (3, 300, 3, 4, 0, 40, "block"),
    (4, 300, 2, 5, 0, 32, "auto"),
    (4, 512, 1, 2, 128, None, "block"),
    (2, 61, 1, 2, 16, 8, "block"),                                                  # ragged grid 32/29 (last block partial): sym_logit_rows windows uneven
    (3, 61, 1, 2, 8, 8, "sub"),                                                      # ragged grid 24/24/13
    (4, 53, 2, 2, 8, 8, "block"),                                                    # ragged grid 16/16/16/5
]


@needs_torch
@pytest.mark.parametrize("P,N,S,nchains,B,rows,form", CASES)
def test_mp_confidence_043(P, N, S, nchains, B, rows, form):
    _report(_mp(P, _entry_conf043, N, S, nchains, B, rows, form))


# ================================================================================================================ in-process checks
@needs_torch
def test_p1_refusals_by_name():
    """The structural n_gpu=1 rule: every 0.4.3 head / reducer entry point refuses a P == 1 (no group) layout by name — for real."""
    import torch
    from opt_core.mem.rowpair import heads
    from opt_core.mem.rowpair.confidence import ContextReducer
    lay = Layout(64, 1, 0)
    z = torch.zeros((1, 64, 64, 4))
    fn = lambda t, *a: t  # noqa: E731
    cases = {
        "logit_rows": lambda: next(heads.logit_rows(fn, z, lay, rows=8)),
        "sym_logit_rows": lambda: next(heads.sym_logit_rows(fn, z, lay, rows=8)),
        "embed_rows": lambda: heads.embed_rows(fn, z, lay, rows=8),
        "ContextReducer": lambda: ContextReducer.for_layout([Ctx("full", torch.ones(64, dtype=torch.bool), torch.ones(4), torch.zeros(64))], lay),
    }
    words = {}
    for name, call in cases.items():
        try:
            call()
            words[name] = None
        except RowpairRefused as e:
            words[name] = e.reason
    print("RESULT " + json.dumps({"case": "p1_refusals", "words": words}, default=str))
    assert all(w is not None and "n_gpu=1" in w for w in words.values()), words


@needs_torch
def test_p1_single_device_optin_is_dense():
    """allow_unsharded=True (a kit's OWN single-device lever, named in its line): the row-blocked heads and the reducer ARE the dense
    statements evaluated in row blocks (the transpose term = fn on this process's own column slab)."""
    import torch
    from opt_core.mem.rowpair import heads
    from opt_core.mem.rowpair.confidence import ContextReducer
    N, S = 96, 2
    lay = Layout(N, 1, 0)
    d = _synthetic(N, S, 3)
    ctxs = _contexts(d, torch.device("cpu"), torch.float32)
    pae_fn, pde_fn, dg_fn = _heads(d)
    cen = _centers("cpu")
    ref = _dense_reference(d, ctxs)
    zij = heads.embed_rows(_embed_rows_fn(d), d["z"].clone(), lay, rows=16, lead_out=(S,), allow_unsharded=True, bins=NB_DIST)
    zij = _stack_fn(d)(zij)
    ck = {"embed": bool((zij - ref["zij"]).abs().max().item() <= TOL)}
    pde = torch.empty((S, N, N))
    for i0, i1, lg in heads.sym_logit_rows(pde_fn, zij, lay, rows=16, allow_unsharded=True):
        pde[:, i0:i1] = torch.sum(torch.softmax(lg, -1) * cen, -1)
    ck["pde_sym"] = bool((pde - ref["pde"]).abs().max().item() <= TOL)
    for form in ("block", "sub", "auto"):
        red = ContextReducer.for_layout(ctxs, lay, form=form, allow_unsharded=True)
        for i0, i1, lg in heads.logit_rows(pae_fn, zij, lay, rows=16, allow_unsharded=True):
            red.consume(i0, i1, torch.softmax(lg, -1))
        Ts = red.finalize()
        ck[f"T_{form}_bitwise"] = all(bool(torch.equal(Ts[k], ref["T"][k])) for k in range(len(ctxs)))
        ck[f"T_{form}"] = all((Ts[k] - ref["T"][k]).abs().max().item() <= TOL for k in range(len(ctxs)))
    print("RESULT " + json.dumps({"case": "p1_optin_dense", "checks": ck}))
    assert all(v for k, v in ck.items() if not k.endswith("_bitwise")), ck


@needs_torch
def test_rowblockreducer_forms_agree_with_042_statement():
    """RowBlockReducer's TM pieces now come from ContextReducer(form='sub'): the literal 0.4.2 statement
    ``(pae_prob[:, rows][:, :, cols] * w).sum(-1)`` — asserted bit-exact against that statement written out here, and against form='block'."""
    import torch
    from opt_core.mem.rowpair.confidence import ChainIndex, ContextReducer, tm_bin_centers
    N = 80
    g = torch.Generator().manual_seed(3)
    probs = torch.softmax(torch.randn((1, N, N, 64), generator=g) * 2.0, -1)
    asym = torch.cat([torch.full((30,), 0), torch.full((30,), 1), torch.full((20,), 4)])
    chains = ChainIndex(asym, torch.ones(N, dtype=torch.bool))
    ctxs = chains.contexts(tm_bin_centers(0.0, 32.0, 64), "cpu")
    lay = Layout(N, 1, 0)
    out = {}
    for form in ("sub", "block"):
        red = ContextReducer.for_layout(ctxs, lay, form=form, allow_unsharded=True)
        for i0 in range(0, N, 24):
            i1 = min(N, i0 + 24)
            red.consume(i0, i1, probs[:, i0:i1])
        out[form] = red.finalize()
    ok = {}
    for k, ctx in enumerate(ctxs):
        sub = probs if ctx.mask is None else probs[:, ctx.mask][:, :, ctx.mask]
        T042 = (sub * ctx.w).sum(dim=-1)                                          # the 0.4.2 statement, whole
        ok[f"sub_{k}"] = bool(torch.equal(out["sub"][k], T042))
        ok[f"block_{k}"] = bool(torch.equal(out["block"][k], T042))
    print("RESULT " + json.dumps({"case": "reducer_forms", "bitwise": ok}))
    assert all(ok[k] for k in ok if k.startswith("sub_")), ok                     # same statement, same shapes: bit-exact by construction
    assert all((out["block"][k] - out["sub"][k]).abs().max().item() <= 1e-6 for k in range(len(ctxs)))


@needs_torch
def test_frames_row_blocked_helpers():
    """nearest_atoms_rows == the stock all-pairs statements gathered at the query rows (indices bit-exact); count_pairs_within blocked == whole."""
    import torch
    from opt_core.mem.rowpair.frames import count_pairs_within, nearest_atoms_rows
    g = torch.Generator().manual_seed(9)
    S, NA, NT = 2, 300, 70
    x = torch.randn((S, NA, 3), generator=g) * 8.0
    atom_mask = (torch.rand((NA,), generator=g) > 0.05).float()
    group = torch.randint(0, 3, (NA,), generator=g)
    start = torch.sort(torch.randperm(NA, generator=g)[:NT]).values                # token start atoms
    eps, inf = 1e-8, 1e9
    # stock (all pairs) statements
    pair_mask = atom_mask[..., None] * atom_mask[..., None, :]
    pair_mask = pair_mask * (group[..., None] == group[..., None, :])
    dmat = torch.sum(eps + (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1) ** 0.5
    dmat = dmat * pair_mask + inf * (1 - pair_mask)
    _, idx_all = torch.topk(dmat, k=3, dim=-1, largest=False)                     # [S, NA, 3]
    want = torch.gather(idx_all, 1, start[None, :, None].expand(S, NT, 3))
    got = nearest_atoms_rows(x, start[None, :].expand(S, NT), atom_mask, group, k=3, eps=eps, inf=inf, rows=7)
    ck = {"nearest_bitwise": bool(torch.equal(got, want)), "shape": tuple(got.shape) == (S, NT, 3)}
    a, b = x[0, :120], x[0, 120:]
    whole = int((torch.cdist(a, b) < 4.0).sum())
    ck["count_blocked"] = count_pairs_within(a, b, 4.0, rows=16, max_pair_bytes=1.0) == whole
    ck["count_whole"] = count_pairs_within(a, b, 4.0) == whole
    print("RESULT " + json.dumps({"case": "frames", "checks": ck, "count": whole}))
    assert all(ck.values()), ck


@needs_torch
def test_conf_rows_sources():
    from opt_core.mem.rowpair import evidence, heads
    os.environ.pop("ROWPAIR_CONF_ROWS", None)
    assert heads.conf_rows(10147, 64) == (128, "default"), heads.conf_rows(10147, 64)
    assert heads.conf_rows(600000, 64) == (55, "default+cap")                     # int32 rule: 55*600000*64 < 2**31 <= 56*600000*64
    assert heads.conf_rows(10147, 64, lead=4) == (128, "default") and heads.conf_rows(10147, 64, lead=64) == (51, "default+cap")
    assert heads.conf_rows(1000, 64, rows=40) == (40, "given")
    os.environ["ROWPAIR_CONF_ROWS"] = "24"
    try:
        assert heads.conf_rows(1000, 64) == (24, "env:ROWPAIR_CONF_ROWS"), heads.conf_rows(1000, 64)
    finally:
        os.environ.pop("ROWPAIR_CONF_ROWS", None)
    assert dict(evidence.schedule_fields())["conf_rows"] == 24
    from opt_core.mem.rowpair import frames
    assert frames.frame_rows(62000, rows=512) == (512, "given")
    assert frames.frame_rows(62000) == (1024 * 2 ** 20 // (62000 * 40), "default"), frames.frame_rows(62000)


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
