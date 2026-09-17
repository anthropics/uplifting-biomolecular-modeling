"""opt_core.mem.rowpair.template — the template embedder on row shards: P in {2, 3, 4} multi-process checks under gloo on CPU (the launcher
is the launcher: ``launch.run_sharded(P, entry, cpu_ok=True, backend="gloo")``) against an INDEPENDENT dense reference written in this file
(a synthetic AF3-style template embedder: 8 feature linears, a 2-block pair stack of triangle-attention-starting + transition with a final
LayerNorm, mean over slots / relu / output linear), plus in-process checks of the slicing seam, the slot census / all-dummy event, the row
accessor, the feature-row statements, the agreed slot de-duplication, host parking and the P == 1 refusal.

Every sharded-vs-dense comparison asserts max|diff| within the dtype class (fp32 1e-4, whole-shard blocks 1e-5; fp64 1e-9) and REPORTS the
value and the bit-exact flag (torch.equal); lazy-dummy rows vs materialised dummy rows and dedup vs no-dedup are asserted BIT-EXACT.

Run: ``python -m pytest tests/test_rowpair_template_043.py -q -rfE`` or ``python tests/test_rowpair_template_043.py`` (RESULT lines + SUMMARY).
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                      # the opt_core checkout under test

from opt_core.mem.rowpair import RowpairRefused, evidence, launch  # noqa: E402
from opt_core.mem.rowpair.dist import Layout  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False

import pytest  # noqa: E402

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch not importable")
# Sharded-vs-dense max|diff| bounds, ONE PER NUMERICS CLASS (never a global loosening); every case also REPORTS max|diff| and torch.equal per rank.
# fp64, any schedule: the row statements are the dense statements on row slices, so fp64 agrees to rounding of rounding — observed 0.0 (bit-exact).
TOL_FP64 = 1e-9
# fp32, row block == the whole shard (rows=None): each rank issues ONE launch per statement over its rows; observed bit-exact on CPU.
TOL_FP32_WHOLESHARD = 1e-5
# fp32, SUB-SHARD row blocks (rows=5/7/9 < R): the same statements launched with a smaller M than the dense N-row launch; CPU fp32 GEMM /
# reduction kernels are not bit-exact M-invariant, and this file's synthetic pair stack ends in a LayerNorm over c_t=6 channels that amplifies a
# 1e-6 input difference to ~1e-5 at low-variance elements (one element of one row in the P=2, N=61 case) — the package's fp32 class.
TOL_FP32_SUBSHARD = 1e-4
TOL = {"float32": TOL_FP32_SUBSHARD, "float64": TOL_FP64}
PAIR_KEYS = ("template_distogram", "template_unit_vector")
B_CANDS = (32, 16, 8)


# ================================================================================================================ synthetic engine (seeded, CPU)
def _synthetic(N: int, T: int, config: str, seed: int = 5, c_z: int = 8, c_t: int = 6, n_bins: int = 7, n_res: int = 5, H: int = 2, dh: int = 3,
               dtype: str = "float32"):
    """Inputs + weights of a synthetic AF3-style template embedder, seeded on CPU (identical on every rank by construction).
    config: 'real' (all slots real, distinct) | 'dummy' (all slots dummies: zero masks, +0.0 pair features) | 'mixed' (T=4: slot 2 == slot 0
    real duplicate, slots 1 and 3 dummies) | 'rowsplit' (slot 1 == slot 0 except pair rows >= N//2: a row-local verdict differs from the dense one)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    rnd = lambda *s: torch.randn(s, generator=g, dtype=torch.float64).to(dt)  # noqa: E731
    dummy = {"real": [], "dummy": list(range(T)), "mixed": [1, 3], "rowsplit": []}[config]
    tok = {}
    idx = torch.randint(0, n_res - 1, (1, T, N), generator=g)
    tok["restype"] = torch.nn.functional.one_hot(idx, n_res).to(dt)                             # [1, T, N, n_res]; class n_res-1 = gap
    tok["pbm"] = (torch.rand((1, T, N), generator=g) > 0.3).to(dt)
    tok["bfm"] = (torch.rand((1, T, N), generator=g) > 0.3).to(dt)
    tok["asym_id"] = torch.cat([torch.ones(N // 2), 2 * torch.ones(N - N // 2)]).to(torch.int64)[None]   # [1, N] two chains
    x = (torch.randn((1, T, N, 3), generator=g, dtype=torch.float64) * 6.0)                       # pseudo-beta coordinates, float64 (featuriser class)
    uv = torch.nn.functional.normalize(rnd(1, T, N, N, 3), dim=-1)                               # synthetic unit vectors (the frame statement is the engine's)
    for t in dummy:
        tok["restype"][:, t] = torch.nn.functional.one_hot(torch.full((N,), n_res - 1), n_res).to(dt)
        tok["pbm"][:, t] = 0.0
        tok["bfm"][:, t] = 0.0
        x[:, t] = float("nan")
        uv[:, t] = 0.0
    if config in ("mixed",):
        for k in ("restype", "pbm", "bfm"):
            tok[k][:, 2] = tok[k][:, 0]
        x[:, 2] = x[:, 0]
        uv[:, 2] = uv[:, 0]
    if config == "rowsplit":
        for k in ("restype", "pbm", "bfm"):
            tok[k][:, 1] = tok[k][:, 0]
        x[:, 1] = x[:, 0]
        uv[:, 1] = uv[:, 0]
        uv[:, 1, N // 2:] = torch.nn.functional.normalize(rnd(1, N - N // 2, N, 3), dim=-1)[0]   # rows >= N//2 differ (pair feature only)
    edges = torch.linspace(3.25, 20.75, n_bins, dtype=torch.float64) ** 2
    lower, upper = edges, torch.cat([edges[1:], torch.tensor([1e8], dtype=torch.float64)])
    from opt_core.mem.rowpair import template as TM
    from opt_core.mem.rowpair import trunk as TR
    same = TM.same_chain_rows(tok["asym_id"], 0, N)                                              # [1, 1, N, N, 1]
    dg = TM.distogram_rows(x, lower, upper, list(range(T)), 0, N, out_dtype=dt)                 # [1, T, N, N, n_bins]
    dg = dg * TR.pair_mask_rows(tok["pbm"], 0, N)[..., None] * same
    uvm = uv * TR.pair_mask_rows(tok["bfm"], 0, N)[..., None] * same
    feats = {"template_distogram": dg.contiguous(), "template_unit_vector": uvm.contiguous(), "x": x, "lower": lower, "upper": upper}
    W = {"ln_z_w": rnd(c_z).abs() + 0.5, "ln_z_b": rnd(c_z) * 0.1, "z": rnd(c_z, c_t) / c_z ** 0.5, "dg": rnd(n_bins, c_t), "pbm": rnd(1, c_t),
         "aa1": rnd(n_res, c_t), "aa2": rnd(n_res, c_t), "x": rnd(1, c_t), "y": rnd(1, c_t), "zc": rnd(1, c_t), "bb": rnd(1, c_t),
         "ln_o_w": rnd(c_t).abs() + 0.5, "ln_o_b": rnd(c_t) * 0.1, "t": rnd(c_t, c_z) / c_t ** 0.5, "blocks": []}
    for _ in range(2):
        W["blocks"].append({"ln1_w": rnd(c_t).abs() + 0.5, "ln1_b": rnd(c_t) * 0.1, "b": rnd(c_t, H), "q": rnd(c_t, H * dh) / c_t ** 0.5,
                            "k": rnd(c_t, H * dh) / c_t ** 0.5, "v": rnd(c_t, H * dh) / c_t ** 0.5, "o": rnd(H * dh, c_t) / (H * dh) ** 0.5,
                            "ln2_w": rnd(c_t).abs() + 0.5, "ln2_b": rnd(c_t) * 0.1, "w1": rnd(c_t, 2 * c_t) / c_t ** 0.5, "w2": rnd(2 * c_t, c_t) / c_t ** 0.5})
    z = rnd(1, N, N, c_z)
    mask = (torch.rand((N, N), generator=g) > 0.05).to(dt)                                       # pair mask [N, N] (replicated)
    dims = {"c_z": c_z, "c_t": c_t, "n_bins": n_bins, "n_res": n_res, "H": H, "dh": dh, "dummy": dummy, "dtype": dtype}
    return tok, feats, W, z, mask, dims


def _ln(x, w, b):
    import torch
    return torch.nn.functional.layer_norm(x, (int(w.shape[0]),), w, b, 1e-5)


def _embed_feats_rows(W, tok, rows_of, slots, i0, i1, g0, g1):
    """The engine's template featuriser on pair ROWS (AF3-style: 8 linears): ``rows_of(key, slots, i0, i1)`` serves the pair-feature rows
    (local i0:i1), the per-token features are indexed with GLOBAL rows g0:g1 -> a [1, S, r, N, c_t]."""
    import torch
    from opt_core.mem.rowpair import template as TM
    from opt_core.mem.rowpair import trunk as TR
    dt = W["dg"].dtype
    same = TM.same_chain_rows(tok["asym_id"], g0, g1)
    pbm2 = (TR.pair_mask_rows(tok["pbm"][..., list(slots), :], g0, g1)[..., None] * same).to(dt)
    bfm2 = (TR.pair_mask_rows(tok["bfm"][..., list(slots), :], g0, g1)[..., None] * same).to(dt)
    dg = rows_of("template_distogram", slots, i0, i1)
    uv = rows_of("template_unit_vector", slots, i0, i1)
    ux, uy, uz = uv.unbind(dim=-1)
    rt = tok["restype"][..., list(slots), :, :]
    N = int(rt.shape[-2])
    rti = rt[..., g0:g1, :][..., None, :].expand(*rt.shape[:-2], g1 - g0, N, -1)
    rtj = rt[..., None, :, :].expand(*rt.shape[:-2], g1 - g0, -1, -1)
    a = dg @ W["dg"]
    a = a + pbm2 @ W["pbm"]
    a = a + rti @ W["aa1"]
    a = a + rtj @ W["aa2"]
    a = a + ux[..., None] @ W["x"]
    a = a + uy[..., None] @ W["y"]
    a = a + uz[..., None] @ W["zc"]
    a = a + bfm2 @ W["bb"]
    return a


def _attn_rows(x_rows, tb_full, g, mask_full, Bp, H, dh):
    """Triangle attention (starting node) of pair rows g0:g1 given the whole triangle bias tb_full [N, N, H]: row-local statement."""
    import torch
    g0, g1 = g
    r, N, _c = x_rows.shape
    xl = _ln(x_rows, Bp["ln1_w"], Bp["ln1_b"])
    q = (xl @ Bp["q"]).view(r, N, H, dh)
    k = (xl @ Bp["k"]).view(r, N, H, dh)
    v = (xl @ Bp["v"]).view(r, N, H, dh)
    logits = torch.einsum("rjhd,rkhd->rhjk", q, k) / dh ** 0.5 + tb_full.permute(2, 0, 1)[None] + (mask_full[g0:g1][:, None, None, :] - 1.0) * 1e9
    p = torch.softmax(logits, dim=-1)
    o = torch.einsum("rhjk,rkhd->rjhd", p, v).reshape(r, N, H * dh)
    return o @ Bp["o"]


def _tb_of(x_rows, Bp):
    return _ln(x_rows, Bp["ln1_w"], Bp["ln1_b"]) @ Bp["b"]                                       # [r, N, H]


def _trans(x_rows, Bp):
    import torch
    return torch.relu(_ln(x_rows, Bp["ln2_w"], Bp["ln2_b"]) @ Bp["w1"]) @ Bp["w2"]


def _stack_dense(u, mask_full, W, H, dh):
    """The per-slot template pair stack, dense: u [1, 1, N, N, c_t] -> same shape (2 blocks + final LayerNorm)."""
    x = u[0, 0]
    N = int(x.shape[0])
    for Bp in W["blocks"]:
        x = x + _attn_rows(x, _tb_of(x, Bp), (0, N), mask_full, Bp, H, dh)
        x = x + _trans(x, Bp)
    return _ln(x, W["ln_o_w"], W["ln_o_b"])[None, None]


def _finish(t, W):
    """Closing statement on t [1, T, rows, N, c_t]: mean over slots, relu, output linear -> [1, rows, N, c_z]."""
    import torch
    n_templ = int(t.shape[-4])
    t = torch.sum(t, dim=-4) / n_templ
    return torch.relu(t) @ W["t"]


def _dense_reference(N, T, tok, feats, W, z, mask, dims):
    """INDEPENDENT dense statement of the whole embedder (no opt_core driver): z + linear(relu(mean_t stack(linear_z(LN(z)) + a_t)))."""
    import torch
    rows_of = lambda key, slots, i0, i1: feats[key][..., list(slots), i0:i1, :, :]  # noqa: E731  (whole tensors, r0 = 0)
    zz = _ln(z, W["ln_z_w"], W["ln_z_b"]) @ W["z"]                                             # [1, N, N, c_t]
    us = []
    for t in range(T):
        a = _embed_feats_rows(W, tok, rows_of, [t], 0, N, 0, N)                                 # [1, 1, N, N, c_t]
        us.append(_stack_dense(zz[:, None] + a, mask, W, dims["H"], dims["dh"]))
    tt = torch.cat(us, dim=-4)                                                                   # [1, T, N, N, c_t]
    return z + _finish(tt, W), len(us)


# ================================================================================================================ multi-process entry (one rank)
def _entry_template(N: int, T: int, config: str, rows, provider: str, dedupe: bool, park: bool, add: bool, dtype: str = "float32"):
    import torch
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair import template as TM
    from opt_core.mem.rowpair import trunk as TR
    from opt_core.mem.rowpair.shard import shard_rows, unshard_rows
    from opt_core.mem.rowpair.transition import transition_rows
    from opt_core.mem.rowpair.triatt import gather_triangle_bias, triatt_starting
    torch.set_num_threads(1)
    P, r = D.world()
    lay = D.Layout.auto(N, P, r, candidates=B_CANDS)
    tok, feats, W, z, mask, dims = _synthetic(N, T, config, dtype=dtype)
    H, dh, c_t = dims["H"], dims["dh"], dims["c_t"]
    tol = TOL[dtype] if rows is not None else min(TOL[dtype], TOL_FP32_WHOLESHARD)
    res = {"case": "template", "P": P, "N": N, "T": T, "config": config, "rows": rows, "provider": provider, "dedupe": dedupe, "park": park,
           "add": add, "dtype": dtype, "tol": tol, "B": lay.B, "bounds": lay.bounds, "checks": {}, "metrics": {}}
    calls = {"stack": 0, "unit": 0}
    with torch.no_grad():
        z_ref, dense_stack_calls = _dense_reference(N, T, tok, feats, W, z, mask, dims)
        # ---- entry seam: pair-shaped keys become rows once
        fr = TM.slice_template_inputs_to_rows(feats, lay, PAIR_KEYS)
        for k in PAIR_KEYS:
            assert tuple(fr[k].shape) == (1, T, lay.R, N, feats[k].shape[-1]), (k, fr[k].shape)
            assert torch.equal(fr[k], feats[k][:, :, lay.r0:lay.r1]), k
            assert fr[k]._base is feats[k] or fr[k].data_ptr() == feats[k][:, :, lay.r0:lay.r1].data_ptr(), "slice must be a view"
        assert fr["x"] is feats["x"]
        census = TM.slot_census(tok["pbm"], tok["bfm"], slot_dim=-2)
        assert list(census.dummy) == dims["dummy"], (census, dims["dummy"])
        note = TM.note_all_dummy(census, templated=True, refuse=False)
        assert (note is not None) == (config == "dummy"), note
        # ---- row accessor
        if provider == "rows":
            prov = TM.TemplatePairRows(lay, T, rows={k: fr[k] for k in PAIR_KEYS}, lead=(1,))
        elif provider == "lazy":
            prov = TM.TemplatePairRows(lay, T, lazy_dummy=census, feature_dims={"template_distogram": dims["n_bins"], "template_unit_vector": 3},
                                       lead=(1,), dtype=z.dtype, device=z.device)
        else:   # computed: distogram rows from coordinates per row block; unit vectors served from the sliced rows (the frame statement is the engine's)
            def dg_rows(slots, g0, g1):
                d = TM.distogram_rows(feats["x"], feats["lower"], feats["upper"], slots, g0, g1, out_dtype=z.dtype)
                return d * TR.pair_mask_rows(tok["pbm"][..., list(slots), :], g0, g1)[..., None] * TM.same_chain_rows(tok["asym_id"], g0, g1)
            prov = TM.TemplatePairRows(lay, T, computed={"template_distogram": dg_rows,
                                                        "template_unit_vector": lambda slots, g0, g1: fr["template_unit_vector"][..., list(slots), g0 - lay.r0:g1 - lay.r0, :, :]})
        # ---- agreed slot groups (keys: per-token feats + this rank's pair ROWS, slot on dim 0)
        keys = [tok["restype"].movedim(1, 0), tok["pbm"].movedim(1, 0), tok["bfm"].movedim(1, 0)] + [fr[k].movedim(1, 0) for k in PAIR_KEYS]
        groups = TM.template_slot_groups(keys, T, lay, dedupe=dedupe)
        expect = {"real": [(t, [t]) for t in range(T)], "dummy": [(0, list(range(T)))], "mixed": [(0, [0, 2]), (1, [1, 3])],
                  "rowsplit": [(t, [t]) for t in range(T)]}[config] if dedupe else [(t, [t]) for t in range(T)]
        if config == "dummy" and T == 1:
            expect = [(0, [0])]
        assert groups == expect, (groups, expect)
        # local verdict of the rowsplit case differs across ranks (rows < N//2 see slots 0/1 equal) -> the agreement is what makes groups dense-correct
        if config == "rowsplit" and dedupe:
            local_equal = torch.equal(fr["template_unit_vector"][:, 0], fr["template_unit_vector"][:, 1])
            res["metrics"].setdefault("rowsplit_local_equal_rank", {})[str(r)] = bool(local_equal)

        # ---- engine callables
        def unit_rows_fn(z_rows, slot, g):
            g0, g1 = g
            calls["unit"] += 1
            a = _embed_feats_rows(W, tok, prov.rows, [slot], g0 - lay.r0, g1 - lay.r0, g0, g1)
            zz = _ln(z_rows, W["ln_z_w"], W["ln_z_b"]) @ W["z"]
            return zz[..., None, :, :, :] + a

        def pair_stack_fn(u, mask_loc):
            calls["stack"] += 1
            x = u[0, 0].contiguous()                                                             # [R, N, c_t] this rank's rows
            for Bp in W["blocks"]:
                tb_full = gather_triangle_bias(_tb_of(x, Bp), lay)                               # the one collective of the starting node
                y = triatt_starting(lambda zr, tb, gg, Bp=Bp: _attn_rows(zr, tb, gg, mask, Bp, H, dh), x, tb_full, lay, q_rows=8)
                x = x + y
                transition_rows(lambda blk, Bp=Bp: _trans(blk, Bp), x, lay, rows=8, add=True)
            return _ln(x, W["ln_o_w"], W["ln_o_b"])[None, None]

        finish_fn = lambda t: _finish(t, W)  # noqa: E731
        mask_loc = mask[lay.r0:lay.r1][None, None]                                              # [1, 1, R, N] (the engine's slot-dim expansion)
        z_loc = shard_rows(z, lay, dim=-3).contiguous().clone()                                  # [1, R, N, c_z] owns its storage (parkable)
        z_obj = z_loc
        out = TM.template_embed_rows(z_loc, lay, n_templ=T, unit_rows_fn=unit_rows_fn, pair_stack_fn=pair_stack_fn, finish_fn=finish_fn,
                                     mask_loc=mask_loc, slot_groups=groups, rows=rows, c_t=c_t, add=add, park_z=park, park_u=park)
        if add:
            assert out is z_obj and out.data_ptr() == z_loc.data_ptr()
        got = out
        mine_ref = z_ref[:, lay.r0:lay.r1]
        maxabs = float((got.double() - mine_ref.double()).abs().max())
        bitwise = bool(torch.equal(got, mine_ref))
        full = unshard_rows(got[0].contiguous(), lay)                                            # [N, N, c_z] on every rank
        maxabs_full = float((full.double() - z_ref[0].double()).abs().max())
        # the assembly itself is pure data movement: bit-exact on the (ragged) grid, so any full-vs-dense excess is a rank's OWN arithmetic
        res["checks"]["gather_own_rows_bitexact"] = bool(torch.equal(full[lay.r0:lay.r1], got[0]))
        res["checks"]["gather_roundtrip_bitexact"] = bool(torch.equal(unshard_rows(shard_rows(z_ref[0], lay, dim=0).contiguous(), lay), z_ref[0]))
        res["checks"]["rows_within_tol"] = maxabs <= tol
        res["checks"]["full_within_tol"] = maxabs_full <= tol
        res["checks"]["stack_calls_eq_groups"] = calls["stack"] == len(groups)
        sched = evidence.schedule()
        if sched.get("templ_order") == "zparked":                                                 # ROWPAIR_TEMPL_ORDER=zparked (test_mp_template_zparked, 0.5.220.6):
            want_park_z = "parked"                                                                # z is ALWAYS parked in that order; park_u parks the NON-last slabs
            want_park_u = (f"{len(groups) - 1}of{len(groups)}" if len(groups) > 1 else "off") if park else "off"
        else:                                                                                     # the classic order (word unset): unchanged expectations
            want_park_z = "parked" if park else "off"
            want_park_u = (f"{len(groups)}of{len(groups)}" if len(groups) > 1 else "off") if park else "off"
        res["checks"]["schedule_recorded"] = (sched.get("templ_slots") == T and sched.get("templ_groups") == TM.groups_word(groups)
                                             and (sched.get("templ_rows_source") == "given" if rows is not None else str(sched.get("templ_rows_source")).startswith("default"))
                                             and sched.get("templ_mode") == {"rows": "rows", "lazy": "lazy_dummy", "computed": "computed"}[provider]
                                             and sched.get("templ_pair_keys") == ",".join(PAIR_KEYS) and sched.get("templ_replicated") == "token_feats"
                                             and sched.get("templ_park_z") == want_park_z
                                             and sched.get("templ_park_u") == want_park_u)
        res["schedule"] = {k: v for k, v in sched.items() if str(k).startswith("templ")}
        # ---- lazy-dummy rows == materialised dummy rows, BIT-EXACT (same P, same schedule)
        if config == "dummy" and provider == "lazy":
            prov_rows = TM.TemplatePairRows(lay, T, rows={k: fr[k] for k in PAIR_KEYS}, lead=(1,))
            for (i0, i1) in ((0, lay.R), (0, min(3, lay.R)), (max(0, lay.R - 5), lay.R)):
                for k in PAIR_KEYS:
                    assert torch.equal(prov.rows(k, list(range(T)), i0, i1), prov_rows.rows(k, list(range(T)), i0, i1)), (k, i0, i1)
            prov_save = prov
            prov = prov_rows                                                                     # unit_rows_fn reads `prov`
            z2 = shard_rows(z, lay, dim=-3).contiguous().clone()
            out2 = TM.template_embed_rows(z2, lay, n_templ=T, unit_rows_fn=unit_rows_fn, pair_stack_fn=pair_stack_fn, finish_fn=finish_fn,
                                          mask_loc=mask_loc, slot_groups=groups, rows=rows, c_t=c_t, add=True)
            res["checks"]["lazy_eq_materialised_bitwise"] = bool(torch.equal(got, out2))
            prov = prov_save
        # ---- dedup on == dedup off, BIT-EXACT (identical slots give identical u; the closing mean sees the same operands in the same order)
        if dedupe and len(groups) < T:
            g_off = TM.template_slot_groups(keys, T, lay, dedupe=False)
            assert g_off == [(t, [t]) for t in range(T)]
            before = calls["stack"]
            z3 = shard_rows(z, lay, dim=-3).contiguous().clone()
            out3 = TM.template_embed_rows(z3, lay, n_templ=T, unit_rows_fn=unit_rows_fn, pair_stack_fn=pair_stack_fn, finish_fn=finish_fn,
                                          mask_loc=mask_loc, slot_groups=g_off, rows=rows, c_t=c_t, add=True)
            res["checks"]["dedup_eq_nodedup_bitwise"] = bool(torch.equal(got, out3))
            res["checks"]["nodedup_stack_calls_eq_T"] = calls["stack"] - before == T
        # ---- per-rank metrics to rank 0 (one gather every rank issues)
        piece = torch.tensor([[maxabs, 1.0 if bitwise else 0.0, float(calls["stack"]), float(lay.R), maxabs_full]], dtype=torch.float64)
        allm = D.gather_cat_to_rank0(piece, [1] * P)
        res["metrics"]["local"] = {"rank": r, "maxabs_rows": maxabs, "maxabs_full": maxabs_full, "bitwise": bitwise,
                                   "per_row_maxabs_full": [float(v) for v in (full.double() - z_ref[0].double()).abs().amax(dim=(1, 2)).tolist()],
                                   "ref_absmax": float(z_ref.abs().max()), "term_absmax": float((z_ref - z).abs().max())}
    for k, v in res["checks"].items():
        assert v or os.environ.get("ROWPAIR_TEMPLATE_DIAG") == "1", (k, res)
    if allm is None:
        return None
    res["metrics"]["maxabs_rows_per_rank"] = [float(x) for x in allm[:, 0].tolist()]
    res["metrics"]["bitwise_rows_per_rank"] = [bool(x) for x in (allm[:, 1] > 0.5).tolist()]
    res["metrics"]["stack_calls_per_rank"] = [int(x) for x in allm[:, 2].tolist()]
    res["metrics"]["rows_per_rank"] = [int(x) for x in allm[:, 3].tolist()]
    res["metrics"]["maxabs_full"] = float(allm[0, 4])
    res["metrics"]["dense_stack_calls"] = dense_stack_calls
    res["ok"] = all(res["checks"].values()) and max(res["metrics"]["maxabs_rows_per_rank"]) <= tol and len(set(res["metrics"]["stack_calls_per_rank"])) == 1
    return res


def _mp(P, entry, *args, **kw):
    return launch.run_sharded(P, entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=120, run_timeout_s=600, **kw)


def _report(res):
    print("RESULT " + json.dumps(res, sort_keys=True, default=str))
    assert res["ok"], res


# ================================================================================================================ multi-process cases
# (P, N, T, config, rows, provider, dedupe, park, add, dtype): uneven N (61: ragged last rank / B per Layout.auto), T in {1, 4}, every provider
# mode, multi-block row schedules (rows=5/7/9) and the whole-shard schedule (rows=None -> the 512 MiB default clamps to the shard), parking on,
# out-of-place; fp32 asserted at the package's fp32 class (1e-4; whole-shard blocks 1e-5) and fp64 at 1e-9 (statement equality beyond fp32
# rounding: a row-blocked launch differs from the dense launch only in M, which CPU fp32 kernels do not keep bit-exact).
MP_CASES = [
    (2, 61, 4, "real", 5, "rows", True, False, True, "float32"),
    (2, 61, 4, "real", 5, "rows", True, False, True, "float64"),
    (2, 61, 1, "real", None, "computed", True, False, True, "float32"),
    (2, 61, 4, "dummy", 5, "lazy", True, False, True, "float32"),
    (3, 61, 4, "mixed", 7, "rows", True, True, True, "float32"),
    (3, 61, 4, "mixed", 7, "rows", True, True, True, "float64"),
    (3, 61, 4, "rowsplit", 5, "rows", True, False, False, "float32"),
    (4, 61, 4, "mixed", None, "computed", True, False, True, "float32"),
    (4, 64, 1, "dummy", 5, "lazy", True, True, True, "float32"),
    (4, 64, 4, "real", 9, "rows", False, False, True, "float32"),
    (4, 61, 4, "real", 5, "computed", True, False, True, "float64"),
]


@needs_torch
@pytest.mark.parametrize("P,N,T,config,rows,provider,dedupe,park,add,dtype", MP_CASES)
def test_mp_template(P, N, T, config, rows, provider, dedupe, park, add, dtype):
    assert os.environ.get("ROWPAIR_TEMPL_ORDER", "") == ""                                       # the CLASSIC order (word unset) — a leaked word from another module fails HERE by name
    _report(_mp(P, _entry_template, N, T, config, rows, provider, dedupe, park, add, dtype))


ZPARKED_MP_CASES = [MP_CASES[0], MP_CASES[4], MP_CASES[6], MP_CASES[8], MP_CASES[10]]         # real / mixed+parked slabs / rowsplit out= form / dummy-lazy T=1 / computed fp64; P 2..4


@needs_torch
@pytest.mark.parametrize("P,N,T,config,rows,provider,dedupe,park,add,dtype", ZPARKED_MP_CASES)
def test_mp_template_zparked(monkeypatch, P, N, T, config, rows, provider, dedupe, park, add, dtype):
    """``ROWPAIR_TEMPL_ORDER=zparked`` under REAL gloo process groups (0.5.220.6; the spawned ranks inherit the word): the z-parked order runs the
    same statements — every check of :func:`test_mp_template` holds (own rows / the assembled pair tensor within the fp32 / fp64 class of the dense
    reference, gather bit-exact, one pair-stack call per slot group) and the census carries ``templ_order=zparked`` with z parked on every rank.
    (zparked == classic BITWISE is asserted rank-thread-wise in test_rowpair_templ_zparked_0515; this is the process-group leg.)"""
    monkeypatch.setenv("ROWPAIR_TEMPL_ORDER", "zparked")
    res = _mp(P, _entry_template, N, T, config, rows, provider, dedupe, park, add, dtype)
    _report(res)                                                                                  # rank 0's record (every rank asserted its own checks in _entry_template)
    assert res["schedule"].get("templ_order") == "zparked" and res["schedule"].get("templ_park_z") == "parked", res["schedule"]


# ================================================================================================================ in-process (P == 1 / no group)
@needs_torch
def test_p1_template_embed_rows_refuses_by_name():
    import torch
    from opt_core.mem.rowpair import template as TM
    lay = Layout(40, 1, 0, 16)
    z = torch.zeros(1, 40, 40, 4)
    try:
        TM.template_embed_rows(z, lay, n_templ=1, c_t=4, unit_rows_fn=lambda *a: None, pair_stack_fn=lambda *a: None, finish_fn=lambda *a: None, rows=4)
        raise AssertionError("expected refusal")
    except RowpairRefused as e:
        assert "template_embed_rows: refused at n_gpu=1" in e.reason and "the adapter installs nothing at n_gpu=1" in e.reason, e.reason
    lay2 = Layout(40, 2, 0, 128)                                   # replicated grid (N < P*B) -> refused too
    assert lay2.replicated
    with pytest.raises(RowpairRefused):
        TM.template_embed_rows(z, lay2, n_templ=1, c_t=4, unit_rows_fn=lambda *a: None, pair_stack_fn=lambda *a: None, finish_fn=lambda *a: None, rows=4)


@needs_torch
def test_p1_dense_helper_equals_independent_reference():
    """template_embed_dense (the module's dense statement) == this file's independent dense reference, bit-exact (one device, same statements)."""
    import torch
    from opt_core.mem.rowpair import template as TM
    N, T = 24, 4
    tok, feats, W, z, mask, dims = _synthetic(N, T, "mixed")
    with torch.no_grad():
        z_ref, n = _dense_reference(N, T, tok, feats, W, z, mask, dims)
        rows_of = lambda key, slots, i0, i1: feats[key][..., list(slots), i0:i1, :, :]  # noqa: E731
        unit_fn = lambda zr, slot, g: (_ln(zr, W["ln_z_w"], W["ln_z_b"]) @ W["z"])[..., None, :, :, :] + _embed_feats_rows(W, tok, rows_of, [slot], g[0], g[1], g[0], g[1])  # noqa: E731
        got = TM.template_embed_dense(z, n_templ=T, unit_fn=unit_fn, pair_stack_fn=lambda u, m: _stack_dense(u, mask, W, dims["H"], dims["dh"]),
                                      finish_fn=lambda t: _finish(t, W), mask=None)
    assert n == T and torch.equal(got, z_ref), float((got - z_ref).abs().max())
    assert not torch.equal(got, z), "the template term must be non-zero in this synthetic case"


@needs_torch
def test_slice_template_inputs_to_rows_words():
    import torch
    from opt_core.mem.rowpair import template as TM
    N, T = 40, 2
    feats = {"template_distogram": torch.randn(1, T, N, N, 7), "template_unit_vector": torch.randn(1, T, N, N, 3), "template_restype": torch.randn(1, T, N, 5)}
    lay1 = Layout(N, 1, 0, 16)
    out1 = TM.slice_template_inputs_to_rows(feats, lay1, PAIR_KEYS)
    assert all(out1[k] is feats[k] for k in feats)                                              # equality at P == 1
    lays = [Layout(N, 3, q, 16) for q in range(3)]                                              # (0,16),(16,32),(32,40): no group needed for the view
    for lay in lays:
        out = TM.slice_template_inputs_to_rows(feats, lay, PAIR_KEYS)
        for k in PAIR_KEYS:
            assert tuple(out[k].shape) == (1, T, lay.R, N, feats[k].shape[-1]) and torch.equal(out[k], feats[k][:, :, lay.r0:lay.r1])
        assert out["template_restype"] is feats["template_restype"]
    assert evidence.schedule()["templ_pair_keys"] == "template_distogram,template_unit_vector" and evidence.schedule()["templ_pair_rows"] == lays[-1].R
    for bad, words in (({"template_distogram": torch.randn(1, T, 1, 1, 7)}, "dims -3 and -2"), ({}, "is not in the features")):
        try:
            TM.slice_template_inputs_to_rows(bad, lays[0], ["template_distogram"])
            raise AssertionError("expected refusal")
        except RowpairRefused as e:
            assert words in e.reason, e.reason
    assert TM.slice_template_inputs_to_rows({}, lays[0], ["template_distogram"], missing="skip") == {}


@needs_torch
def test_slot_census_and_all_dummy_event():
    import torch
    from opt_core.mem.rowpair import template as TM
    T, N = 4, 10
    pbm = torch.zeros(1, T, N)
    bfm = torch.zeros(1, T, N)
    c = TM.slot_census(pbm, bfm)
    assert c.all_dummy and c.real == () and c.dummy == (0, 1, 2, 3) and c.words() == {"templ_slots": 4, "templ_real": 0, "templ_dummy": 4}
    assert TM.note_all_dummy(c, templated=False) is None                                      # untemplated run: nothing to note
    note = TM.note_all_dummy(c, templated=True, refuse=False, tag="kit")
    assert note.startswith("NOTE kit: templates were requested but all 4 template slots featurised as dummies (event=all_slots_dummy")
    assert evidence.schedule()["templ_event"] == TM.ALL_DUMMY_EVENT == "all_slots_dummy"
    try:
        TM.note_all_dummy(c, templated=True, refuse=True, tag="kit")
        raise AssertionError("expected refusal")
    except RowpairRefused as e:
        assert e.reason.startswith("refused: kit: templates were requested but all 4 template slots featurised as dummies"), e.reason
    bfm[0, 2, 5] = 1.0
    c2 = TM.slot_census(pbm, bfm)
    assert not c2.all_dummy and c2.real == (2,) and c2.dummy == (0, 1, 3) and c2.is_dummy(1) and not c2.is_dummy(2)
    assert TM.note_all_dummy(c2, templated=True, refuse=True) is None
    # the lazy-dummy accessor refuses by name when a slot is real; serves +0.0 rows otherwise
    lay = Layout(N, 1, 0, 4)
    try:
        TM.TemplatePairRows(lay, T, lazy_dummy=c2, feature_dims={"d": 7}, dtype=torch.float32, device="cpu")
        raise AssertionError("expected refusal")
    except RowpairRefused as e:
        assert "slots [2] are REAL templates" in e.reason, e.reason
    pr = TM.TemplatePairRows(lay, T, lazy_dummy=c, feature_dims={"d": 7}, dtype=torch.float32, device="cpu", lead=(1,))
    zr = pr.rows("d", [0, 3], 2, 6)
    assert tuple(zr.shape) == (1, 2, 4, N, 7) and zr.dtype == torch.float32 and bool((zr == 0).all()) and not bool(torch.signbit(zr).any())
    assert pr.mode == "lazy_dummy" and evidence.schedule()["templ_mode"] == "lazy_dummy" and pr.keys() == ["d"]
    with pytest.raises(RowpairRefused):
        TM.TemplatePairRows(lay, T)                                                                 # no source
    with pytest.raises(RowpairRefused):
        TM.TemplatePairRows(lay, T, rows={"d": torch.zeros(1, T, N + 1, N, 7)})                    # wrong row count


@needs_torch
def test_feature_row_statements_equal_dense_slices():
    import torch
    from opt_core.mem.rowpair import template as TM
    from opt_core.mem.rowpair import trunk as TR
    N, T = 23, 3
    g = torch.Generator().manual_seed(3)
    x = torch.randn((2, T, N, 3), generator=g, dtype=torch.float64) * 7                          # leading batch dim 2
    x[0, 1, 4] = float("nan")
    edges = torch.linspace(3.25, 50.75, 39, dtype=torch.float64) ** 2
    lower, upper = edges, torch.cat([edges[1:], torch.tensor([1e8], dtype=torch.float64)])
    dense = TM.distogram_rows(x, lower, upper, list(range(T)), 0, N)
    assert dense.shape == (2, T, N, N, 39) and dense.dtype == torch.float64
    # independent statement: cdist-free explicit loop over bins
    d2 = ((x[:, :, :, None, :] - x[:, :, None, :, :]) ** 2)
    d2 = ((d2[..., 0] + d2[..., 1]) + d2[..., 2])[..., None]
    ref = ((d2 > lower) & (d2 < upper)).to(torch.float64)
    assert torch.equal(dense, ref)
    assert float(dense[0, 1, 4].abs().sum()) == 0.0 and float(dense[0, 1, :, 4].abs().sum()) == 0.0      # NaN coordinate -> all-zero one-hot rows/cols
    assert bool((dense.sum(-1) <= 1).all())
    for (g0, g1) in ((0, N), (0, 5), (7, 19), (22, 23)):
        for slots in ([0], [2, 0], [0, 1, 2]):
            assert torch.equal(TM.distogram_rows(x, lower, upper, slots, g0, g1), dense[:, slots, g0:g1])
            assert torch.equal(TM.distogram_rows(x, lower, upper, slots, g0, g1, out_dtype=torch.float32), dense[:, slots, g0:g1].float())
    # masks
    m = (torch.rand((2, T, N), generator=g) > 0.5).float()
    pm_dense = m[..., :, None] * m[..., None, :]
    aid = torch.tensor([1] * 10 + [2] * 8 + [3] * 5)[None].expand(2, N)
    grp = torch.full((2, T, N), -1, dtype=torch.int32)
    grp[:, 1, :18] = 0                                                                          # slot 1: chains 1+2 grouped
    grp[:, 2, 10:] = 5                                                                          # slot 2: chains 2+3 grouped
    same_dense = (aid[..., :, None] == aid[..., None, :])
    sog_dense = TM.same_or_group_dense(aid, grp)
    assert sog_dense.shape == (2, T, N, N, 1) and sog_dense.dtype == torch.bool
    brute = torch.zeros(2, T, N, N, dtype=torch.bool)
    for t in range(T):
        for i in range(N):
            for j in range(N):
                brute[:, t, i, j] = (aid[0, i] == aid[0, j]) | ((grp[0, t, i] >= 0) & (grp[0, t, i] == grp[0, t, j]))
    assert torch.equal(sog_dense[..., 0], brute)
    for (g0, g1) in ((0, N), (3, 11), (18, 23)):
        for slots in ([1], [2, 0], [0, 1, 2]):
            assert torch.equal(TR.pair_mask_rows(m[:, slots], g0, g1), pm_dense[:, slots, g0:g1])
            assert torch.equal(TM.same_chain_rows(aid, g0, g1)[..., 0, :, :, 0], same_dense[:, g0:g1])
            assert torch.equal(TM.same_or_group_rows(aid, grp, slots, g0, g1), sog_dense[:, slots, g0:g1])
    # pairs_unmasked closed form vs brute force count of newly un-masked ordered pairs (asym_i != asym_j)
    pu = TM.pairs_unmasked(aid, grp)
    brute_pu = [int(((brute[0, t]) & ~same_dense[0]).sum()) for t in range(T)]
    assert pu == brute_pu == [0, 2 * 10 * 8, 2 * 8 * 5], (pu, brute_pu)


@needs_torch
def test_slot_groups_local_and_words():
    import torch
    from opt_core.mem.rowpair import template as TM
    T = 4
    a = torch.randn(T, 3, 5)
    a[2] = a[0]
    a[3] = a[1]
    b = torch.randn(T, 7)
    b[2] = b[0]
    b[3] = b[1]
    b[3, 0] = float("nan")
    b[1, 0] = float("nan")                                                                      # NaN-aware equality
    assert TM.template_slot_groups([a, b], T) == [(0, [0, 2]), (1, [1, 3])]
    assert evidence.schedule()["templ_groups"] == "0+2/1+3" and evidence.schedule()["templ_dedupe"] == "on"
    assert TM.template_slot_groups([a, b], T, dedupe=False) == [(t, [t]) for t in range(T)] and evidence.schedule()["templ_dedupe"] == "off"
    os.environ[TM.ENV_TEMPL_NODEDUPE] = "1"
    try:
        assert TM.template_slot_groups([a, b], T) == [(t, [t]) for t in range(T)]
    finally:
        os.environ.pop(TM.ENV_TEMPL_NODEDUPE)
    b[3, 1] += 1.0
    assert TM.template_slot_groups([a, b], T, Layout(20, 1, 0, 8)) == [(0, [0, 2]), (1, [1]), (3, [3])]
    assert TM.groups_word([(0, [0, 2]), (1, [1]), (3, [3])]) == "0+2/1/3"
    with pytest.raises(RowpairRefused):
        TM.template_slot_groups([a.movedim(0, 1)], T)                                           # slot dim must be 0


@needs_torch
def test_parked_storage_roundtrip_cpu():
    import torch
    from opt_core.mem.rowpair import template as TM
    z = torch.randn(1, 12, 9, 4)
    ref = z.clone()
    view = z[:, 2:5]
    logs = []
    p = TM.ParkedStorage(z, name="z", log=logs.append)
    assert p.ok and z.untyped_storage().nbytes() == 0 and "storage released" in logs[-1]
    assert torch.equal(p.block(3, 7), ref[:, 3:7])
    back = p.restore()
    assert back is z and torch.equal(z, ref) and torch.equal(view, ref[:, 2:5]) and z.untyped_storage().nbytes() == ref.numel() * 4
    q = TM.ParkedStorage(z[:, 1:], name="view")                                                  # does not own its storage -> not parked, said
    assert not q.ok and "does not own" in q.how and q.restore() is not None and torch.equal(q.block(0, 2), ref[:, 1:3])


@needs_torch
def test_templ_block_rows_sources():
    from opt_core.mem.rowpair import template as TM
    from opt_core.mem.rowpair.shard import choose_block_rows
    ch = 64 * (2 + 4) + 128 + 64
    os.environ.pop("ROWPAIR_ROWBLK_MB", None)
    assert TM.templ_block_rows(1000, 64, 4, 128) == choose_block_rows(1000, ch, 4) == (2 ** 29 // (1000 * ch * 4), "default")   # 512 MiB target
    assert evidence.schedule()["templ_rows"] == 2 ** 29 // (1000 * ch * 4) and evidence.schedule()["templ_rows_source"] == "default"
    assert TM.templ_block_rows(1000, 64, 4, 128, rows=7) == (7, "given") and evidence.schedule()["templ_rows_source"] == "given"
    assert TM.templ_block_rows(1000, 64, 4, 128, n_max=33) == (33, "default+n_max")                 # never longer than the longest shard
    os.environ["ROWPAIR_ROWBLK_MB"] = "1"
    try:
        assert TM.templ_block_rows(1000, 64, 4, 128) == (max(1, 2 ** 20 // (1000 * ch * 4)), "env:ROWPAIR_ROWBLK_MB")
        assert TM.templ_block_rows(1000, 64, 4, 128, rows=5) == (5, "given")                     # the argument wins over the env
    finally:
        os.environ.pop("ROWPAIR_ROWBLK_MB")
    assert TM.templ_block_rows(10, 1, 1, 1, elem_bytes=8)[0] >= 1


def test_import_is_light():
    """Importing the module imports no torch (the package contract): a fresh interpreter with torch masked imports it."""
    import subprocess
    code = ("import sys\n"
            "class _Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            raise ImportError('masked')\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Block())\n"
            f"sys.path.insert(0, {os.path.dirname(HERE)!r})\n"
            "import opt_core.mem.rowpair.template as T\n"
            "assert 'template_embed_rows' in T.__all__ and 'torch' not in sys.modules\n"
            "print('ok')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]


if __name__ == "__main__":
    import inspect
    fails, ran = [], 0
    mod = sys.modules[__name__]
    t_all = time.monotonic()
    for name, fn in sorted(inspect.getmembers(mod, inspect.isfunction)):
        if not name.startswith("test_"):
            continue
        marks = getattr(fn, "pytestmark", [])
        params = [m for m in marks if m.name == "parametrize"]
        if any(m.name == "skipif" and m.args and m.args[0] for m in marks):
            print(f"SKIP {name}")
            continue
        combos = [dict()]
        for m in params:
            keys = [k.strip() for k in m.args[0].split(",")]
            combos = [dict(c, **dict(zip(keys, (v if isinstance(v, (tuple, list)) else (v,))))) for c in combos for v in m.args[1]]
        for kw in combos:
            ran += 1
            t0 = time.monotonic()
            try:
                fn(**kw)
                print(f"PASS {name} {kw} {time.monotonic() - t0:.1f}s")
            except BaseException as e:  # noqa: BLE001
                fails.append((name, kw, repr(e)[:400]))
                print(f"FAIL {name} {kw} {e!r}"[:600])
    print(f"SUMMARY ran={ran} failed={len(fails)} wall_s={time.monotonic() - t_all:.0f}")
    for f in fails:
        print("FAILED", f)
    sys.exit(1 if fails else 0)
