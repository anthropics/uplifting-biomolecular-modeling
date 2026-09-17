"""opt_core.mem.rowpair 0.4.3 — cross-seam END-TO-END CPU proof on the synthetic AF3-family model (``tests/synthetic_af3.py``).

The dense forward (``synthetic_af3.forward_dense``: whole ``[N, N, c_z]`` statements, no layout, no comm — the independent reference) is
compared with the row-sharded forward below, in which the pair tensor is a ROW SHARD ``z_loc = z[r0:r1]`` from the statement that creates it
(pair init) through recycling, the template embedder, the MSA module, the Pairformer, the distogram / confidence heads and diffusion
conditioning + the DiffusionTransformer, wired through the seams of ``opt_core.mem.rowpair`` (``SEAM`` table: seam name -> the core
callable this file drives; an entry whose core function is absent from the tree runs the same statement composition from the primitives
present and says so in ``SEAM_LOCAL``). Ranks are threads of one interpreter (``opt_core.testing.run_ranks``, the ``threaded`` comm backend).

Asserted per case (P in {2, 3, 4}; N even and uneven; grid and chunk-aligned layouts): every output (coords, distogram contact map, PAE / PDE
matrices, pLDDT logits, pTM / ipTM / per-chain tables, s / z after the trunk) equals the dense output to ``<= TOL`` fp32 max|diff|, and
``torch.equal`` is REPORTED per output; no rank ever produces a tensor with ``numel >= N*N*c_z`` inside the sharded forward (allocation guard,
a ``TorchFunctionMode`` over every torch call of the rank thread); the schedule census names the block sizes used and every
replicated-by-design tensor. P = 1: the sharded entry points REFUSE BY NAME (the structural n_gpu=1 rule; exercised for real, no monkeypatch).

Run: ``python -m pytest tests/test_rowpair_e2e_043.py -q -rfE`` or ``python tests/test_rowpair_e2e_043.py`` (RESULT lines + SUMMARY; rc != 0
on failure). CPU only; ``OMP_NUM_THREADS=1`` recommended (the suite command pins it).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                      # the opt_core checkout under test

import pytest  # noqa: E402

try:
    import torch  # noqa: E402
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False

pytestmark = pytest.mark.skipif(not HAVE_TORCH, reason="torch not importable")

if HAVE_TORCH:
    from torch.overrides import TorchFunctionMode  # noqa: E402

    from opt_core.mem.rowpair import RowpairRefused  # noqa: E402
    from opt_core.mem.rowpair import confidence as CONF  # noqa: E402
    from opt_core.mem.rowpair import dist as D  # noqa: E402
    from opt_core.mem.rowpair import evidence as EV  # noqa: E402
    from opt_core.mem.rowpair import shard as SH  # noqa: E402
    from opt_core.mem.rowpair import transition as TR  # noqa: E402
    from opt_core.mem.rowpair import triatt as TA  # noqa: E402
    from opt_core.mem.rowpair import trimul as TM  # noqa: E402
    from opt_core.mem.rowpair.dist import Layout  # noqa: E402
    from opt_core.testing import run_ranks  # noqa: E402

    from tests import synthetic_af3 as SA  # noqa: E402

TOL = 1e-5                       # fp32 max|diff| per output, times max(1, max|dense output|) (coordinates are O(10-100)); torch.equal is REPORTED
# every per-rank-derived block size is an explicit argument (0.4.2 schedule-desync rule) and lands in the schedule census
BLK = dict(init_rows=8, recycle_rows=8, templ_rows=8, opm_rows=8, pwa_rows=8, trans_rows=8, triatt_qrows=4, trimul_RA=8, trimul_RB=8,
           logits_rows=8, cond_rows=8, conf_rows=8)
REPLICATED_BY_DESIGN = ("s", "s_input", "m", "msa_feat", "x_atoms", "a_tokens", "plddt")

# (P, N, align) — align=None: the grid policy via Layout.auto; align='B<k>': the grid policy with block k; align=k: chunk-aligned balanced
# parts (dist.row_parts), last rank ragged. Large geometries (P=8, one block per rank, ragged last rank) run through the CLI:
#   python tests/test_rowpair_e2e_043.py --cases '8:1012:-,2:992:-,8:992:B64,4:1012:-'
CASES = [(2, 37, 1), (2, 64, None), (3, 37, 1), (3, 64, 4), (4, 37, 1), (4, 64, 4)]


# =====================================================================================================================================
# seam table: seam name ("<seam>.<statement>") -> (core module, attribute). Resolved at import; a name whose core function
# is absent maps to None, the forward then runs the same statement composition from the primitives present, and the name is listed in
# SEAM_LOCAL (printed per RESULT line).
# =====================================================================================================================================
def _mod(modname: str):
    try:
        return __import__(f"opt_core.mem.rowpair.{modname}", fromlist=["_"])
    except Exception:  # noqa: BLE001 — a seam module not in this tree
        return None


SEAM_TABLE = {
    # substrate
    "shard.produce_rows_": ("shard", "produce_rows_"),
    "shard.transpose_shard": ("ring", "transpose_shard"),
    "shard.gather_rows": ("shard", "unshard_rows"),
    "shard.gather_rows_to": ("shard", "unshard_rows_to_rank0"),
    "testing.run_ranks": ("testing", None),
    # trunk (seams 1, 2, 4)
    "trunk.input_embedder_sharded": ("trunk", "init_pair_shard"),
    "trunk.recycle_z_": ("trunk", "recycle_shard_"),
    "trunk._guard_replicated": ("trunk", "guard_replicated"),
    # template (seam 3)
    "template.template_embedder_add_sharded_": ("template", "template_embed_rows"),
    # MSA (seams 5, 6)
    "msa.opm_rows": ("transition", "opm_rows"),
    "msa.msa_pair_avg_sharded": ("msa", "pwa_rows"),
    "msa.pwa_bias_rows": ("msa", "pwa_bias_rows"),
    "msa.msa_transition_maybe_sharded": ("msa", "msa_transition_rows"),
    "msa.msa_block_sharded": ("msa", "msa_block_sharded"),
    # pair stack (seams 8-13)
    "trimul.TriMulFns": ("trimul", "TriMulFns"),
    "triatt.TriAttFns": ("triatt", "TriAttFns"),
    "pairstack.bind": ("pairstack", "bind"),
    "pairstack.pair_block_forward_sharded": ("pairstack", "pair_block_"),
    "pairstack.pairformer_stack_sharded": ("pairstack", "pair_stack_"),
    "pairstack.attn_pair_bias_sharded": ("transition", "apb_local_queries"),
    # L2 statements the local compositions use
    "trimul.outgoing": ("trimul", "trimul_outgoing"),
    "trimul.incoming": ("trimul", "trimul_incoming"),
    "triatt.gather_bias": ("triatt", "gather_triangle_bias"),
    "triatt.triatt_sharded": ("triatt", "triatt_starting"),
    "pairstack.tri_att_end": ("triatt", "triatt_ending"),
    # heads + confidence (seam 17)
    "confidence.sym_logit_rows": ("heads", "sym_logit_rows"),
    "confidence.logit_rows": ("heads", "logit_rows"),
    "confidence.embed_rows": ("heads", "embed_rows"),
    "confidence.RowBlockReducer": ("confidence", "RowBlockReducer"),
    "confidence.ChainIndex": ("confidence", "ChainIndex"),
    # diffusion (seams 14, 15)
    "diffusion.cond_pair_local": ("diffusion", "pair_cond_rows"),
    "diffusion.pair_bias_rows": ("diffusion", "pair_bias_rows"),
    "diffusion.DiTBlockFns": ("diffusion", "DiTBlockFns"),
    "diffusion.dit_block_sharded": ("diffusion", "dit_block_sharded"),
    "diffusion.sync_replicated": ("diffusion", "sync_replicated"),
}
SEAM: Dict[str, Optional[Callable]] = {}
SEAM_LOCAL: List[str] = []
if HAVE_TORCH:
    for seam_name, (modname, attr) in SEAM_TABLE.items():
        m_ = _mod(modname) if modname != "testing" else __import__("opt_core.testing", fromlist=["_"])
        fn = m_ if attr is None else (getattr(m_, attr, None) if m_ is not None else None)
        SEAM[seam_name] = fn
        if fn is None:
            SEAM_LOCAL.append(seam_name)


def have(*names) -> bool:
    return all(SEAM.get(n) is not None for n in names)


# =====================================================================================================================================
# allocation guard
# =====================================================================================================================================
if HAVE_TORCH:
    class AllocGuard(TorchFunctionMode):
        """Fails loudly the moment any torch call of this thread returns a tensor with ``numel >= limit``; records the largest seen."""

        def __init__(self, limit: int, rank: int):
            super().__init__()
            self.limit, self.rank, self.max_numel, self.max_where = int(limit), int(rank), 0, ""

        def __torch_function__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            stack = [out]
            while stack:
                o = stack.pop()
                if isinstance(o, torch.Tensor):
                    n = o.numel()
                    if n > self.max_numel:
                        self.max_numel, self.max_where = n, f"{getattr(func, '__name__', func)} -> {tuple(o.shape)}"
                    if n >= self.limit:
                        raise AssertionError(f"rank {self.rank}: allocation guard: {getattr(func, '__name__', func)} produced {tuple(o.shape)} "
                                             f"numel={n} >= N*N*c_z={self.limit}")
                elif isinstance(o, (list, tuple)):
                    stack.extend(o)
            return out


# =====================================================================================================================================
# the row-sharded forward (the engine's control flow; statements enter as the model's callables; comm / layout / blocking from the core)
# =====================================================================================================================================
def _sched(**kv):
    EV.record_schedule(**kv)


def _reset_census():
    """a case starts from an empty schedule census so a missing key cannot be masked by the previous case (main thread, before the ranks start)."""
    if hasattr(EV, "reset_schedule"):
        EV.reset_schedule()
        return
    store = getattr(EV, "_SCHEDULE", None)
    if isinstance(store, dict):
        store.clear()


def guard_replicated(t, name: str):
    """replicated-by-design tensors drawn from the replicated RNG stream are PROVEN identical across ranks (checksum all-gather; raises by name)."""
    fn = SEAM["trunk._guard_replicated"]
    if fn is not None:
        fn(t, name)
    else:
        D.allreduce_checksum(t, name)
    return t


# ---------------------------------------------------------------------------------------------------------------- seams 1, 2
def pair_init_sharded(model, feats, lay):
    """seam 1: z_init BORN sharded (relpos one-hot rows exist per row block only)."""
    s_input, s_init, emb_i, emb_j = model.pair_init.single(feats)
    row_fn = lambda g0, g1: model.pair_init.rows(feats, emb_i, emb_j, g0, g1)  # noqa: E731
    if have("trunk.input_embedder_sharded"):
        z0 = SEAM["trunk.input_embedder_sharded"](lay, row_fn, emb_j, rows=BLK["init_rows"])
    else:
        z0 = SH.produce_rows_(emb_j.new_empty((lay.n_loc, lay.N, model.cfg.c_z)), lay, row_fn, op="set", block_rows=BLK["init_rows"])
    _sched(init_rows=BLK["init_rows"])
    return s_input, s_init, z0


def recycle_sharded_(model, z_loc, z0_loc, lay):
    """seam 2: ``z = z_init + linear(LN(z_prev))`` row-blockwise IN PLACE (row i depends on row i only); cycle 0 (z_loc None) = zeros row block."""
    if have("trunk.recycle_z_"):
        z_loc = SEAM["trunk.recycle_z_"](z_loc, z0_loc, model.recycle, lay, rows=BLK["recycle_rows"])
    else:
        if z_loc is None:
            z_loc = torch.zeros_like(z0_loc)
        L = lay.r0
        zz = z_loc
        SH.produce_rows_(z_loc, lay, lambda g0, g1: z0_loc[g0 - L:g1 - L] + model.recycle(zz[g0 - L:g1 - L]), op="set", block_rows=BLK["recycle_rows"])
    _sched(recycle_rows=BLK["recycle_rows"])
    return z_loc


# ---------------------------------------------------------------------------------------------------------------- seams 8-13
def apb_sharded(apb, a, z_loc, mask, lay, s=None):
    """seam 13: per-token projections on ALL tokens (M = N), attention for LOCAL query rows with the bias from local z rows, gather rows."""
    a_ln, q, k, v = apb.prep(a, s)
    attn_fn = lambda _q_rows, _a_full, z_rows: apb.core_rows(q[:, lay.r0:lay.r1], k, v, apb.bias_rows(z_rows), mask)  # noqa: E731
    o_full = SEAM["pairstack.attn_pair_bias_sharded"](attn_fn, a, z_loc, lay, gather=True)
    return apb.wrap_up(o_full, a_ln, s)


def bind_pair_block(pb, cfg, token_mask=None, apb_mod=None, single_transition=None):
    """The engine's PairBlock (+ the Pairformer single track) as the core's PairBlockFns (``pairstack.bind`` over TriMulFns / TriAttFns)."""
    TriMulFns, TriAttFns, bind = SEAM["trimul.TriMulFns"], SEAM["triatt.TriAttFns"], SEAM["pairstack.bind"]

    def tm_fns(tm):
        def proj(zb, mb, is_a):
            h = tm.layer_norm_in(zb)
            if is_a:
                return torch.sigmoid(tm.linear_a_g(h)) * tm.linear_a_p(h) * mb
            return torch.sigmoid(tm.linear_b_g(h)) * tm.linear_b_p(h) * mb
        return TriMulFns(proj=proj, out=lambda x: tm.linear_z(tm.layer_norm_out(x)), gate=lambda zb: torch.sigmoid(tm.linear_g(tm.layer_norm_in(zb))),
                         C_h=cfg.c_mul)

    def ta_fns(ta):
        return TriAttFns(ln=ta.layer_norm, bias=lambda x: ta.bias_rows(x).contiguous(), attend=lambda x, mrows, tb, rng: ta.attn_rows(x, mrows, tb))

    apb = None
    st = None
    if apb_mod is not None:
        apb = lambda s, z, lay: s + apb_sharded(apb_mod, s, z, token_mask, lay)          # noqa: E731
        st = lambda s: s + single_transition(s, token_mask)                              # noqa: E731  replicated (single rep)
    return bind(trimul_out=tm_fns(pb.tri_mul_out), trimul_in=tm_fns(pb.tri_mul_in), triatt_start=ta_fns(pb.tri_att_start),
                triatt_end=ta_fns(pb.tri_att_end), transition=lambda x, mu: pb.pair_transition(x, mu[..., 0]), chunk=BLK["triatt_qrows"],
                apb=apb, single_transition=st, trimul_kw=dict(RB=BLK["trimul_RB"], inplace_chunk=256))


def pair_block_sharded(pb, z_loc, mask_loc, maskT_loc, lay, cfg, s=None, token_mask=None, apb_mod=None, single_transition=None):
    """seams 8-12 (+13): one pair block on the row shard through the core driver; local composition of the L2 statements otherwise."""
    if have("pairstack.bind", "pairstack.pair_block_forward_sharded", "trimul.TriMulFns", "triatt.TriAttFns"):
        fns = bind_pair_block(pb, cfg, token_mask, apb_mod, single_transition)
        z_loc, s = SEAM["pairstack.pair_block_forward_sharded"](fns, z_loc.contiguous(), mask_loc, lay, s=s, maskT_shard=maskT_loc)
        _sched(trans_rows=BLK["triatt_qrows"])
        return z_loc, s
    # ---- local composition (trees without the pair-block driver)
    L = lay.r0

    def trimul(tm_mod, z, outgoing):
        a_, b_ = tm_mod.projections(z, mask_loc)
        fn = SEAM["trimul.outgoing"] if outgoing else SEAM["trimul.incoming"]
        x_, _ = fn(a_.contiguous(), b_.contiguous(), lay, RA=BLK["trimul_RA"], RB=BLK["trimul_RB"])
        return tm_mod.epilogue(x_, z)

    z_loc = z_loc + trimul(pb.tri_mul_out, z_loc, True)
    z_loc = z_loc + trimul(pb.tri_mul_in, z_loc, False)
    ta = pb.tri_att_start
    x_ln = ta.layer_norm(z_loc)
    tb_full = SEAM["triatt.gather_bias"](ta.bias_rows(x_ln).contiguous(), lay)
    z_loc = z_loc + SEAM["triatt.triatt_sharded"](lambda xr, tb, rng: ta.attn_rows(xr, mask_loc[rng[0] - L:rng[1] - L], tb), x_ln, tb_full, lay,
                                                     q_rows=BLK["triatt_qrows"])
    te = pb.tri_att_end
    z_loc = z_loc + SEAM["pairstack.tri_att_end"](lambda zTr, tb, rng: te.attn_rows(te.layer_norm(zTr), maskT_loc[rng[0] - L:rng[1] - L], tb), z_loc,
                                                    lambda zT_rows: te.bias_rows(te.layer_norm(zT_rows)).contiguous(), lay, q_rows=BLK["triatt_qrows"])
    zz = z_loc
    SH.produce_rows_(z_loc, lay, lambda g0, g1: pb.pair_transition(zz[g0 - L:g1 - L], mask_loc[g0 - L:g1 - L]), op="add", block_rows=BLK["trans_rows"])
    _sched(trans_rows=BLK["trans_rows"])
    if apb_mod is not None:
        s = s + apb_sharded(apb_mod, s, z_loc, token_mask, lay)
        s = s + single_transition(s, token_mask)
    return z_loc, s


def pairformer_block_sharded(blk, s, z_loc, token_mask, mask_loc, maskT_loc, lay, cfg):
    z_loc, s = pair_block_sharded(blk.pair, z_loc, mask_loc, maskT_loc, lay, cfg, s=s, token_mask=token_mask, apb_mod=blk.attn_pair_bias,
                                  single_transition=blk.single_transition)
    return s, z_loc


# ---------------------------------------------------------------------------------------------------------------- seam 3
def template_sharded_(model, feats, z_loc, mask_loc, maskT_loc, lay):
    """seam 3: per slot ``u = linear_z(LN(z rows)) + embed(feature rows)`` (feature rows synthesised per block from [T, N] precursors), the
    template pair stack on the row shard (+ final LN), the closing statements per row block added into z_loc IN PLACE."""
    te, cfg = model.template_embedder, model.cfg

    def stack_fn(u3, m):                                               # u3 [R, N, c_t]
        for pblk in te.pair_stack:
            u3, _ = pair_block_sharded(pblk, u3, m, maskT_loc, lay, cfg)
        return te.layer_norm_out(u3)

    if have("template.template_embedder_add_sharded_"):
        z_loc = SEAM["template.template_embedder_add_sharded_"](
            z_loc, lay, n_templ=cfg.T, c_t=cfg.c_t,
            unit_rows_fn=lambda z_rows, slot, g: te.u_rows(feats, z_rows, [int(slot)], g[0], g[1]),          # [1, rows, N, c_t]
            pair_stack_fn=lambda u, m: stack_fn(u[0], m).unsqueeze(0),
            finish_fn=lambda t: te.closing_rows(t),                                                      # t [T, rows, N, c_t]
            mask_loc=mask_loc, rows=BLK["templ_rows"], add=True)
    else:
        L = lay.r0
        us = []
        for t in range(cfg.T):
            u = SH.produce_rows_(z_loc.new_empty((lay.n_loc, lay.N, cfg.c_t)), lay,
                                 lambda g0, g1, t=t: te.u_rows(feats, z_loc[g0 - L:g1 - L], [t], g0, g1)[0], op="set", block_rows=BLK["templ_rows"])
            us.append(stack_fn(u, mask_loc))
        SH.produce_rows_(z_loc, lay, lambda g0, g1: te.closing_rows(torch.stack([u[g0 - L:g1 - L] for u in us], 0)), op="add", block_rows=BLK["templ_rows"])
    _sched(templ_rows=BLK["templ_rows"])
    return z_loc


# ---------------------------------------------------------------------------------------------------------------- seams 5-7
def msa_block_sharded(blk, m, z_loc, msa_mask, pair_mask, mask_loc, maskT_loc, lay, cfg):
    """seams 5-7: OPM output rows local (a rows x b all, row blocks, IN PLACE into the shard); PWA logits on local (complete) rows, v / gate at
    M = S·N, per-row outputs gathered, linear_o on the gathered slab; MSA transition replicated; the pair block sharded."""
    opm, pwa = blk.outer_product_mean, blk.msa_att_row
    L = lay.r0

    def opm_(m_, z_):
        a_, b_ = opm.operands(m_, msa_mask)
        outer_fn = lambda a_blk, b_all, g0, g1: opm.outer_rows(a_blk, b_all, opm.norm_rows(msa_mask, g0, g1))  # noqa: E731
        if have("msa.opm_rows"):
            try:
                return SEAM["msa.opm_rows"](a_, b_, lay, outer_fn, rows=BLK["opm_rows"], out=z_, add=True, global_rows=True)
            except TypeError:                                          # a tree whose opm_rows predates global_rows/add
                pass
        return SH.produce_rows_(z_, lay, lambda g0, g1: outer_fn(a_[:, g0:g1], b_, g0, g1), op="add", block_rows=BLK["opm_rows"])

    def pwa_(m_, z_):
        S_, N_, H, c = int(m_.shape[0]), int(m_.shape[1]), pwa.H, pwa.c
        prep_fn = lambda z_rows, g0, g1: (pwa.linear_z(pwa.layer_norm_z(z_rows)).permute(2, 0, 1) + (pwa.inf * (pair_mask[g0:g1] - 1.0))[None])  # noqa: E731

        def values_fn(mc):
            m_ln = pwa.layer_norm_m(mc)
            v = pwa.linear_v(m_ln).view(int(mc.shape[0]), N_, H, c).permute(2, 0, 1, 3)           # [H, chunk, N, c]
            g = torch.sigmoid(pwa.linear_g(m_ln)).view(int(mc.shape[0]), N_, H, c)                # [chunk, N, H, c]
            return v, g

        def attend_fn(w, state, g0, g1):
            v, g = state
            o = torch.einsum("hij,hsjc->sihc", w, v)                                              # [chunk, q, H, c]
            return (o * g[:, g0:g1]).reshape(int(v.shape[1]), g1 - g0, H * c)

        if have("msa.msa_pair_avg_sharded", "msa.pwa_bias_rows"):
            bias_loc = SEAM["msa.pwa_bias_rows"](prep_fn, z_, lay, rows=BLK["pwa_rows"])          # [H, R, N]
            upd = SEAM["msa.msa_pair_avg_sharded"](m_, bias_loc, lay, values_fn=values_fn, attend_fn=attend_fn, out_fn=pwa.linear_o,
                                                     q_block=BLK["pwa_rows"])
        else:
            pieces = [pwa.rows_fn(m_, z_[b0:b1], g0, g1, pair_mask) for b0, b1, g0, g1 in SH.iter_row_blocks(lay, BLK["pwa_rows"])]
            o_loc = torch.cat(pieces, dim=1)                                                        # [S, n_loc, H*c]
            upd = pwa.linear_o(D.all_gather_rows(o_loc.movedim(1, 0).contiguous(), lay).movedim(0, 1))
        return upd

    def msa_update(m_, z_):
        m_ = m_ + pwa_(m_, z_)
        tr = lambda mm, g0, g1: blk.msa_transition(mm, msa_mask[:, g0:g1])  # noqa: E731
        if have("msa.msa_transition_maybe_sharded"):
            return m_ + SEAM["msa.msa_transition_maybe_sharded"](tr, m_, lay, shard_tokens=False)
        return m_ + tr(m_, 0, lay.N)

    pair_block = lambda z_: pair_block_sharded(blk.pair, z_, mask_loc, maskT_loc, lay, cfg)[0]  # noqa: E731
    if have("msa.msa_block_sharded"):
        m, z_loc = SEAM["msa.msa_block_sharded"](m, z_loc, lay, opm=opm_, pair_block=pair_block, msa_update=msa_update, opm_first=True)
    else:
        z_loc = opm_(m, z_loc)
        m = msa_update(m, z_loc)
        z_loc = pair_block(z_loc)
    _sched(opm_rows=BLK["opm_rows"], pwa_rows=BLK["pwa_rows"])
    return m, z_loc


# ---------------------------------------------------------------------------------------------------------------- seam 17 (heads)
def _lockstep(gen_a, gen_b):
    """walk two row-block generators of ONE schedule together; both are exhausted (their exchange steps are collectives)."""
    for xa in gen_a:
        xb = next(gen_b)
        if (xa[0], xa[1]) != (xb[0], xb[1]):
            raise AssertionError(f"row-block schedules disagree: {xa[:2]} vs {xb[:2]}")
        yield xa[0], xa[1], xa[2], xb[2]
    for _ in gen_b:
        raise AssertionError("second generator yielded more blocks than the first")


def distogram_contact_sharded(model, z_loc, lay):
    """distogram: ``logits = linear(z) + linear(z)^T`` per row block (column-slab exchange for the transpose term); only the [rows, N, 64]
    block exists; the contact-map rows [n_loc, N] are kept (an O(N^2) output)."""
    head = model.distogram
    contact_loc = z_loc.new_empty((lay.n_loc, lay.N))
    if have("confidence.sym_logit_rows"):
        for i0, i1, logits in SEAM["confidence.sym_logit_rows"](head.linear, z_loc, lay, rows=BLK["logits_rows"], bins=model.cfg.n_dist_bins):
            contact_loc[i0:i1] = head.contact_from_logits(logits)
            del logits
    else:
        L = lay.r0
        zT = SEAM["shard.transpose_shard"](z_loc, lay)
        SH.produce_rows_(contact_loc, lay, lambda g0, g1: head.contact_from_logits(head.linear(z_loc[g0 - L:g1 - L]) + head.linear(zT[g0 - L:g1 - L])),
                         op="set", block_rows=BLK["logits_rows"], row_dim=0)
        del zT
    _sched(logits_rows=BLK["logits_rows"])
    return contact_loc


def confidence_sharded(model, feats, s_input, s_trunk, z_loc, contact_loc, x, token_mask, mask_loc, maskT_loc, lay):
    """seam 17: per sample, the head's pair input rows from the trunk shard; pairformer blocks sharded; PAE (plain) / PDE (symmetrised) logits
    produced per row block on one schedule and consumed on the fly by the exact-finish reducer; pLDDT from the replicated single rep."""
    cfg, ch = model.cfg, model.confidence
    L = lay.r0
    chains = SEAM["confidence.ChainIndex"](feats["asym_id"], feats["has_frame"])
    out = {}
    for si in range(cfg.n_samples):
        emb = lambda z_rows, g0, g1, si=si: ch.embed_rows(s_input, z_rows, x[si], g0, g1)  # noqa: E731
        if have("confidence.embed_rows"):
            zc = SEAM["confidence.embed_rows"](emb, z_loc, lay, rows=BLK["conf_rows"])
        else:
            zc = SH.produce_rows_(z_loc.new_empty((lay.n_loc, lay.N, cfg.c_z)), lay, lambda g0, g1: emb(z_loc[g0 - L:g1 - L], g0, g1), op="set",
                                  block_rows=BLK["conf_rows"])
        s = ch.single(s_input, s_trunk)
        for blk in ch.blocks:
            s, zc = pairformer_block_sharded(blk, s, zc, token_mask, mask_loc, maskT_loc, lay, cfg)
        red = SEAM["confidence.RowBlockReducer"](chains, lay.r0, lay.r1, zc.device, finish="exact")
        if have("confidence.logit_rows", "confidence.sym_logit_rows"):
            pae_g = SEAM["confidence.logit_rows"](ch.linear_pae, zc, lay, rows=BLK["conf_rows"], bins=cfg.n_pae_bins)
            pde_g = SEAM["confidence.sym_logit_rows"](ch.linear_pde, zc, lay, rows=BLK["conf_rows"], bins=cfg.n_pae_bins)
            for i0, i1, pae_rows, pde_rows in _lockstep(pae_g, pde_g):
                red.consume(L + i0, L + i1, pae_rows, pde_rows, contact_loc[i0:i1])
                del pae_rows, pde_rows
        else:
            zcT = SEAM["shard.transpose_shard"](zc, lay)
            for b0, b1, g0, g1 in SH.iter_row_blocks(lay, BLK["conf_rows"]):
                red.consume(g0, g1, ch.pae_rows(zc[b0:b1]), ch.pde_rows(zc[b0:b1], zcT[b0:b1]), contact_loc[b0:b1])
            del zcT
        res = red.finalize(lay.bounds, collect_full=True)
        out[f"plddt_logits.{si}"] = ch.linear_plddt(s)
        if res is not None:
            for k in ("ptm", "iptm", "chain_ptm", "chain_pair_iptm", "chain_iptm", "gpde", "chain_gpde", "chain_pair_gpde"):
                out[f"{k}.{si}"] = res[k]
            out[f"token_pair_pde.{si}"] = res["token_pair_pde_f32"]
            out[f"token_pair_pae_f16.{si}"] = res["token_pair_pae_f16"]
            out[f"contact_probs.{si}"] = res["contact_probs_f32"]
        del zc, red
    _sched(conf_rows=BLK["conf_rows"], conf_finish="exact")
    return out


# ---------------------------------------------------------------------------------------------------------------- seams 14, 15
def bind_dit_block(blk, token_mask):
    """One DiffusionTransformer block as the core's DiTBlockFns (norm on all rows; k/v from all rows once; attention for query rows; the
    AdaLN-zero gate + conditioned transition on this rank's rows)."""
    attn = blk.attn

    def norm(a, s):
        return attn.layer_norm_a(a, s)                                   # AdaLN(a, s), all rows

    def kv(x):
        N = int(x.shape[0])
        heads = lambda t: t.view(N, attn.H, attn.c).transpose(0, 1)     # noqa: E731  [H, N, c]
        return heads(attn.linear_k(x)), heads(attn.linear_v(x))

    def attend(x_q, kvs, bias_q, g):
        k, v = kvs
        q = (attn.linear_q(x_q).view(int(x_q.shape[0]), attn.H, attn.c).transpose(0, 1)) * (1.0 / (attn.c ** 0.5))
        o = attn.core_rows(q, k, v, bias_q, token_mask)                  # [q, H*c]
        return attn.linear_o(torch.sigmoid(attn.linear_g(x_q)) * o)      # wrap_up minus the ada gate (needs s rows: in update)

    def update(a, o_rows, s, rr):
        r0, r1 = rr
        a_rows = a[r0:r1] + torch.sigmoid(attn.linear_ada_out(s[r0:r1])) * o_rows
        return a_rows + blk.transition_rows(a_rows, s[r0:r1])

    return SEAM["diffusion.DiTBlockFns"](norm=norm, kv=kv, attn=attend, update=update, bias=lambda z_rows: attn.linear_z(attn.layer_norm_z(z_rows)))


def diffusion_sharded(model, feats, s_input, s_trunk, z_loc, token_mask, lay, gen):
    """seams 14-15: z_cond rows ONCE per rollout (noise-independent); per DiT block: norm / k,v on all tokens, attention for local query rows
    with bias rows from z_cond_loc, gate + conditioned transition on local rows, ONE gather of the token rows per block."""
    cfg, dm, cond = model.cfg, model.diffusion, model.cond
    L = lay.r0
    embed = lambda z_rows, g0, g1: cond.linear_z(cond.layer_norm_z(torch.cat([z_rows, cond._relpos.relpos_rows(feats, g0, g1)], dim=-1)))  # noqa: E731
    transitions = [(lambda x, g0, g1, tr=tr: tr(x)) for tr in cond.trans_z]
    if have("diffusion.cond_pair_local"):
        zc = SEAM["diffusion.cond_pair_local"](embed, z_loc, lay, c_out=cfg.c_z, rows=BLK["cond_rows"], transitions=transitions)
    else:
        zc = SH.produce_rows_(z_loc.new_empty((lay.n_loc, lay.N, cfg.c_z)), lay, lambda g0, g1: cond.pair_rows(feats, z_loc[g0 - L:g1 - L], g0, g1),
                              op="set", block_rows=BLK["cond_rows"])
    _sched(cond_rows=BLK["cond_rows"], diff_cond="once_per_rollout")
    use_core_dit = have("diffusion.DiTBlockFns", "diffusion.dit_block_sharded", "diffusion.pair_bias_rows")
    fns = [bind_dit_block(blk, token_mask) for blk in dm.blocks] if use_core_dit else None
    biases = None
    sync = SEAM["diffusion.sync_replicated"] or (lambda t, name: guard_replicated(t, name))
    sig = SA.noise_schedule(cfg)
    N = lay.N
    xs = []
    for _s in range(cfg.n_samples):
        x = sig[0] * torch.randn((N, 3), generator=gen)
        sync(x, "x_atoms")
        for i in range(1, cfg.n_steps + 1):
            c_prev, c = sig[i - 1], sig[i]
            gamma = 0.4 if c > 1.0 else 0.0
            t_hat = c_prev * (gamma + 1.0)
            eps = 1.003 * torch.sqrt((t_hat ** 2 - c_prev ** 2).clamp(min=0)) * torch.randn((N, 3), generator=gen)
            sync(eps, "eps_atoms")
            x_noisy = x + eps
            s_cond = cond.single(feats, s_trunk, s_input, t_hat)
            a = dm.token_act(x_noisy, t_hat, s_cond)
            if use_core_dit:
                if biases is None:                                     # per-block pair biases depend on z_cond only: once per rollout set
                    biases = [SEAM["diffusion.pair_bias_rows"](f.bias, zc, lay, rows=BLK["cond_rows"]) for f in fns]
                for f, bias in zip(fns, biases):
                    a = SEAM["diffusion.dit_block_sharded"](f, a, s_cond, bias, lay, q_rows=BLK["triatt_qrows"], gather=True)
            else:
                for blk in dm.blocks:
                    a_ln, q, k, v = blk.attn.prep(a, s_cond)
                    o_loc = blk.attn.core_rows(q[:, lay.r0:lay.r1], k, v, blk.attn.bias_rows(zc), token_mask)
                    a_loc = a[lay.r0:lay.r1] + blk.attn.wrap_up(o_loc, a_ln[lay.r0:lay.r1], s_cond[lay.r0:lay.r1])
                    a_loc = a_loc + blk.transition_rows(a_loc, s_cond[lay.r0:lay.r1])
                    a = D.all_gather_rows(a_loc.contiguous(), lay)
            x_den = dm.finish(a, x_noisy, t_hat)
            delta = (x_noisy - x_den) / t_hat
            x = x_noisy + 1.5 * (c - t_hat) * delta
        xs.append(x)
    _sched(dit="core" if use_core_dit else "local")
    return torch.stack(xs, 0)


# ---------------------------------------------------------------------------------------------------------------- the whole forward
def forward_sharded(rank: int, P: int, model, feats, N: int, align, guard: bool = True, x_conf=None):
    """One rank's whole forward. Returns (outputs dict on rank 0 | {} elsewhere, census, alloc report). ``x_conf``: the coordinates the
    confidence stage consumes — the DENSE reference roll-out's (seam isolation: the confidence head one-hot-bins pair distances, a
    discontinuous statement, so feeding it this rank's own roll-out (equal to dense only within the fp32 GEMM class) would compare a bin flip at
    a bin edge, not the confidence seams; the roll-out is compared as its own output ``coords``). None = this rank's own roll-out."""
    torch.set_num_threads(1)
    cfg = model.cfg
    if align is None:
        lay = Layout.auto(N, P, rank)                                              # the grid policy, B chosen by the family rule
    elif isinstance(align, str) and align.startswith("B"):
        lay = Layout.checked(N, P, rank, int(align[1:]))                          # the grid policy with an explicit block B (a kit's pinned B)
    else:
        lay = Layout(N, P, rank, align=int(align))                                 # chunk-aligned balanced parts
    D.require_sharded(lay, "forward_sharded")                     # the structural n_gpu=1 rule: P == 1 refuses by name HERE
    _sched(**lay.facts())
    _sched(replicated=",".join(REPLICATED_BY_DESIGN), seams_local=",".join(SEAM_LOCAL) or "none")
    gen = SA.rng(feats)
    token_mask = feats["token_mask"]
    pair_mask = token_mask[:, None] * token_mask[None, :]          # [N, N] input mask (O(N^2), like token_bonds)
    mask_loc = pair_mask[lay.r0:lay.r1].contiguous()
    maskT_loc = pair_mask.t()[lay.r0:lay.r1].contiguous()
    limit = N * N * cfg.c_z
    g = AllocGuard(limit, rank) if guard else None
    if g is not None:
        g.__enter__()
    try:
        s_input, s_init, z0 = pair_init_sharded(model, feats, lay)
        s = torch.zeros_like(s_init)
        z_loc = None
        for _cycle in range(cfg.n_cycles):
            z_loc = recycle_sharded_(model, z_loc, z0, lay)
            z_loc = template_sharded_(model, feats, z_loc, mask_loc, maskT_loc, lay)
            m, msa_mask = model.msa_embedder(feats, s_input, gen)  # seam 4: replicated by design (identical RNG stream), proven
            guard_replicated(m, "m")
            for blk in model.msa_blocks:
                m, z_loc = msa_block_sharded(blk, m, z_loc, msa_mask, pair_mask, mask_loc, maskT_loc, lay, cfg)
            del m
            s = s_init + model.linear_s(model.layer_norm_s(s))
            for blk in model.pairformer:
                s, z_loc = pairformer_block_sharded(blk, s, z_loc, token_mask, mask_loc, maskT_loc, lay, cfg)
        contact_loc = distogram_contact_sharded(model, z_loc, lay)
        x = diffusion_sharded(model, feats, s_input, s, z_loc, token_mask, lay, gen)
        out = {"coords": x, "s_trunk": s}
        xc = x if x_conf is None else x_conf.to(device=x.device, dtype=x.dtype)
        out.update(confidence_sharded(model, feats, s_input, s, z_loc, contact_loc, xc, token_mask, mask_loc, maskT_loc, lay))
        _sched(conf_coords="own_rollout" if x_conf is None else "dense_reference")
    finally:
        if g is not None:
            g.__exit__(None, None, None)
    # ---- outside the guard: assemble full matrices on rank 0 for the comparison only
    z_full = SEAM["shard.gather_rows_to"](z_loc.contiguous(), lay, 0) if have("shard.gather_rows_to") else D.gather_rows_to_rank0(z_loc.contiguous(), lay)
    contact_full = D.gather_rows_to_rank0(contact_loc.contiguous(), lay)
    census = EV.schedule()
    alloc = {"max_numel": g.max_numel if g else -1, "max_where": g.max_where if g else "", "limit": limit}
    if rank != 0:
        return {}, census, alloc
    out["z_trunk"] = z_full
    out["contact_probs"] = contact_full
    return {k: v.detach().clone() for k, v in out.items()}, census, alloc


# =====================================================================================================================================
# comparison
# =====================================================================================================================================
def compare(dense: dict, shard: dict) -> Tuple[dict, List[str]]:
    """per output: (max|diff|, torch.equal). PAE full matrix comes back f16 from the reducer's writer rows -> compared with dense.half()."""
    rows, fails = {}, []
    for k, v in shard.items():
        quantum = 0.0
        if k.startswith("token_pair_pae_f16."):                   # the writer's f16 PAE rows: a QUANTISED copy of a tier-2 fp32 value ->
            ref32 = dense["token_pair_pae." + k.split(".", 1)[1]]  # within one f16 ulp of the dense fp32 value; bit-exact vs dense.half() reported
            ref = ref32.half()
            quantum = 2.0 ** (math.floor(math.log2(max(float(ref32.abs().max()), 2.0 ** -14))) - 10)
        elif k.startswith("contact_probs."):
            ref = dense["contact_probs"]
        else:
            ref = dense[k]
        v = v.reshape(ref.shape) if v.numel() == ref.numel() else v
        if tuple(v.shape) != tuple(ref.shape):
            fails.append(f"{k}: shape {tuple(v.shape)} vs dense {tuple(ref.shape)}")
            continue
        eq = bool(torch.equal(v, ref))
        d = float((v.float() - ref.float()).abs().max()) if v.numel() else 0.0           # REPORTED: vs the dense value in the output's own dtype
        scale = max(1.0, float(ref.float().abs().max())) if ref.numel() else 1.0
        d_assert = float((v.float() - ref32.float()).abs().max()) if (quantum and v.numel()) else d
        rows[k] = (d, eq)
        if not (d_assert <= TOL * scale + quantum):               # fp32: 1e-5 x max(1, max|dense|); quantised outputs: vs dense fp32 + one quantum
            fails.append(f"{k}: max|diff|={d_assert:.3e} > {TOL:g} x scale {scale:.3g}" + (f" + f16 ulp {quantum:.3g}" if quantum else ""))
    missing = [k for k in dense if k not in shard and not k.startswith(("pae_logits.", "distogram_logits", "token_pair_pae."))]
    for k in missing:
        fails.append(f"{k}: missing from the sharded outputs")
    return rows, fails


def run_case(P: int, N: int, align, seed: int = 0, verbose: bool = True) -> dict:
    cfg = SA.Config(N=N)
    model = SA.build_model(cfg, seed)
    feats = SA.make_features(cfg, seed)
    t0 = time.time()
    dense = SA.forward_dense(model, feats)
    t1 = time.time()
    _reset_census()
    res = run_ranks(P, forward_sharded, model, feats, N, align, x_conf=dense["coords"])      # seam isolation: confidence on the reference coords
    t2 = time.time()
    out0, census, _ = res[0]
    allocs = [r[2] for r in res]
    rows, fails = compare(dense, out0)
    # census assertions: every explicit block size + the replicated-by-design names + layout facts
    for key in ("replicated", "init_rows", "recycle_rows", "opm_rows", "pwa_rows", "trans_rows", "logits_rows", "cond_rows", "conf_rows", "conf_finish"):
        if key not in census:
            fails.append(f"schedule census lacks {key!r} (have {sorted(census)})")
    if "replicated" in census:
        for name in ("m", "s", "x_atoms"):
            if name not in str(census["replicated"]).split(","):
                fails.append(f"schedule census 'replicated' does not name {name!r}: {census['replicated']}")
    for a in allocs:
        if a["max_numel"] >= a["limit"]:
            fails.append(f"allocation guard: max numel {a['max_numel']} ({a['max_where']}) >= {a['limit']}")
    n_eq = sum(1 for d, eq in rows.values() if eq)
    worst = max((d for d, _ in rows.values()), default=0.0)
    line = (f"RESULT e2e_043 P={P} N={N} align={align} {'PASS' if not fails else 'FAIL'} outputs={len(rows)} bitwise={n_eq}/{len(rows)} "
            f"max|diff|={worst:.3e} tol={TOL:g} max_numel/limit={max(a['max_numel'] for a in allocs)}/{allocs[0]['limit']} "
            f"seams_local={len(SEAM_LOCAL)} dense_s={t1 - t0:.2f} sharded_s={t2 - t1:.2f}")
    if verbose:
        print(line)
        for k in sorted(rows):
            d, eq = rows[k]
            print(f"    {k:24s} max|diff|={d:.3e} torch.equal={eq}")
        if SEAM_LOCAL:
            print(f"    seams composed locally from landed primitives (core function not landed): {SEAM_LOCAL}")
        for f in fails:
            print(f"    FAIL: {f}")
    return {"P": P, "N": N, "align": align, "ok": not fails, "fails": fails, "rows": {k: [d, eq] for k, (d, eq) in rows.items()},
            "line": line, "census": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in census.items()},
            "alloc": allocs, "seams_local": list(SEAM_LOCAL)}


# =====================================================================================================================================
# pytest entry points
# =====================================================================================================================================
@pytest.mark.parametrize("P,N,align", CASES, ids=[f"P{P}-N{N}-{'grid' if a is None else (a if isinstance(a, str) else 'align' + str(a))}" for P, N, a in CASES])
def test_e2e_sharded_equals_dense(P, N, align):
    r = run_case(P, N, align)
    assert r["ok"], "\n".join(r["fails"])


def test_p1_refuses_by_name():
    """The structural n_gpu=1 rule: at P == 1 the sharded forward's first L2 statement refuses BY NAME (no monkeypatch; the solo comm)."""
    cfg = SA.Config(N=37)
    model = SA.build_model(cfg, 0)
    feats = SA.make_features(cfg, 0)
    with pytest.raises(RowpairRefused) as ei:
        forward_sharded(0, 1, model, feats, 37, None, guard=False)
    msg = str(ei.value)
    assert "n_gpu=1" in msg and "installs nothing" in msg, msg
    # and every L2 entry point the forward drives refuses a P=1 layout by name on its own
    lay = Layout(37, 1, 0, B=1)
    z = torch.zeros(37, 37, 4)
    for name, call in {
        "trimul_outgoing": lambda: TM.trimul_outgoing(z, z, lay),
        "trimul_incoming": lambda: TM.trimul_incoming(z, z, lay),
        "gather_triangle_bias": lambda: TA.gather_triangle_bias(z, lay),
        "triatt_starting": lambda: TA.triatt_starting(lambda *a: a[0], z, z, lay),
        "triatt_ending": lambda: TA.triatt_ending(lambda *a: a[0], z, lambda t: t, lay),
        "transition_rows": lambda: TR.transition_rows(lambda t: t, z, lay),
        "apb_local_queries": lambda: TR.apb_local_queries(lambda q, s, zz: q, torch.zeros(37, 3), z, lay),
        "RowBlockReducer": lambda: CONF.RowBlockReducer(CONF.ChainIndex(feats["asym_id"], feats["has_frame"]), 0, 37, "cpu"),
    }.items():
        with pytest.raises(RowpairRefused) as ei:
            call()
        assert "n_gpu=1" in str(ei.value), (name, str(ei.value))
    print(f"RESULT e2e_043 P=1 PASS refused-by-name: {msg.splitlines()[0][:120]}")


def test_dense_is_deterministic():
    cfg = SA.Config(N=37)
    model = SA.build_model(cfg, 0)
    feats = SA.make_features(cfg, 0)
    a, b = SA.forward_dense(model, feats), SA.forward_dense(model, feats)
    assert all(torch.equal(a[k], b[k]) for k in a)


# =====================================================================================================================================
def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default=None, help="e.g. '2:37:1,4:64:4' (P:N:align, align '-' = grid)")
    ap.add_argument("--json", default=None, help="write the per-case report here")
    a = ap.parse_args(argv)
    if not HAVE_TORCH:
        print("SUMMARY e2e_043 SKIP torch not importable")
        return 0
    cases = CASES if not a.cases else [(int(p), int(n), None if al in ("-", "None", "grid") else (al if al.startswith("B") else int(al)))
                                       for p, n, al in (c.split(":") for c in a.cases.split(","))]
    reports, ok = [], True
    try:
        test_p1_refuses_by_name()
        p1 = "PASS"
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        p1, ok = "FAIL", False
    for P, N, al in cases:
        try:
            r = run_case(P, N, al)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            r = {"P": P, "N": N, "align": al, "ok": False, "fails": [f"{type(e).__name__}: {e}"], "line": f"RESULT e2e_043 P={P} N={N} align={al} ERROR {e}"}
            print(r["line"])
        ok = ok and r["ok"]
        reports.append(r)
    print(f"SUMMARY e2e_043 {'PASS' if ok else 'FAIL'} cases={len(reports)} pass={sum(1 for r in reports if r['ok'])} p1_refusal={p1} "
          f"seams_local={SEAM_LOCAL if SEAM_LOCAL else 'none'}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"ok": ok, "p1_refusal": p1, "cases": reports, "seams_local": SEAM_LOCAL, "tol": TOL}, fh, indent=1, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
