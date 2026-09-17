"""The trunk / template / relpos / distogram BINDINGS (protenix_opt.tp_bind) on the shared core's row-sharded drivers, CPU: a seeded
STUB Protenix (the sub-modules the bindings read: input embedder, zinit / relpos / token-bond / recycling / single linears, a template
embedder with one c_s=0 pair block, an MSA seam, a pairformer seam, a distogram linear) runs (a) DENSE — the stock statements written whole
in this file — and (b) through the bindings on P in {2, 3} gloo ranks (``opt_core.mem.rowpair.launch.run_sharded(cpu_ok=True)``): z is only
ever this rank's rows; every piece is compared with the dense statement's rows (fp32, one BLAS thread; max|diff| <= 1e-5 asserted,
torch.equal REPORTED in RESULT lines); the sharded trunk is held to "no tensor of numel >= N*N*c_z produced on a rank" (TorchDispatchMode
census); the schedule census carries the bindings' words. The carried unit and protenix are NOT imported: the seam registry
(``ptx_tp.impl``), its log / phase hooks and protenix's contact-probability helpers are stubbed per rank.

Run: ``OMP_NUM_THREADS=1 python -m pytest protenix_opt/tests/test_tp_rebase_trunk.py -q -rfE -s`` (needs torch + opt_core 0.4.3)."""
from __future__ import annotations

import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.path.dirname(os.path.dirname(HERE))                      # opt/ (protenix_opt's parent)
if PKG_PARENT not in sys.path:
    sys.path.insert(0, PKG_PARENT)

import pytest  # noqa: E402

from protenix_opt.tests.conftest import needs_gloo  # noqa: E402

try:
    import torch  # noqa: F401
    import opt_core.mem.rowpair  # noqa: F401
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False

needs = pytest.mark.skipif(not HAVE, reason="torch + opt_core.mem.rowpair required")
TOL = 1e-5
N, B, C_Z, C_S, C_SI, C_T, T_SLOTS, R_MAX, S_MAX, N_BINS, N_CYCLE = 272, 128, 64, 12, 10, 8, 4, 4, 2, 6, 3
BLK = 128


# ================================================================================================================ stub engine (seeded, CPU)
def _lin(g, i, o):
    import torch
    m = torch.nn.Linear(i, o, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.randn((o, i), generator=g) / (i ** 0.5))
    return m


def _ln(g, c):
    import torch
    m = torch.nn.LayerNorm(c)
    with torch.no_grad():
        m.weight.copy_(1.0 + 0.1 * torch.randn((c,), generator=g)); m.bias.copy_(0.1 * torch.randn((c,), generator=g))
    return m


class NS(types.SimpleNamespace):
    pass


def build_model(seed: int = 3):
    """The attribute paths the bindings read, seeded identically on every rank."""
    import torch
    g = torch.Generator().manual_seed(seed)
    m = NS()
    m.c_z = C_Z
    m.configs = NS(mc_dropout_rate=0.25, triangle_multiplicative="torch", triangle_attention="torch",
                   loss=NS(distogram=NS(min_bin=2.0, max_bin=10.0, no_bins=N_BINS)))
    W_in = _lin(g, C_SI, C_SI)
    m.input_embedder = lambda feats, inplace_safe=False, chunk_size=None: W_in(feats["s_raw"])
    m.linear_no_bias_sinit = _lin(g, C_SI, C_S)
    m.linear_no_bias_zinit1, m.linear_no_bias_zinit2 = _lin(g, C_S, C_Z), _lin(g, C_S, C_Z)
    nf = 2 * (2 * R_MAX + 2) + 1 + (2 * S_MAX + 2)
    m.relative_position_encoding = NS(r_max=R_MAX, s_max=S_MAX, linear_no_bias=_lin(g, nf, C_Z))
    m.linear_no_bias_token_bond = _lin(g, 1, C_Z)
    m.layernorm_z_cycle, m.linear_no_bias_z_cycle = _ln(g, C_Z), _lin(g, C_Z, C_Z)
    m.layernorm_s, m.linear_no_bias_s = _ln(g, C_S), _lin(g, C_S, C_S)
    m.constraint_embedder = NS()
    te = NS(n_blocks=1, c=C_T)
    te.linear_no_bias_z, te.layernorm_z, te.linear_no_bias_a = _lin(g, C_Z, C_T), _ln(g, C_Z), _lin(g, 108, C_T)
    te.pairformer_stack = NS(blocks=[NS(lin=_lin(g, C_T, C_T), ln=_ln(g, C_T)), NS(lin=_lin(g, C_T, C_T), ln=_ln(g, C_T))])
    te.layernorm_v, te.linear_no_bias_u, te.relu = _ln(g, C_T), _lin(g, C_T, C_Z), torch.nn.ReLU()
    m.template_embedder = te
    m.msa_module = NS(lin=_lin(g, C_Z, C_Z), ln=_ln(g, C_Z))
    m.pairformer_stack = NS(lin_s=_lin(g, C_S, C_S), lin_z=_lin(g, C_Z, C_Z), ln_z=_ln(g, C_Z), lin_sz=_lin(g, C_S, C_Z))
    m.distogram_head = NS(linear=_lin(g, C_Z, N_BINS))
    for mod in (W_in, m.linear_no_bias_sinit, m.linear_no_bias_zinit1, m.linear_no_bias_zinit2, m.relative_position_encoding.linear_no_bias,
                m.linear_no_bias_token_bond, m.layernorm_z_cycle, m.linear_no_bias_z_cycle, m.layernorm_s, m.linear_no_bias_s, te.linear_no_bias_z,
                te.layernorm_z, te.linear_no_bias_a, te.layernorm_v, te.linear_no_bias_u, m.msa_module.lin, m.msa_module.ln,
                m.pairformer_stack.lin_s, m.pairformer_stack.lin_z, m.pairformer_stack.ln_z, m.pairformer_stack.lin_sz, m.distogram_head.linear):
        mod.requires_grad_(False)
    for blk in te.pairformer_stack.blocks:
        blk.lin.requires_grad_(False); blk.ln.requires_grad_(False)
    return m


# the seams the trunk binding reaches through the carried registry: row-local stand-ins (the same callables serve dense (whole z) and rows)
def pair_block(blk, v, layout=None, **kw):                      # c_s = 0 PairformerBlock stand-in: v + lin(ln(v)) (row-local)
    return v + blk.lin(blk.ln(v))


def msa_seam(msa_module, feats, z, s_inputs, layout=None, **kw):  # MSA module stand-in: z + lin(ln(z)) * mean(s_inputs) (row-local, reads s_inputs)
    return z + msa_module.lin(msa_module.ln(z)) * float(s_inputs.float().mean())


def pairformer_seam(stack, s, z, layout=None, g0=None, g1=None, **kw):   # pairformer stand-in: s <- s + lin_s(s); z <- z + lin_z(ln_z(z)) + lin_sz(s)_i (row term)
    s = s + stack.lin_s(s)
    rows = stack.lin_sz(s)                                        # [N, C_Z]; z[i, j] += rows[i]
    if g0 is None:
        g0, g1 = 0, rows.shape[0]
    z = z + stack.lin_z(stack.ln_z(z)) + rows[g0:g1, None, :]
    return s, z


def compute_contact_prob(distogram_logits, min_bin, max_bin, no_bins, thr=6.0):
    import torch
    p = torch.softmax(distogram_logits.float(), dim=-1)
    centers = torch.linspace(min_bin, max_bin, no_bins)
    return (p * (centers < thr).float()).sum(-1)


def install_stubs(layout=None):
    """Per-rank stand-ins for the carried unit's registry / instrumentation and protenix's contact helpers (the bindings import them lazily)."""
    px = types.ModuleType("ptx_tp")
    px.log = lambda msg: sys.stderr.write(f"[ptx_tp:stub] {msg}\n")
    px.ledger = lambda: "stub"
    seams = {"pairformer": types.SimpleNamespace(tp_pairformer_stack=lambda stack, s, z, lay, **kw: pairformer_seam(stack, s, z, lay, lay.r0, lay.r1)),
             "msa": types.SimpleNamespace(tp_msa_module=msa_seam, tp_pair_stack_block=pair_block)}
    px.impl = lambda seam: seams[seam]
    tr = types.ModuleType("ptx_tp.trunk"); tr.phase = lambda *a, **k: None
    px.trunk = tr
    sys.modules.update({"ptx_tp": px, "ptx_tp.trunk": tr})
    pm = types.ModuleType("protenix"); pmm = types.ModuleType("protenix.model"); sc = types.ModuleType("protenix.model.sample_confidence")
    sc.compute_contact_prob = compute_contact_prob
    sc.get_bin_params = lambda cfg: {"min_bin": cfg.min_bin, "max_bin": cfg.max_bin, "no_bins": cfg.no_bins}
    pu = types.ModuleType("protenix.utils"); tu = types.ModuleType("protenix.utils.torch_utils")
    tu.autocasting_disable_decorator = lambda flag: (lambda fn: fn)
    pm.model, pmm.sample_confidence, pm.utils, pu.torch_utils = pmm, sc, pu, tu
    sys.modules.update({"protenix": pm, "protenix.model": pmm, "protenix.model.sample_confidence": sc, "protenix.utils": pu,
                        "protenix.utils.torch_utils": tu})


# ================================================================================================================ features + DENSE references
def build_feats(kind: str, seed: int = 11):
    """kind: 'compact' (dummy-free masks nonzero, distogram/unit-vector zero: the compacted dict) | 'dense' (random real pair features, dense keys)
    | 'dummy' (the featuriser's all-dummy compacted dict)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    n1, n2 = 100, 200
    f = {"asym_id": torch.cat([torch.zeros(n1), torch.ones(n2 - n1), 2 * torch.ones(N - n2)]).long(),
         "entity_id": torch.cat([torch.zeros(n2), torch.ones(N - n2)]).long(),
         "sym_id": torch.cat([torch.zeros(n1), torch.ones(n2 - n1), torch.zeros(N - n2)]).long(),
         "residue_index": torch.cat([torch.arange(n1), torch.arange(n2 - n1), torch.arange(N - n2) // 3]).long(),
         "token_index": torch.arange(N).long(),
         "s_raw": torch.randn((N, C_SI), generator=g)}
    tb = torch.zeros(N, N)
    for i in range(n2, N - 1, 2):
        tb[i, i + 1] = tb[i + 1, i] = 1.0
    f["token_bonds"] = tb
    f["template_aatype"] = torch.randint(0, 32, (T_SLOTS, N), generator=g)
    if kind == "dense":
        pb1d = (torch.rand((T_SLOTS, N), generator=g) > 0.3).float(); bb1d = (torch.rand((T_SLOTS, N), generator=g) > 0.3).float()
        f["template_distogram"] = (torch.rand((T_SLOTS, N, N, 39), generator=g) > 0.9).float()
        f["template_unit_vector"] = torch.randn((T_SLOTS, N, N, 3), generator=g)
        f["template_pseudo_beta_mask"] = pb1d[:, :, None] * pb1d[:, None, :]
        f["template_backbone_frame_mask"] = bb1d[:, :, None] * bb1d[:, None, :]
        f["template_aatype"][1] = f["template_aatype"][0]; f["template_distogram"][1] = f["template_distogram"][0]     # slots 0/1 identical: a dedup group
        f["template_unit_vector"][1] = f["template_unit_vector"][0]; f["template_pseudo_beta_mask"][1] = f["template_pseudo_beta_mask"][0]
        f["template_backbone_frame_mask"][1] = f["template_backbone_frame_mask"][0]
    else:
        live = kind == "compact"
        f["template_pseudo_beta_mask_1d"] = (torch.rand((T_SLOTS, N), generator=g) > 0.3).float() * (1.0 if live else 0.0)
        f["template_backbone_frame_mask_1d"] = (torch.rand((T_SLOTS, N), generator=g) > 0.3).float() * (1.0 if live else 0.0)
        if not live:
            f["template_aatype"] = torch.full((T_SLOTS, N), 31, dtype=torch.long)
        f["template_pair_dense_free"] = torch.ones((), dtype=torch.long)
    return f


def dense_relp(f):
    """Protenix RelativePositionEncoding.generate_relp, whole (the arithmetic form of the stock module)."""
    import torch
    import torch.nn.functional as F
    same_chain = (f["asym_id"][:, None] == f["asym_id"][None, :]).long()
    same_res = (f["residue_index"][:, None] == f["residue_index"][None, :]).long()
    same_ent = (f["entity_id"][:, None] == f["entity_id"][None, :]).long()
    d_res = torch.clip(f["residue_index"][:, None] - f["residue_index"][None, :] + R_MAX, 0, 2 * R_MAX) * same_chain + (1 - same_chain) * (2 * R_MAX + 1)
    rel_tok = torch.clip(f["token_index"][:, None] - f["token_index"][None, :] + R_MAX, 0, 2 * R_MAX) * same_chain * same_res \
        + (1 - same_chain * same_res) * (2 * R_MAX + 1)
    d_chain = torch.clip(f["sym_id"][:, None] - f["sym_id"][None, :] + S_MAX, 0, 2 * S_MAX) * same_ent + (1 - same_ent) * (2 * S_MAX + 1)
    return torch.cat([F.one_hot(d_res, 2 * R_MAX + 2), F.one_hot(rel_tok, 2 * R_MAX + 2), same_ent[..., None], F.one_hot(d_chain, 2 * S_MAX + 2)], -1).float()


def dense_template_pair_feats(f):
    """(dgram [T,N,N,39], pb2d [T,N,N], uv [T,N,N,3], bb2d [T,N,N]) whole, whatever the dict's form."""
    import torch
    if "template_distogram" in f:
        return f["template_distogram"], f["template_pseudo_beta_mask"], f["template_unit_vector"], f["template_backbone_frame_mask"]
    pb, bb = f["template_pseudo_beta_mask_1d"], f["template_backbone_frame_mask_1d"]
    return (torch.zeros((T_SLOTS, N, N, 39)), pb[:, :, None] * pb[:, None, :], torch.zeros((T_SLOTS, N, N, 3)), bb[:, :, None] * bb[:, None, :])


def dense_template(m, f, z):
    """Stock TemplateEmbedder.forward, whole (pair block = the stand-in)."""
    import torch
    import torch.nn.functional as F
    te = m.template_embedder
    dg, pb2d, uv, bb2d = dense_template_pair_feats(f)
    mcm = (f["asym_id"][:, None] == f["asym_id"][None, :]).to(z.dtype)
    pm = z.new_ones((N, N))
    u = 0
    for t in range(T_SLOTS):
        aat = F.one_hot(f["template_aatype"][t], 32)
        at = torch.cat([dg[t] * mcm[..., None] * pm[..., None], (pb2d[t] * mcm * pm)[..., None], aat[None, :, :].expand(N, -1, -1),
                        aat[:, None, :].expand(-1, N, -1), uv[t] * mcm[..., None] * pm[..., None], (bb2d[t] * mcm * pm)[..., None]], dim=-1)
        v = te.linear_no_bias_z(te.layernorm_z(z)) + te.linear_no_bias_a(at)
        for blk in te.pairformer_stack.blocks:
            v = pair_block(blk, v)
        v = te.layernorm_v(v)
        u = u + v
    u = u / (1e-7 + T_SLOTS)
    return te.linear_no_bias_u(te.relu(u))


def mc_dropout_dense(u, p, cycle):
    """The block-seeded mask, whole tensor (the carried statement above its full-Philox gate: private Generator per 128-row global block)."""
    import torch
    out = torch.empty_like(u)
    base = int(torch.initial_seed()) % (2 ** 62)
    for c0 in range(0, u.shape[0], BLK):
        c1 = min(u.shape[0], c0 + BLK)
        g = torch.Generator(device=u.device); g.manual_seed(base + 1_000_003 * (cycle + 1) + (c0 // BLK))
        keep = torch.rand((c1 - c0,) + tuple(u.shape[1:]), generator=g, device=u.device, dtype=torch.float32) >= p
        out[c0:c1] = (u[c0:c1].float() * keep * (1.0 / (1.0 - p))).to(u.dtype)
    return out


def dense_trunk(m, f, n_cycle, mc_dropout):
    """Protenix.get_pairformer_output, whole (stock statement order), with the stand-in seams."""
    import torch
    s_inputs = m.input_embedder(f)
    s_init = m.linear_no_bias_sinit(s_inputs)
    z_init = m.linear_no_bias_zinit1(s_init)[:, None, :] + m.linear_no_bias_zinit2(s_init)[None, :, :]
    z_init += m.relative_position_encoding.linear_no_bias(dense_relp(f))
    z_init += m.linear_no_bias_token_bond(f["token_bonds"][..., None])
    z, s = torch.zeros_like(z_init), torch.zeros_like(s_init)
    per_cycle = []
    for c in range(n_cycle):
        u = m.linear_no_bias_z_cycle(m.layernorm_z_cycle(z))
        if mc_dropout:
            u = mc_dropout_dense(u, m.configs.mc_dropout_rate, c)
        z = z_init + u
        if "template_aatype" in f:
            z = z + dense_template(m, f, z)
        z = msa_seam(m.msa_module, f, z, s_inputs)
        s = s_init + m.linear_no_bias_s(m.layernorm_s(s))
        s, z = pairformer_seam(m.pairformer_stack, s, z)
        per_cycle.append(z.clone())
    L = m.distogram_head.linear(z)
    contact = compute_contact_prob(L + L.transpose(0, 1), **{"min_bin": 2.0, "max_bin": 10.0, "no_bins": N_BINS})
    return {"s_inputs": s_inputs, "s": s, "z": z, "z_init": z_init, "contact": contact, "per_cycle": per_cycle}


# ================================================================================================================ plumbing
def _metrics(a, b):
    import torch
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return {"max_abs": float((a - b).abs().max()) if a.numel() else 0.0, "equal": bool(torch.equal(a, b)), "shape": list(a.shape)}


def _census(fn):
    """Run fn() under a TorchDispatchMode recording the largest tensor numel any op produced -> (result, max_numel)."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    top = {"numel": 0}

    class Census(TorchDispatchMode):
        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for o in tree_flatten(out)[0]:
                if isinstance(o, torch.Tensor):
                    top["numel"] = max(top["numel"], o.numel())
            return out
    with Census():
        res = fn()
    return res, top["numel"]


def _mp(P, entry, *args, **kw):
    from opt_core.mem.rowpair import launch
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("ROWPAIR_TEMPL_NODEDUPE", "0")
    return launch.run_sharded(P, entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=180, run_timeout_s=900, **kw)


def _entry(kind: str, mc_dropout: bool, align: int = 0, grid_b: int = B):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    install_stubs()
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair import evidence
    from opt_core.mem.rowpair.shard import unshard_rows
    from protenix_opt.tp_bind import relpos as BR, template as BT, trunk as BK
    P, r = D.world()
    lay = D.Layout.checked(N, P, r, grid_b, align=align) if align else D.Layout.checked(N, P, r, grid_b)   # aligned policy / a finer grid: R < the 128-row block
    m, f = build_model(), build_feats(kind)
    res = {"kind": kind, "mc_dropout": mc_dropout, "align": align, "grid_b": grid_b, "P": P, "rank": r, "rows": [lay.r0, lay.r1], "metrics": {}, "checks": {},
           "schedule": {}}
    mine = slice(lay.r0, lay.r1)
    with torch.no_grad():
        ref = dense_trunk(m, f, N_CYCLE, mc_dropout)
        # ---- relpos rows (both bound signatures) vs the dense feature
        rd = dense_relp(f)
        res["metrics"]["relp_rows"] = _metrics(BK.relp_rows(m.relative_position_encoding, f, lay.r0, lay.r1), rd[mine])
        res["metrics"]["relp_rows_pkg"] = _metrics(BR.relp_rows(f, 5, 17, R_MAX, S_MAX), rd[5:17])
        # ---- z_init shard vs dense
        s_init = m.linear_no_bias_sinit(m.input_embedder(f))
        zi = BK.zinit_rows(m, s_init, f, lay)
        res["metrics"]["zinit_rows"] = _metrics(zi, ref["z_init"][mine])
        # ---- template term on rows (add=False) vs dense, on the cycle-0 z
        z0 = ref["z_init"] + m.linear_no_bias_z_cycle(m.layernorm_z_cycle(torch.zeros_like(ref["z_init"])))
        if mc_dropout:
            z0 = ref["z_init"] + mc_dropout_dense(m.linear_no_bias_z_cycle(m.layernorm_z_cycle(torch.zeros_like(ref["z_init"]))), 0.25, 0)
        ft = BT.slice_template_inputs_to_rows(f, lay)
        zt_ref = (z0 + dense_template(m, f, z0))[mine]
        zt = BT.template_update_(m.template_embedder, ft, z0[mine].clone().contiguous(), lay, pair_block=pair_block, add=True)     # in place
        res["metrics"]["template_rows_inplace"] = _metrics(zt, zt_ref)
        z0_rows = z0[mine].clone().contiguous()
        zt2 = BT.template_update_(m.template_embedder, ft, z0_rows, lay, pair_block=pair_block, add=False)                      # z rows + term, z untouched
        res["metrics"]["template_rows_outofplace"] = _metrics(zt2, zt_ref)
        res["checks"]["template_outofplace_left_z"] = bool(torch.equal(z0_rows, z0[mine]))
        res["checks"]["templ_mode"] = evidence.schedule().get("templ_mode")
        res["checks"]["templ_groups"] = evidence.schedule().get("templ_groups")
        # ---- the whole trunk through the binding (numel census) vs dense
        (out, numel) = _census(lambda: BK.get_pairformer_output_tp(m, f, N_CYCLE, layout=lay, mc_dropout=mc_dropout))
        s_inputs, s, z_sh, lay2 = out
        # no N x N x c_z tensor on a rank; a dict carrying DENSE template pair features holds their row slices [T, R, N, 39] (never [T, N, N, 39])
        limit = N * N * C_Z if kind != "dense" else max(N * N * C_Z, T_SLOTS * lay.Rmax * N * 39 + 1)
        res["checks"]["max_numel"], res["checks"]["numel_limit"] = int(numel), int(limit)
        res["checks"]["dense_templ_full_numel"] = int(T_SLOTS * N * N * 39)
        res["checks"]["z_is_shard"] = list(z_sh.shape) == [lay.R, N, C_Z] and lay2 is lay
        res["metrics"]["trunk_z_rows"] = _metrics(z_sh, ref["z"][mine])
        res["metrics"]["trunk_z_full"] = _metrics(unshard_rows(z_sh, lay, dim=-3), ref["z"])
        res["metrics"]["trunk_s"] = _metrics(s, ref["s"])
        res["metrics"]["trunk_s_inputs"] = _metrics(s_inputs, ref["s_inputs"])
        # ---- distogram / contact rows (symmetrised head, one all-to-all window per block) vs dense
        cr = BK.contact_probs_rows(m, ref["z"][mine].contiguous(), lay)
        res["metrics"]["contact_rows"] = _metrics(cr, ref["contact"][mine])
        sched = evidence.schedule()
        res["schedule"] = {k: sched.get(k) for k in ("trunk_bind", "trunk_mc_dropout", "trunk_bind_rows", "trunk_bind_replicated", "trunk_init_rows",
                                                      "trunk_init_rows_source", "trunk_recycle_rows", "trunk_recycle_rows_source", "park_z_init",
                                                      "templ_bind", "templ_mode", "templ_rows", "templ_rows_source", "templ_slots",
                                                      "templ_real", "templ_dummy", "templ_event", "trunk_guard", "trunk_replicated")}
    import torch.distributed as tdist
    allv = [None] * P
    tdist.all_gather_object(allv, res)
    return allv


def _report_and_assert(allv, tag):
    bad = []
    for res in allv:
        for k, mt in res["metrics"].items():
            print(f"RESULT {tag} P={res['P']} B={res['grid_b']} align={res['align']} rank={res['rank']} rows={res['rows']} {k}: max_abs={mt['max_abs']:.3e} "
                  f"torch.equal={mt['equal']} shape={mt['shape']}")
            if mt["max_abs"] > TOL:
                bad.append((res["rank"], k, mt["max_abs"]))
        print(f"CENSUS {tag} P={res['P']} rank={res['rank']} checks={json.dumps(res['checks'])} schedule={json.dumps(res['schedule'])}")
        assert res["checks"]["z_is_shard"], res["checks"]
        assert res["checks"]["max_numel"] < res["checks"]["numel_limit"], ("an op produced a tensor of numel >= the shard bound on a rank", res["checks"])
        assert res["checks"]["max_numel"] < res["checks"]["dense_templ_full_numel"], res["checks"]
        assert res["checks"]["template_outofplace_left_z"], "template_update_(add=False) modified z"
        sch = res["schedule"]
        assert sch["trunk_bind"] == "protenix_v2" and sch["templ_bind"] == "protenix_v2"
        assert sch["trunk_mc_dropout"] == ("block_seeded" if res["mc_dropout"] else "off")
        assert sch["trunk_init_rows"] <= BLK and str(sch["trunk_init_rows_source"]).startswith("given"), sch
        assert sch["trunk_recycle_rows"] <= BLK and str(sch["trunk_recycle_rows_source"]).startswith("given"), sch
        assert sch["templ_rows"] <= BLK and str(sch["templ_rows_source"]).startswith("given"), sch
    assert not bad, bad


GRID = [(2, 128), (3, 64)]        # (P, B): the B=128 grid shards N=272 two ways (256 / 16); three ranks need the B=64 grid (128 / 128 / 16)


@needs
@needs_gloo
@pytest.mark.parametrize("P,grid_b", GRID)
@pytest.mark.parametrize("kind", ["compact", "dense", "dummy"])
def test_bindings_equal_dense_statements(P, grid_b, kind):
    allv = _mp(P, _entry, kind, False, 0, grid_b)
    _report_and_assert(allv, f"{kind}")
    modes = {res["checks"]["templ_mode"] for res in allv}
    assert modes == {"rows" if kind == "dense" else "computed"}, modes
    if kind == "dummy":                                                      # the featuriser's use_template=False dict: every slot a dummy, no event (not a templated run)
        assert all(int(res["schedule"]["templ_real"]) == 0 and int(res["schedule"]["templ_dummy"]) == T_SLOTS for res in allv), [r["schedule"] for r in allv]


@needs
@needs_gloo
@pytest.mark.parametrize("P,grid_b", GRID)
def test_mc_dropout_block_seeded_is_p_invariant(P, grid_b):
    allv = _mp(P, _entry, "compact", True, 0, grid_b)
    _report_and_assert(allv, "mc_dropout")


@needs
@needs_gloo
@pytest.mark.parametrize("P,grid_b", [(3, 64), (5, 32)])
@pytest.mark.parametrize("mc_dropout", [False, True])
def test_fine_grid_small_R_ragged_last_rank(P, grid_b, mc_dropout):
    """Finer grids at N=272: B=64, P=3 -> 128 / 128 / 16 rows; B=32, P=5 -> 64 / 64 / 64 / 64 / 16 rows (every rank's R below the statements'
    128-row block, the MC-dropout mask's 128-row global blocks straddle the shards, the last rank ragged)."""
    allv = _mp(P, _entry, "compact", mc_dropout, 0, grid_b)
    if grid_b == 32:
        assert all(res["rows"][1] - res["rows"][0] < BLK for res in allv), [res["rows"] for res in allv]
    _report_and_assert(allv, f"grid{grid_b}{'_mcd' if mc_dropout else ''}")


@needs
@needs_gloo
@pytest.mark.parametrize("P,align", [(4, 16), (3, 16)])
@pytest.mark.parametrize("mc_dropout", [False, True])
def test_aligned_layout_small_R_ragged_last_rank(P, align, mc_dropout):
    """The ALIGNED layout policy (dist.Layout align=16 at N=272 = 17 units over P ranks): every rank's R is below the 128-row block of the
    statements, the ranks are uneven, and the MC-dropout mask's 128-row global blocks straddle rank boundaries."""
    allv = _mp(P, _entry, "compact", mc_dropout, align)
    rows = [tuple(res["rows"]) for res in allv]
    assert all(r1 - r0 < BLK for r0, r1 in rows), rows
    _report_and_assert(allv, f"aligned{align}{'_mcd' if mc_dropout else ''}")


@needs
def test_refusals_by_name():
    import torch
    install_stubs()
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair import dist as D
    from protenix_opt.tp_bind import template as BT, trunk as BK
    m, f = build_model(), build_feats("compact")
    lay1 = D.Layout.checked(N, 1, 0, B)                                       # P == 1: the engine's own trunk runs; the binding refuses
    lay2 = D.Layout.checked(N, 2, 0, B)
    with pytest.raises(RowpairRefused):
        BK.get_pairformer_output_tp(m, f, 2, layout=lay1)
    with pytest.raises(RowpairRefused):
        BT.tp_template_embedder(m.template_embedder, f, torch.zeros(lay2.R, N, C_Z), lay2, pair_block=pair_block)   # term-returning form -> refused
    with pytest.raises(RowpairRefused):
        BT.tp_template_embedder(m.template_embedder, f, torch.zeros(lay2.R, N, C_Z), lay2, add_to_z=True)          # pair_block missing -> refused
    torch.manual_seed(7)                                                                          # the mask is a function of (seed, cycle, global block): any row split agrees
    u = torch.randn(N, 9, 5)
    whole = BK.mc_dropout_rows(u, 0, N, 0.25, 1)
    parts = torch.cat([BK.mc_dropout_rows(u[a:b], a, N, 0.25, 1) for a, b in ((0, 50), (50, 130), (130, 271), (271, N))], dim=0)
    assert torch.equal(whole, parts) and torch.equal(whole, mc_dropout_dense(u, 0.25, 1))
    f2 = dict(f); f2["constraint_feature"] = {"contact": torch.zeros(N, N, 2)}
    with pytest.raises(RowpairRefused):
        BK.get_pairformer_output_tp(m, f2, 2, layout=lay2)                                       # refused before any collective
