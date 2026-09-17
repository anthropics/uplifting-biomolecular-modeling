"""The tp line's adapter routes every pair statement of OpenFold3's trunk through the core's row seams and never holds an N x N pair tensor:
a stub OpenFold3 model (the attribute paths the adapter binds, tiny CPU modules) runs ``tp_rowpair.trunk.run_trunk_rows`` for two recycles on
ONE rank's rows of a P=2 layout with RECORDING fakes served as the core seams (``tp_rowpair.core.inject``); the census of calls is asserted
(trunk rows born once, recycle / template / MSA-block / pair-block / attention-pair-bias rows per cycle, zero gathers of the pair
representation inside the trunk, every z the seams saw is ``[1, n_loc, N, C]``). Also: the structural n_gpu=1 rule (``install()`` outside a
rank process refuses by name and installs nothing), the switch-name map (``OF3TP_*`` -> ``ROWPAIR_*``; an unknown ``OF3TP_*`` name refused),
and a missing core seam refused by name. Needs torch (CPU) and opt_core on the path; openfold3 itself is stubbed.
"""
import os
import sys
import types
from collections import Counter

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


N, P, RANK, C_Z, C_S, C_M, N_MSA = 8, 2, 0, 4, 6, 5, 3
R0, R1 = 0, 4


# ----------------------------------------------------------------------------------------------------------------- openfold3 stubs (import surface only)
def _install_openfold3_stubs(monkeypatch):
    """The openfold3 import surface the trunk binding touches, as FRESH stand-in modules SHADOWING ``sys.modules`` for this test only
    (``monkeypatch.setitem``: a real openfold3 imported earlier in the session is put back at teardown, an absent one stays absent) — the real
    package's module objects are never mutated (a raising ``_attention`` left on the real module would poison every later dense reference)."""
    def add(a, b, inplace=False):
        if inplace:
            a += b
            return a
        return a + b

    class OffloadModules:
        class _V:
            def __init__(self, v): self.value = v
        MSA_MODULE = _V("msa_module")
        TEMPLATE_MODULE = _V("template_module")
        CONFIDENCE_HEADS = _V("confidence_heads")

    class FusedTriangleMultiplicativeUpdate(nn.Module):
        pass

    def binned_one_hot(d, bounds):
        return torch.nn.functional.one_hot(d.long().clamp(0, len(bounds) - 1), len(bounds)).float()

    def _attention(q, k, v, biases, use_high_precision=False):
        raise AssertionError("the census fakes never reach the attention core")

    mods = {
        "openfold3": {}, "openfold3.core": {}, "openfold3.core.utils": {}, "openfold3.core.utils.tensor_utils": {"add": add, "binned_one_hot": binned_one_hot},
        "openfold3.core.utils.device_utils": {"autocast_device_type": lambda x: (x.device.type if hasattr(x, "device") else "cpu")},   # 0.5.0's device-generic autocast helper (module or tensor -> device type)
        "openfold3.core.model": {}, "openfold3.core.model.layers": {}, "openfold3.core.model.primitives": {}, "openfold3.core.model.primitives.attention": {"_attention": _attention},
        "openfold3.core.model.layers.triangular_multiplicative_update": {"FusedTriangleMultiplicativeUpdate": FusedTriangleMultiplicativeUpdate},
        "openfold3.projects": {}, "openfold3.projects.of3_all_atom": {}, "openfold3.projects.of3_all_atom.model": {"OffloadModules": OffloadModules},
    }
    stubs = {}
    for name, attrs in mods.items():
        stub = types.ModuleType(name)
        stub.__dict__.update(attrs)
        stub.__path__ = []                                             # package-like: `from a.b import c` resolves the shadowed children
        stubs[name] = stub
        monkeypatch.setitem(sys.modules, name, stub)                   # shadows a real module (restored at undo) or fills an absent one (removed at undo)
    for name, stub in stubs.items():                                   # children reachable as attributes of their stand-in parents
        parent, _, child = name.rpartition(".")
        if parent in stubs:
            setattr(stubs[parent], child, stub)


# ----------------------------------------------------------------------------------------------------------------- recording fake core seams
class Recorder:
    def __init__(self):
        self.calls = Counter()
        self.z_shapes = []

    def see_z(self, z):
        z = z[0] if isinstance(z, list) else z
        if torch.is_tensor(z):
            self.z_shapes.append(tuple(z.shape))
        return z


def _fake_seams(rec: Recorder):
    class Layout:
        def __init__(self):
            self.N, self.P, self.rank, self.r0, self.r1, self.align = N, P, RANK, R0, R1, 4
            self.n_loc, self.parts = R1 - R0, [0, R1, N]
            self.R, self.Rmax, self.n_max, self.bounds = R1 - R0, R1 - R0, R1 - R0, [0, R1, N]

    class Comm:
        rank, world, P, verbose, backend, active = RANK, P, P, False, "fake", True
        def log(self, msg): rec.calls["log"] += 1
        def checksum(self, t, tag): rec.calls["checksum"] += 1; return True

    comm = Comm()

    def ops_record(kind):
        def make(*a, **kw):
            rec.calls[f"ops:{kind}"] += 1
            return types.SimpleNamespace(kind=kind, args=a, **kw)
        return make

    # ---- trunk (S2 surface): row primitives + the recycling loop taking the adapter's callables
    def same_rows(x, g0, g1):
        return x[..., g0:g1, None] == x[..., None, :]

    def reloffset_rows(pos, g0, g1, clip, condition=None):               # the clipped offset bins (the adapter one-hots them itself: the cyclic chains' wrap sits in between)
        rec.calls["trunk.reloffset_rows"] += 1
        d = ((pos[..., g0:g1, None] - pos[..., None, :]).long() + clip).clamp(0, 2 * clip)
        if condition is not None:
            d = torch.where(condition, d, torch.full_like(d, 2 * clip + 1))
        return d

    def relpos_onehot_rows(pos, g0, g1, clip, condition=None, dtype=None, one_hot=None):
        rec.calls["trunk.relpos_onehot_rows"] += 1
        d = (pos[..., g0:g1, None] - pos[..., None, :]).long().clamp(-clip, clip) + clip
        if condition is not None:
            d = torch.where(condition, d, torch.full_like(d, 2 * clip + 1))
        n = 2 * clip + 2
        oh = torch.nn.functional.one_hot(d, n) if one_hot is None else one_hot(d, n)
        return oh.to(dtype=dtype or torch.float32)

    def outer_sum_rows(a, b, g0, g1):
        rec.calls["trunk.outer_sum_rows"] += 1
        return a[..., g0:g1, :].unsqueeze(-2) + b.unsqueeze(-3)

    def feature_rows(x, g0, g1, device=None, *, row_dim=-2, non_blocking=False):
        return x.narrow(row_dim % x.dim(), g0, g1 - g0)

    def run_trunk_sharded(lay, *, n_cycles, init_rows_fn, init_like, recycle_update_fn, s_init, single_recycle_fn, template_fn, msa_fn, pairstack_fn,
                          init_channels=None, s_input=None, gather="none", ckpt=None, on_cycle_start=None, census_fn=None, log=None, **kw):
        rec.calls["trunk.run_trunk_sharded"] += 1
        assert gather == "none", gather                       # the adapter never asks the core to gather the pair representation
        z_init = rec.see_z(init_rows_fn(lay.r0, lay.r1)); rec.calls["trunk.pair_init_rows"] += 1
        s, z = s_init, None
        for cycle in range(n_cycles):
            if on_cycle_start is not None:
                on_cycle_start(cycle, cycle == n_cycles - 1)
            z = rec.see_z(z_init + recycle_update_fn(torch.zeros_like(z_init) if z is None else z)); rec.calls["trunk.recycle_rows_"] += 1
            z = rec.see_z(template_fn(z, cycle))
            z = rec.see_z(msa_fn(z, cycle))
            s = single_recycle_fn(s, cycle)
            s, z = pairstack_fn(s, z, cycle)
            rec.see_z(z)
        return types.SimpleNamespace(s=s, z=z, layout=lay, sharded=True)

    def pair_mask_rows(token_mask, r0, r1):
        rec.calls["trunk.pair_mask_rows"] += 1
        return token_mask[..., r0:r1, None] * token_mask[..., None, :]

    def template_add_rows_(z_loc, pair_mask_loc, lay, *, rows, embed, pair_stack, close, census_tag=None, dedupe=True, park_z=False, park_u=False):
        rec.calls["template.template_add_rows_"] += 1
        u = torch.zeros(1, 1, lay.n_loc, N, 3)
        u = pair_stack(u, pair_mask_loc)                    # drives the adapter's pair_stack_rows over the template blocks
        return rec.see_z(z_loc + close(u.mean(dim=-4)))

    def TemplatePairRows(feats, *, asym_id, token_mask, layout, lazy):
        rec.calls["template.TemplatePairRows"] += 1
        return types.SimpleNamespace(feats=feats, lazy=lazy)

    def opm_rows(ops, m, msa_mask, lay, chunk_size=None, z_acc=None):
        rec.calls["msa.opm_rows"] += 1
        out = torch.zeros(1, lay.n_loc, N, C_Z)
        if z_acc is not None:
            z_acc += out
            return rec.see_z(z_acc)
        return out

    def pwa_rows(ops, m, z_loc, pair_mask_loc, lay, chunk_size=None):
        rec.calls["msa.pwa_rows"] += 1
        rec.see_z(z_loc)
        return torch.zeros_like(m)

    def bind(**kw):
        rec.calls["ops:pairblock"] += 1
        return types.SimpleNamespace(**kw)

    def pair_block_(fns, z, mask_shard, lay, *, s=None, maskT_shard=None, census=False, transition_mask=True, **kw):
        rec.calls["pairstack.pair_block_rows"] += 1
        assert z.dim() == 3 and z.shape[0] == lay.n_loc and z.shape[1] == N, tuple(z.shape)      # the core sees [R, N, C]
        rec.see_z(z)
        z = z + 0.5
        if getattr(fns, "apb", None) is not None:
            s = fns.apb(s, z, lay)
            s = fns.single_transition(s)
        return rec.see_z(z), s

    def pair_stack_(blocks, z, mask_shard, lay, *, s=None, block_callback=None, between_blocks=None, **kw):
        for i, b in enumerate(blocks):
            z, s = pair_block_(b, z, mask_shard, lay, s=s, **kw)
            if block_callback is not None:
                block_callback(i, s, z)
        return z, s

    def apb_local_queries(attn_fn, s_full, z_shard, lay, *, gather=True):
        rec.calls["pairstack.attn_pair_bias_rows"] += 1
        rec.see_z(z_shard)
        return torch.zeros(tuple(s_full.shape[:-1]) + (C_S,))

    def blocks4(n, b):
        b = max(4, (int(b) // 4) * 4)
        return [(i, min(n, i + b)) for i in range(0, n, b)]

    def gather_rows(x, lay, dim=0):
        rec.calls["shard.gather_rows"] += 1                # the census asserts this stays 0 inside the trunk
        return x

    def env_int(name, default=0):
        v = os.environ.get(name, "")
        return int(v) if v.strip() else default

    return {
        "dist": types.SimpleNamespace(comm=lambda: comm, default_ctx=lambda n: Layout(), env_int=env_int, Layout=Layout),
        "shard": types.SimpleNamespace(gather_rows=gather_rows),
        "trunk": types.SimpleNamespace(same_rows=same_rows, reloffset_rows=reloffset_rows, relpos_onehot_rows=relpos_onehot_rows, outer_sum_rows=outer_sum_rows, feature_rows=feature_rows,
                                       pair_mask_rows=pair_mask_rows, run_trunk_sharded=run_trunk_sharded),
        "template": types.SimpleNamespace(TemplatePairRows=TemplatePairRows, PairEmbedderOps=ops_record("template"), template_add_rows_=template_add_rows_),
        "msa": types.SimpleNamespace(OpmOps=ops_record("opm"), PwaOps=ops_record("pwa"), opm_rows=opm_rows, pwa_rows=pwa_rows),
        "pairstack": types.SimpleNamespace(bind=bind, pair_block_=pair_block_, pair_stack_=pair_stack_),
        "trimul": types.SimpleNamespace(TriMulFns=ops_record("trimul")),
        "trimul_fused": types.SimpleNamespace(fused_trimul_fns=lambda weights, stock, eps=None, **kw: stock),        # the provider around the recorded statements: the statements (this test records seams, not kernels)
        "triatt": types.SimpleNamespace(TriAttFns=ops_record("triatt"), attend_query_blocks=lambda *a, **k: None, blocks4=blocks4, lever_rows=lambda name: None,
                                        attention_core=lambda core, **kw: (lambda q, k, v, biases: core(q, k, v, biases))),
        "transition": types.SimpleNamespace(apb_local_queries=apb_local_queries),
    }


# ----------------------------------------------------------------------------------------------------------------- stub OpenFold3 (attribute paths only)
def _tmu():
    return types.SimpleNamespace(layer_norm_in=nn.LayerNorm(C_Z), layer_norm_out=nn.LayerNorm(C_Z), linear_a_p=nn.Linear(C_Z, C_Z), linear_a_g=nn.Linear(C_Z, C_Z),
                                 linear_b_p=nn.Linear(C_Z, C_Z), linear_b_g=nn.Linear(C_Z, C_Z), linear_g=nn.Linear(C_Z, C_Z), linear_z=nn.Linear(C_Z, C_Z))


def _ta():
    return types.SimpleNamespace(starting=True, layer_norm=nn.LayerNorm(C_Z), linear_z=nn.Linear(C_Z, 2), mha=None, inf=1e9)


def _pair_block():
    return types.SimpleNamespace(tri_mul_out=_tmu(), tri_mul_in=_tmu(), tri_att_start=_ta(), tri_att_end=_ta(), tri_mul_first=True,
                                 pair_transition=lambda x, mask=None, chunk_size=None: torch.zeros_like(x))


def _apb():
    H = 2
    mha = types.SimpleNamespace(_prep_qkv=None, _wrap_up=lambda o, a: o.reshape(tuple(o.shape[:-2]) + (-1,)))
    return types.SimpleNamespace(use_ada_layer_norm=False, layer_norm_a=nn.LayerNorm(C_S), layer_norm_z=nn.LayerNorm(C_Z), linear_z=nn.Linear(C_Z, H), inf=1e9, mha=mha)


def _stub_model(n_msa_blocks=2, n_pf_blocks=3, n_templ_blocks=1):
    mem = types.SimpleNamespace(chunk_size=4, use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False, use_triton_triangle_kernels=False, use_lma=False,
                                msa_module=types.SimpleNamespace(swiglu_chunk_token_cutoff=None, swiglu_seq_chunk_size=None))
    ie = types.SimpleNamespace(atom_attn_enc=lambda batch, use_high_precision_attention: (torch.zeros(1, N, 2), None, None, None),
                               linear_s=nn.Linear(2 + 3 + 3 + 1, C_S), linear_z_i=nn.Linear(9, C_Z), linear_z_j=nn.Linear(9, C_Z), linear_relpos=nn.Linear(2 * (2 * 2 + 2) + 1 + (2 * 1 + 2), C_Z),
                               linear_token_bonds=nn.Linear(1, C_Z), max_relative_idx=2, max_relative_chain=1)
    tpe = types.SimpleNamespace(**{n: nn.Identity() for n in ("dgram_linear", "pseudo_beta_mask_linear", "x_linear", "y_linear", "z_linear", "backbone_mask_linear", "layer_norm_z", "linear_z")})
    te = types.SimpleNamespace(template_pair_embedder=tpe, template_pair_stack=types.SimpleNamespace(blocks=[_pair_block() for _ in range(n_templ_blocks)]), linear_t=nn.Linear(3, C_Z))
    msa_blk = lambda: types.SimpleNamespace(opm_first=True, skip_msa_update=False, msa_dropout_layer=nn.Identity(), pair_stack=_pair_block(),  # noqa: E731
                                              outer_product_mean=types.SimpleNamespace(layer_norm=nn.LayerNorm(C_M), linear_1=nn.Linear(C_M, 2), linear_2=nn.Linear(C_M, 2), linear_out=nn.Linear(4, C_Z), eps=1e-3),
                                              msa_att_row=types.SimpleNamespace(layer_norm_m=nn.LayerNorm(C_M), layer_norm_z=nn.LayerNorm(C_Z), linear_z=nn.Linear(C_Z, 2), linear_v=nn.Linear(C_M, 4),
                                                                                linear_g=nn.Linear(C_M, 4), linear_o=nn.Linear(4, C_M), no_heads=2, inf=1e9),
                                              msa_transition=lambda m, mask=None, chunk_size=None, ckpt_chunk_size=None: torch.zeros_like(m))
    pf_blk = lambda: types.SimpleNamespace(pair_stack=_pair_block(), attn_pair_bias=_apb(), single_transition=lambda s, mask=None, chunk_size=None: torch.zeros_like(s))  # noqa: E731
    model = types.SimpleNamespace(
        training=False, _get_mode_mem_settings=lambda: mem, _do_inference_offload=lambda seq_len, module_name: False, clear_autocast_cache=lambda: None,
        input_embedder=ie, layer_norm_z=nn.LayerNorm(C_Z), linear_z=nn.Linear(C_Z, C_Z), template_embedder=te,
        msa_module_embedder=lambda batch, s_input: (torch.zeros(1, N_MSA, N, C_M), torch.ones(1, N_MSA, N)),
        msa_module=types.SimpleNamespace(chunk_size_tuner=None, clear_cache_between_blocks=False, blocks=[msa_blk() for _ in range(n_msa_blocks)]),
        layer_norm_s=nn.LayerNorm(C_S), linear_s=nn.Linear(C_S, C_S),
        pairformer_stack=types.SimpleNamespace(chunk_size_tuner=None, blocks_per_ckpt=None, clear_cache_between_blocks=False, blocks=[pf_blk() for _ in range(n_pf_blocks)]),
    )
    batch = {"token_mask": torch.ones(1, N), "restype": torch.zeros(1, N, 3), "profile": torch.zeros(1, N, 3), "deletion_mean": torch.zeros(1, N),
             "asym_id": torch.zeros(1, N), "residue_index": torch.arange(N)[None].float(), "entity_id": torch.zeros(1, N), "token_index": torch.arange(N)[None].float(),
             "sym_id": torch.zeros(1, N), "token_bonds": torch.zeros(1, N, N)}
    return model, batch


# ----------------------------------------------------------------------------------------------------------------- tests
@pytest.fixture()
def adapter(monkeypatch):
    for k in [k for k in os.environ if k.startswith(("OF3TP_", "ROWPAIR_"))]:
        monkeypatch.delenv(k, raising=False)
    saved = {k: m for k, m in sys.modules.items() if k == "openfold3" or k.startswith(("openfold3.", "openfold3_ob0_opt.tp_rowpair"))}   # a real openfold3 in this session is restored at teardown
    _install_openfold3_stubs(monkeypatch)
    for n in [n for n in sys.modules if n.startswith("openfold3_ob0_opt.tp_rowpair")]:
        del sys.modules[n]
    import openfold3_ob0_opt.tp_rowpair as tp
    from openfold3_ob0_opt.tp_rowpair import core
    core.reset()
    rec = Recorder()
    for name, mod in _fake_seams(rec).items():
        core.inject(name, mod)
    # the MSA-module and template bindings have their own stock-oracle tests (test_tp_rowpair_{msa,template}_cpu.py); here their trunk-facing
    # entry points are recording stand-ins so the census is the trunk's + the pair-stack binding's routing
    from openfold3_ob0_opt.tp_rowpair import msa as MSA, template as TEMPL, pairstack as PS

    def msa_module_rows(stack, m, z_loc, msa_mask, pair_mask_loc, lay, chunk_size=None, **kw):
        rec.calls["msa.module_rows"] += 1
        for blk in stack.blocks:
            rec.calls["msa.block_rows"] += 1
            z_loc = PS.pair_block_rows(blk.pair_stack, z_loc, pair_mask_loc, lay, chunk_size=chunk_size, inplace_safe=True)
        return rec.see_z(z_loc)

    def template_embedder_add_rows_(te, batch, z_loc, pair_mask_loc, lay, census_tag=None, chunk_size=None, **kw):
        rec.calls["template.add_rows_"] += 1
        for blk in te.template_pair_stack.blocks:
            PS.pair_block_rows(blk, z_loc.new_zeros(z_loc.shape[:-1] + (C_Z,)), pair_mask_loc, lay, chunk_size=chunk_size, inplace_safe=True)
        return rec.see_z(z_loc)

    monkeypatch.setattr(MSA, "msa_module_rows", msa_module_rows)
    monkeypatch.setattr(MSA, "log_batch_residency", lambda batch, tag, top=12: None)
    monkeypatch.setattr(MSA, "log_live_tensors", lambda tag, min_gib=1.0: None)
    monkeypatch.setattr(TEMPL, "template_embedder_add_rows_", template_embedder_add_rows_)
    yield tp, core, rec
    core.reset()
    for k in [k for k in sys.modules if k == "openfold3" or k.startswith(("openfold3.", "openfold3_ob0_opt.tp_rowpair"))]:
        del sys.modules[k]
    sys.modules.update(saved)


def test_trunk_seam_census(adapter):
    tp, core, rec = adapter
    from openfold3_ob0_opt.tp_rowpair import trunk
    n_msa, n_pf, n_templ, cycles = 2, 3, 1, 2
    model, batch = _stub_model(n_msa, n_pf, n_templ)
    with torch.no_grad():
        s_input, s, z_loc, lay = trunk.run_trunk_rows(model, batch, num_cycles=cycles, inplace_safe=True)
    c = rec.calls
    census = {"trunk_rows": c["trunk.pair_init_rows"], "recycle_rows": c["trunk.recycle_rows_"], "template_rows": c["template.add_rows_"],
              "msa_module_rows": c["msa.module_rows"], "msa_blocks_rows": c["msa.block_rows"], "presharded_calls": c["pairstack.pair_block_rows"],
              "apb_rows": c["pairstack.attn_pair_bias_rows"], "gathers": c["shard.gather_rows"]}
    assert census == {"trunk_rows": 1, "recycle_rows": cycles, "template_rows": cycles, "msa_module_rows": cycles, "msa_blocks_rows": cycles * n_msa,
                      "presharded_calls": cycles * (n_msa + n_pf + n_templ), "apb_rows": cycles * n_pf, "gathers": 0}, census
    assert tuple(z_loc.shape) == (1, R1 - R0, N, C_Z) and tuple(s.shape) == (1, N, C_S) and tuple(s_input.shape) == (1, N, 9)
    assert rec.z_shapes and all(sh[-3] == R1 - R0 and sh[-2] == N for sh in rec.z_shapes if len(sh) >= 3 and sh[-1] == C_Z), rec.z_shapes   # never [N, N, C] on a rank
    assert c["ops:trimul"] == 2 * census["presharded_calls"] and c["ops:triatt"] == 2 * census["presharded_calls"] and c["trunk.run_trunk_sharded"] == 1


def test_install_refuses_outside_a_rank_process_and_installs_nothing(adapter, monkeypatch):
    tp, core, rec = adapter
    monkeypatch.delenv("OF3TP_WORLD", raising=False)
    with pytest.raises(tp.NotARank, match="installs nothing at n_gpu=1"):
        tp.install()
    assert tp.installed_names() == [] and not tp.active()
    monkeypatch.setenv("OF3TP_WORLD", "1")
    monkeypatch.setenv("OF3TP_RANK", "0")
    with pytest.raises(tp.NotARank):
        tp.install()
    assert tp.world_of({"ROWPAIR_WORLD": "4"}) == 4 and tp.rank_of({"OF3TP_RANK": "3"}) == 3 and not tp.is_rank_process({"OF3TP_WORLD": "2"})


def test_env_map_names(adapter, monkeypatch):
    tp, core, rec = adapter
    from openfold3_ob0_opt.tp_rowpair import env
    e = {"OF3TP_CHUNK": "64", "OF3TP_TRIATT_QBLOCK": "256", "OF3TP_OPM_ROWS": "128", "ROWPAIR_OPM_ROWS": "32"}
    out = env.core_env(e)
    assert out == {"ROWPAIR_CHUNK_ALIGN": "64", "ROWPAIR_TRIATT_QBLOCK": "256"}, out               # an explicit ROWPAIR_ value wins
    written = env.export_core_env(dict(e))
    assert written == {**out, **env.LINE_CONSTANTS} and written["ROWPAIR_MSA_HOST"] == "rank0"     # the line's constants written beside the mapped sizes
    with pytest.raises(env.UnknownSwitch, match="OF3TP_NOT_A_SWITCH"):
        env.check_known({"OF3TP_NOT_A_SWITCH": "1"})
    with pytest.raises(env.UnknownSwitch, match="OF3TP_MODE"):                                  # the folded switches are unknown names now
        env.export_core_env({"OF3TP_MODE": "S"})
    assert all(v.startswith("ROWPAIR_") for v in env.ENV_MAP.values()) and all(k.startswith("ROWPAIR_") for k in env.LINE_CONSTANTS)


def test_missing_core_seam_refuses_by_name(adapter, monkeypatch):
    tp, core, rec = adapter
    core.reset()
    monkeypatch.setattr(core, "PACKAGE", "opt_core_absent_for_this_test.mem.rowpair")
    with pytest.raises(Exception, match=r"refused: the tp line needs opt_core_absent_for_this_test\.mem\.rowpair\.trunk"):
        core.seam("trunk")
    with pytest.raises(KeyError):
        core.seam("not_a_seam")


def test_every_seam_name_resolves_in_the_installed_core():
    """``core.SEAMS`` names only functions/classes the installed ``opt_core.mem.rowpair`` modules define (a renamed or missing core name fails here by name)."""
    import importlib
    pytest.importorskip("opt_core.mem.rowpair.dist")
    for n in [n for n in sys.modules if n.startswith("openfold3_ob0_opt.tp_rowpair")]:
        del sys.modules[n]
    from openfold3_ob0_opt.tp_rowpair import core
    core.reset()
    missing = []
    for name, attrs in core.SEAMS.items():
        mod = importlib.import_module(f"{core.PACKAGE}.{name}")
        missing += [f"{name}.{a}" for a in attrs if not hasattr(mod, a)]
    assert not missing, missing
