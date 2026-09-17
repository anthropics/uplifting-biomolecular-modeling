"""big: the mode's composition (base arm, the row, the README row exports), the memory levers on the core registry (flags, refusals by
name, the patches on stub modules), the census per unit and the exit rule, and the chunked bodies against the stock statements on CPU
tensors."""
import json
import os
import re
import sys
import types
import io
from contextlib import redirect_stderr

import pytest

from protenix_opt import big, modes, registry, stack
from protenix_opt import runner_hooks as rh


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    saved_path = list(sys.meta_path)
    saved_mods = {k: sys.modules.get(k) for k in ("protenix", big.FEATURIZER, big.DIFFUSION, big.TRANSFORMER, big.MODEL, big.EMBEDDERS, big.PAIRFORMER,
                                                     big.CONFIDENCE, rh.TARGET)}
    for k in saved_mods:
        sys.modules.pop(k, None)
    sys.modules["protenix"] = types.ModuleType("protenix")                     # the stub stock package: the levers' import precondition
    monkeypatch.setattr(big, "_STATE", {"record": None, "ctx": None, "unit": None, "n_unit": 0, "peaks": [], "seams": {}, "counters": {},
                                          "finders": {}, "patched": {}, "install_error": None, "decisions": {}, "verdict": None, "diffcache": None})
    monkeypatch.setattr(rh, "_STATE", {"guard_lift": False, "patched": False, "guard_lifted_items": 0, "max_n_token": 0, "finder": None, "failures": [], "handler": None, "listeners": []})
    for k in list(os.environ):
        if k.startswith("PROTENIX_V2_BIG_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(big, "SETTINGS", {lv: dict(kv) for lv, kv in big.SETTINGS.items()})   # a test's small settings never leak into the module table
    yield
    sys.meta_path[:] = saved_path
    for k, v in saved_mods.items():
        sys.modules.pop(k, None)
        if v is not None:
            sys.modules[k] = v


# ------------------------------------------------------------------------------------------------------------ composition


def test_big_row_is_the_base_without_the_graph_and_pad_levers_plus_the_memory_levers():
    for base in (modes.BIG_BASE,):
        row = modes.big_levers()
        assert not set(row) & set(modes.BIG_DROPPED)
        n_mem = len(modes.MEM_LEVERS)
        assert [n for n in modes.MODES[base] if n not in modes.BIG_DROPPED] == row[:len(row) - 1 - n_mem]
        assert row[-1 - n_mem:-n_mem] == ["guard_lift"]                        # the kit's own runner-hook lever
        assert tuple(row[-n_mem:]) == modes.MEM_LEVERS == big.LINE          # the memory line, in order
    assert modes.MODES["big"] == modes.big_levers()
    assert modes.BIG_POST == {"PTX_GUARD_LIFT": "1", "PTX_KEEP_POOL": "0"}          # the runner hook on; keep_pool (BIG_DROPPED) switched off after env.sh; pf_attn rides the base row's own export (carried since 0.3.33)


def test_big_has_one_base_and_no_selector():
    assert modes.BIG_BASE == "fast" and modes.ARM["big"] == modes.ARM["fast"]          # one lever set per mode: no base / line selector exists
    assert not hasattr(modes, "big_" + "base") and not hasattr(modes, "BIG_BASE_" + "ENV")


def test_big_readme_row_is_the_base_row_with_the_lean_pre_exports_and_no_pad8():
    for key in list(modes.README_ROWS) + [None]:
        fast = modes.readme_row(key, "big", "fast"); exact = modes.readme_row(key, "big", "exact")
        for row, base in ((fast, "fast"), (exact, "exact")):
            src = modes.readme_row(key, base)
            assert row["pre"] == {**src["pre"], **modes.BIG_PRE, **({"PTX_T_ATT": "big"} if src["pre"].get("PTX_T_ATT") == "fast" else {})}   # big names its own tier word to the shared core's tri-attention provider wherever the base row binds it
            assert not set(row["post"]) & set(modes.BIG_POST_DROPPED)
            assert row["post"] == {**{k: v for k, v in src["post"].items() if k not in modes.BIG_POST_DROPPED}, **modes.BIG_POST}
    assert modes.readme_row("9.0|3.7", "big") == modes.readme_row("9.0|3.7", "big", "fast")   # the fast base by default


def test_every_memory_lever_has_a_registry_row_and_a_marker():
    for name in modes.MEM_LEVERS:
        lv = registry.LEVERS[name]
        assert lv.probe == "marker" and lv.extra and lv.env_keys == (), "a memory lever has no switch: the line is one lever set"
        assert stack.MARKERS[name][0] == f"{big.MARK}{name}"
    assert registry.LEVERS["cond_chunk"].tier == registry.TOLERANCE and registry.LEVERS["apb_bias_chunk"].tier == registry.TOLERANCE
    assert registry.LEVERS["drop_bond_mask"].tier == registry.EXACT and registry.LEVERS["cache_release"].tier == registry.EXACT
    assert registry.LEVERS["relp_lazy"].tier == registry.TOLERANCE and registry.LEVERS["msa_zfree"].tier == registry.EXACT and registry.LEVERS["diffcache_free"].tier == registry.EXACT


# ------------------------------------------------------------------------------------------------------------ the core registry


def _core():
    return pytest.importorskip("opt_core.mem")


def test_line_levers_register_on_the_core_and_the_line_selects_whole():
    """The line's levers register on the core; the kit passes no switches, so the selection is the whole line in order (a switch, were one
    passed, would be named by the core: off_by_flag / switch.<name> refusals — the kit exposes none)."""
    mem = _core()
    big._register_levers()
    for name in big.LINE:
        assert name in mem.LEVERS or name == "cache_release"
    assert mem.LEVERS["cond_chunk"].exact == "band" and mem.LEVERS["drop_bond_mask"].exact == "bitwise"
    mem.discover()
    sel = mem.selection(big.LINE)
    assert sel.levers == big.LINE and sel.off_by_flag == () and not sel.refusals and sel.allow_partial is False
    assert mem.selection(big.LINE, allow_partial=True).allow_partial is True
    bad = mem.selection(big.LINE, switches={"nosuch": "1", "cond_chunk": "maybe"})
    assert {r.precondition for r in bad.refusals} == {"switch.nosuch", "switch.cond_chunk"}


def _stub_stock(with_torch=True):
    """Stub protenix modules: the featurizer, the diffusion conditioning (a per-row body), the transformer's pair-bias site, the model."""
    torch = pytest.importorskip("torch") if with_torch else None
    feat = types.ModuleType(big.FEATURIZER)

    class Featurizer:
        def get_mask_features(self):
            return {"bond_mask": torch.ones((7, 7), dtype=torch.long), "other": 1}
    feat.Featurizer = Featurizer

    diff = types.ModuleType(big.DIFFUSION)

    class DiffusionConditioning:
        def __init__(self):
            g = torch.Generator().manual_seed(0)
            self.w = torch.randn(4, 8, generator=g)

        def prepare_cache(self, relp_feature, z_trunk, inplace_safe=False):
            x = torch.cat([z_trunk, relp_feature], dim=-1)                      # per-(i, j) vector ops only, as the stock body
            return torch.nn.functional.layer_norm(x, (x.shape[-1],)) @ self.w.T
    diff.DiffusionConditioning = DiffusionConditioning

    tr = types.ModuleType(big.TRANSFORMER)

    def permute_final_dims(t, inds):
        zero_index = -1 * len(inds)
        first = list(range(len(t.shape[:zero_index])))
        return t.permute(first + [zero_index + i for i in inds])

    class AttentionPairBias:
        def __init__(self):
            g = torch.Generator().manual_seed(1)
            self.layernorm_z = torch.nn.LayerNorm(4)
            self.linear_nobias_z = torch.nn.Linear(4, 2, bias=False)
            with torch.no_grad():
                self.linear_nobias_z.weight.copy_(torch.randn(2, 4, generator=g))
            self.calls = []

        def attention(self, q_x, kv_x, attn_bias, inplace_safe=False):
            self.calls.append(attn_bias)
            return q_x + attn_bias.sum()

        def standard_multihead_attention(self, q, kv, z, inplace_safe=False, enable_efficient_fusion=False):
            bias = self.linear_nobias_z(self.layernorm_z(z))
            bias = permute_final_dims(bias, [2, 0, 1])
            return self.attention(q_x=q, kv_x=kv, attn_bias=bias, inplace_safe=inplace_safe)
    tr.AttentionPairBias = AttentionPairBias
    tr.permute_final_dims = permute_final_dims

    emb = types.ModuleType(big.EMBEDDERS)

    class RelativePositionEncoding:                                             # the stock body (protenix/model/modules/embedders.py generate_relp / forward), transcribed
        def __init__(self, r_max=32, s_max=2, c_z=8):
            self.r_max, self.s_max, self.c_z = r_max, s_max, c_z
            g = torch.Generator().manual_seed(5)
            self.linear_no_bias = torch.nn.Linear(4 * r_max + 2 * s_max + 7, c_z, bias=False)
            with torch.no_grad():
                self.linear_no_bias.weight.copy_(torch.randn(c_z, 4 * r_max + 2 * s_max + 7, generator=g))

        def forward(self, relp_feature):
            return self.linear_no_bias(relp_feature)

        def generate_relp(self, input_feature_dict):
            F = torch.nn.functional
            with torch.no_grad():
                asym_id = input_feature_dict["asym_id"]; residue_index = input_feature_dict["residue_index"]; entity_id = input_feature_dict["entity_id"]
                token_index = input_feature_dict["token_index"]; sym_id = input_feature_dict["sym_id"]
                b_same_chain = (asym_id[..., :, None] == asym_id[..., None, :]).long()
                b_same_residue = (residue_index[..., :, None] == residue_index[..., None, :]).long()
                b_same_entity = (entity_id[..., :, None] == entity_id[..., None, :]).long()
                d_residue = torch.clip(input=residue_index[..., :, None] - residue_index[..., None, :] + self.r_max, min=0, max=2 * self.r_max) * b_same_chain + (1 - b_same_chain) * (2 * self.r_max + 1)
                a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))
                d_token = torch.clip(input=token_index[..., :, None] - token_index[..., None, :] + self.r_max, min=0, max=2 * self.r_max) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (2 * self.r_max + 1)
                a_rel_token = F.one_hot(d_token, 2 * (self.r_max + 1))
                d_chain = torch.clip(input=sym_id[..., :, None] - sym_id[..., None, :] + self.s_max, min=0, max=2 * self.s_max) * b_same_entity + (1 - b_same_entity) * (2 * self.s_max + 1)
                a_rel_chain = F.one_hot(d_chain, 2 * (self.s_max + 1))
                input_feature_dict["relp"] = torch.cat([a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain], dim=-1).float()
            return input_feature_dict
    emb.RelativePositionEncoding = RelativePositionEncoding

    pf = types.ModuleType(big.PAIRFORMER)

    class _PairStack:
        def __call__(self, s, z, pair_mask, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
            return s, z * 2.0 + 1.0

    class MSABlock:                                                              # the stock MSABlock.forward (protenix/model/modules/pairformer.py), transcribed; tiny sub-modules
        def __init__(self, is_last_block=False):
            self.is_last_block = is_last_block
            self.pair_stack = _PairStack()

        def outer_product_mean_msa(self, m, inplace_safe=False, chunk_size=None):
            o = m.sum(dim=-3).sum(dim=-1)                                       # [n_token]
            return (o[:, None] * o[None, :])[..., None].expand(-1, -1, 3) * 0.5   # [n_token, n_token, c_z]: a fresh tensor, as the stock residual's operand

        def msa_stack(self, m, z):
            return m + z.sum()

        def forward(self, m, z, pair_mask, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
            # Communication
            z = z + self.outer_product_mean_msa(
                m, inplace_safe=inplace_safe, chunk_size=chunk_size
            )
            if not self.is_last_block:
                # MSA stack
                m = self.msa_stack(m, z)
            # Pair stack
            _, z = self.pair_stack(
                s=None,
                z=z,
                pair_mask=pair_mask,
                triangle_multiplicative=triangle_multiplicative,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            if not self.is_last_block:
                return m, z
            else:
                return None, z  # to ensure that `m` will not be used.
    pf.MSABlock = MSABlock

    conf = types.ModuleType(big.CONFIDENCE)

    class ConfidenceHead:
        def forward(self, x):
            return x + 1
    conf.ConfidenceHead = ConfidenceHead

    model = types.ModuleType(big.MODEL)

    class Protenix:
        def get_pairformer_output(self, input_feature_dict, n=1):
            return "trunk"

        def sample_diffusion(self, **kw):
            return "samples"
    model.Protenix = Protenix
    return feat, diff, tr, model, emb, pf, conf


def _unwrapped(f):
    """The stock function under every lever wrapper (each carries `__wrapped__`)."""
    while hasattr(f, "__wrapped__"):
        f = f.__wrapped__
    return f


def test_activate_patches_imported_modules_arms_the_rest_and_the_census_closes(tmp_path, monkeypatch):
    mem = _core()
    torch = pytest.importorskip("torch")
    feat, diff, tr, model, emb, pf, conf = _stub_stock()
    sys.modules[big.FEATURIZER] = feat                                          # imported before activation: patched now
    big.SETTINGS["cond_chunk"].update(rows=2, above_tok=1); big.SETTINGS["apb_bias_chunk"].update(rows=2, above_tok=1); big.SETTINGS["relp_lazy"].update(rows=4)
    marks = big.activate("fast", os.environ)
    assert marks == ["MEM:drop_bond_mask(applied)", "MEM:cond_chunk(armed)", "MEM:apb_bias_chunk(armed)", "MEM:cache_release(applied)", "MEM:relp_lazy(armed)",
                     "MEM:msa_zfree(armed)", "MEM:diffcache_free(armed)"]
    released = []
    monkeypatch.setattr(big._allocator, "release", lambda ctx, point, unit=None: (released.append(point), ctx.record.mark("cache_release", detail=point), True)[2])   # no CUDA on a CPU box: the seam's call stands in
    rec = big._STATE["record"]
    assert rec.base == "fast" and rec.levers == big.LINE and not rec.refused and rec.exact == "band"   # the test activates on the fast base explicitly
    assert rec.exact_per_lever == {"drop_bond_mask": "bitwise", "cond_chunk": "band", "apb_bias_chunk": "band", "cache_release": "bitwise", "relp_lazy": "band",
                                   "msa_zfree": "bitwise", "diffcache_free": "bitwise"}
    # the featurizer patch
    feats = feat.Featurizer().get_mask_features()
    assert feats["bond_mask"].shape == (0, 0) and feats["bond_mask"].dtype == torch.long and feats["other"] == 1
    # the armed patches fire when the modules are executed through the import system
    for m in (diff, tr, model, emb, pf, conf):
        fired = 0
        for f in list(sys.meta_path):
            if isinstance(f, rh.PostImportFinder) and f.target == m.__name__:      # every lever arming the module gets its turn (the finders chain)
                f.on_import(m)                                                     # what exec_module's wrapper does after the module body
                sys.meta_path.remove(f)
                fired += 1
        assert fired >= 1, f"no post-import patch armed for {m.__name__}"
    assert big._STATE["patched"] == {"drop_bond_mask": True, "cond_chunk": True, "apb_bias_chunk": True, "big_seams": True, "relp_lazy": True, "msa_zfree": True,
                                      "diffcache_free": True, "diffcache_free:confidence": True}
    # two units through the runner's per-item hook, with the seams
    m_runner = types.ModuleType(rh.TARGET)
    m_runner.update_inference_configs = lambda configs, n_token: configs
    sys.modules[rh.TARGET] = m_runner
    rh.install(True)
    cfg = types.SimpleNamespace(model_name="protenix-v2", infer_setting=types.SimpleNamespace(sample_diffusion_chunk_size=5),
                                skip_amp=types.SimpleNamespace(confidence_head=False, sample_diffusion=True))   # the runner's configs shape
    m_runner.update_inference_configs(cfg, 3000)
    px = model.Protenix()
    g2 = torch.Generator().manual_seed(3); n2_ = 6
    feats2 = {k: torch.randint(0, 4, (n2_,), generator=g2) for k in ("asym_id", "residue_index", "entity_id", "token_index", "sym_id")}
    rpe = emb.RelativePositionEncoding()
    d1 = rpe.generate_relp(dict(feats2))                                          # the relpe site runs on every item (the model's first statement): a LazyRelp, no plane
    assert isinstance(d1["relp"], big.LazyRelp) and d1["relp"].shape == (n2_, n2_, 139)
    with torch.no_grad():
        zi = rpe.forward(d1["relp"])                                              # the trunk's z_init term: the relpe linear per row block (rows=4 of 6: two blocks)
        assert zi.shape == (n2_, n2_, 8) and torch.allclose(zi, _unwrapped(rpe.forward)(rpe, _unwrapped(rpe.generate_relp)(rpe, dict(feats2))["relp"]), rtol=0, atol=1e-5)
        blk = pf.MSABlock(); mm = torch.randn(2, n2_, 3); zz = torch.randn(n2_, n2_, 3); zz_ref = zz.clone()
        m1, z1 = blk.forward(mm, zz, None)                                        # msa_zfree: the stock statements, the block-input z's storage released
        assert zz.untyped_storage().nbytes() == 0
        m0, z0 = _unwrapped(pf.MSABlock.forward)(pf.MSABlock(), mm, zz_ref, None)
        assert torch.equal(z1, z0) and torch.equal(m1, m0)
    px.get_pairformer_output({"bond_mask": torch.zeros((0, 0), dtype=torch.long)})
    dc = diff.DiffusionConditioning()
    z = torch.randn(5, 5, 3); relp = torch.randn(5, 5, 5)
    with torch.no_grad():
        out = dc.prepare_cache(relp, z)
        ref = _unwrapped(dc.prepare_cache)(dc, relp, z)
        assert out.shape == ref.shape and torch.allclose(out, ref, atol=1e-6)
        px.sample_diffusion()
        nb = out.untyped_storage().nbytes()
        assert nb > 0 and conf.ConfidenceHead().forward(torch.zeros(1)).item() == 1.0   # diffcache_free: the confidence head's entry releases the pair cache
        assert out.untyped_storage().nbytes() == 0 and big._STATE["diffcache"] is None
    m_runner.update_inference_configs(cfg, 3001)                                 # the second unit: an un-levered featurization (the named fallback)
    emb.RelativePositionEncoding().generate_relp(dict(feats2))
    px.get_pairformer_output({"bond_mask": torch.ones((7, 7), dtype=torch.long)})
    rep = big.reconcile({"mode": "big", "levers_applied": list(big.LINE), "levers_fallback": [], "fallback_reasons": {}})
    c = rep["big"]["census"]
    assert c["n_units"] == 3 and c["partial_units"] == ["item2:3001tok"]            # pre-item (expects nothing) + two items
    assert c["units"][big.PRE_ITEM_UNIT]["expected"] == [] and c["units"][big.PRE_ITEM_UNIT]["partial"] == []
    u1, u2 = c["units"]["item1:3000tok"], c["units"]["item2:3001tok"]
    assert set(u1["ran"]) == {"drop_bond_mask", "cond_chunk", "cache_release", "relp_lazy", "msa_zfree", "diffcache_free"} and u1["skipped"] == {"apb_bias_chunk": "site not reached on this unit"}
    assert set(u2["skipped"]) >= {"relp_lazy", "msa_zfree", "diffcache_free"}          # item 2 ended before those sites: named skips (the patches are live), never absent
    assert u2["fallback"]["drop_bond_mask"].startswith("bond_mask present with shape (7, 7)")
    assert rep["partial"] and "drop_bond_mask" in rep["levers_fallback"] and rep["fallback_reasons"]["drop_bond_mask"].startswith("big: item2:3001tok:")
    assert "peak_file" not in rep["big"], "the seams' torch counters stay in memory (big._STATE['peaks']); no side file"
    seams = [r["seam"] for r in big._STATE["peaks"]]
    assert seams[:7] == ["unit_end", "unit_begin", "relp", "trunk_end", "cond_cache", "sampling_end", "unit_end"] and released[:3] == ["stage"] * 3   # the pre-item unit closes at item 1


def test_apb_bias_chunk_matches_the_stock_statement_and_skips_the_fusion_branch(monkeypatch):
    _core(); torch = pytest.importorskip("torch")
    feat, diff, tr, model, emb, pf, conf = _stub_stock()
    sys.modules[big.TRANSFORMER] = tr
    big.SETTINGS["apb_bias_chunk"].update(rows=2, above_tok=1)
    marks = big.activate("fast", os.environ)
    assert "MEM:apb_bias_chunk(applied)" in marks
    m = tr.AttentionPairBias()
    q = torch.zeros(5, 3); z = torch.randn(5, 5, 4)
    m.standard_multihead_attention(q, q, z, enable_efficient_fusion=True)      # the fusion branch is stock's (skipped, named)
    m.calls.clear()
    out = m.standard_multihead_attention(q, q, z)
    ref = m.standard_multihead_attention.__wrapped__(m, q, q, z)
    assert m.calls[0].shape == (2, 5, 5) and torch.allclose(m.calls[0], m.calls[1], atol=1e-6) and torch.allclose(out, ref, atol=1e-5)
    rec = big._STATE["record"]
    assert rec.units[big.PRE_ITEM_UNIT].ran == ["apb_bias_chunk"] and "efficient-fusion" in rec.units[big.PRE_ITEM_UNIT].skipped.get("apb_bias_chunk", "")


def test_big_variables_are_refused_by_name_and_a_refused_lever_refuses_the_mode(monkeypatch):
    """A PROTENIX_V2_BIG_* variable selects nothing: the kit's undeclared-name gate refuses it by name (exit 3 on every route). A
    memory lever that cannot apply makes the mode refuse by name: the core's record is applied with allow_partial=False and there is
    no flag and no variable that changes it."""
    from protenix_opt import _autoload
    assert _autoload.undeclared({"PROTENIX_V2_BIG_COND_CHUNK": "0", "PROTENIX_V2_BIG_ALLOW_PARTIAL": "1", "PROTENIX_V2_BIG_COND_CHUNK_ROWS": "64", "PROTENIX_OPT": "big"}) == \
        ["PROTENIX_V2_BIG_ALLOW_PARTIAL", "PROTENIX_V2_BIG_COND_CHUNK", "PROTENIX_V2_BIG_COND_CHUNK_ROWS"]
    mem = _core()
    marks = big.activate("fast", os.environ)
    for n in big.LINE:
        assert any(m.startswith(f"MEM:{n}(") for m in marks)
    rec = big._STATE["record"]
    assert rec.off_by_flag == () and rec.allow_partial is False
    assert not hasattr(big, "request_allow_partial") and "allow_partial" not in big._STATE


def test_modes_big_literal_is_the_composition():
    """MODES["big"] is a LITERAL inside the MODES dict (tools outside this tree read the literal block's keys and tokens); it equals the kit's
    own composition big_levers() — a drift on either side fails here."""
    assert modes.MODES["big"] == modes.big_levers()
    src = open(modes.__file__, encoding="utf-8").read()
    block = src[src.index("MODES: Dict[str, List[str]] = {"):]
    block = block[:block.index("\n}\n")]
    assert '"big": [' in block and 'MODES["big"] =' not in src


def test_classify_has_no_off_switch_for_a_memory_lever(monkeypatch):
    """A PROTENIX_V2_BIG_<LEVER>=0 word in the environment changes nothing in `_classify` (the gate refuses it before activation):
    every line lever with its marker is applied; PTX_SAMPLER_FUSE=0 is the one switch-off `_classify` records (sampler_fuse skipped)."""
    applied = [f"MEM:{n}(applied)" for n in big.LINE] + ["GUARD_LIFT:1(armed)"]
    env = {"PROTENIX_V2_BIG_CACHE_RELEASE": "0", "PTX_GUARD_LIFT": "1"}
    on, fb, skipped, why = stack._classify("big", applied, env, row=modes.readme_row("9.0|3.7", "big", "fast"), row_key="9.0|3.7")
    for n in big.LINE:
        assert n in on and n not in skipped

def test_seams_record_the_peak_counters_in_memory(tmp_path, monkeypatch):
    """Every seam appends its torch counters to the in-memory record and releases the allocator cache at the stage; nothing is written."""
    mem = _core()
    torch = pytest.importorskip("torch")
    feat, diff, tr, model, emb, pf, conf = _stub_stock()
    empty = tmp_path / "cwd"; empty.mkdir(); monkeypatch.chdir(empty)
    big.activate("fast", os.environ)
    released = []
    monkeypatch.setattr(big._allocator, "release", lambda ctx, point, unit=None: (released.append(point), True)[1])
    big._seam("trunk_end"); big._seam("cond_cache")
    assert [r["seam"] for r in big._STATE["peaks"]] == ["trunk_end", "cond_cache"] and released == ["stage", "stage"]
    assert os.listdir(empty) == [], "no file written"


def test_lever_line_evidence_values_are_single_tokens(monkeypatch):
    """report.lever_lines renders every memory lever's evidence through opt_core.report.lever_line, which refuses a value with a blank:
    big.evidence() tokenises settings (drop_bond_mask's placeholder `[0, 0] int64`) and the census counts for every lever of the line."""
    mem = _core()
    from protenix_opt import report
    _stub_stock()
    marks = big.activate("fast", os.environ)
    assert all(m.startswith(big.MARK) and m.endswith(("(applied)", "(armed)")) for m in marks), marks
    block = big.state()
    rep = {"active": True, "mode": "big", "base": "fast", "levers_applied": list(modes.big_levers()), "levers_fallback": [],
           "levers_not_in_arm": [], "fallback_reasons": {}, "big": block}
    lines = report.lever_lines(rep)                                       # raises ValueError on a value with a blank
    for lv in big.LINE:
        mine = [l for l in lines if l.endswith(f" lever={lv}")]
        assert len(mine) == 1 and " state=on " in mine[0] and " exact=" in mine[0], mine
        for pair in big.evidence(lv, block):
            assert " " not in str(pair[1]), pair
    assert " placeholder=[0,_0]_int64 " in [l for l in lines if l.endswith(" lever=drop_bond_mask")][0]


# ------------------------------------------------------------------------------------------------------------ relp_lazy / msa_zfree / diffcache_free


def _token_feats(torch, n, seed=7):
    g = torch.Generator().manual_seed(seed)
    return {"asym_id": torch.randint(0, 3, (n,), generator=g), "residue_index": torch.randint(0, 40, (n,), generator=g), "entity_id": torch.randint(0, 2, (n,), generator=g),
            "token_index": torch.randint(0, 60, (n,), generator=g), "sym_id": torch.randint(0, 3, (n,), generator=g)}


def test_relp_lazy_rows_are_the_stock_plane_rows_and_the_forward_is_the_stock_linear_per_block(monkeypatch):
    """relp_lazy: input_feature_dict['relp'] is a LazyRelp; its rows (rows(i0, i1) and the [..., i0:i1, :, :] slice cond_chunk takes) are
    torch.equal to the stock generate_relp plane's rows (transcribed from the pin) at several block boundaries; RelativePositionEncoding.forward
    on the LazyRelp is torch.equal to the stock forward on the stock plane (CPU fp32; one `ran` event per chunked call); an index form the
    lever does not serve raises TypeError by name; the plane is never an attribute of the object."""
    mem = _core()
    torch = pytest.importorskip("torch")
    feat, diff, tr, model, emb, pf, conf = _stub_stock()
    big.SETTINGS["relp_lazy"].update(rows=7)
    sys.modules[big.EMBEDDERS] = emb
    marks = big.activate("fast", os.environ)
    assert "MEM:relp_lazy(applied)" in marks
    n = 23
    feats = _token_feats(torch, n)
    rpe = emb.RelativePositionEncoding()
    ref = _unwrapped(rpe.generate_relp)(rpe, dict(feats))["relp"]
    lazy = rpe.generate_relp(dict(feats))["relp"]
    assert isinstance(lazy, big.LazyRelp) and lazy.shape == tuple(ref.shape) == (n, n, 139) and lazy.dtype == ref.dtype and lazy.numel() == ref.numel()
    for i0, i1 in ((0, n), (0, 7), (7, 14), (21, 23), (5, 6)):
        assert torch.equal(lazy.rows(i0, i1), ref[i0:i1]) and torch.equal(lazy[..., i0:i1, :, :], ref[..., i0:i1, :, :])
    with pytest.raises(TypeError, match="LazyRelp: unsupported index"):
        lazy[0]
    with pytest.raises(TypeError, match="LazyRelp: unsupported index"):
        lazy[..., 0:4, 0:4, :]
    with torch.no_grad():
        out = rpe.forward(lazy)                                                    # 23 rows in blocks of 7: 4 blocks into one [23, 23, c_z] output
        assert out.shape == (n, n, 8) and torch.allclose(out, _unwrapped(rpe.forward)(rpe, ref), rtol=0, atol=1e-5)   # a CPU GEMM at another M: 1-ulp class (bitwise on the pinned GPU stack)
        for i0, i1 in ((0, 7), (7, 14), (14, 21), (21, 23)):                        # per block the statement is the stock linear on those rows
            assert torch.allclose(out[i0:i1], _unwrapped(rpe.forward)(rpe, ref[i0:i1]), rtol=0, atol=1e-5)
        assert torch.equal(rpe.forward(ref[..., 3:9, :, :]), _unwrapped(rpe.forward)(rpe, ref[3:9]))   # a tensor (cond_chunk's row block): the stock statement
    rec = big._STATE["record"]
    assert "relp_lazy" in rec.units[big.PRE_ITEM_UNIT].ran and big.unit_calls(big.PRE_ITEM_UNIT, "relp_lazy") == 1
    assert any(r["seam"] == "relp" for r in big._STATE["peaks"])
    assert not any(torch.is_tensor(getattr(lazy, s, None)) and getattr(lazy, s).dim() == 3 for s in big.LazyRelp.__slots__)   # no plane held


def test_msa_zfree_releases_the_block_input_and_refuses_a_foreign_owner(monkeypatch):
    mem = _core()
    torch = pytest.importorskip("torch")
    feat, diff, tr, model, emb, pf, conf = _stub_stock()
    sys.modules[big.PAIRFORMER] = pf
    marks = big.activate("fast", os.environ)
    assert "MEM:msa_zfree(applied)" in marks
    blk, last = pf.MSABlock(), pf.MSABlock(is_last_block=True)
    m = torch.randn(2, 6, 3); z = torch.randn(6, 6, 3); z_ref = z.clone()
    with torch.no_grad():
        m1, z1 = blk.forward(m, z, None)
        assert z.untyped_storage().nbytes() == 0                                   # the block-input pair tensor released
        m0, z0 = _unwrapped(pf.MSABlock.forward)(pf.MSABlock(), m, z_ref, None)
        assert torch.equal(m1, m0) and torch.equal(z1, z0)
        n1, z2 = last.forward(m, z1, None)
        assert n1 is None and torch.equal(z2, _unwrapped(pf.MSABlock.forward)(pf.MSABlock(is_last_block=True), m, z0, None)[1])
    z3 = torch.randn(6, 6, 3, requires_grad=True)
    blk.forward(m, z3, None)                                                       # autograd on (a training-form call): nothing released, a named skip
    assert z3.untyped_storage().nbytes() > 0
    rec = big._STATE["record"]
    assert "msa_zfree" in rec.units[big.PRE_ITEM_UNIT].ran                       # (the autograd call's named skip is not a second census state: the unit already ran the lever)
    # a foreign owner of MSABlock.forward at apply time is refused by name (the lever re-states the stock body: it never drops another patch silently)
    feat2, diff2, tr2, model2, emb2, pf2, conf2 = _stub_stock()

    def foreign(self, *a, **k):
        return None
    pf2.MSABlock.forward = foreign
    with pytest.raises(mem.RefusalError) as ei:
        big._msa_zfree_apply_module(pf2)
    assert ei.value.refusal.precondition == "site_owned" and "foreign" in ei.value.refusal.reason


def test_diffcache_free_releases_the_pair_cache_at_the_confidence_head_and_wraps_cond_chunk(monkeypatch):
    mem = _core()
    torch = pytest.importorskip("torch")
    feat, diff, tr, model, emb, pf, conf = _stub_stock()
    sys.modules[big.DIFFUSION] = diff; sys.modules[big.CONFIDENCE] = conf
    big.SETTINGS["cond_chunk"].update(rows=2, above_tok=1)
    marks = big.activate("fast", os.environ)
    assert "MEM:diffcache_free(applied)" in marks and "MEM:cond_chunk(applied)" in marks
    dc = diff.DiffusionConditioning(); head = conf.ConfidenceHead()
    assert getattr(diff.DiffusionConditioning.prepare_cache, "_big_diffcache", False) and _unwrapped(dc.prepare_cache) is not dc.prepare_cache
    z = torch.randn(5, 5, 3); relp = torch.randn(5, 5, 5)
    with torch.no_grad():
        out = dc.prepare_cache(relp, z)                                             # cond_chunk's row-blocked body ran UNDER the wrapper (called, not replaced)
        assert torch.allclose(out, _unwrapped(dc.prepare_cache)(dc, relp, z), atol=1e-6) and big.unit_calls(big.PRE_ITEM_UNIT, "cond_chunk") == 1
        keep = out.clone()
        assert head.forward(torch.zeros(2)).shape == (2,)                           # confidence entry: the cache's storage released, the head's own statement untouched
        assert out.untyped_storage().nbytes() == 0 and keep.untyped_storage().nbytes() > 0
        head.forward(torch.zeros(2))                                                # a second entry with no live cache: a named skip, no error
    rec = big._STATE["record"]
    assert "diffcache_free" in rec.units[big.PRE_ITEM_UNIT].ran


def test_diffcache_free_stays_outermost_whatever_the_apply_order(monkeypatch):
    """cond_chunk applied after diffcache_free (the reverse of the line order) still leaves diffcache_free's wrapper outermost: the weak
    reference is the assembled pair cache, never a row block of it."""
    mem = _core()
    torch = pytest.importorskip("torch")
    feat, diff, tr, model, emb, pf, conf = _stub_stock()
    sys.modules[big.CONFIDENCE] = conf
    big.SETTINGS["cond_chunk"].update(rows=2, above_tok=1)
    big.activate("fast", os.environ)                                            # DIFFUSION not imported: both levers armed
    big._diffcache_free_apply_diffusion(diff)                                   # the reverse order by hand: diffcache first,
    big._cond_chunk_apply_module(diff, 2, 1)                                    # cond_chunk second
    assert getattr(diff.DiffusionConditioning.prepare_cache, "_big_diffcache", False) and getattr(diff.DiffusionConditioning.prepare_cache.__wrapped__, "_big", False)
    dc = diff.DiffusionConditioning()
    with torch.no_grad():
        out = dc.prepare_cache(torch.randn(5, 5, 5), torch.randn(5, 5, 3))
        assert big._STATE["diffcache"]() is out and big.unit_calls(big.PRE_ITEM_UNIT, "cond_chunk") == 1
        conf.ConfidenceHead().forward(torch.zeros(1))
        assert out.untyped_storage().nbytes() == 0


def test_apb_bias_chunked_serves_an_instance_binding_bitwise_ln_and_counts_the_unit(monkeypatch):
    """protenix_opt 0.3.55: a binding that owns AttentionPairBias.standard_multihead_attention on an INSTANCE (apb_core's pf_attn; the DiT hoist /
    fused DiT producers) asks the memory line for the chunked pair-bias statement: above the gate it gets linear_nobias_z(layernorm_z(z)) computed on
    `rows` token rows at a time, permuted to [.., H, N, N] -- the row-blocked LayerNorm equals the full-N one bitwise on CPU fp32 -- and the unit counts
    under units_ran / calls exactly as the class-level re-statement; below the gate / outside the line it gets None."""
    _core(); torch = pytest.importorskip("torch")
    assert big.apb_bias_chunked(object(), torch.zeros(3, 3, 4)) is None                        # the lever not applied in this process: None (the binding's own producer)
    feat, diff, tr, model, emb, pf, conf = _stub_stock()
    sys.modules[big.TRANSFORMER] = tr
    big.SETTINGS["apb_bias_chunk"].update(rows=256, above_tok=1023)                            # the shipped words: 256 rows, engaged above 1023 tokens
    marks = big.activate("big", os.environ)
    assert "MEM:apb_bias_chunk(applied)" in marks and big._STATE["apb_policy"] is not None
    m = tr.AttentionPairBias()
    z_small = torch.randn(1, 40, 40, 4)
    assert big.apb_bias_chunked(m, z_small) is None                                            # 40 tokens: below the gate, named once, the binding produces the bias itself
    N = 1200
    z = torch.randn(1, N, N, 4)
    out = big.apb_bias_chunked(m, z)                                                           # 1200 tokens: engaged, 256-row blocks
    ln_full = m.layernorm_z(z)
    ln_rows = torch.cat([m.layernorm_z(z[:, i:i + 256]) for i in range(0, N, 256)], dim=1)
    assert torch.equal(ln_rows, ln_full)                                                         # the row-blocked LayerNorm IS the full-N LayerNorm, bitwise (per-row statistics)
    ref = tr.permute_final_dims(m.linear_nobias_z(ln_full), [2, 0, 1])
    assert out.shape == ref.shape == (1, m.linear_nobias_z.out_features, N, N) and torch.allclose(out, ref, atol=1e-5, rtol=0)
    rec = big._STATE["record"]
    unit = rec.units[big.PRE_ITEM_UNIT]
    assert unit.ran == ["apb_bias_chunk"]
    # the DiT blocks' own modules, wrapped from the outside (the hoist's / fused stack's producers evaluate linear_nobias_z(layernorm_z(z)) themselves)
    m2 = tr.AttentionPairBias()
    assert big._wrap_pair_bias_modules(m2) is True and big._wrap_pair_bias_modules(m2) is False       # idempotent
    small = m2.linear_nobias_z(m2.layernorm_z(z_small))                                          # below the gate: the modules' own forwards (a tensor, not the carrier)
    assert torch.is_tensor(small) and torch.equal(small, m2.linear_nobias_z.__call__.__self__.forward.__wrapped__(m2.layernorm_z.forward.__wrapped__(z_small)))
    full = tr.permute_final_dims(m2.linear_nobias_z(m2.layernorm_z(z)), [2, 0, 1])                # the producers' statement at 1200 tokens: row-blocked underneath
    ref2 = tr.permute_final_dims(m2.linear_nobias_z.forward.__wrapped__(m2.layernorm_z.forward.__wrapped__(z)), [2, 0, 1])
    assert full.shape == ref2.shape and torch.allclose(full, ref2, atol=1e-5, rtol=0)
    # the runner-seam installer finds the DiffusionModule's 24 token blocks and names the result
    diffmod = types.ModuleType("protenix.model.modules.diffusion")
    class DiffusionModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            for _ in range(24):
                b = torch.nn.Module(); b.attention_pair_bias = torch.nn.Module()
                b.attention_pair_bias.layernorm_z = torch.nn.LayerNorm(4); b.attention_pair_bias.linear_nobias_z = torch.nn.Linear(4, 3, bias=False)
                blocks.append(b)
            self.diffusion_transformer = torch.nn.Module(); self.diffusion_transformer.blocks = torch.nn.ModuleList(blocks)
    diffmod.DiffusionModule = DiffusionModule
    mods_pkg = sys.modules.get("protenix.model.modules") or types.ModuleType("protenix.model.modules")
    monkeypatch.setitem(sys.modules, "protenix.model.modules", mods_pkg); monkeypatch.setitem(sys.modules, "protenix.model.modules.diffusion", diffmod)
    mods_pkg.diffusion = diffmod
    top = torch.nn.Module(); top.diffusion_module = DiffusionModule()
    err = io.StringIO()
    with redirect_stderr(err):
        big._install_dit_pair_bias_rows(types.SimpleNamespace(model=top))
    assert "[protenix-opt] MEM:apb_bias_chunk(dit_sites=24 dit_stacks_per_block=0; " in err.getvalue(), err.getvalue()
    blk0 = top.diffusion_module.diffusion_transformer.blocks[0].attention_pair_bias
    assert isinstance(blk0.layernorm_z(z), big._ZRows) and torch.is_tensor(blk0.layernorm_z(z_small))
    from protenix_opt import runner_seam
    assert "apb_bias_chunk" in runner_seam.names()                                               # registered on the seam when the lever applied
    pairs = dict(big.evidence("apb_bias_chunk", big.state()))                                # the exit record (closes the unit): the LEVER counters
    assert pairs["units_ran"] >= 1 and pairs["calls"] >= 2 and str(pairs["rows"]) == "256" and str(pairs["above_tok"]) == "1023", pairs


def test_dit_bias_per_block_produces_the_fused_stacks_block_biases_one_at_a_time(monkeypatch):
    """protenix_opt 0.3.56 (big): the fused DiT stack asks for its 24 block biases before the block loop; bound through big the list holds
    handles and each block's planes are produced right before that block's attention by the stack's OWN producer (same tensor), then let go --
    one [16, N, N] fp32 plane set resident at a time instead of 24 (12.9 GiB at 3000 tokens on the eager / stock sampler routes)."""
    torch = pytest.importorskip("torch")
    monkeypatch.delitem(sys.modules, "protenix.model.modules.diffusion", raising=False)
    received = []

    class _Apb:                                   # the stack's attention module (fpf_apb): dit_apb(qkvg, bias, S, N, ...)
        VERSION = "x"
        def dit_apb(self, qkvg, bias, S, N, **k):
            received.append(bias); return qkvg

    class _Stack:                                 # the fused stack's shape: _bias(i, z, eff) per block, all 24 asked before the loop
        nb = 24
        def __init__(self):
            self._apb = _Apb(); self.produced = []
        def _bias(self, i, z, eff):
            t = torch.full((1, 16, 4, 4), float(i)); self.produced.append(i); return t
        def forward(self, z):
            biases = [self._bias(i, z, False) for i in range(self.nb)]
            live = sum(torch.is_tensor(b) for b in biases)
            for i in range(self.nb):
                self._apb.dit_apb(torch.zeros(2), biases[i], 5, 4, gate=True)
            return live, biases

    st = _Stack()
    model = types.SimpleNamespace(diffusion_module=types.SimpleNamespace(diffusion_transformer=types.SimpleNamespace(_protenix_fpf_ditfast=st)))
    live0, _ = _Stack().forward(None)
    assert live0 == 24 and len(received) == 24                                                    # the stack as shipped: 24 plane sets resident across the loop
    received.clear()
    assert big.dit_bias_per_block(model) == 1 and big.dit_bias_per_block(model) == 0          # bound once (idempotent)
    assert isinstance(st._apb, big._ApbPerBlock) and st._apb.VERSION == "x"                     # everything but dit_apb is the module itself
    live, biases = st.forward(None)
    assert live == 0 and all(isinstance(b, big._LazyBias) for b in biases)                      # handles, not planes, across the loop
    assert [float(b.flatten()[0]) for b in received] == [float(i) for i in range(24)] and st.produced == list(range(24))   # each block got ITS planes from the stack's own producer, in order
    assert big._STATE["dit_bias_per_block"] == 24 and big._STATE["dit_stacks_per_block"] == 1
    assert (big._STATE["dit_bias_per_block"], big._STATE["dit_stacks_per_block"]) == (24, 1)   # the LEVER line's dit_bias_per_block / dit_stacks_per_block counters read these


def test_znorm_permuted_rows_is_the_stock_normalize_permute_contiguous_bitwise_in_one_tensor(monkeypatch):
    """protenix_opt 0.3.57 (big): the sampler's per-call normalize(z) + permute(.., [2,0,1]).contiguous() produced as ONE channel-major tensor, the
    LayerNorm row-blocked straight into it -- bitwise the stock two-step result (LayerNorm statistics are per (i, j)); None below the gate / outside
    the line; and the composite pair-bias chunk hands chunk_rows a [rows, N, n_heads] slab per block (never a channel-wide intermediate)."""
    _core(); torch = pytest.importorskip("torch")
    ln = torch.nn.LayerNorm(8)
    with torch.no_grad():
        ln.weight.copy_(torch.randn(8)); ln.bias.copy_(torch.randn(8))
    z = torch.randn(1, 1100, 6, 8)
    assert big.znorm_permuted_rows(ln, z) is None                                               # the line not active: None (the stock statement)
    feat, diff, tr, model, emb, pf, conf = _stub_stock()
    sys.modules[big.TRANSFORMER] = tr
    big.SETTINGS["apb_bias_chunk"].update(rows=256, above_tok=1023)
    assert "MEM:apb_bias_chunk(applied)" in big.activate("big", os.environ)
    assert big.znorm_permuted_rows(ln, torch.randn(1, 40, 6, 8)) is None                        # below the gate
    out = big.znorm_permuted_rows(ln, z)
    ref = ln(z).permute(0, 3, 1, 2).contiguous()                                                  # stock: normalize, then permute_final_dims([2,0,1]).contiguous()
    assert out.shape == (1, 8, 1100, 6) and out.is_contiguous() and torch.equal(out, ref)         # ONE channel-major tensor, bitwise the stock result
    assert big._STATE["znorm_rows_calls"] == 1
    # the composite pair-bias chunk: every block handed to chunk_rows is linear_nobias_z(layernorm_z(z_blk)) -> [.., rows, N, n_heads]
    m = tr.AttentionPairBias(); shapes = []
    lin = m.linear_nobias_z
    m.linear_nobias_z = lambda x, _l=lin: (shapes.append(tuple(x.shape)), _l(x))[1]
    zb = torch.randn(1, 1100, 5, 4)
    b = big.apb_bias_chunked(m, zb)
    H = lin.out_features
    assert b.shape == (1, H, 1100, 5) and shapes and all(sh[-1] == 4 and sh[-3] <= 256 for sh in shapes) and len(shapes) == -(-1100 // 256)   # 5 row blocks of <= 256 rows, 4 channels in, H out
    src = open(os.path.join(os.path.dirname(big.__file__), "..", "forward", "flashpairformer", "src", "dit_hoist.py")).read()
    assert "_big_znorm(_o, z)" in src and 'getattr(B, "znorm_permuted_rows", None)' in src          # the hoist's normalize producer asks the line by name
