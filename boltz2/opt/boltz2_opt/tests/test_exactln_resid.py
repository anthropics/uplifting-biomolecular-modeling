"""Lever exactln_resid [0.3.10]: the pair residual add fused with the NEXT block's LayerNorm (the carried exactln package's exactln_resid_fwd, opt_core kernels.ln) behind the trunk
levers' RESID_FUSER hook (opt/forward/trunk_levers/boltz_trunk_levers.py `_fused`). CPU tests of the wiring: the hook's operand rule and call
form, the sites it offers, the adapter's per-site park classes, the row word (exact only: fast / big keep z bf16-resident under PAIRFUSE) and
the registry / attach / needs entries. The kernel's bits are proven on the card at run time (first call per class bit-compared; the GPU-host
acceptance of CHANGES 0.3.10 is the byte record)."""
import importlib.util
import os
import sys
import types

import pytest

from .. import exactln as XL
from .. import modes, registry

HERE = os.path.dirname(os.path.abspath(__file__))
LEV_PATH = os.path.join(HERE, "..", "..", "forward", "trunk_levers", "boltz_trunk_levers.py")


def _load_trunk_levers():
    torch = pytest.importorskip("torch")
    if "opt_core.oom" not in sys.modules:                       # the module imports the core's OOM predicate; a stub keeps this a wiring test
        try:
            import opt_core.oom  # noqa: F401
        except Exception:  # noqa: BLE001
            oc = sys.modules.setdefault("opt_core", types.ModuleType("opt_core")); oom = types.ModuleType("opt_core.oom"); oom.is_oom = lambda e: False
            sys.modules["opt_core.oom"] = oom; setattr(oc, "oom", oom)
    spec = importlib.util.spec_from_file_location("boltz_trunk_levers_under_test", LEV_PATH)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return torch, mod


class _T:                                                     # a tensor stand-in: the hook decides on dtype / shape / device only
    def __init__(self, dtype, shape=(1, 8, 8, 128), cuda=True):
        self.dtype, self.shape, self.is_cuda = dtype, tuple(shape), cuda


def test_hook_offers_the_fp32_promote_add_only_and_calls_the_fuser_in_its_form():
    torch, L = _load_trunk_levers()
    assert L.RESID_FUSER is None, "no fuser registered = the lever absent: every residual statement runs as written"
    ln = torch.nn.LayerNorm(128)
    z, u = _T(torch.float32), _T(torch.bfloat16)
    assert L._fused(z, u, ln, "transition_z") is None and L.STATS.get("resid_fused", 0) == 0 == L.STATS.get("resid_unfused", 0)
    calls = []

    def fuser(z_, u_, next_ln, transpose_ln, ln_autocast, site):
        calls.append((z_, u_, next_ln, transpose_ln, ln_autocast, site)); return "SUM" if site != "refuse" else None
    L.RESID_FUSER = fuser
    try:
        assert L._fused(z, u, ln, "tri_att_end", transpose_ln=True) == "SUM" and calls[-1] == (z, u, ln, True, True, "tri_att_end")
        assert L._fused(z, _T(torch.float32), ln, "proj_z", ln_autocast=False) == "SUM" and calls[-1][3:] == (False, False, "proj_z"), "an fp32 update is the same one-add statement"
        assert L._fused(z, u, ln, "refuse") is None and L.STATS["resid_unfused"] == 1 and L.STATS["resid_fused"] == 2, "the fuser's None hands the statement back, counted"
        n = len(calls)
        for args in ((z, u, None, "s"), (_T(torch.bfloat16), u, ln, "s"), (z, _T(torch.float16), ln, "s"), (z, _T(torch.bfloat16, (1, 8, 8, 64)), ln, "s"), (_T(torch.float32, cuda=False), u, ln, "s")):
            assert L._fused(*args) is None
        assert len(calls) == n, "no consumer LayerNorm / a bf16 z / an fp16 or other-shape update / a CPU z never reach the fuser: those statements are not the fp32 promote-add of one shape"
        assert L._resid_ln(z, u, ln, "tri_att_start") == "SUM" and calls[-1][2:] == (ln, False, True, "tri_att_start")
    finally:
        L.RESID_FUSER = None


def test_hook_sites_and_the_pair_bias_layernorm():
    torch, L = _load_trunk_levers()
    import ast
    src = open(LEV_PATH).read(); tree = ast.parse(src)
    body = {f.name: ast.get_source_segment(src, f) for f in tree.body if isinstance(f, ast.FunctionDef) and f.name in ("_pf_layer_forward", "_pf_noseq_layer_forward")}
    for name, b in body.items():
        assert b.count("_resid_ln(") == 3 and '"tri_att_start")' in b and '"tri_att_end", transpose_ln=True)' in b and '"transition_z")' in b, name
        assert "z = _resid(z, self.tri_mul_out(" in b, f"{name}: the TriMul-out residual is not offered (tri_mul_in normalises inside the TriMul kernels)"
    assert '_fused(z, t, _pair_bias_ln(self), "proj_z", ln_autocast=False)' in body["_pf_layer_forward"] and "_pair_bias_ln" not in body["_pf_noseq_layer_forward"], \
        "the transition's sum is offered with the sequence attention's pair-bias LayerNorm (fp32, autocast off) in PairformerLayer only"
    apb = types.SimpleNamespace(proj_z=torch.nn.Sequential(torch.nn.LayerNorm(128), torch.nn.Linear(128, 16, bias=False)))
    assert L._pair_bias_ln(types.SimpleNamespace(attention=apb)) is apb.proj_z[0]
    assert L._pair_bias_ln(types.SimpleNamespace(attention=types.SimpleNamespace(proj_z=torch.nn.Linear(128, 16)))) is None and L._pair_bias_ln(types.SimpleNamespace()) is None
    assert "resid_fuser" in L.report()


def test_adapter_learns_the_park_dtype_per_site():
    torch = pytest.importorskip("torch")
    z = torch.zeros(1, 4, 4, 128)
    k = XL._site_key(z, False, True, "tri_att_start")
    assert k == ("tri_att_start", 128, 16, False, True) and k != XL._site_key(z, False, True, "transition_z") != XL._site_key(z, True, True, "transition_z") != XL._site_key(z, True, False, "transition_z")
    import inspect
    assert list(inspect.signature(XL.resid_fuser).parameters) == ["z", "u", "next_ln", "transpose_ln", "ln_autocast", "site"], "boltz_trunk_levers._fused calls RESID_FUSER(z, u, next_ln, transpose_ln, ln_autocast, site)"
    assert XL.RESID_MIN_NUMEL == 1 << 26 and {"resid_below_min_numel", "resid_probe_unused", "resid_probe_dtype", "resid_site_off"} <= set(XL.EXPECTED) and "resid_form" not in XL.EXPECTED, \
        "the hook offers only the fusable form: an operand-form refusal would be an undeclared word (the gate refuses)"


def test_the_word_is_the_exact_rows_alone():
    assert registry.LEVERS["exactln_resid"]["switch"] == XL.SWITCH_RESID == "BOLTZ_EXACTLN_RESID" and registry.LEVERS["exactln_resid"]["value"] == "on" and registry.LEVERS["exactln_resid"]["tier"] == 1
    assert modes.LEVER_ATTACH["exactln_resid"] == "exactln" and modes.NEEDS["exactln_resid"] == ("exactln",)
    ex = modes.levers("exact")
    assert ex[ex.index("exactln") + 1] == "exactln_resid" and modes.env_row("exact")["BOLTZ_EXACTLN_RESID"] == "on"
    assert "BOLTZ_EXACTLN_RESID" in modes._EXACT_ONLY and all("exactln_resid" not in modes.levers(m) and "BOLTZ_EXACTLN_RESID" not in modes.env_row(m) for m in ("fast", "big")), \
        "fast / big keep z bf16-resident under the PAIRFUSE layer driver: no fp32 residual stream on the C=128 stacks to fuse"
    try:
        modes.set_card("10.0")
        r = modes.resolve("exact")
        assert r["card_off"].get("exactln") == "cc_unproven:sm_100" == r["card_off"].get("exactln_resid") and "BOLTZ_EXACTLN_RESID" not in r["env"], "where the replica leaves the row by name its residual pass leaves with it (NEEDS)"
        modes.set_card("8.0")
        r = modes.resolve("exact")
        assert "exactln_resid" in r["levers"] and r["env"]["BOLTZ_EXACTLN_RESID"] == "on" and "exactln_resid" not in r["card_off"], "8.0 is a proven card of the replica (exactln.PROVEN_CC): the row carries the pass there"
    finally:
        modes.set_card(None)
