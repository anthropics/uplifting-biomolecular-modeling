"""boltz2_opt.exactln — the bitwise ATen layer_norm replica (registry `exactln`): the row's word and refusals, the registry / mode-table / attach
wiring, the report shape the evidence rule reads, the carried package's layout plan (which leading layouts are read strided, which are copied,
which ATen dispatches to its non-vectorized kernels and are refused by name).  The CPU tests need no GPU; the bitwise identity itself is a GPU
fact (``test_bitwise_vs_torch``, requires_gpu) — the track's dev tool runs the full matrix."""
import importlib.util
import os
import sys

import pytest

from .. import exactln as XL, modes, registry, stack
from ..worker_launch import ATTACH
from .conftest import requires_gpu


def _pkg():
    """The carried exactln package (the shared core's kernels.ln row; the kit copy forward/exactln left the tree at 0.3.16), imported the adapter's way."""
    return XL._load_package()


def test_switch_word_and_variants():
    assert XL.SWITCH == "BOLTZ_EXACTLN" and XL.VARIANTS == ("core",) and XL.RETIRED_VARIANTS == ("on",) and XL.LEVERS == ("exactln",)
    assert XL.variant({"BOLTZ_EXACTLN": "core"}) == "core" and XL.CORE_ROW == "exactln"          # `core` = the same package as opt_core's kernels.ln row carries it
    assert XL._provider_word("core").startswith("core")
    assert XL.variant({}) is None and not XL.requested({})
    assert XL.variant({"BOLTZ_EXACTLN": " CORE "}) == "core"
    assert XL.variant({"BOLTZ_EXACTLN": "1"}) is None and XL.variant({"BOLTZ_EXACTLN": "off"}) is None   # only the tabled word engages it
    assert XL.apply(spec="") == [] and XL.report()["applied"] == []                                     # switch absent: nothing replaced, nothing reported applied
    assert XL.variant({"BOLTZ_EXACTLN": "on"}) is None                                                  # the retired kit copy's word is no variant …
    with pytest.raises(RuntimeError, match="BOLTZ_EXACTLN=on refused by name — exactln:on:kit_copy_retired"):
        XL.apply(spec="on")                                                                              # … and refuses BY NAME at apply, before anything is replaced (worker_launch: REFUSED, exit 3)
    assert XL.report()["applied"] == []


def test_registry_modes_attach_wiring():
    L = registry.LEVERS["exactln"]
    assert L["switch"] == XL.SWITCH and L["value"] == "core" and L["tier"] == 1 and L["file"].endswith("boltz2_opt/exactln.py")
    assert (L["origin"], L["impl"]) == ("core", "opt_core.kernels.ln:exactln/exactln_fwd.cu") == (registry.LEVERS["exactln_resid"]["origin"], registry.LEVERS["exactln_resid"]["impl"])
    assert ATTACH["exactln"] == {"module": "boltz2_opt.exactln", "trigger": "boltz.model.models.boltz2", "report_key": "exactln_report"}
    for m in ("exact", "fast", "big"):                                                                   # big = fast minus the levers with a measured memory cost: the replica has none, it rides
        row = modes.MODES[m]
        assert "exactln" in row["levers"] and "exactln" in row["attach"] and row["env"].get("BOLTZ_EXACTLN") == "core", m
    assert registry.tier_of(modes.MODES["exact"]["levers"]) == 1                                          # an exact-class lever: the exact row stays tier 1


def test_no_card_drop():
    """8.0 (A100) is a proven card: no card_off entry; a card the core LayerNorm provider does not admit steps aside at apply BY NAME with the face's word (dispositions) [v1.3.3, 0.3.30]."""
    modes.set_card("8.0")
    try:
        row = modes.resolve("exact")
        assert "exactln" in row["levers"] and row["env"].get("BOLTZ_EXACTLN") == "core"
        assert "exactln" not in (row.get("card_off") or {})
    finally:
        modes.set_card(None)
    assert not hasattr(XL, "PROVEN_CC") and not hasattr(XL, "TORCH_PIN"), "no architecture table or torch pin in the adapter: the core LayerNorm provider's face answers (face_word)"
    from opt_core.kernels import ln as LN
    assert set(LN.EXACTLN_PROVEN_CC) >= {"9.0", "8.0"}
    assert XL.face_word((9, 0), "2.12.0+cu130") is None and XL.face_word((8, 0), "2.12.0+cu130") is None, "proven cards on the pinned torch: admitted"
    w = XL.face_word((10, 0), "2.12.0+cu130")
    assert w and w.startswith("cc:10.0_not_proven") and " " not in w, ("an unproven card steps aside BY NAME with the face's word", w)
    w = XL.face_word((9, 0), "2.13.0+cu130")
    assert w == "torch_version:2.13.0+cu130!=2.12.0", ("another torch than the carried package's pin: aside by name", w)
    assert XL.dispositions() == {}

def test_pins_strip_the_word_from_stock_processes():
    pins = stack.load_pins()
    prefixes = pins["stock_environment"]["must_be_absent_prefixes"]
    assert any(XL.SWITCH.startswith(p) for p in prefixes), prefixes


def test_report_shape_is_the_evidence_rules():
    """stack.gate's adapter loop reads applied / variant / census.served / gate.{ok,idle,reason}; report.lever_lines reads census.fallback/errors."""
    r = XL.report()
    assert set(r) >= {"applied", "line", "census", "gate", "variant"}
    # a stand-in applied state (no GPU): the shape with counts
    saved = dict(XL._STATE)
    try:
        XL._STATE.update(applied=["exactln"], variant="core", served={"C128:float32:contig": 5, "C128:float32:strided": 2}, fallback={"below_min_numel": 7}, errors={}, calls=14)
        r = XL.report()
        assert r["applied"] == ["exactln"] and r["variant"] == "core" and r["impl"] == "opt_core.kernels.ln:exactln/exactln_fwd.cu"
        assert r["census"]["served_total"] == 7 and r["census"]["fallback"] == {"below_min_numel": 7} and r["census"]["errors"] == {}
        assert r["gate"] == {"ok": True, "idle": False, "reason": None}
        assert r["line"].startswith("LEVER name=exactln state=on variant=core served=7 fallback=7 errors=0")
        XL._STATE["fallback"]["mystery_word"] = 1                                   # an undeclared census word refuses the gate
        assert XL.verdict()["ok"] is False and "undeclared_fallback:mystery_word" in XL.verdict()["reason"]
        XL._STATE["fallback"].pop("mystery_word"); XL._STATE["fallback"]["width:100"] = 3   # plan()'s parametrised words are declared by prefix? no: width is NOT expected -> refuses
        assert XL.verdict()["ok"] is False
        XL._STATE["fallback"].pop("width:100"); XL._STATE["errors"]["RuntimeError"] = 1
        assert XL.verdict()["ok"] is False and XL.verdict()["reason"].startswith("errors:")
        XL._STATE["errors"] = {}; XL._STATE["served"] = {}
        assert XL.verdict() == {"ok": True, "idle": True, "reason": XL.verdict()["reason"]} and XL.verdict()["idle"]
    finally:
        XL._STATE.clear(); XL._STATE.update(saved)


def test_package_files_and_pin():
    E = _pkg()
    assert os.path.isfile(E.SOURCE_FILE) and E.TORCH_PIN == "2.12.0"
    src = open(E.SOURCE_FILE).read()
    for token in ("__frcp_rn", "__fdiv_rn", "rsqrtf", "__fmaf_rn(delta, coef, w.mean)", "__shfl_down_sync", "combine(R[0], R[2])"):
        assert token in src, token                                                  # the recovered arithmetic's landmarks are in the carried source


def test_collapse_leading_layouts():
    """Which leading layouts the kernel reads in place (rows = an (outer, inner) stride pair) and which take a contiguous copy first."""
    E = _pkg()
    C = 128; N = 7
    f = E._collapse_leading
    assert f((N, N), (N * C, C)) == (N * N, N * N, 0, C)                            # contiguous [N,N,C]: one group
    assert f((1, N, N), (N * N * C, N * C, C)) == (N * N, N * N, 0, C)             # size-1 batch dropped
    assert f((N, N), (C, N * C)) == (N * N, N, C, N * C)                            # z.transpose(-2,-3): rows gathered with stride N*C, outer stride C
    assert f((N, 3), (N * C * 2, C)) == (N * 3, 3, N * C * 2, C)                    # a sliced view: two groups
    assert f((3, N, 5), (5 * C, 3 * 5 * C * 2, C)) is None                          # three stride groups: copy path
    assert f((), ()) == (1, 1, 0, 0)


def test_plan_refusals_cpu():
    E = _pkg()
    import torch
    x = torch.zeros(4, 128)
    with pytest.raises(E.Unsupported) as e:
        E.plan(x, (128,), None, None)
    assert e.value.reason == "not_cuda"


@requires_gpu
def test_plan_words_gpu():
    E = _pkg()
    import torch
    dev = "cuda"
    z = torch.randn(1, 9, 9, 128, device=dev)
    p = E.plan(z, (128,), None, None); assert p["copy"] is False and p["stride_outer"] == 0 and p["affine"] == 0
    p = E.plan(z.transpose(1, 2), (128,), None, None); assert p["copy"] is False and p["stride_outer"] == 128 and p["stride_inner"] == 9 * 128
    w = torch.ones(128, device=dev); b = torch.zeros(128, device=dev)
    assert E.plan(z, (128,), w, b)["affine"] == 3 and E.plan(z, (128,), w, None)["affine"] == 1
    for bad, word in (((64,), "normalized_shape"), ((9, 128), "normalized_shape")):
        with pytest.raises(E.Unsupported) as e:
            E.plan(z, bad, None, None)
        assert e.value.reason == word
    with pytest.raises(E.Unsupported) as e:
        E.plan(torch.randn(4, 6, device=dev), (6,), None, None)
    assert e.value.reason == "width:6"
    with pytest.raises(E.Unsupported) as e:
        E.plan(z.double(), (128,), None, None)
    assert e.value.reason == "dtype:float64"
    flat = torch.randn(64 * 128 + 8, device=dev); xc = flat[2:2 + 64 * 128].view(64, 128)
    with pytest.raises(E.Unsupported) as e:
        E.plan(xc, (128,), None, None)
    assert e.value.reason == "aten_rowwise_path"                                    # contiguous but not 16-byte aligned: ATen's non-vectorized kernels, another arithmetic
    with pytest.raises(E.Unsupported) as e:
        E.plan(z, (128,), w.to(torch.bfloat16), b)
    assert e.value.reason == "weight_form"
    p = E.plan(z.to(torch.bfloat16), (128,), w, b, widen=True); assert p["tin"] == "bfloat16" and p["tpar"] == "float32" and p["tout"] == "float32"


@requires_gpu
def test_bitwise_vs_torch():
    """The identity itself on the local GPU: NaN-strict bit equality with torch.nn.functional.layer_norm."""
    E = _pkg()
    import torch
    import torch.nn.functional as F
    if not torch.__version__.startswith(E.TORCH_PIN):
        pytest.skip(f"the replica is torch {E.TORCH_PIN}'s arithmetic; this is {torch.__version__}")
    g = torch.Generator(device="cuda").manual_seed(0)
    for C in (64, 128, 384):
        w = torch.randn(C, device="cuda", generator=g); b = torch.randn(C, device="cuda", generator=g)
        z = torch.randn(1, 48, 48, C, device="cuda", generator=g) * 3 + 1
        for x in (z, z.transpose(1, 2), z[:, ::2]):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ref = F.layer_norm(x, (C,), w, b, 1e-5)
            y = E.layer_norm(x, (C,), w, b, 1e-5)
            assert y.dtype == ref.dtype and y.shape == ref.shape and y.is_contiguous()
            assert torch.equal(y.view(torch.int32), ref.contiguous().view(torch.int32)), (C, x.stride())
            yb = E.layer_norm(x, (C,), w, b, 1e-5, out_dtype=torch.bfloat16)
            assert torch.equal(yb.view(torch.int16), ref.to(torch.bfloat16).contiguous().view(torch.int16))
