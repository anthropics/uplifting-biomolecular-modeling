"""kernels/fpf_trimul (the TM-K3 line) cell verdicts under P12: VERIFIED classes serve exactly as recorded; the MEASURED_OFF classes (bf16 C=64, fp32
C=64 outside exact, fp32 C=128 / C=256) stay on the stock TriMul BY NAME; any other bf16 / fp32 class is UNKNOWN and served on the default tiles,
named; a dtype the kernels do not compute is refused by name.  CPU only (the package's tables; torch + triton importable)."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")


@pytest.fixture(scope="module")
def Tm():
    os.environ.setdefault("FPF_TRIMUL_MODE", "exact")
    from opt_core import kernels as CK
    core_copy = os.path.join(os.path.dirname(CK.__file__), "fpf_trimul")
    top = sys.modules.get("fpf_trimul")
    if top is not None and not os.path.abspath(getattr(top, "__file__", "") or "").startswith(os.path.abspath(core_copy)):
        for m in [m for m in sys.modules if m == "fpf_trimul" or m.startswith("fpf_trimul.")]:
            del sys.modules[m]
    CK.route("fpf_trimul")
    import fpf_trimul.trimul as Tm
    assert os.path.abspath(Tm.__file__).startswith(os.path.abspath(core_copy)), Tm.__file__
    return Tm


def test_verified_classes_serve_as_recorded(Tm):
    for cls, C in list(Tm.VERIFIED) + (list(Tm.VERIFIED_EXACT) if Tm._MODE == "exact" else []):
        dt = torch.bfloat16 if cls == "bf16" else torch.float32
        kind, rec = Tm.cell_verdict(dt, C)
        assert kind == "verified" and rec and Tm.is_verified(dt, C), (cls, C)
    assert {("bf16", 256), ("bf16", 128), ("fp32", 384)} <= set(Tm.VERIFIED)


def test_measured_off_classes_keep_the_stock_trimul_by_name(Tm):
    for (cls, C), why in Tm.MEASURED_OFF.items():
        dt = torch.bfloat16 if cls == "bf16" else torch.float32
        kind, word = Tm.cell_verdict(dt, C)
        if Tm._MODE == "exact" and (cls, C) in Tm.VERIFIED_EXACT:
            assert kind == "verified", (cls, C)                                        # fp32 C=64: certified through the exact per-call proof, served in exact
            continue
        assert kind == "off" and word == f"cell:{cls}_C{C}+off(not-measured)" and not Tm.is_verified(dt, C) and why, (cls, C)
    assert {("bf16", 64), ("fp32", 128), ("fp32", 256)} <= set(Tm.MEASURED_OFF)


def test_unknown_classes_are_served_on_the_default_tiles_named(Tm, capfd):
    assert Tm.cell_verdict(torch.bfloat16, 512) == ("unverified", "unverified(bf16_C512)")
    assert Tm.cell_verdict(torch.float32, 512) == ("unverified", "unverified(fp32_C512)")
    cfg = Tm._select_cfg(torch.bfloat16, 705, 512)                                          # the default tiles serve the class (TILES_DEFAULT): settings exist to launch
    assert cfg["A"]["BM"] > 0 and cfg["C"]["BM"] > 0
    Tm.STATS["unverified"].clear()
    Tm._note_unverified("unverified(bf16_C512)"); Tm._note_unverified("unverified(bf16_C512)")
    err = capfd.readouterr().err
    assert err.count("[fpf_trimul] cells=unverified(bf16_C512)") == 1 and Tm.STATS["unverified"] == {"unverified(bf16_C512)": 2}


def test_dtypes_and_widths_the_kernels_do_not_compute_are_refused_by_name(Tm):
    assert Tm.cell_verdict(torch.float16, 128) == ("unsupported", "dtype:float16")
    assert Tm.cell_verdict(torch.bfloat16, 96) == ("unsupported", "c:96_not_multiple_of_64")
    before = dict(Tm.STATS["fallback_by"]); n = Tm.STATS["fallback"]
    Tm._count_stock("dtype:float16")
    assert Tm.STATS["fallback"] == n + 1 and Tm.STATS["fallback_by"]["dtype:float16"] == before.get("dtype:float16", 0) + 1


def test_arch_table_state_is_named(Tm):
    assert Tm.ARCH_TABLE_USED is None or Tm.ARCH_TABLE_USED.startswith(("sm_", "error:"))     # CPU: no device table consulted (None); an unreadable table says error:<why> (and ONE stderr line at import)
