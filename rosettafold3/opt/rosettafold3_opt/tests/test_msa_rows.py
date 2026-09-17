"""The MSA-module rows of the FPF add-on (``rf3fpf/fpf_rf3_msa_rows.py``: arm components ``msa[.opm|.pwa]`` (fast class) and ``smsa`` (exact
candidate under a run-time bit-compare lock), on the shared core's fused MSA-module cells ``opt_core.ops.msa_{opm,pwa,pwa2}``):
today's arms are untouched; the components parse; the gates refuse by NAME (never silently) off CUDA / outside bf16 autocast / on foreign dims;
no kit copy of the cells is carried. CPU only."""
import ast
import os
import re
import sys

import pytest

from rosettafold3_opt import modes, stack

RF3FPF = os.path.join(stack.fpf_home(), "rf3fpf")
CORE_CELLS = ("msa_opm", "msa_pwa", "msa_pwa2")                  # the shared core's fused MSA-module cells the rows bind (opt_core.ops.<name>); no kit copy is carried


def _src(name):
    return open(os.path.join(RF3FPF, name), encoding="utf-8").read()


def test_rows_module_declares_units_knobs_and_bindings():
    src = _src("fpf_rf3_msa_rows.py")
    tree = ast.parse(src)
    names = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assert {"UNITS", "MIN_I", "STATE", "EXACT", "_ORIG"} <= names
    knobs = modes.flag_table(os.path.join(RF3FPF, "fpf_rf3_msa_rows.py"))
    assert set(knobs) == {"FPF_RF3_MSA_OPM_MIN_I", "FPF_RF3_MSA_PWA_MIN_I"} and all(k.startswith(modes.FPF_ENV_PREFIXES) for k in knobs)
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {"enable", "enable_exact", "describe", "_opm_forward", "_pwa_forward", "_pwa_exact_forward", "pwa_exact_probe", "pwa_exact_candidate"} <= funcs
    for cell in CORE_CELLS:                                      # the rows import the shared core's cells by name; no kit-namespaced copy exists
        assert re.search(r"from opt_core\.ops import %s\b" % cell, src), cell
    assert not any(f.startswith("rf3msa_") for f in os.listdir(RF3FPF)), "the carried MSA cells left the tree: the core's serve"


def test_the_adapter_parses_msa_and_smsa_and_binds_the_rows():
    src = open(os.path.join(stack.fpf_home(), modes.FPF_ADAPTER_RELPATH), encoding="utf-8").read()
    assert re.search(r"head == [\"']msa[\"']", src) and re.search(r"head == [\"']smsa[\"']", src)
    assert "import fpf_rf3_msa_rows as MR" in src and "enable_msa" in src and "enable_smsa" in src
    assert all(('"%s"' % k) in src for k in ("msa", "msa_units", "msa_aside", "smsa"))      # the arm record's keys (FPF APPLIED msa=<word> msa_units=[...] msa_aside={...} smsa=<bool>)


def test_card_rows_resolve_the_bare_component_per_compute_capability(monkeypatch):
    pytest.importorskip("torch"); pytest.importorskip("triton")  # the rows module imports torch at top and the core cells import triton (CPU suffices: no kernel launches here)
    monkeypatch.syspath_prepend(os.path.join(stack.opt_root(), "..", "..", "common", "opt_core")); monkeypatch.syspath_prepend(RF3FPF)
    import fpf_rf3_msa_rows as MR
    from rosettafold3_opt import big
    assert MR.COMPONENT_WORDS == {"msa": modes.FPF_SUBWORDS["msa"]} and MR.DEFAULT_WORD["msa"] == "card" == modes.FPF_SUBWORDS["msa"][0]
    assert MR.units_for("card", (9, 0)) == (("opm", "pwa"), {"cc": "9.0", "word": "card", "row": "both", "aside": {}})           # 9.0: both cells
    u, c = MR.units_for("card", (8, 0))
    assert u == ("pwa",) and c["row"] == "pwa" and set(c["aside"]) == {"opm"} and "cc8.0" in c["aside"]["opm"]                    # 8.0: the opm cell steps aside BY NAME with its number
    assert MR.units_for("card", (12, 0))[0] == ("pwa",) == MR.units_for("card", None)[0]                                            # no row for the device: CARD_ELSE
    assert MR.units_for("both", (8, 0))[0] == ("opm", "pwa") and MR.units_for("opm", (8, 0))[0] == ("opm",) and MR.units_for("pwa", (9, 0))[0] == ("pwa",)   # a named cell is served as named
    with pytest.raises(ValueError):
        MR.units_for("fast", (9, 0))
    arm, gone = big.fpf_arm_for(modes.FPF_ARM)                 # big keeps the OPM site for opm_chunk: the component rides as msa.pwa (big.UNIT_WORD), not disengaged
    assert "msa.pwa" in arm.split("@")[0].split("+") and "msa" not in gone and big.UNIT_WORD["msa"][0] == "msa.pwa"
    step = "" if "RF3_CUDAGRAPH" in big.DISENGAGED_SWITCHES else "@L1"      # the arm's lever step leaves with the sampler graph when the memory row disengages it (big.DISENGAGED_SWITCHES)
    assert big.fpf_arm_for("fast+msa.pwa+tg@L1")[0] == "fast+msa.pwa" + step and big.fpf_arm_for("fast+msa.both@L1")[0] == "fast+msa.pwa" + step   # tg stripped, msa.* -> msa.pwa


def test_gates_refuse_by_name_on_cpu_tensors(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("opt_core.oom")
    monkeypatch.syspath_prepend(RF3FPF)
    sys.modules.pop("fpf_rf3_msa_rows", None)
    import fpf_rf3_msa_rows as MR

    class OPM(torch.nn.Module):
        def __init__(s, ch=32, cz=128):
            super().__init__(); s.norm = torch.nn.LayerNorm(64); s.proj_left = torch.nn.Linear(64, ch); s.proj_right = torch.nn.Linear(64, ch); s.proj_out = torch.nn.Linear(ch * ch, cz)

    class PWA(torch.nn.Module):
        def __init__(s, heads=8, c=32, gate=True):
            super().__init__(); s.n_heads = heads; s.weighted_average_channels = c; s.msa_channels = 64; s.separate_gate_for_every_channel = gate
            s.to_out = torch.nn.Linear(heads * c, 64, bias=False)

    msa4 = torch.zeros(1, 8, 20, 64); msa3 = torch.zeros(8, 20, 64); pair = torch.zeros(20, 20, 128)
    assert MR._opm_gate(OPM(), msa3) == "rank"
    assert MR._opm_gate(OPM(), msa4) == "device"                 # CPU tensors: the stock forward by name
    assert MR._pwa_gate(PWA(), msa3, pair) == "device"
    assert MR._pwa_gate(PWA(), msa4, pair) == "rank"
    monkeypatch.setattr(MR, "_autocast_bf16", lambda: True)
    cuda_like = type("T", (), {"dim": lambda s: 4, "is_cuda": True, "shape": (1, 8, 20, 64)})()
    assert MR._opm_gate(OPM(ch=16), cuda_like) == "dims"
    monkeypatch.setitem(MR.MIN_I, "opm", 64)
    assert MR._opm_gate(OPM(), cuda_like) == "I<64"
    cl3 = type("T", (), {"dim": lambda s: 3, "is_cuda": True, "shape": (8, 20, 64), "dtype": torch.bfloat16})()
    pr3 = type("T", (), {"dim": lambda s: 3, "is_cuda": True, "shape": (20, 20, 128), "dtype": torch.float32})()
    assert MR._pwa_gate(PWA(gate=False), cl3, pr3) == "gate_per_head"
    assert MR._pwa_gate(PWA(heads=4), cl3, pr3) == "dims"
    assert MR._pwa_gate(PWA(), cl3, pr3) is None
    assert MR._pwa_exact_gate(PWA(), cl3, pr3).startswith("floor:")  # S=8 < 16: the exact cell's size floor, by name
    d = MR.describe()
    assert d["on"] is False and set(d["counts"]) == set(MR.UNITS) and d["exact"]["on"] is False
    sys.modules.pop("fpf_rf3_msa_rows", None)
