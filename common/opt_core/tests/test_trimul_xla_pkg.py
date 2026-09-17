"""kernels/trimul_xla: the sealed trimul_native cubins for JAX / XLA programs (CPU facts: carry, recipes, names, refusals; the GPU equality record is vectors on the box)."""
import hashlib
import json
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.dirname(HERE)
import sys
sys.path.insert(0, CORE)
from opt_core.kernels import trimul_xla as TX  # noqa: E402
from opt_core.kernels import pallas as PP  # noqa: E402

PKG = TX.PKG_DIR


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_carried_files_are_the_sealed_release_bytes():
    sums = TX.sha256sums()
    assert TX.package_version() == "1.0.1"
    carried = []
    for root, _dirs, files in os.walk(PKG):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), PKG)
            if rel == "SHA256SUMS":
                continue
            assert rel in sums, "carried file %s is not in the release's SHA256SUMS" % rel
            assert _sha(os.path.join(PKG, rel)) == sums[rel], rel
            carried.append(rel)
    for unit, d in TX.UNITS.items():
        assert d["cubin"] in carried, unit
        assert d["manifest_unit"] in TX.manifest()["units"], unit
    assert len(carried) >= 10


def test_every_served_launch_names_a_kernel_the_unit_contains_and_fits_the_parameter_blocks():
    m = TX.manifest()
    for unit, d in TX.UNITS.items():
        kernels = set(m["units"][d["manifest_unit"]]["kernels"])
        c_z, c_h = d["c_z"], d["c_hidden"]
        for form in ("b", "f"):
            if form not in d:
                continue
            for n in (16, 200, 256, 400, 808, 1200, 1536, 1537, 2048, 4096):
                for masked in (False, True):
                    for transpose in (False, True):
                        k1 = TX.k1_launch(c_z, c_h, n, form == "f", masked, transpose, 1e-5)
                        assert k1["kname"] in kernels, (unit, k1["kname"])
                        assert k1["shared"] <= 232448, k1                      # the sm_90 opt-in dynamic shared-memory limit
                        _check_struct(k1["params"][0], TX.K1_PARAMS_SIZE, ntmaps=len(k1["tmaps"]))
                        assert k1["grid"][0] == (-(-k1["np"] // k1["cfg"]["k1"][0])) * (-(-k1["np"] // k1["cfg"]["k1"][1])) and k1["grid_rule"] == 1
                for residual in (False, True):
                    k3 = TX.k3_launch(c_z, c_h, n, form == "f", residual, 1e-5)
                    assert k3["kname"] in kernels, (unit, k3["kname"])
                    assert k3["shared"] <= 232448
                    _check_struct(k3["params"][0], TX.K3_PARAMS_SIZE, ntmaps=len(k3["tmaps"]))
    # the served names are exactly the release's `serves` lists for the fast class (K1 m0/m1 + K3, both forms) at the table defaults
    for unit, d in TX.UNITS.items():
        for rec in m["units"][d["manifest_unit"]]["serves"]:
            if not rec.get("fast"):
                continue
            zf32 = rec["form"] == "f"
            mine = {TX.k1_launch(d["c_z"], d["c_hidden"], 400, zf32, mk, False, 1e-5)["kname"] for mk in (False, True)} | {TX.k3_launch(d["c_z"], d["c_hidden"], 400, zf32, False, 1e-5)["kname"]}
            assert mine <= set(rec["kernels"]), (unit, rec["form"], mine - set(rec["kernels"]))


def _check_struct(tok, size, ntmaps):
    """Re-count the launcher's byte accounting of an S<size>:<fields> token (kinds b/o/n/q = 8, i/f = 4, z<k> = k, t = 128 at a 64-byte offset)."""
    m = re.fullmatch(r"S(\d+):(.*)", tok)
    assert m and int(m.group(1)) == size
    off = 0
    for f in m.group(2).split(";"):
        c = f[0]
        if c in "bonq":
            assert off % 8 == 0, (f, off); off += 8
        elif c in "if":
            assert off % 4 == 0, (f, off); off += 4
            if c == "f":
                float.fromhex(f[1:])
        elif c == "z":
            off += int(f[1:])
        elif c == "t":
            assert off % 64 == 0, (f, off)
            mm = re.fullmatch(r"t([bo])(\d+):(\d+)", f); assert mm and int(mm.group(3)) < ntmaps
            off += 128
        else:
            raise AssertionError(f)
    assert off == size, (off, size)


def test_tensor_map_geometry_restates_the_release():
    k1 = TX.k1_launch(128, 128, 400, False, True, False, 1e-5)
    assert k1["tmaps"][0] == "dt=bf16,dims=128/400/400,str=256/102400,box=64/64/2,swz=128,l2=128,oob=0"        # z: box [128/esz, BJ, BI] over dims [c_z, N, N]
    assert k1["tmaps"][1] == "dt=bf16,dims=128/512,str=256,box=64/64,swz=128,l2=256,oob=0"                      # w1 [4 c_h, c_z]
    k1t = TX.k1_launch(128, 128, 400, False, True, True, 1e-5)
    assert k1t["tmaps"][0] == "dt=bf16,dims=128/400/400,str=102400/256,box=64/64/2,swz=128,l2=128,oob=0"        # incoming: token axes swapped
    assert "i1;i400;" in k1t["params"][0] and "i400;i1;" in k1["params"][0]                                       # mask strides (ms_i, ms_j)
    k3 = TX.k3_launch(128, 128, 400, False, False, 1e-5)
    assert k3["tmaps"] == ["dt=bf16,dims=128/400/400,str=256/102400,box=64/64/2,swz=128,l2=128,oob=0", "dt=bf16,dims=400/400/128,str=800/320000,box=64/1/64,swz=128,l2=128,oob=0",
                           "dt=bf16,dims=128/128,str=256,box=64/32,swz=128,l2=256,oob=0", "dt=bf16,dims=128/128,str=256,box=64/32,swz=128,l2=256,oob=0"]
    k3f = TX.k3_launch(128, 128, 808, True, True, 1e-5)
    assert k3f["kname"] == "tmn_k3_z128_h128_f_t2x64_s8a1_l1" and k3f["tmaps"][0].startswith("dt=f32,dims=128/808/808,str=512/413696,box=32/64/2")
    assert TX.k1_launch(128, 128, 808, True, False, False, 1e-5)["kname"] == "tmn_k1_z128_h128_f_t2x64_s8k2_m0_l1_v0"
    assert TX.k1_launch(64, 64, 300, False, True, False, 1e-5)["kname"] == "tmn_k1_z64_h64_b_t2x64_s4k1_m1_l2_v0"
    # by_n buckets of the 128 unit: K3 with two accumulator sets up to N 1536, one above; uncovered above 4096 is refused by size first
    assert TX.k3_launch(128, 128, 1536, False, False, 1e-5)["kname"].endswith("s8a2_l1") and TX.k3_launch(128, 128, 1537, False, False, 1e-5)["kname"].endswith("s8a1_l1")
    assert TX.k1_smem(128, 2, 8, 2) == 181504 and TX.k3_smem(128, 128, 2, 2, 64, 8) == 149760


def test_refusals_are_by_name_with_the_fallback(monkeypatch):
    import numpy as np
    bf = np.dtype("float32")   # dtype words are taken from .name
    class DT:                   # noqa: E306
        def __init__(self, n): self.name = n
    ok = TX.cannot_serve("9.0", DT("bfloat16"), 128, 128, 400)
    assert ok is None
    assert TX.cannot_serve("8.0", DT("bfloat16"), 128, 128, 400)[0] == "arch"
    assert TX.cannot_serve("9.0", DT("bfloat16"), 128, 64, 400)[0] == "width"
    assert TX.cannot_serve("9.0", DT("float16"), 128, 128, 400)[0] == "dtype"
    assert TX.cannot_serve("9.0", DT("float32"), 128, 128, 400) is None                 # fp32-resident form
    assert TX.cannot_serve("9.0", DT("bfloat16"), 128, 128, 8)[0] == "size" and TX.cannot_serve("9.0", DT("bfloat16"), 128, 128, 4097)[0] == "size"
    assert TX.cannot_serve("9.0", DT("bfloat16"), 128, 128, 400, biases=True)[0] == "biases"
    with pytest.raises(TX.Refused) as e:
        TX.select("8.6", DT("bfloat16"), 128, 128, 400)
    assert "refused by name (arch)" in str(e.value) and "fallback: kernels.pallas cd_trimul" in str(e.value) and e.value.kind == "arch"
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "trimul_xla:native_xla")
    assert TX.cannot_serve("9.0", DT("bfloat16"), 128, 128, 400)[0] == "levers_off"
    monkeypatch.delenv("MODEL_OPT_LEVERS_OFF")
    assert TX.select("9.0", DT("bfloat16"), 64, 64, 1200) == "native_xla"


def test_math_layout_mapping_and_bias_refusal():
    np = pytest.importorskip("numpy")
    C, Ch = 8, 4
    p = {"ln_in_scale": np.ones(C), "ln_in_offset": np.zeros(C), "left_w": np.full((C, Ch), 1.0), "right_w": np.full((C, Ch), 2.0), "left_gate_w": np.full((C, Ch), 3.0),
         "right_gate_w": np.full((C, Ch), 4.0), "ln_c_scale": np.ones(Ch), "ln_c_offset": np.zeros(Ch), "out_w": np.arange(Ch * C, dtype=float).reshape(Ch, C), "gate_w": np.eye(C)}
    try:
        import jax.numpy  # noqa: F401
    except Exception:
        pytest.skip("jax not importable here")
    w = TX.weights_from_math_layout(p, "outgoing")
    assert float(w["w_ap"][0, 0]) == 1.0 and float(w["w_bp"][0, 0]) == 2.0 and float(w["w_ag"][0, 0]) == 3.0 and float(w["w_bg"][0, 0]) == 4.0 and tuple(w["w_o"].shape) == (C, Ch)
    wi = TX.weights_from_math_layout(p, "incoming")                                     # 'kjc,kic->ijc' on (left, right) == the kernels' incoming with a = right, b = left
    assert float(wi["w_ap"][0, 0]) == 2.0 and float(wi["w_ag"][0, 0]) == 4.0 and float(wi["w_bp"][0, 0]) == 1.0
    with pytest.raises(TX.Refused) as e:
        TX.weights_from_math_layout(dict(p, left_b=np.zeros(Ch)), "outgoing")
    assert e.value.kind == "biases"


def test_the_row_is_registered_in_the_pallas_provider_forward_only_with_its_fallback():
    assert "native_xla" in PP.ROW_NAMES and "native_xla" in PP.FORWARD_ONLY
    info = PP.row_info("native_xla")
    assert info["fallback"] == "cd_trimul" and info["backward"] is False and "trimul" in " ".join(info["ops"])
    assert not PP.has_backward("native_xla")
    src = open(os.path.join(CORE, "opt_core", "kernels", "pallas", "serve.py"), encoding="utf-8").read()
    assert 'if row == "native_xla":' in src and "TX.weights_from_math_layout(params, direction_word)" in src


def test_forward_only_and_words_are_stated():
    assert "FORWARD ONLY" in TX.__doc__ and "Refused" in TX.__doc__ and "biases" in TX.__doc__.lower()
    meta = json.load(open(os.path.join(CORE, "opt_core", "kernels", "META", "trimul_xla.json")))
    assert meta["name"] == "trimul_xla" and "MODEL_OPT_LEVERS_OFF" in meta["exports"]
    rep = TX.report()
    assert rep["forward_only"] is True and rep["package_version"] == "1.0.1" and set(rep["units"]) == {"128x128", "64x64"}
    for u in rep["units"].values():
        assert re.fullmatch(r"[0-9a-f]{64}", u["sha256"]) and u["kernels"] >= 20
