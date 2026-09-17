"""kernels.trimul row esm_v61: the ESM-family engine's sealed package (prebuilt sm_90a cubin K3 through the CUDA driver) behind one face.
No framework here: admission / refusals by name, tier words never choose it, payload integrity (DIGESTS, not_carried, the kernel-source
digest the cubins' manifest records), the face and its driver binding are standard library at import, and the ctypes binding mirrors the
names / enumerators / return convention the carried module uses."""
import ast
import hashlib
import importlib
import json
import os

import pytest

from opt_core import kernels
from opt_core.kernels import trimul as T

BASE = os.path.join(kernels.KERNELS_DIR, "trimul", "esm_v61")
H100 = "H100:2.13.0+cu130/3.7.1/cueq0.11.1"
A100 = "A100:2.13.0+cu130/3.7.1/cueq0.11.1"


def test_row_facts_and_table_entry():
    assert "esm_v61" in T.ROW_NAMES and "esm_v61" not in T.NEEDS_ESM_IMAGE and "esm_v61" not in T.BACKWARD_ROWS and "esm_v61" not in T.EXACT_ROWS
    assert "esm_v61" not in T.STOCK_ROWS and T.ESM_V61_C == (256,) and T.ESM_V61_CC == "9.0"
    tab = T.table()
    row = tab["rows"]["esm_v61"]
    assert row["class"] == "fast" and row["exact_vs"] is None and row["backward"] is False and row["fallback"].startswith("esm_v5_fwd")
    assert row["admits"]["c_z"] == [256] and row["admits"]["n_min"] == 16 and row["levers"]["k3cute"] is True and row["levers"]["lnfold"] is True
    assert "esm_v61" in tab["tiers"]["fast"] and "esm_v61" not in tab["tiers"]["exact"]
    named = [(k, st) for k, c in tab["cells"].items() for st, r in c.get("fast_per_stack", {}).items() if r == "esm_v61"]
    assert named, "the race runs name esm_v61 on the stacks without a tx_sm90a key"
    ruled = {(e[0], e[1]) for v in (tab.get("in_model") or {}).values() if isinstance(v, dict) for e in (v.get("applies_to") or []) if e[2] == "fast"}
    for k, st in named:                                                                # a cell names it only with a same-machine run record on that stack (any race_* /
        c_ = tab["cells"][k]                                                           # shapes_* block), an in-model decision listing (cell, stack, tier), or a policy record
        assert any(st in v for b_, v in c_.items() if isinstance(v, dict) and (b_.startswith("race_") or b_.startswith("shapes_"))) or (k, st) in ruled or st in (c_.get("policy") or {}) or st in ((c_.get("carried_columns") or {}).get("columns") or {}), (k, st)
    assert ruled and all(k.startswith("9.0|bf16|C256|H256|") and k.endswith("|fwd") for k, _st in ruled)     # in_model.c256_fwd_h100_t211: the torch-2.11 H100 column


def test_admits_and_refuses_by_name():
    R = T.Refusal
    for d in ("outgoing", "incoming"):
        for n in (16, 400, 800, 1200):
            s = T.select("9.0", "bf16", 256, 256, n, d, word="esm_v61", stack=H100)
            assert s.row == "esm_v61" and (s.cls == "fast" or s.cls.startswith("tol(")) and s.backward is False and s.fallback.startswith("esm_v5_fwd")
    assert T.select("9.0", "bf16", 256, 256, 800, "outgoing", word="esm_v61", residency="fp32").row == "esm_v61"       # fp32-resident z under bf16 autocast: cast once
    assert T.select("9.0", "bf16", 256, 256, 800, "outgoing", word="esm_v61", stack="H100:2.10.0+cu128/3.6.0/cueq0.10.0/ds4s").row == "esm_v61"
    for args, kind, fb in ((("8.0", "bf16", 256, 256, 800), "cc:8.0!=9.0", "esm_v5_fwd"),          # the cubin is sm_90a: capability 8.0 serves the line through esm_v5_fwd
                           (("10.0", "bf16", 256, 256, 800), "cc:10.0!=9.0", "esm_v5_fwd"),
                           (("7.5", "bf16", 256, 256, 800), "cc:7.5!=9.0", "cueq"),
                           (("9.0", "bf16", 128, 128, 800), "c_z:128(not 256)", "esm_v5_fwd"),      # the cofolding trunks' width: the Triton-K3 line
                           (("9.0", "bf16", 384, 384, 800), "c_z:384(not 256)", "cueq"),
                           (("9.0", "bf16", 64, 64, 800), "c_z:64(not 256)", "esm_v5_fwd"),
                           (("9.0", "bf16", 64, 128, 800), "c_hidden=128!=c_z=64", "torch_math"),
                           (("9.0", "fp32", 256, 256, 800), "dtype:fp32!=bf16", "v4"),
                           (("9.0", "tf32", 256, 256, 800), "dtype:tf32!=bf16", "v4"),
                           (("9.0", "bf16", 256, 256, 8), "n<16", "cueq")):
        with pytest.raises(R) as e:
            T.admits("esm_v61", *args)
        assert e.value.kind.startswith(kind) and e.value.row == "esm_v61" and e.value.fallback == fb, (args, e.value.kind, e.value.fallback)
    with pytest.raises(R) as e:                                                        # the 2.7 stacks' triton has no tensor descriptors: named, v4 serves the line there
        T.select("9.0", "bf16", 256, 256, 800, "outgoing", word="esm_v61", stack="H100:2.7.1+cu128/3.3.1/cueq0.10.0")
    assert e.value.kind == "triton:3.3.1<3.6(tl.make_tensor_descriptor)" and e.value.fallback == "v4"
    with pytest.raises(R) as e:
        T.select("9.0", "bf16", 256, 256, 800, "outgoing", word="esm_v61", backward=True)
    assert e.value.kind == "no_backward" and e.value.fallback == "ef2_fused"


def test_tier_words_choose_the_row_only_where_a_race_run_names_it():
    """esm_v61 is a tier word's value only on the stacks whose same-machine measurement named it (no tx_sm90a key there); by row word everywhere."""
    tab = T.table()
    for word in ("fast", "big", "exact"):
        for cc, st in (("9.0", H100), ("8.0", A100)):
            for C in (128, 256):
                for n in (400, 800, 1200):
                    for d in ("outgoing", "incoming"):
                        s = T.select(cc, "bf16", C, C, n, d, word=word, stack=st)
                        if s.row == "esm_v61":
                            cell = tab["cells"][s.cell]
                            assert word != "exact" and (s.stack in cell.get("race_v3", {}) or s.stack in cell.get("shapes_v1", {})), (s.cell, s.stack, word)


def test_payload_digests_not_carried_and_kernel_source_record():
    from opt_core.kernels.trimul import esm_v61 as E                                   # standard library at import: loads here without a framework
    assert E.ACTIVE_PKG == "v6.1" and E.ARCH == "sm_90a" and E.C == 256 and E.N_MIN == 16 and E.LOADCHECK == {"regs": 168, "local_bytes": 0}
    root = E.pkg_dir()
    assert os.path.isdir(root) and root.endswith(os.path.join("esm_v61", "pkg", "v6.1"))
    assert E.verify_digests() == E.DIGESTS
    on_disk = sorted(os.path.relpath(os.path.join(r, f), root) for r, _, fs in os.walk(root) for f in fs if "__pycache__" not in r)
    assert on_disk == sorted(E.DIGESTS), "a payload file DIGESTS does not pin (or a pinned file absent)"
    for rel in E.NOT_CARRIED:
        assert not os.path.exists(os.path.join(root, rel)), rel
    meta = kernels.sums("trimul")
    for rel in E.NOT_CARRIED:
        assert "esm_v61/pkg/v6.1/" + rel in meta["not_carried"]
    for rel in E.RESTATED:
        assert "esm_v61/pkg/v6.1/" + rel in meta["not_byte_identical"]
    cub = E.cubins()
    assert sorted(cub) == [(0, 1), (1, 1)]
    man = E.manifest()
    for ent in man["kernels"]:
        assert ent["arch"] == "sm_90a" and ent["kernel"] == "k3v6" and int(ent["lnfold"]) == 1
        assert E.DIGESTS["bin/" + ent["cubin"]] == ent["cubin_sha256"] and os.path.getsize(os.path.join(root, "bin", ent["cubin"])) == ent["cubin_bytes"]
        assert ent["source_sha256"]["K3_SRC"] == E.K3_SRC_SHA256 and int(ent["regs"]) == E.LOADCHECK["regs"] and int(ent["lmem"]) == 0
    src = open(os.path.join(root, "src", "ef2_trimul_v6.py"), encoding="utf-8").read()
    a = src.index("K3_SRC = r'''") + len("K3_SRC = r'''")
    k3 = src[a:src.index("'''", a)]
    assert hashlib.sha256(k3.encode()).hexdigest() == E.K3_SRC_SHA256                   # the carried kernel source is the one both cubins were compiled from
    assert k3 == open(os.path.join(root, "src", "k3v6.cu"), encoding="utf-8").read() or hashlib.sha256(k3.encode()).hexdigest() == E.K3_SRC_SHA256
    vec = json.load(open(os.path.join(root, "tests", "vectors", "vectors.json"), encoding="utf-8"))
    assert vec["pt_sha256"] == E.DIGESTS["tests/vectors/vectors.pt"]                   # the vectors placed under tests/vectors are the ones its record names
    with pytest.raises(E.Unavailable) as e:
        E.verify_digests("v0.0")
    assert e.value.kind.startswith("digest:")


def test_face_and_binding_are_standard_library_at_import_and_no_model_package_anywhere():
    banned = {"transformers", "esm", "cuequivariance_torch", "cuequivariance_ops_torch"}     # (kit-package imports: tests/test_selfcontained's scan)
    stdlib_face = {"contextlib", "hashlib", "importlib", "json", "os", "sys", "threading"}
    stdlib_drv = {"ctypes", "sys", "types"}
    for dirpath, _, files in os.walk(BASE):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dirpath, f)
            tree = ast.parse(open(p, encoding="utf-8").read())
            mods = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods.add(node.module.split(".")[0])
            assert not (mods & banned), (os.path.relpath(p, BASE), sorted(mods & banned))
            top = set()
            for node in tree.body:
                if isinstance(node, ast.Import):
                    top |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    top.add((node.module or "").split(".")[0])
            rel = os.path.relpath(p, BASE)
            if rel == "__init__.py":
                assert top <= stdlib_face, sorted(top)
            if rel == "cudrv.py":
                assert top <= stdlib_drv, sorted(top)
    from opt_core.kernels.trimul.esm_v61 import cudrv as D                             # loads without a framework and without a GPU
    assert D.binding() is None
    drv, nv = D._shim_modules()
    for n in ("cuModuleLoadData", "cuModuleGetFunction", "cuFuncSetAttribute", "cuFuncGetAttribute", "cuLaunchKernel", "cuTensorMapEncodeTiled", "cuGetErrorName",
              "CUresult", "CUfunction_attribute", "CUtensorMapDataType", "CUtensorMapInterleave", "CUtensorMapSwizzle", "CUtensorMapL2promotion", "CUtensorMapFloatOOBfill",
              "cuuint64_t", "cuuint32_t"):                                            # every driver name the carried module touches
        assert hasattr(drv, n), n
    assert hasattr(nv, "nvrtcResult") and hasattr(nv, "nvrtcVersion") and int(nv.nvrtcVersion()[0]) != int(nv.nvrtcResult.NVRTC_SUCCESS)   # the compiler half refuses
    A = drv.CUfunction_attribute
    assert int(A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) == 8 and int(A.CU_FUNC_ATTRIBUTE_NUM_REGS) == 4 and int(A.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES) == 3
    assert int(drv.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16) == 9 and int(drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B) == 3
    assert int(drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B) == 2 and int(drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE) == 0
    assert int(drv.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B) == 2 and int(drv.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE) == 0
    assert int(drv.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == 0
    ok = drv.CUresult(0)
    assert isinstance(ok, drv.CUresult) and ok == drv.CUresult.CUDA_SUCCESS and drv.CUresult(719) != drv.CUresult.CUDA_SUCCESS and int(drv.cuuint64_t(5)) == 5
    tm = drv.CUtensorMap()
    assert len(tm.opaque) == 16 and tm._address() % 64 == 0
    with pytest.raises(ValueError):
        D.ensure(binding="nvrtc")
    try:                                                                               # no wheel here: either libcuda resolves (a driver is mounted) or the refusal is named
        word = D.ensure()
        assert word.startswith(("cuda.bindings", "ctypes:libcuda"))
    except D.Unresolvable as e:
        assert "libcuda" in str(e) or "cuda.bindings" in str(e)


def test_the_carried_source_modules_search_the_engine_cells_the_way_esm_v5_does():
    """The package's K1 / bmm line is generation-matched to its K3: its own cell table (src/ef2_w4_fpf_trimul_v4_cells.json) is byte-identical to
    row esm_v5_fwd's carried table, so both rows resolve the same (cc | triton) launch cell."""
    a = os.path.join(BASE, "pkg", "v6.1", "src", "ef2_w4_fpf_trimul_v4_cells.json")
    b = os.path.join(kernels.KERNELS_DIR, "trimul", "esm_v5", "ef2_w4_fpf_trimul_v4_cells.json")
    assert open(a, "rb").read() == open(b, "rb").read()


def test_foreign_module_rule_admits_equal_sealed_bytes_and_refuses_a_mismatch_by_name():
    """An engine's own copy of the package module registered under the package's name: admitted only when its K3 source digest (and any cubin
    digest its manifest lists) equals ours -- the carried copy serves either way; a differing copy is refused BY NAME with both digests."""
    import re
    import types
    E = importlib.import_module("opt_core.kernels.trimul.esm_v61")
    src = open(os.path.join(E.pkg_dir(), "src", "ef2_trimul_v6.py"), encoding="utf-8").read()
    k3 = re.search(r"K3_SRC = r'''(.*?)'''", src, re.S).group(1)
    good = types.ModuleType("ef2_trimul_v6"); good.K3_SRC = k3; good.__file__ = "/elsewhere/src/ef2_trimul_v6.py"
    assert E.foreign_check("ef2_trimul_v6", good) == E.K3_SRC_SHA256
    bad = types.ModuleType("ef2_trimul_v6"); bad.K3_SRC = k3 + "// edited"; bad.__file__ = "/elsewhere/src/ef2_trimul_v6.py"
    with pytest.raises(E.Unavailable) as e:
        E.foreign_check("ef2_trimul_v6", bad)
    assert e.value.reason.startswith("foreign_module_digest_mismatch:ef2_trimul_v6:kit_") and ("!=core_" + E.K3_SRC_SHA256[:8]) in e.value.reason
    blank = types.ModuleType("ef2_trimul_v6"); blank.__file__ = "/elsewhere/x.py"
    with pytest.raises(E.Unavailable) as e:
        E.foreign_check("ef2_trimul_v6", blank)
    assert e.value.reason.startswith("foreign_module_digest_unknown:")
    assert E.foreign_check("ef2_trimul_v5", types.ModuleType("ef2_trimul_v5")) is None


def test_a_foreign_module_under_the_package_name_is_parked_while_the_carried_modules_bind():
    """An engine that imported ITS OWN ef2_trimul_v6 first (a module of the package's name without the sealed package's attributes): the face
    binds the CARRIED modules inside carried_binding() -- the foreign entry is parked for the duration and restored after, so the engine keeps its
    module -- and install() in a process that cannot engage answers with an Unavailable word BY NAME, never an AttributeError off that module."""
    import sys
    import types
    E = importlib.import_module("opt_core.kernels.trimul.esm_v61")
    fake = types.ModuleType("ef2_trimul_v6"); fake.__file__ = "/engine/kit/ef2_trimul_v6.py"      # no _Kernel, no _STATE, no K3_SRC
    class StubFace:                                                                                 # the face's registry of loaded (carried) modules
        _MODS = {}
    prev = sys.modules.get("ef2_trimul_v6")
    sys.modules["ef2_trimul_v6"] = fake
    try:
        assert E.foreign_modules().get("ef2_trimul_v6") is fake
        with E.carried_binding(StubFace()) as parked:
            assert parked.get("ef2_trimul_v6") is fake and "ef2_trimul_v6" not in sys.modules   # parked: name lookups cannot reach the foreign copy
        assert sys.modules.get("ef2_trimul_v6") is fake                                          # restored on exit
        assert E.carried_module("ef2_trimul_v6") is None                                         # nothing carried is bound in this process yet
        with pytest.raises(E.Unavailable) as e:
            E.install()
        assert e.value.kind.split(":")[0] in ("foreign_module_digest_unknown", "driver", "device", "import", "cc", "triton", "digest", "prebuilt", "load_failed"), e.value.kind
        assert sys.modules.get("ef2_trimul_v6") is fake                                          # a refusal leaves the engine's module in place too
    finally:
        if prev is None:
            sys.modules.pop("ef2_trimul_v6", None)
        else:
            sys.modules["ef2_trimul_v6"] = prev
