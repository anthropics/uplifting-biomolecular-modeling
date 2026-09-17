"""triattn_xla (the JAX triangle-attention bridge): its tables are well-formed, every shipped binary is listed with a matching sha256, the
CUDA kernel sources are the sealed ones, the cells reference binaries that exist, refusals are named with the fallback, the FFI target
names agree between the launcher source and the Python side, and the test-input generator reproduces the recorded input digests.
CPU only: nothing here imports jax (the rows themselves are verified on GPU stacks; see the release notes)."""
import hashlib
import json
import os
import re

import pytest

from opt_core import kernels
from opt_core.kernels import triattn_xla as X
from opt_core.kernels.triattn_xla import _cuda, _k2b, _launch, _vectors

PKG = X.PKG_DIR
KERNELS = os.path.dirname(PKG)
SEALED_CUDA_SOURCES = {          # sha256 of the CUDA kernel sources (carried by kernels/triattn, the torch provider) as sealed by their producing package
    "kernels/triattn/cuda_sm90a/csrc/triattn_mw.cu": "14eb59c6a2028df19fdc7ff34c180bd9acf0c292484bf3bb7ec013f447be33a5",
    "kernels/triattn/cuda_sm90a/csrc/mw_ptx.h": "aa22a8a237f1915f467b2091de84c7060790c1641f6c4d8b3758c33172cf1e33",
}


def src_path(rel):
    """manifest source keys: relative to the package, or `kernels/...` relative to opt_core/ (sources another carried package holds)."""
    return os.path.join(os.path.dirname(KERNELS), rel) if rel.startswith("kernels/") else os.path.join(PKG, rel)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def _sha(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_is_a_carried_package_with_metadata():
    doc = kernels.sums("triattn_xla")
    assert doc["kind"] == "package" and doc["name"] == "triattn_xla" and doc["version"] == X.__version__
    assert "CELLS.json" in doc["files"] and "manifest.json" in doc["files"] and "__init__.py" in doc["files"]


def test_cells_table_is_well_formed():
    c = X.cells()
    assert c["schema"] == "triattn_xla/cells/v1" and c["fallback"] == X.FALLBACK == "stock: xla"
    assert set(c["rows"]) == set(X.ROWS)
    for cc, ent in c["by_cc"].items():
        assert re.fullmatch(r"\d+\.\d+", cc), cc
        assert ent["arch"] in ("sm_90", "sm_80") and ent["order"] and set(ent["order"]) <= set(X.ROWS)
        if "cuda_sm90a" in ent["order"]:
            assert cc == "9.0"                                        # the CUDA binary is sm_90a only
    assert c["by_cc"]["9.0"]["order"][0] in X.ROWS and c["by_cc"]["8.0"]["order"] == ["cuda_80", "k2b_aot"] and c["by_cc"]["8.6"]["order"] == ["k2b_aot"]
    k = c["k2b"]
    assert k["bias16_min_sk"] == 256 and k["nflag"] == 1024 and k["prep"] == {"PB": 32, "PK": 128} and k["exp_mode"] == {"bf16": 1, "fp32": 0}
    for m in c.get("measured", []):                                   # evidence rows: a number or nothing
        assert {"cc", "dtype", "head_dim", "N", "row", "ms"} <= set(m), m


def test_manifest_lists_every_binary_with_its_sha256():
    man = X.manifest()
    assert man["schema"] == "triattn_xla/manifest/v1"
    listed = {b["file"]: b for b in man["binaries"]}
    on_disk = sorted(os.path.relpath(os.path.join(r, f), PKG) for r, _, fs in os.walk(os.path.join(PKG, "bin")) for f in fs if not f.endswith(".json"))
    assert on_disk, "no binaries under bin/"
    assert sorted(listed) == on_disk, (sorted(set(listed) ^ set(on_disk)))
    for rel, b in listed.items():
        assert sha(os.path.join(PKG, rel)) == b["sha256"], rel
        assert b["kind"] in ("launcher", "cuda", "k2b_cubin")
    kinds = [b["kind"] for b in man["binaries"]]
    assert kinds.count("cuda") == 4 and kinds.count("launcher") >= 2 and kinds.count("k2b_cubin") >= 8
    apis = sorted(b["ffi_api_version"] for b in man["binaries"] if b["kind"] == "launcher")
    assert len(set(apis)) == len(apis)                                 # one launcher per XLA-FFI API version
    for rel, digest in man["sources_sha256"].items():                  # the sources the binaries were built from are the ones shipped
        assert sha(src_path(rel)) == digest, rel


def test_cuda_kernel_sources_are_the_sealed_ones():
    for rel, digest in SEALED_CUDA_SOURCES.items():                    # the torch provider's carried copy is the sealed one ...
        assert sha(src_path(rel)) == digest, rel
    cuda = [b for b in X.manifest()["binaries"] if b["kind"] == "cuda" and b.get("role", "fwd") == "fwd"][0]
    for rel, digest in SEALED_CUDA_SOURCES.items():                    # ... and the shipped binary was built from exactly those bytes
        assert cuda["sources_sha256"][rel] == digest, rel
    assert cuda["arch"] == "sm_90a" and cuda["cudart"] == "static" and cuda["abi_version"] == 1
    for f in ("triattn_mw_cuda.cu", "triattn_cuda_abi.h"):             # the shipped host sources are the built ones
        assert cuda["sources_sha256"][f] == sha(os.path.join(PKG, "csrc", f)), f
    assert cuda["arch"] == "sm_90a" and cuda["cudart"] == "static" and cuda["abi_version"] == 1
    assert cuda["sources_sha256"]["triattn_mw_cuda.cu"] == sha(os.path.join(PKG, "csrc", "triattn_mw_cuda.cu"))      # the shipped host source is the built one


def test_cells_reference_binaries_that_exist():
    idx = _k2b.cubin_index()
    for cc, ent in X.cells()["by_cc"].items():
        arch = ent["arch"]
        if "k2b_aot" in ent["order"]:
            for D in (16, 32):
                for mask in (0, 1):
                    for div in ("s16", "any"):
                        for b16 in (0, 1):
                            assert "%s/k2b_fwd_bf16_d%d_mask%d_b16%d_%s" % (arch, D, mask, b16, div) in idx, (cc, arch, D, mask, b16, div)
                        assert "%s/k2b_fwd_fp32_d%d_mask%d_b160_%s" % (arch, D, mask, div) in idx
                        assert "%s/k2b_prep_mk0_%s" % (arch, div) in idx and "%s/k2b_prep_mk1_%s" % (arch, div) in idx
        if "cuda_sm90a" in ent["order"]:
            assert _cuda.lib_entry() is not None and os.path.isfile(os.path.join(PKG, _cuda.lib_entry()["file"]))
    for key, b in idx.items():                                          # every cubin carries its launch recipe
        assert b["kernel"]["name"] and isinstance(b["kernel"]["shared"], int) and b["params"] and b["n_trailing_null"] in (0, 1, 2)
        assert b["triton"].startswith("3.7"), b["triton"]                # the compiler line these cells were built with
        if b["role"] == "fwd":
            assert b["cell"]["BLOCK_M"] and b["cell"]["BLOCK_N"] and b["cell"]["ROWS"] and b["constexprs"]["IP"] == "tf32"
            assert int(b["constexprs"]["EXP_MODE"]) == (1 if b["dtype"] == "bf16" else 0)


def test_refusals_are_named_with_the_fallback(monkeypatch):
    monkeypatch.delenv("MODEL_OPT_LEVERS_OFF", raising=False)
    row, cell = X.select("9.0", "bfloat16", 32, S=384, N=384, H=4, B=1, has_mask=True)
    assert row == "cuda_sm90a"                                                          # below triattn_native's min_S the next row of the order serves
    for masked in (False, True):                                                        # 1.6.1: the family's tie policy -- auto takes triattn_native from S 500 in every mask form,
        for S_, want in ((256, "k2b_aot"), (383, "k2b_aot"), (384, "cuda_sm90a"), (448, "cuda_sm90a"), (499, "cuda_sm90a"), (500, "triattn_native"), (640, "triattn_native"),
                         (1024, "triattn_native"), (1536, "triattn_native"), (2048, "triattn_native"), (2560, "triattn_native"), (4096, "triattn_native")):   # cuda_sm90a 384-499, k2b_aot below 384
            row, cell = X.select("9.0", "bfloat16", 32, S=S_, N=64, H=4, B=1, has_mask=masked)
            assert row == want, (S_, row, want, masked)
    rl = X.cells()["by_cc"]["9.0"]["rules"]["triattn_native"]
    assert rl["min_S"] == 500 and "max_S_excl" not in rl and "TIE" in rl["why"] and rl["history"]["1.4.2-1.6.0"]["min_S"] == 2048
    assert set(X.cells()["by_cc"]["9.0"]["forms"]) >= {"design_model_bf16_h4_d32_keymask", "structure_model_bf16_h4_d32_pad"} and "line-independent" in X.cells()["by_cc"]["8.0"]["lines"]
    row, cell = X.select("9.0", "bfloat16", 32, S=640, N=640, H=4, B=1, has_mask=True, impl="triattn_native")   # by word anywhere in its envelope
    assert row == "triattn_native"
    # cc 8.0 (1.6.0): cuda_80 under auto where its cells won on both jax lines, per head_dim and mask form; k2b_aot elsewhere; by word anywhere in its envelope
    for D_, S_, masked, want in ((32, 256, True, "k2b_aot"), (32, 384, True, "k2b_aot"), (32, 400, True, "cuda_80"), (32, 2048, True, "cuda_80"), (32, 256, False, "k2b_aot"), (32, 400, False, "cuda_80"),
                                 (16, 200, True, "k2b_aot"), (16, 256, True, "cuda_80"), (16, 1200, True, "cuda_80"), (16, 400, False, "k2b_aot"), (16, 800, False, "cuda_80")):
        row, _ = X.select("8.0", "bfloat16", D_, S=S_, N=S_, H=4, has_mask=masked)
        assert row == want, (D_, S_, masked, row, want)
    with pytest.raises(X.Refused) as e:                                                # head_dim 64 on cc 8.0: no cell measured -> not under auto (k2b_aot has no head_dim-64 cubins either)
        X.select("8.0", "bfloat16", 64, S=2048, N=2048, H=4, has_mask=True)
    assert "head_dim 64 not in [16, 32]" in str(e.value) and X.FALLBACK in str(e.value)
    row, _ = X.select("8.0", "bfloat16", 64, S=2048, N=2048, H=4, has_mask=True, impl="cuda_80")   # by word: served
    assert row == "cuda_80"
    row, _ = X.select("8.6", "bf16", 16, S=200, N=200, H=4, has_mask=False)          # same-major rule: sm_80 cubins
    assert row == "k2b_aot"
    row, _ = X.select("9.0", "float32", 32, S=256, N=256, H=4, has_mask=True)        # fp32: the CUDA row cannot, K2B (TF32) can
    assert row == "k2b_aot"
    for kw, words in (
        (dict(cc="7.5", dtype="bf16", head_dim=32, S=256), ["7.5", "stock: xla"]),
        (dict(cc="9.0", dtype="float16", head_dim=32, S=256), ["fp16", "triattn_native", "cuda_sm90a", "k2b_aot", "stock: xla"]),
        (dict(cc="9.0", dtype="bf16", head_dim=48, S=256), ["head_dim 48", "stock: xla"]),
        (dict(cc="8.0", dtype="bf16", head_dim=64, S=256), ["head_dim 64", "stock: xla"]),
    ):
        with pytest.raises(X.Refused) as ei:
            X.select(**kw)
        msg = str(ei.value)
        for w in words:
            assert w in msg, (w, msg)
        assert ei.value.fallback == "stock: xla" and isinstance(ei.value, NotImplementedError)
    with pytest.raises(X.Refused) as ei:
        X.select("9.0", "bf16", 32, S=100000, N=8, H=4)
    assert "stock: xla" in str(ei.value)
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "triattn_xla:cuda_sm90a")
    row, _ = X.select("9.0", "bf16", 32, S=384, N=384, H=4)
    assert row == "k2b_aot"
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "triattn_xla:triattn_native")
    with pytest.raises(X.Refused) as ei:
        X.select("9.0", "bf16", 32, S=640, N=640, H=4, impl="triattn_native")
    assert "switched off by MODEL_OPT_LEVERS_OFF=triattn_xla:triattn_native" in str(ei.value)
    row, _ = X.select("9.0", "bf16", 32, S=2048, N=64, H=4)
    assert row == "cuda_sm90a"                                                          # the band's row switched off by word -> the incumbent
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "triattn_xla:cuda_sm90a")
    row, _ = X.select("9.0", "bf16", 32, S=640, N=640, H=4)
    assert row == "triattn_native"                                                          # 1.6.1: triattn_native serves from S 500 whether or not cuda_sm90a is switched off
    row, _ = X.select("9.0", "bf16", 32, S=448, N=448, H=4)
    assert row == "k2b_aot"                                                             # below triattn_native's floor with cuda_sm90a off: the cubins
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "some_other_lever,triattn_xla")
    with pytest.raises(X.Refused) as ei:
        X.select("9.0", "bf16", 32, S=384, N=384, H=4)
    assert "switched off by MODEL_OPT_LEVERS_OFF=triattn_xla" in str(ei.value)
    with pytest.raises(X.Refused):
        X.select("9.0", "bf16", 32, S=384, impl="no_such_row")


def test_ffi_target_names_agree_between_launcher_source_and_python():
    cc = open(os.path.join(PKG, "csrc", "cubin_launch.cc"), encoding="utf-8").read()
    assert "XLA_FFI_DEFINE_HANDLER_SYMBOL(TriattnXlaLaunch," in cc and "XLA_FFI_DEFINE_HANDLER_SYMBOL(TriattnXlaCudaFwd," in cc and "XLA_FFI_DEFINE_HANDLER_SYMBOL(TriattnXlaM1Fwd," in cc
    from opt_core.kernels.triattn_xla import _native
    assert _native.TARGET == "triattn_xla_m1_fwd" and _native.TARGET + ":" in cc
    assert _launch.TARGET_LAUNCH == "triattn_xla_launch" and _launch.TARGET_CUDA == "triattn_xla_cuda_fwd"
    assert _launch.TARGET_LAUNCH + ":" in cc and _launch.TARGET_CUDA + ":" in cc            # the handler error texts name their target
    py = open(os.path.join(PKG, "_launch.py"), encoding="utf-8").read()
    assert "lib.TriattnXlaLaunch" in py and "lib.TriattnXlaCudaFwd" in py and "api_version=1" in py
    abi = open(os.path.join(PKG, "csrc", "triattn_cuda_abi.h"), encoding="utf-8").read()
    assert "#define TRIATTN_CUDA_ABI_VERSION 1" in abi
    for sym in ("triattn_mw_cuda_fwd", "triattn_mw_cuda_fix_elems", "triattn_mw_cuda_smem_bytes", "triattn_mw_cuda_describe"):
        assert sym in abi and sym in open(os.path.join(PKG, "csrc", "triattn_mw_cuda.cu"), encoding="utf-8").read()


def test_forward_only_is_stated_and_refused_by_name():
    assert "FORWARD ONLY" in X.__doc__ and "Refused" in X.__doc__
    src = open(os.path.join(PKG, "_k2b.py"), encoding="utf-8").read()
    assert "custom_jvp" in src and "forward-only" in src and "defjvp" in src


def test_vector_generator_reproduces_the_recorded_inputs():
    np = pytest.importorskip("numpy")
    rec = _vectors.recorded()
    assert rec.get("schema") == "triattn_xla/vectors/v1" and rec["cases"], "vectors.json missing or empty"
    assert set(rec["cases"]) == set(_vectors.CASES)
    for name, case in rec["cases"].items():
        assert _vectors.input_digests(name) == case["inputs_sha256"], name
        for key in (("k2b_torch",) if name in _vectors.PER_ARCH_OPTIONAL else ("k2b_torch", "k2b_torch_sm80")):   # one digest per cubin arch (sm_90, sm_80); the S >= 512 cases: sm_90
            assert key in case["rows"] and re.fullmatch(r"[0-9a-f]{64}", case["rows"][key]["sha256"]), (name, key)
        meta = case["meta"]
        if meta["dtype"] == "bf16" and meta["D"] == 32:
            assert re.fullmatch(r"[0-9a-f]{64}", case["rows"]["cuda_torch"]["sha256"]), name      # the CUDA kernel's torch launch was digested too
            assert re.fullmatch(r"[0-9a-f]{64}", case["rows"]["native_m1_torch"]["sha256"]), name  # and the fleet kernel's (row triattn_native reproduces these bytes)
    x = np.array([1.0, -2.5, 3.0e-3, 65504.0, 1e-9], dtype=np.float32)
    assert _vectors.bf16_bits_to_f32(_vectors.f32_to_bf16_bits(x)).dtype == np.float32
    assert _vectors.f32_to_bf16_bits(np.float32([1.0]))[0] == 0x3F80


# ----------------------------------------------------------------------------------------------- the differentiable row (vjp=<word>)
def test_lse_cubins_mirror_the_forward_cubins():
    man = X.manifest()
    fwd = {(b["arch"], b["key"]): b for b in man["binaries"] if b["kind"] == "k2b_cubin" and b.get("role") == "fwd"}
    lse = [b for b in man["binaries"] if b["kind"] == "k2b_cubin" and b.get("role") == "fwd_lse"]
    assert lse, "no k2bl_fwd_* cubins"
    for arch in ("sm_90", "sm_80"):
        mine = [b for b in lse if b["arch"] == arch]
        assert len(mine) == 12, (arch, len(mine))                        # {bf16 b16 0/1, fp32} x mask 0/1 x {s16, any}, head_dim 32
        assert {b["dtype"] for b in mine} == {"bf16", "fp32"} and {b["head_dim"] for b in mine} == {32}
        for b in mine:
            twin = fwd[(arch, "k2b" + b["key"][len("k2bl"):])]           # same cell, constexprs and specialisation as the forward-only cubin
            assert b["cell"] == twin["cell"] and b["constexprs"] == twin["constexprs"] and b["div16"] == twin["div16"], b["key"]
            assert b["params"][:9] == ["Q", "K", "V", "Bias32", "Bias16", "Flags", "Mask", "Out", "Lse"] and b["params"][9:] == twin["params"][8:], b["key"]
            assert b["lse"] == {"dtype": "fp32", "layout": "[B,N,H,SQ] dense", "units": "log2", "exp_mode": b["constexprs"]["EXP_MODE"]}
            assert b["kernel"]["name"] == "_flash_triattn_fwd_lse" and b["triton"] == twin["triton"]


def test_lse_kernel_source_is_the_carried_forward_plus_the_marked_edits():
    """csrc/k2b_lse_kernel.py = kernels/fpf_triatt_k2b/triatt_k2b.py::_flash_triattn_fwd byte for byte + the `Lse` parameter + the marked epilogue."""
    src = open(os.path.join(KERNELS, "fpf_triatt_k2b", "triatt_k2b.py"), encoding="utf-8").read()
    mine = open(os.path.join(PKG, "csrc", "k2b_lse_kernel.py"), encoding="utf-8").read()
    s = src.index("@triton.jit\ndef _flash_triattn_fwd("); e = src.index("\n# ----", s) + 1        # up to the next column-0 section rule
    region = src[s:e].rstrip("\n") + "\n"
    body = mine[mine.index("@triton.jit\ndef _flash_triattn_fwd_lse("):]
    mark = "    # ---- added epilogue:"
    assert mark in body
    body = body[:body.index(mark)]
    body = body.replace("def _flash_triattn_fwd_lse(\n    Q, K, V, Bias32, Bias16, Flags, Mask, Out, Lse,", "def _flash_triattn_fwd(\n    Q, K, V, Bias32, Bias16, Flags, Mask, Out,", 1)
    body = re.sub(r"     # Lse: [^\n]*\n", "\n", body, count=1)
    assert body == region, "k2b_lse_kernel.py drifted from the carried forward kernel"
    man = X.manifest()
    assert man["sources_sha256"]["csrc/k2b_lse_kernel.py"] == sha(os.path.join(PKG, "csrc", "k2b_lse_kernel.py"))
    for b in man["binaries"]:
        if b["kind"] == "k2b_cubin" and b.get("role") == "fwd_lse":
            assert b["source_sha256"] == man["sources_sha256"]["csrc/k2b_lse_kernel.py"], b["key"]      # the cubins were compiled from the shipped source


def test_vjp_words_cells_and_refusals_by_name():
    from opt_core.kernels.triattn_xla import _vjp
    assert X.VJP_WORDS == ("auto",) + _vjp.BWDS
    vc = X.cells()["vjp"]
    for cc_, ent in vc["by_cc"].items():
        for dt, order in ent.items():
            assert dt in ("bf16", "fp32") and order and set(order) <= set(_vjp.BWDS), (cc_, dt, order)
    assert set(vc["knobs"]) == {"attbwd", "flash", "cuda_lse"} and vc["fwd_by_cc"]["9.0"] == ["cuda_lse", "k2bl"]
    assert "XLA autodiff" in _vjp.VJP_FALLBACK
    # refusals by name (CPU: no cubin is launched; reasons are computed from static facts)
    assert "BNSHD" in _vjp.cannot_serve("9.0", "bf16", 32, 256, 256, 8, 4, 1, False, 1, "attbwd")
    assert "B == 1" in _vjp.cannot_serve("9.0", "bf16", 32, 256, 256, 8, 4, 2, False, 0, "attbwd")
    assert "head_dim 16" in _vjp.cannot_serve("9.0", "bf16", 16, 256, 256, 8, 4, 1, False, 0, "attbwd")
    assert "compute capability" in _vjp.cannot_serve("7.5", "bf16", 32, 256, 256, 8, 4, 1, False, 0, "attbwd")
    assert "unknown backward" in (_vjp.cannot_serve("9.0", "bf16", 32, 256, 256, 8, 4, 1, False, 0, "nope") or "")
    bwd, reasons = _vjp.select_bwd("9.0", "bf16", 32, 256, 256, 8, 4, 1, False, 0, "auto")
    try:
        import jax  # noqa: F401
        have_jax = True
    except ImportError:
        have_jax = False
    if not have_jax:                                                     # the framework-free interpreter: every backward is refused, quoting the import error
        assert bwd is None and reasons and all("does not import on this jax" in r for r in reasons), reasons
    assert "vjp" in X.report() and X.report()["vjp"]["lse_cubins"] == {"sm_90": 12, "sm_80": 12}
    assert "triattn_xla:vjp" in X.__doc__ and "vjp=" in X.__doc__


def test_pallas_triatt_is_carried_with_recorded_digests():
    import json as _json
    meta = _json.load(open(os.path.join(KERNELS, "META", "pallas_triatt.json"), encoding="utf-8"))
    d = os.path.join(KERNELS, "pallas_triatt")
    notice = open(os.path.join(d, "NOTICE"), encoding="utf-8").read()
    for f, rec in meta["carried_digests"].items():
        assert sha(os.path.join(d, f)) == rec["carried"], f
        assert rec["carried"] in notice and rec["upstream"] in notice and rec["carried"] != rec["upstream"], f
        src = open(os.path.join(d, f), encoding="utf-8").read()
        assert "from . import " in src and "colab" + "design_opt" not in src, f          # the one rewritten import; no kit package path left
    assert meta["kind"] == "package" and meta["name"] == "pallas_triatt"
    tm = _json.load(open(os.path.join(KERNELS, "META", "triattn_xla.json"), encoding="utf-8"))
    assert set(tm["runtime_imports"]) >= {"pallas_triatt", "pallas_attn"}


def test_cuda_lse_payload_is_the_sealed_kernel_plus_the_recorded_patch():
    """bin/cuda/sm_90a/libtriattn_mw_cuda_lse.so: built from the carried device section + csrc/cuda_lse.patch (strict apply, no fuzz) and the same host
    translation unit compiled with -DTXLA_LSE; call-struct revision 2; the sealed library beside it is untouched (revision 1)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("txla_build", os.path.join(PKG, "csrc", "build.py")); B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
    src = os.path.join(KERNELS, "triattn", "cuda_sm90a", "csrc", "triattn_mw.cu")
    prefix = B.device_prefix(src).decode("utf-8")
    patch = open(os.path.join(PKG, "csrc", "cuda_lse.patch"), encoding="utf-8").read()
    patched = B.apply_patch_strict(prefix, patch)
    assert patched != prefix and patched.count("float* lse;") == 1 and patched.count("p.lse != nullptr") == 1
    added = [l[1:].strip() for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++")]
    removed = [l for l in patch.splitlines() if l.startswith("-") and not l.startswith("---")]
    assert removed == [] and len(added) == 5, (len(added), removed)          # the patch only ADDS: a Params member (3 lines w/ comment) + the epilogue store (2 lines)
    bins = {b.get("role", "fwd"): b for b in X.manifest()["binaries"] if b["kind"] == "cuda"}
    assert set(bins) == {"fwd", "fwd_lse", "m1_fwd", "sm80_fwd"}
    plain, lse = bins["fwd"], bins["fwd_lse"]
    assert plain["abi_version"] == 1 and plain["entry"] == "triattn_mw_cuda_fwd" and "csrc/cuda_lse.patch" not in plain["sources_sha256"]
    assert lse["abi_version"] == 2 and lse["entry"] == "triattn_mw_cuda_fwd_lse" and lse["arch"] == "sm_90a" and lse["cudart"] == "static"
    assert lse["sources_sha256"]["csrc/cuda_lse.patch"] == sha(os.path.join(PKG, "csrc", "cuda_lse.patch"))
    assert lse["sources_sha256"]["triattn_mw.cu[device prefix + cuda_lse.patch]"] == hashlib.sha256(patched.encode("utf-8")).hexdigest()
    for k in ("kernels/triattn/cuda_sm90a/csrc/triattn_mw.cu", "kernels/triattn/cuda_sm90a/csrc/mw_ptx.h", "triattn_mw_cuda.cu", "triattn_cuda_abi.h"):
        assert lse["sources_sha256"][k] == plain["sources_sha256"][k], k        # same carried kernel sources, same host TU and header as the sealed-row build
    assert lse["lse"] == {"dtype": "fp32", "layout": "[B,N,H,S] dense", "units": "log2", "dead_row": "log2(key count) (unseeded offset 0)"}
    abi = open(os.path.join(PKG, "csrc", "triattn_cuda_abi.h"), encoding="utf-8").read()
    assert "TRIATTN_CUDA_ABI_LSE_VERSION 2" in abi and "void* lse;" in abi and abi.index("void* lse;") > abi.index("char err[512];")   # appended last
    notice = open(os.path.join(PKG, "NOTICE"), encoding="utf-8").read()
    assert "cuda_lse.patch" in notice
    c = X.cells()["vjp"]
    assert c["fwd_by_cc"]["9.0"][0] == "cuda_lse" and all(v == ["k2bl"] for k_, v in c["fwd_by_cc"].items() if k_ != "9.0")


def test_the_launch_is_self_describing():
    """1.3.0: the K2B launches go through the `triattn_xla_run` FFI target whose ATTRIBUTES carry the cubin key + digest, kernel name, grid, shared memory
    and the parameter recipe (nothing registered per shape in the launcher), so executables restored from a persistent compilation cache run in a
    fresh process; the launcher reads cubins from the package root set at registration; the CUDA entries are installed at registration."""
    from opt_core.kernels.triattn_xla import _launch
    cc = open(os.path.join(PKG, "csrc", "cubin_launch.cc"), encoding="utf-8").read()
    assert "XLA_FFI_DEFINE_HANDLER_SYMBOL(TriattnXlaRun, RunImpl," in cc and 'Attr<std::string_view>("key")' in cc and 'Attr<std::string_view>("params")' in cc
    assert "txla_set_root" in cc and '"/bin/k2b/"' in cc and "hex_sha256_prefix16" in cc
    assert "XLA_FFI_DEFINE_HANDLER_SYMBOL(TriattnXlaLaunch" in cc                      # previous interface still exported (one version)
    src = open(os.path.join(PKG, "_launch.py"), encoding="utf-8").read()
    assert _launch.TARGET_RUN == "triattn_xla_run" and "call(*inputs, **attrs)" in src and "txla_set_root(PKG_DIR" in src and "_cuda.load()" in src and "_cuda.load_lse()" in src
    # the recipe encoding: buffers by index, integers, exact floats, trailing nulls
    _launch._SPECS.pop("t|x", None)
    real_load = _launch.load
    _launch.load = lambda: None                                                         # no launcher on a CPU host: register_spec itself needs none
    try:
        spec = _launch.register_spec("t|x", {"arch": "sm_90", "key": "k2b_fwd_bf16_d32_mask1_b161_s16", "sha256": "ab" * 32}, "_flash_triattn_fwd", 65536, 4, (3, 5),
                                     [0, 0, 1, 2, 3, 4, 5], [0, 1, 0, 7, 1 << 40, 0, 0], [0, 0, 0, 0, 0, 0.125, 0], 2)
    finally:
        _launch.load = real_load
    assert spec["key"] == "sm_90/k2b_fwd_bf16_d32_mask1_b161_s16" and spec["sha"] == "ab" * 8 and (spec["gx"], spec["gy"], spec["gz"]) == (3, 5, 1)
    assert spec["params"] == "b0,b1,o0,i7,l1099511627776,f0x1.0000000000000p-3,n" and spec["nnull"] == 2 and spec["shared"] == 65536
    assert float.fromhex(spec["params"].split(",")[5][1:]) == 0.125
    for b in X.manifest()["binaries"]:
        if b["kind"] == "k2b_cubin":
            assert b["file"] == "bin/k2b/%s/%s.cubin" % (b["arch"], b["key"])                         # the path rule the launcher applies to the key attribute


def test_triattn_native_row_is_the_carried_payload_kernel_unmodified():
    """Row triattn_native: bin/cuda/sm_90a/libtriattn_m1_xla.so compiles the M1 kernel header of kernels/triattn/triattn_native generation 11 (cuda_b kernel directory)
    UNMODIFIED (manifest digests == the carried files), with csrc/triattn_m1_xla.cu restating that kernel's torch host code; C ABI shared with the launcher."""
    from opt_core.kernels.triattn_xla import _native
    kdir = os.path.join(KERNELS, "triattn", "triattn_native", "pkg", "v11", "triattn_pkg", "cuda_b", "csrc")
    assert os.path.isfile(os.path.join(kdir, "m1", "triattn_m1_sm90.cuh")) and os.path.isfile(os.path.join(kdir, "fa3_utils.h"))
    tu = open(os.path.join(PKG, "csrc", "triattn_m1_xla.cu"), encoding="utf-8").read()
    assert '#include "triattn_m1_sm90.cuh"' in tu and "launch_m1x<Traits<0>>" in tu and "launch_m1x<Traits<1024>>" in tu and "#include <torch" not in tu and "at::Tensor" not in tu
    for k in ("stage_bias_m1_kernel", "mask_words_kernel", "mask_rows_kernel", "uniform_rows_kernel"):       # that kernel's staging kernels, restated
        assert k in tu and k in open(os.path.join(kdir, "m1", "m1_binding.cu"), encoding="utf-8").read()
    abi = open(os.path.join(PKG, "csrc", "triattn_m1_abi.h"), encoding="utf-8").read()
    assert "TRIATTN_M1_ABI_VERSION 1" in abi and "triattn_m1_xla_fwd" in abi and '#include "triattn_m1_abi.h"' in open(os.path.join(PKG, "csrc", "cubin_launch.cc"), encoding="utf-8").read()
    ent = _native.lib_entry()
    assert ent is not None, "manifest has no role m1_fwd library"
    src = ent["sources_sha256"]
    assert src["kernels/triattn/triattn_native/pkg/v11/triattn_pkg/cuda_b/csrc/m1/triattn_m1_sm90.cuh"] == _sha(os.path.join(kdir, "m1", "triattn_m1_sm90.cuh"))
    assert src["kernels/triattn/triattn_native/pkg/v11/triattn_pkg/cuda_b/csrc/fa3_utils.h"] == _sha(os.path.join(kdir, "fa3_utils.h"))
    assert src["triattn_m1_xla.cu"] == _sha(os.path.join(PKG, "csrc", "triattn_m1_xla.cu")) and ent["cudart"] == "static" and ent["arch"] == "sm_90a"
    # scratch layout the handler and the library agree on
    sh = _native.scratch_shapes(1, 700, 4, 700, True)
    assert [n for n, _, _ in sh] == ["bias_staged", "fix", "fix_total", "words", "keyany", "rowkind", "kcend", "kcstart", "rowkc0", "rowkc1", "counts"]
    nq = 6; W4 = 28
    assert sh[0][1] == (4 * nq * W4 * 4096,) and sh[1][1] == (1 + 3 * nq * (234 + 1) * 4,) and [n for n, _, _ in _native.scratch_shapes(1, 700, 4, 700, False)] == ["bias_staged", "fix", "fix_total"]
    assert _native.cannot_serve("8.0", "bf16", 32, 640, 640, 640, 4, 1, True) and _native.cannot_serve("9.0", "fp32", 32, 640, 640, 640, 4, 1, True) and _native.cannot_serve("9.0", "bf16", 32, 640, 512, 640, 4, 1, True)


def test_launcher_1_5_generic_targets_are_built_and_wired():
    """launcher 1.5: the generic self-describing cubin target (named roots, by-value struct parameters, tensor maps encoded at call time, SM-count grid rule) and
    the cuBLAS strided-batched GEMM target that kernels/trimul_xla uses; both launcher builds carry them and keep the old glibc / libstdc++ floor."""
    src = open(os.path.join(PKG, "csrc", "cubin_launch.cc"), encoding="utf-8").read()
    for sym in ("XlaCubinCall", "XlaCublasBgemm", "txla_add_root", "txla_set_cublas", "cuTensorMapEncodeTiled", "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT"):
        assert sym in src, sym
    assert "dlopen" not in src                                                     # the cuBLAS entry points come from the Python side (ctypes), not dlopen (glibc floor)
    lsrc = open(os.path.join(PKG, "_launch.py"), encoding="utf-8").read()
    for name in ("def cubin_call(", "def cublas_bgemm_nt(", "def add_root(", "def struct_token(", "def tmap_token(", "def install_cublas(", 'TARGET_GENERIC = "xla_cubin_call"', 'TARGET_BGEMM = "xla_cublas_bgemm_nt"'):
        assert name in lsrc, name
    assert _launch.struct_token(16, ["b0", "q3"]) == "S16:b0;q3" and _launch.f32_token(0.125) == "f0x1.0000000000000p-3"
    assert _launch.tmap_token("bf16", [128, 400, 400], [256, 102400], [64, 64, 2]) == "dt=bf16,dims=128/400/400,str=256/102400,box=64/64/2,swz=128,l2=128,oob=0"
    for b in X.manifest()["binaries"]:
        if b["kind"] == "launcher":
            req = b["requires"]
            assert tuple(int(x) for x in req["GLIBC_max"].split(".")) <= (2, 17) and tuple(int(x) for x in req["GLIBCXX_max"].split(".")) <= (3, 4, 21), b["file"]
            assert b["sources_sha256"]["cubin_launch.cc"] == hashlib.sha256(open(os.path.join(PKG, "csrc", "cubin_launch.cc"), "rb").read()).hexdigest(), b["file"]


def test_cuda_80_row_is_the_carried_sm80_member_and_refuses_by_name():
    """Row cuda_80: kernels/triattn/triattn_native generation 11's sm_80 kernel directory compiled unmodified into bin/cuda/sm_80/libtriattn_sm80_xla.so behind a C ABI;
    served on cc 8.0 for bf16 head_dim 16/32/64; every other case is refused by name with the fallback (the selection then walks to k2b_aot)."""
    from opt_core.kernels.triattn_xla import _sm80
    ent = _sm80.lib_entry()
    assert ent is not None and ent["file"] == "bin/cuda/sm_80/libtriattn_sm80_xla.so" and ent["arch"] == "sm_80" and ent["entry"] == "triattn_sm80_xla_fwd"
    assert os.path.isfile(os.path.join(PKG, ent["file"])) and hashlib.sha256(open(os.path.join(PKG, ent["file"]), "rb").read()).hexdigest() == ent["sha256"]
    req = ent["requires"]                                                              # static cudart: the same libc / libstdc++ floor as the sm_90a libraries beside it
    assert tuple(int(x) for x in req["GLIBC_max"].split(".")) <= (2, 34) and tuple(int(x) for x in req["GLIBCXX_max"].split(".")) <= (3, 4, 21)
    srcs = ent["sources_sha256"]
    carried = [k for k in srcs if k.startswith("kernels/triattn/triattn_native/pkg/")]
    assert sorted(os.path.basename(k) for k in carried) == sorted(["inst_d16.cu", "inst_d32.cu", "inst_d64.cu", "inst_sm80.h", "launch_sm80.cuh", "triattn_sm80.cuh"])
    import importlib.util
    spec = importlib.util.spec_from_file_location("txla_build", os.path.join(PKG, "csrc", "build.py")); B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
    if os.path.isdir(B.SM80_SRC):                                                       # the kernel directory is reachable from this checkout: digests must match the record
        for k in carried:
            assert hashlib.sha256(open(os.path.join(B.SM80_SRC, os.path.basename(k)), "rb").read()).hexdigest() == srcs[k], k
    for k in ("triattn_sm80_xla.cu", "triattn_sm80_abi.h"):
        assert hashlib.sha256(open(os.path.join(PKG, "csrc", k), "rb").read()).hexdigest() == srcs[k], k
    tu = open(os.path.join(PKG, "csrc", "triattn_sm80_xla.cu"), encoding="utf-8").read()
    assert "#include <torch" not in tu and "at::Tensor" not in tu and "stage_bias_kernel<" in tu and "mask_rows_kernel<" in tu and "in.run(a, variant, small, st, &stage)" in tu
    abi = open(os.path.join(PKG, "csrc", "triattn_sm80_abi.h"), encoding="utf-8").read()
    assert "TRIATTN_SM80_ABI_VERSION 1" in abi and "typedef struct TriattnSm80Call" in abi
    # fix-buffer sizing (trace time, no device) == the kernel directory's rule: FIX_LIST + 3 * max over both geometries of ceil(S/BM)*ceil(N/R)*B*H
    assert _sm80.fix_elems(1, 700, 4, 700, 32) == 3 + 3 * max(6 * 175 * 4, 11 * 175 * 4) and _sm80.fix_elems(2, 129, 4, 129, 16) == 3 + 3 * max(2 * 65 * 8, 3 * 33 * 8)
    names = [n for n, _s, _d in _sm80.scratch_shapes(1, 64, 4, 64, 32, True, True)]
    assert names == ["lse", "bias_staged", "fix", "census", "keyany", "rows", "maskw", "rgflag"]
    # selection: cc 8.0 bf16 -> cuda_80 first; fp32 / other 8.x / head_dim 128 -> refused by name, k2b_aot or the stock statement named
    os.environ.pop("MODEL_OPT_LEVERS_OFF", None)
    row, _cell = X.select("8.0", "bfloat16", 32, S=768, N=768, H=4, B=1, has_mask=True)
    assert row == "cuda_80"
    row, _cell = X.select("8.0", "bfloat16", 16, S=256, N=256, H=4, B=1, has_mask=True)
    assert row == "cuda_80"
    row, _cell = X.select("8.0", "float32", 32, S=768, N=768, H=4, B=1, has_mask=True)
    assert row == "k2b_aot" and "bf16 only" in X.LAST_PASSED.get("cuda_80", "")
    row, _cell = X.select("8.6", "bfloat16", 32, S=768, N=768, H=4, B=1, has_mask=True)
    assert row == "k2b_aot"
    with pytest.raises(X.Refused) as e:
        X.select("8.0", "bfloat16", 32, S=768, N=768, H=4, B=1, has_mask=True, impl="cuda_80", SK=512)
    assert "keys == queries" in str(e.value) and X.FALLBACK in str(e.value)
    os.environ["MODEL_OPT_LEVERS_OFF"] = "triattn_xla:cuda_80"
    try:
        row, _cell = X.select("8.0", "bfloat16", 32, S=768, N=768, H=4, B=1, has_mask=True)
        assert row == "k2b_aot"
    finally:
        os.environ.pop("MODEL_OPT_LEVERS_OFF", None)
