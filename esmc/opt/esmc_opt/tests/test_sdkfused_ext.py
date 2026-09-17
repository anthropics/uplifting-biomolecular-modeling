"""The fused item's extension loader (esmc_opt/kits/residual_ln: load_ext + build.py + EXTENSION.json), on CPU with the builder and
the import stubbed: a runtime whose per-user cache holds no build builds kernels.cu ONCE by the EXTENSION.json recipe (one info line),
and a later process imports that build from the cache (one info line); the install step's build_cache() fills the same cache up
front; no toolchain / a failed build is a KitRefused naming what is missing / the log — and the package's exit rule names the escapes
around it."""
import hashlib
import json
import os
import sys

import pytest

from esmc_opt import report

from ._paths import KITS

KIT_DIR = os.path.join(KITS, "residual_ln")
FOREIGN = {"torch": "2.8.0+cu128", "cuda": "12.8", "sm": "89", "python": "3.11"}      # some runtime's ABI key


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    """The item module, its process state fresh, its cache home under tmp."""
    import importlib
    mod = importlib.import_module("esmc_opt.kits.residual_ln")
    build = importlib.import_module("esmc_opt.kits.residual_ln.build")
    monkeypatch.setattr(mod, "_state", {"ext": None, "ext_load": None, "lin": {}, "inv_b": {}})
    monkeypatch.setattr(mod, "CACHE_HOME", str(tmp_path / "cache"))
    mod.build_mod = build
    yield mod


class _FakeExt:
    def __init__(self, path):
        self.path = path

    def residual_bf16(self, *a):
        return ("residual_bf16", self.path)


def _fake_build_fn(calls):
    def fake_build(build_dir, sm, kit_dir=None, verbose=False):
        os.makedirs(build_dir, exist_ok=True)
        so = os.path.join(build_dir, "esmc_v3_sdkfused_r1.so")
        with open(so, "wb") as f:
            f.write(b"\x7fELF-fake-" + sm.encode())
        calls.append({"build_dir": build_dir, "sm": sm})
        return {"module": _FakeExt(so), "so_path": so, "so_sha256": hashlib.sha256(open(so, "rb").read()).hexdigest(), "size": os.path.getsize(so),
                "wall_s": 12.5, "arch": f"TORCH_CUDA_ARCH_LIST={sm[:-1]}.{sm[-1]}", "nvcc": "Build cuda_fake", "recipe": {"tool": "torch.utils.cpp_extension.load"}}
    return fake_build


# ----------------------------------------------------------------------------------------------------------- EXTENSION.json = the recipe
def test_pin_names_the_recipe_only(sdk):
    p = sdk.build_mod.pin(KIT_DIR)
    assert p["extension"] == "esmc_v3_sdkfused_r1" and p["source"] == "kernels.cu" and os.path.isfile(os.path.join(KIT_DIR, "kernels.cu"))
    assert set(p) == {"extension", "source", "build"}, "the kit ships the recipe, no binaries"
    assert not [f for f in os.listdir(KIT_DIR) if f.endswith((".so", ".pyd"))]
    r = sdk.build_mod.recipe(KIT_DIR)
    assert r["tool"] == "torch.utils.cpp_extension.load" and r["extra_cuda_cflags"] == ["-O3"] and r["extra_cflags"] == [] and r["extra_ldflags"] == []
    assert sdk.SERVED_SM == ("80", "90")


def test_key_spellings(sdk):
    B = sdk.build_mod
    assert B.key_dir({"torch": "2.11.0+cu130", "cuda": "13.0", "sm": "90", "python": "3.12"}) == "torch2.11.0+cu130__cuda13.0__sm90__py3.12"
    assert B.arch_of("90") == "9.0" and B.arch_of("80") == "8.0" and B.arch_of("120") == "12.0"
    with pytest.raises(ValueError):
        B.arch_of("?")
    assert sdk.cache_dir(FOREIGN, "ab" * 32) == os.path.join(sdk.CACHE_HOME, "torch2.8.0+cu128__cuda12.8__sm89__py3.11__abababababab")
    tag = B.cache_tag(KIT_DIR)
    assert len(tag) == 64 and tag != B.source_sha256(KIT_DIR)          # source bytes + recipe, not the source alone


# ------------------------------------------------------------------------------------------------------------------ the two paths + the install build
def test_first_process_builds_once_then_later_processes_hit_the_cache(sdk, monkeypatch, capsys):
    monkeypatch.setattr(sdk, "runtime_key", lambda: dict(FOREIGN))
    monkeypatch.setattr(sdk, "_import_so", lambda path: _FakeExt(path))
    monkeypatch.setattr(sdk.build_mod, "toolchain", lambda: {"missing": [], "nvcc_version": "Build cuda_fake"})
    built = []
    monkeypatch.setattr(sdk.build_mod, "build", _fake_build_fn(built))
    cdir = sdk.cache_dir(FOREIGN, sdk.build_mod.cache_tag(KIT_DIR))
    ext = sdk.load_ext()                                                                    # process 1: builds
    out = capsys.readouterr().out.splitlines()
    assert len(built) == 1 and built[0] == {"build_dir": os.path.join(cdir, "build"), "sm": "89"}
    assert out == [sdk.build_line(FOREIGN, 12.5, cdir)] and out[0].startswith("[residual_ln] extension torch2.8.0+cu128__cuda12.8__sm89__py3.11: built from kernels.cu in 12.5s (cached at ")
    rec = sdk.ext_load()
    assert rec["source"] == "built" and rec["cache_dir"] == cdir and rec["path"] == ext.path == os.path.join(cdir, "build", "esmc_v3_sdkfused_r1.so")
    meta = json.load(open(os.path.join(cdir, sdk.BUILT_FILE)))
    assert meta["key"] == FOREIGN and meta["so_rel"] == os.path.join("build", "esmc_v3_sdkfused_r1.so") and meta["cu_sha256"] == sdk.build_mod.source_sha256(KIT_DIR)
    monkeypatch.setattr(sdk, "_state", {"ext": None, "ext_load": None, "lin": {}, "inv_b": {}})     # process 2: the cache
    ext2 = sdk.load_ext()
    out2 = capsys.readouterr().out.splitlines()
    assert len(built) == 1, "a cached build is imported, never rebuilt"
    assert len(out2) == 1 and out2[0] == sdk.cache_line(FOREIGN, sdk.ext_load()["wall_s"], cdir) and out2[0].startswith("[residual_ln] extension torch2.8.0+cu128__cuda12.8__sm89__py3.11: loaded the cached build at ")
    assert sdk.ext_load()["source"] == "cache" and ext2.path == ext.path
    with open(ext.path, "ab") as f:                                                         # a damaged cached binary is rebuilt, not imported
        f.write(b"junk")
    monkeypatch.setattr(sdk, "_state", {"ext": None, "ext_load": None, "lin": {}, "inv_b": {}})
    sdk.load_ext(); capsys.readouterr()
    assert len(built) == 2 and sdk.ext_load()["source"] == "built"


def test_no_build_and_no_toolchain_is_refused_by_name(sdk, monkeypatch, capsys):
    monkeypatch.setattr(sdk, "runtime_key", lambda: dict(FOREIGN))
    monkeypatch.setattr(sdk.build_mod, "toolchain", lambda: {"missing": ["nvcc (the CUDA toolkit compiler; CUDA_HOME=None)", "ninja"], "nvcc_version": None})
    built = []
    monkeypatch.setattr(sdk.build_mod, "build", _fake_build_fn(built))
    with pytest.raises(sdk.KitRefused) as ei:
        sdk.load_ext()
    msg = str(ei.value)
    assert msg.startswith("the extension for torch2.8.0+cu128__cuda12.8__sm89__py3.11 is not built and cannot be built here — missing nvcc (the CUDA toolkit compiler; CUDA_HOME=None), ninja")
    assert "cannot engage" in msg and "run.sh install" in msg
    assert built == [] and capsys.readouterr().out == "" and sdk.ext_load() is None
    line = report.partial_refused_line(report.partial_detail(["fused"], {"fused": {"applied": False, "refused": f"KitRefused: {msg}"}}))   # the mode refuses by name: the lever cannot run here (no binary, no toolchain)
    assert line.startswith("[esmc-opt] NOT ACTIVE: partial activation — fused: KitRefused: the extension for ") and "ESMC_OPT=off" in line and "the load is refused" in line


def test_a_failed_build_is_refused_with_its_log(sdk, monkeypatch, capsys):
    monkeypatch.setattr(sdk, "runtime_key", lambda: dict(FOREIGN))
    monkeypatch.setattr(sdk.build_mod, "toolchain", lambda: {"missing": [], "nvcc_version": "Build cuda_fake"})

    def broken(build_dir, sm, kit_dir=None, verbose=False):
        raise RuntimeError("Error building extension 'esmc_v3_sdkfused_r1': nvcc fatal : Unsupported gpu architecture 'compute_89'\n[2/2] …")
    monkeypatch.setattr(sdk.build_mod, "build", broken)
    with pytest.raises(sdk.KitRefused) as ei:
        sdk.load_ext()
    msg = str(ei.value)
    cdir = sdk.cache_dir(FOREIGN, sdk.build_mod.cache_tag(KIT_DIR))
    assert msg.startswith("the build of kernels.cu for torch2.8.0+cu128__cuda12.8__sm89__py3.11 failed — RuntimeError: Error building extension")
    assert f"(full output: {os.path.join(cdir, 'build_error.txt')})" in msg and "compute_89" in open(os.path.join(cdir, "build_error.txt")).read()
    assert not os.path.exists(os.path.join(cdir, sdk.BUILT_FILE)) and capsys.readouterr().out == ""


def test_unwritable_cache_home_builds_into_a_temporary_dir(sdk, monkeypatch, capsys, tmp_path):
    ro = tmp_path / "ro"
    monkeypatch.setattr(sdk, "CACHE_HOME", str(ro / "cache"))
    makedirs = os.makedirs

    def refusing_makedirs(path, *a, **k):      # the user's cache home cannot be created (a read-only home): as the OS would raise
        if str(path).startswith(str(ro)):
            raise PermissionError(13, "Permission denied", str(path))
        return makedirs(path, *a, **k)
    monkeypatch.setattr(os, "makedirs", refusing_makedirs)
    monkeypatch.setattr(sdk, "runtime_key", lambda: dict(FOREIGN))
    monkeypatch.setattr(sdk, "_import_so", lambda path: _FakeExt(path))
    monkeypatch.setattr(sdk.build_mod, "toolchain", lambda: {"missing": [], "nvcc_version": "Build cuda_fake"})
    built = []
    monkeypatch.setattr(sdk.build_mod, "build", _fake_build_fn(built))
    sdk.load_ext()
    rec = sdk.ext_load()
    assert rec["source"] == "built" and not rec["cache_dir"].startswith(str(ro)) and os.path.isfile(rec["path"])
    assert capsys.readouterr().out.splitlines() == [sdk.build_line(FOREIGN, 12.5, rec["cache_dir"])]


def test_install_build_fills_the_cache_load_ext_reads(sdk, monkeypatch, capsys):
    """build_cache(): one build per capability into the per-user cache (device 0's by default; every served capability with no device);
    a capability already built is left alone; load_ext() then imports without building."""
    monkeypatch.setattr(sdk, "runtime_key", lambda: dict(FOREIGN, sm="?"))                 # no CUDA device: an image builder
    monkeypatch.setattr(sdk, "_import_so", lambda path: _FakeExt(path))
    monkeypatch.setattr(sdk.build_mod, "toolchain", lambda: {"missing": [], "nvcc_version": "Build cuda_fake"})
    built = []
    monkeypatch.setattr(sdk.build_mod, "build", _fake_build_fn(built))
    recs = sdk.build_cache()
    assert [b["sm"] for b in built] == ["80", "90"] and [r["source"] for r in recs] == ["built", "built"]
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2 and all(l.startswith("[residual_ln] extension torch2.8.0+cu128__cuda12.8__sm") and "built from kernels.cu" in l for l in lines)
    recs2 = sdk.build_cache(["90"])                                                         # already built: left alone
    assert len(built) == 2 and recs2[0]["source"] == "cache" and "already built" in capsys.readouterr().out
    monkeypatch.setattr(sdk, "runtime_key", lambda: dict(FOREIGN, sm="90"))                # the run-time process on an sm90 card: imports, builds nothing
    sdk.load_ext()
    assert len(built) == 2 and sdk.ext_load()["source"] == "cache" and capsys.readouterr().out.startswith("[residual_ln] extension torch2.8.0+cu128__cuda12.8__sm90__py3.11: loaded the cached build at ")


def test_ops_go_through_load_ext(sdk, monkeypatch, capsys):
    """residual / residual_ln_qkv / ops() take the extension from load_ext (the one loader); nothing else imports a binary."""
    src = open(os.path.join(KIT_DIR, "__init__.py")).read()
    assert src.count("spec_from_file_location") == 1
    monkeypatch.setattr(sdk, "runtime_key", lambda: dict(FOREIGN))
    monkeypatch.setattr(sdk, "_import_so", lambda path: _FakeExt(path))
    monkeypatch.setattr(sdk.build_mod, "toolchain", lambda: {"missing": [], "nvcc_version": "Build cuda_fake"})
    built = []
    monkeypatch.setattr(sdk.build_mod, "build", _fake_build_fn(built))
    assert set(sdk.ops(None)) == {"residual", "residual_ln_qkv"}
    monkeypatch.setattr(sdk, "inv_b_of", lambda sf: 0.5)
    cdir = sdk.cache_dir(FOREIGN, sdk.build_mod.cache_tag(KIT_DIR))
    assert sdk.residual("x", "y", 2.0) == ("residual_bf16", os.path.join(cdir, "build", "esmc_v3_sdkfused_r1.so")) and len(built) == 1
