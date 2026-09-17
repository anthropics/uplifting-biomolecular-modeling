"""opt_core.seq.warm_witness: snapshot / cache_witness / witness_line / CompileCounter (no GPU, no triton required)."""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import time
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "opt_core")
SRC = os.path.join(PKG, "seq", "warm_witness.py")


def _load():
    """Import through the package when opt_core.seq is a package here; else load the file under a stub sub-package (this branch carries
    warm_witness.py only — opt_core/seq/__init__.py lives on the ARCH branch)."""
    sys.path.insert(0, os.path.dirname(PKG))
    try:
        return importlib.import_module("opt_core.seq.warm_witness")
    except ImportError:
        import opt_core  # noqa: F401
        if "opt_core.seq" not in sys.modules:
            pkg = types.ModuleType("opt_core.seq")
            pkg.__path__ = [os.path.join(PKG, "seq")]
            pkg.__package__ = "opt_core.seq"
            sys.modules["opt_core.seq"] = pkg
        spec = importlib.util.spec_from_file_location("opt_core.seq.warm_witness", SRC)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod


ww = _load()


def _write(p, data: bytes):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as fh:
        fh.write(data)


def test_snapshot_missing_dir_and_only(tmp_path):
    assert ww.snapshot(None) == {} and ww.snapshot(str(tmp_path / "absent")) == {}
    _write(str(tmp_path / "a" / "k.cubin"), b"\x00\x01")
    _write(str(tmp_path / "b.ptx"), b"ptx")
    s = ww.snapshot(str(tmp_path))
    assert set(s) == {os.path.join("a", "k.cubin"), "b.ptx"}
    assert s["b.ptx"]["bytes"] == 3 and len(s["b.ptx"]["sha256"]) == 64
    assert set(ww.snapshot(str(tmp_path), only=["b.ptx"])) == {"b.ptx"}


def test_hit_when_nothing_moves(tmp_path):
    _write(str(tmp_path / "x.cubin"), b"abc")
    before = ww.snapshot(str(tmp_path))
    w = ww.cache_witness(before, ww.snapshot(str(tmp_path)), real_compiles=0)
    assert w["verdict"] == "HIT" and w["n_entries"] == 1 and w["n_new"] == 0 and w["groups"] == [] and w["real_compiles"] == 0


def test_miss_on_new_file_names_the_kernel_group(tmp_path):
    _write(str(tmp_path / "x.cubin"), b"abc")
    before = ww.snapshot(str(tmp_path))
    _write(str(tmp_path / "h1" / "__grp__fused_ln_kernel.json"), b"{}")
    _write(str(tmp_path / "h1" / "fused_ln_kernel.cubin"), b"\x01")
    w = ww.cache_witness(before, ww.snapshot(str(tmp_path)))
    assert w["verdict"] == "MISS" and w["n_new"] == 2 and w["groups"] == ["fused_ln_kernel"] and w["real_compiles"] is None


def test_miss_on_rewrite_same_bytes_and_on_missing(tmp_path):
    p = str(tmp_path / "x.cubin")
    _write(p, b"abc")
    before = ww.snapshot(str(tmp_path))
    time.sleep(0.01)
    _write(p, b"abc")                                   # same bytes, new mtime = a rewrite = MISS by the content rule
    os.utime(p, ns=(before["x.cubin"]["mtime_ns"] + 10_000_000, before["x.cubin"]["mtime_ns"] + 10_000_000))
    w = ww.cache_witness(before, ww.snapshot(str(tmp_path)))
    assert w["verdict"] == "MISS" and w["n_changed"] == 1
    os.remove(p)
    w2 = ww.cache_witness(before, ww.snapshot(str(tmp_path)))
    assert w2["verdict"] == "MISS" and w2["n_missing"] == 1


def test_own_entries_scope_the_judgement(tmp_path):
    _write(str(tmp_path / "kit.cubin"), b"k")
    _write(str(tmp_path / "other.cubin"), b"o")
    before = ww.snapshot(str(tmp_path), only=["kit.cubin"])
    os.remove(str(tmp_path / "other.cubin"))            # someone else's entry vanishing is not the kit's miss
    w = ww.cache_witness(before, ww.snapshot(str(tmp_path)), own=["kit.cubin"])
    assert w["verdict"] == "HIT"
    w2 = ww.cache_witness(before, ww.snapshot(str(tmp_path)), own=["kit.cubin"], real_compiles=2)
    assert w2["verdict"] == "MISS" and w2["real_compiles"] == 2


def test_witness_line_grammar(tmp_path):
    _write(str(tmp_path / "x.cubin"), b"abc")
    s = ww.snapshot(str(tmp_path))
    line = ww.witness_line("enginea-opt", "TRITON_CACHE_DIR", ww.cache_witness(s, s, real_compiles=0), key="torch2.8.0-cu128-sm90", dir_rule="keyed")
    assert line == "[enginea-opt] CACHE name=TRITON_CACHE_DIR verdict=HIT entries=1 changed=0 missing=0 new=0 unreadable=0 groups=none compiles=0 key=torch2.8.0-cu128-sm90 dir_rule=keyed"
    na = ww.witness_line("engineb-opt", "TRITON_CACHE_DIR", None, reason="no compiled code on this route")
    assert na == "[engineb-opt] CACHE name=TRITON_CACHE_DIR verdict=n/a reason=no compiled code on this route"


def test_line_fields_is_the_str_to_str_hook(tmp_path):
    _write(str(tmp_path / "x.cubin"), b"abc")
    s = ww.snapshot(str(tmp_path))
    f = ww.line_fields("TRITON_CACHE_DIR", ww.cache_witness(s, s), key="torch2.8.0-cu128-sm90")
    assert list(f) == ["name", "verdict", "entries", "changed", "missing", "new", "unreadable", "groups", "compiles", "key"]
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in f.items())
    assert f["groups"] == "none" and f["compiles"] == "none" and f["verdict"] == "HIT"
    assert ww.line_fields("CUDA_CACHE_PATH", None) == {"name": "CUDA_CACHE_PATH", "verdict": "n/a", "reason": "no cache directory on this route"}


def test_compile_counter_wraps_and_restores_fake_triton(monkeypatch):
    """A stand-in `triton.runtime.jit` with a module-level compile and a JITFunction class: the counter wraps both, times calls, restores."""
    triton = types.ModuleType("triton"); runtime = types.ModuleType("triton.runtime"); jit = types.ModuleType("triton.runtime.jit")

    class JITFunction:
        def __init__(self, name):
            self.name = name
            self.compile = lambda src, *a, **k: ("compiled", src)      # bound per instance, as newer Tritons do

    def compile(src, *a, **k):  # noqa: A001
        time.sleep(0.02)
        return ("compiled", src)
    jit.compile = compile; jit.JITFunction = JITFunction; runtime.jit = jit; triton.runtime = runtime
    for name, mod in (("triton", triton), ("triton.runtime", runtime), ("triton.runtime.jit", jit)):
        monkeypatch.setitem(sys.modules, name, mod)
    kitmod = types.ModuleType("mykit.kernels"); kitmod.k1 = JITFunction("k1")
    monkeypatch.setitem(sys.modules, "mykit.kernels", kitmod)

    ctr = ww.CompileCounter(real_compile_s=0.01, module_prefixes=("mykit.",)).install()
    assert "module global" in ctr.report()["bound"] and "1 kernel instance" in ctr.report()["bound"]
    sys.modules["triton.runtime.jit"].compile("srcA")          # slow (>= 0.01 s) = a real compile
    kitmod.k1.compile("srcB")                                  # instant = a cache load
    rep = ctr.report()
    assert rep["calls"] == 2 and rep["real_compiles"] == 1 and rep["max_call_s"] >= 0.01
    ctr.uninstall()
    assert sys.modules["triton.runtime.jit"].compile is compile


def test_compile_counter_without_triton_is_recorded_not_raised(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)          # import triton -> ImportError
    ctr = ww.CompileCounter().install()
    assert ctr.report()["bound"].startswith("NOT bound (") and ctr.report()["real_compiles"] == 0


def test_unreadable_file_is_a_named_bucket_never_a_hit(tmp_path, monkeypatch):
    _write(str(tmp_path / "ok.cubin"), b"abc")
    _write(str(tmp_path / "locked.cubin"), b"xyz")
    import builtins
    real_open = builtins.open

    def guarded(path, *a, **k):                        # the process may read everything but one cache file (a permission / lock class error)
        if str(path).endswith("locked.cubin") and "b" in (a[0] if a else k.get("mode", "r")):
            raise PermissionError(13, "Permission denied", str(path))
        return real_open(path, *a, **k)
    monkeypatch.setattr(builtins, "open", guarded)
    before = ww.snapshot(str(tmp_path))
    assert before["locked.cubin"]["unreadable"] == "PermissionError" and before["locked.cubin"]["sha256"] is None
    assert before["ok.cubin"]["unreadable"] is None
    w = ww.cache_witness(before, ww.snapshot(str(tmp_path)), real_compiles=0)
    assert w["verdict"] == "MISS" and w["n_unreadable"] == 1 and w["unreadable"] == ["locked.cubin"] and w["n_changed"] == 0
    assert ww.line_fields("TRITON_CACHE_DIR", w)["unreadable"] == "1"
