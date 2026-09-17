"""lnstream (opendde_opt/lnstream.py), CPU mechanics: the source transform on the pinned stock snapshot (25 stream-less launch sites put on the
current stream, the 3 streamed sites kept, device text unchanged), the pin table = the shipped stock sources, the refusals by name, the
content-keyed source file written once, and the loader's state machine (serving counts calls; a named refusal falls back to upstream's own
loader and is reported as the lever's fallback). GPU equality (bitwise vs the legacy build, capture self-test) is the GPU row, not this file."""
import os
import re
import sys
import types

import pytest

from opendde_opt import lnstream, modes, ran, registry

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.normpath(os.path.join(HERE, "..", "..", "..", "stock", "src", "opendde", "model", "layer_norm", "kernel"))


@pytest.fixture
def clean(monkeypatch, tmp_path):
    monkeypatch.setattr(lnstream, "STATS", {**lnstream.STATS, "installed": False, "armed": False, "state": "off", "reason": None, "calls": 0,
                                            "jit_s": None, "patch": None})
    monkeypatch.setattr(lnstream, "_ST", {"ext": None, "legacy": None, "orig_loader": None, "tried": False, "cells": {}})
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "torch_ext"))
    yield tmp_path


def test_the_pins_are_the_shipped_stock_sources():
    assert os.path.isdir(KDIR), KDIR
    assert lnstream.check_source(KDIR) == lnstream.PINS                      # stock/src carries the pinned 1.1.1 kernel sources, sha256 for sha256
    assert set(lnstream.PINS) == set(lnstream.KERNEL_FILES)


def test_transform_puts_every_streamless_launch_on_the_current_stream_and_keeps_device_text():
    src = open(os.path.join(KDIR, lnstream.CU)).read()
    text, census = lnstream.to_current_stream(src)
    assert {k: census[k] for k in lnstream.EXPECTED_SITES} == lnstream.EXPECTED_SITES and census["other"] == 0
    sites = re.findall(r"<<<(.*?)>>>", text, re.S)
    assert len(sites) == 28 and all(len(lnstream._split_args(s)) == 4 for s in sites)          # every launch now names shmem + stream
    assert sum(s.rstrip().endswith(lnstream.CURRENT_STREAM) for s in sites) == 25             # the 25 formerly stream-less sites: the current stream
    assert sum(lnstream._split_args(s)[-1] == "stream" for s in sites) == 3                    # upstream's own three Step2 sites untouched
    assert lnstream.device_fingerprint(src) == lnstream.device_fingerprint(text)               # nothing outside <<<...>>> changed: same device code text
    assert "LayerNormForwardV2<float, float4><<<grid, block, 0, at::cuda::getCurrentCUDAStream().stream()>>>(" in text
    assert text.count("\n") == src.count("\n")


def test_a_source_off_the_pin_or_with_other_launch_sites_is_refused_by_name(tmp_path):
    for f in lnstream.KERNEL_FILES:
        with open(os.path.join(KDIR, f), "rb") as fh, open(tmp_path / f, "wb") as out:
            out.write(fh.read() + (b"\n// local edit\n" if f == lnstream.CU else b""))
    with pytest.raises(lnstream.LnstreamRefused, match=r"source_not_pinned:layer_norm_cuda_kernel\.cu:"):
        lnstream.check_source(str(tmp_path))
    os.remove(tmp_path / "layer_norm_cuda.cpp")                                   # the first file of KERNEL_FILES: checked (and named) first
    with pytest.raises(lnstream.LnstreamRefused, match=r"source_missing:layer_norm_cuda\.cpp"):
        lnstream.check_source(str(tmp_path))
    src = open(os.path.join(KDIR, lnstream.CU)).read()
    with pytest.raises(lnstream.LnstreamRefused, match=r"launch_sites_differ"):
        lnstream.to_current_stream(src.replace("<<<grid, block>>>", "<<<grid, block, 0, s>>>", 1))


def test_the_transformed_source_is_written_once_at_a_content_keyed_path(clean):
    p1, census, fp = lnstream.write_source(KDIR)
    assert fp is True and census["streamless_2"] == 25
    assert p1.startswith(os.path.join(os.environ["TORCH_EXTENSIONS_DIR"], lnstream.EXT_CS + "_src")) and p1.endswith(lnstream.CU)
    m1 = os.stat(p1).st_mtime_ns
    p2, _c, _f = lnstream.write_source(KDIR)
    assert p2 == p1 and os.stat(p2).st_mtime_ns == m1                        # unchanged content: not rewritten (ninja's incremental build stays warm)


def _fake_loader_module(monkeypatch):
    lm = types.ModuleType(lnstream.TARGET)
    lm.__file__ = os.path.join(os.path.dirname(KDIR), "layer_norm.py")
    lm.fast_layer_norm_cuda_v2 = None
    lm._fast_layer_norm_load_attempted = False
    legacy = types.SimpleNamespace(__file__="/legacy/fast_layer_norm_cuda_v2.so", tag="legacy")

    def orig():                                                               # upstream's loader: early return of the bound global once a load was attempted
        if lm._fast_layer_norm_load_attempted:
            return lm.fast_layer_norm_cuda_v2
        lm.fast_layer_norm_cuda_v2 = legacy
        lm._fast_layer_norm_load_attempted = True
        return legacy
    lm._load_fast_layer_norm_cuda_v2 = orig
    monkeypatch.setitem(sys.modules, lnstream.TARGET, lm)
    return lm, legacy, orig


def test_loader_serves_and_counts_after_a_successful_build(clean, monkeypatch):
    lm, legacy, orig = _fake_loader_module(monkeypatch)
    ext = types.SimpleNamespace(__file__="/jit/fast_layer_norm_cuda_v2_cs/fast_layer_norm_cuda_v2_cs.so", tag="cs")
    monkeypatch.setattr(lnstream, "build", lambda loader_module=None: ext)
    f = lnstream.make_loader(orig)
    lm._load_fast_layer_norm_cuda_v2 = f                                                          # as the core's patch installs it
    assert f() is ext and lnstream.serving() and lnstream.STATS["calls"] == 1
    assert lm.fast_layer_norm_cuda_v2 is ext and lm._fast_layer_norm_load_attempted is True     # bound where upstream binds its own
    assert f() is ext and f() is ext and lnstream.STATS["calls"] == 3 and lnstream.bound()
    assert ran.count("lnstream") == 3
    for _ in range(lnstream.COUNT_WINDOW):
        lm._load_fast_layer_norm_cuda_v2()                                                        # through whatever the attribute is now
    assert lm._load_fast_layer_norm_cuda_v2 is orig and lnstream.STATS["loader"].startswith("upstream(")   # the window over: upstream's own loader serves the bound global
    assert lnstream.STATS["calls"] == lnstream.COUNT_WINDOW and lnstream.serving() and lnstream.fallbacks(["lnstream"]) == []
    lm.fast_layer_norm_cuda_v2 = legacy                                                           # something rebinds the global: named at exit
    assert lnstream.fallbacks(["lnstream"]) == ["lnstream:unbound_after_bind"] and not lnstream.serving()
    lm.fast_layer_norm_cuda_v2 = ext                                                             # the ran-or-refuse counter reads the served calls
    assert lnstream.fallbacks(["lnstream"]) == []                                                   # bound by hand (no patch object in this process): no install event to name


def test_a_refused_build_falls_back_to_upstreams_loader_by_name(clean, monkeypatch, capsys):
    lm, legacy, orig = _fake_loader_module(monkeypatch)

    def refuse(loader_module=None):
        raise lnstream.LnstreamRefused("build_failed:RuntimeError:no nvcc")
    monkeypatch.setattr(lnstream, "build", refuse)
    f = lnstream.make_loader(orig)
    assert f() is legacy and not lnstream.serving() and lnstream.STATS["state"] == "fallback"
    assert lnstream.STATS["reason"] == "build_failed:RuntimeError:no nvcc" and lnstream.STATS["calls"] == 0
    assert "LEVER lnstream FELL BACK by name: build_failed:RuntimeError:no nvcc" in capsys.readouterr().err
    assert f() is legacy                                                                            # every later call: upstream's own loader
    assert "lnstream:build_failed:RuntimeError:no nvcc" in lnstream.fallbacks(["lnstream"])         # the kit's PARTIAL word
    assert lnstream.fallbacks(["chunk_lift"]) == []                                                # not planned: no event


def test_install_patches_the_loader_at_the_modules_import(fresh_kit, monkeypatch):
    lnstream.install()
    assert lnstream.STATS["armed"] and not lnstream.STATS["installed"]                             # upstream's layer_norm not imported: armed on the meta path
    lm, legacy, orig = _fake_loader_module(monkeypatch)                                          # a module appearing in sys.modules by hand is not an import:
    lnstream.install()                                                                              # idempotent while armed
    assert lnstream.STATS["armed"]


@pytest.fixture
def fresh_kit(clean, monkeypatch):
    from opt_core import autoload
    monkeypatch.setattr(autoload, "_PATCHES", {})
    saved_meta = list(sys.meta_path)
    monkeypatch.delitem(sys.modules, lnstream.TARGET, raising=False)
    yield
    sys.meta_path[:] = saved_meta


def test_the_lever_rides_every_line_and_is_tabled():
    assert all("lnstream" in ln.levers for ln in modes.LINES.values()), {n: ln.levers for n, ln in modes.LINES.items()}
    lv = registry.LEVERS["lnstream"]
    assert lv.tier == "exact" and lv.switch == "-" and lv.file == "opendde_opt/lnstream.py" and lv.kit == registry.HOUSE
    assert registry.PIN_STATUS["lnstream"][0] == "tested" and "lnstream" in ran.COUNTERS and modes.LEVER_SWITCHES["lnstream"] == ()
    assert ran.engagement("lnstream", {"layernorm": "fast_layernorm"}) == (True, None)
    ok, why = ran.engagement("lnstream", {"layernorm": "torch"})
    assert not ok and "fast_layernorm" in why
