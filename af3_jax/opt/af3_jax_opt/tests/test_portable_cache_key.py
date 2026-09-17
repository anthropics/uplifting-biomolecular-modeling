"""inprocess/portable_cache_key.py: the persistent-cache key's accelerator component becomes the device kind (jax's own fallback), the
box's topology fingerprint no longer enters it; a jax other than the pinned one is refused by name; det reads the CACHEKEY line."""
import hashlib
import importlib.util
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
MOD_PATH = os.path.join(os.path.dirname(HERE), "inprocess", "portable_cache_key.py")


class _Dev:
    def __init__(self, kind, fingerprint):
        self.device_kind, self.fingerprint = kind, fingerprint


class _Topo:
    def __init__(self, devs):
        self._fp = hash(tuple(d.fingerprint for d in devs)) & (2**63 - 1)
    def fingerprint(self):
        return self._fp


def _fake_jax(monkeypatch, version="0.10.2"):
    """A stand-in `jax` + `jax._src.cache_key` with jax 0.10.2's shapes: get() hashes through the MODULE-GLOBAL _hash_accelerator_config
    (topology fingerprint), _hash_devices hashes device_kind — exactly the two functions the shim swaps between."""
    import numpy as np
    jax = types.ModuleType("jax"); jax.__version__ = version
    src = types.ModuleType("jax._src"); ck = types.ModuleType("jax._src.cache_key")
    def _hash_string(h, s): h.update(s.encode("utf-8").strip())
    def _hash_devices(h, devices):
        for d in devices.flat: _hash_string(h, d.device_kind)
    def _hash_accelerator_config(h, accelerators):
        h.update(_Topo(list(accelerators.flat)).fingerprint().to_bytes(8, "big"))
    def get(devices):
        h = hashlib.sha256(); ck._hash_accelerator_config(h, np.array(devices)); return h.hexdigest()
    ck._hash_string, ck._hash_devices, ck._hash_accelerator_config, ck.get = _hash_string, _hash_devices, _hash_accelerator_config, get
    jax._src = src; src.cache_key = ck
    for name, m in (("jax", jax), ("jax._src", src), ("jax._src.cache_key", ck)):
        monkeypatch.setitem(sys.modules, name, m)
    return jax, ck


def _load():
    spec = importlib.util.spec_from_file_location("af3_jax_opt.inprocess.portable_cache_key_undertest", MOD_PATH)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_the_key_ignores_the_topology_fingerprint_and_honours_device_kind(monkeypatch, capsys):
    jax, ck = _fake_jax(monkeypatch)
    box1 = [_Dev("NVIDIA H100 80GB HBM3", 11613387976906732345)]; box2 = [_Dev("NVIDIA H100 80GB HBM3", 7162338684885338182)]
    other = [_Dev("NVIDIA H200", 11613387976906732345)]
    assert ck.get(box1) != ck.get(box2)                                   # jax's own component: two boxes of one model, two keys
    pck = _load()
    assert pck.install() == "device_kind"
    assert ck.get(box1) == ck.get(box2)                                   # after install: one model, one key
    assert ck.get(box1) != ck.get(other)                                  # the model name still enters the key
    h = hashlib.sha256(); ck._hash_devices(h, __import__("numpy").array(box1))
    assert ck.get(box1) == h.hexdigest()                                  # = jax's own device-kind fallback, byte for byte
    out = capsys.readouterr().out
    assert out.count("[af3-jax-opt] CACHEKEY accelerator=device_kind jax=0.10.2 site=jax._src.cache_key._hash_accelerator_config") == 1
    assert pck.install() == "device_kind" and capsys.readouterr().out == ""   # idempotent, one line per process


@pytest.mark.parametrize("version", ["0.10.3", "0.5.3", "0.11.0"])
def test_under_any_other_jax_the_rebinding_stays_off_named(monkeypatch, capsys, version):
    """Another jax than the pinned one: install() refuses by name (nothing patched) and main_guard NAMES it on the CACHEKEY line —
    accelerator=topology jax=<v> reason=… (jax's own box-specific key applies) — and returns; the model process runs on, never exits over it."""
    jax, ck = _fake_jax(monkeypatch, version)
    before = ck._hash_accelerator_config
    pck = _load()
    with pytest.raises(pck.Refused, match="cache key portability shim expects jax 0.10.2, found jax " + version.replace(".", r"\.")):
        pck.install()
    assert ck._hash_accelerator_config is before                          # nothing patched
    capsys.readouterr()
    assert pck.main_guard() == "topology" == pck.TOPOLOGY
    out = capsys.readouterr().out.strip()
    assert out.startswith(f"[af3-jax-opt] CACHEKEY accelerator=topology jax={version} reason=cache_key_portability_shim_expects_jax_0.10.2,_found_jax_{version}") and "\n" not in out
    assert ck._hash_accelerator_config is before                          # still nothing patched
    from af3_jax_opt import det
    assert det.cache_key_word([out]) == "topology" == det.CACHE_KEY_DEFAULT   # the wrapper reads the word off the line: the CACHE when=after line says cache_key=topology


def test_a_cache_key_module_of_another_shape_is_refused(monkeypatch):
    jax, ck = _fake_jax(monkeypatch)
    monkeypatch.delattr(ck, "_hash_devices")
    pck = _load()
    with pytest.raises(pck.Refused, match="different cache_key module"):
        pck.install()


def test_expected_jax_is_the_pinned_jax():
    import json
    pins = json.load(open(os.path.join(os.path.dirname(os.path.dirname(HERE)), "..", "stock", "PINS.json")))
    lock = os.path.join(os.path.dirname(os.path.dirname(HERE)), "..", pins["freeze"]["file"])                  # the pinned stack's one list (environment/requirements.lock)
    pinned = {l.split("==")[0].lower(): l.split("==")[1] for l in open(lock).read().split() if "==" in l and not l.startswith(("#", "-"))}
    assert _load().EXPECTED_JAX == pinned["jax"] and "jax" in [p.lower() for p in pins["check_packages"]]


def test_det_reads_the_cachekey_line():
    from af3_jax_opt import det
    lines = ["something", "[af3-jax-opt] CACHEKEY accelerator=device_kind jax=0.10.2 site=jax._src.cache_key._hash_accelerator_config", "x"]
    assert det.is_cachekey_line(lines[1]) and not det.is_cachekey_line(lines[0])
    assert det.cache_key_word(lines) == "device_kind" and det.cache_key_word(["no such line"]) == det.CACHE_KEY_DEFAULT == "topology"
    assert det.cache_line(None, "after", key="device_kind").endswith(" state=absent cache_key=device_kind")
    assert " cache_key=" not in det.cache_line(None, "before")
