"""opt_core.mem.peak — the allocator-peak probe, CPU only (no torch, no jax: both are stand-ins put into sys.modules): the module imports no
backend and initialises none (a backend that is not ready is never touched), it samples the counters of both families and keeps the maxima,
the record has the schema's fields with the window and the instrument's sha, the env is recorded verbatim and a disagreeing expectation is
a named refusal that changes nothing, per-process records reduce to one pass record (max per field, union window, the sampler's high-water
over that window only), the hook directory carries this file byte for byte and a stock-style process started through it writes its record
at exit without ever importing opt_core."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import types

import pytest

from opt_core.mem import peak


@pytest.fixture(autouse=True)
def _clean_xla_env(monkeypatch):
    """Each test states its own XLA/JAX environment: names another test module exported into this interpreter (a device-count
    XLA_FLAGS set at collection time, for one) are removed for the test's duration, so the recorded env is exactly what the test set."""
    for k in list(os.environ):
        if k in peak.XLA_NAMES or k.startswith(peak.XLA_PREFIXES) or k.startswith("TF_"):
            monkeypatch.delenv(k, raising=False)
    yield


# ------------------------------------------------------------------------------------------------------------------ stand-ins
def fake_torch(alloc_seq, reserved=5_000_000_000, initialized=True, name="GPU-X"):
    calls = {"device_count": 0}
    seq = list(alloc_seq)
    cuda = types.SimpleNamespace()
    cuda.is_initialized = lambda: initialized
    def device_count():
        calls["device_count"] += 1
        return 1
    cuda.device_count = device_count
    cuda.get_device_name = lambda i: name
    cuda.max_memory_allocated = lambda i: seq.pop(0) if len(seq) > 1 else seq[0]
    cuda.max_memory_reserved = lambda i: reserved
    m = types.ModuleType("torch"); m.cuda = cuda; m._calls = calls
    return m


class FakeDev:
    def __init__(self, stats_seq, ident="cuda:0", kind="GPU-X"):
        self.seq = list(stats_seq); self.ident = ident; self.device_kind = kind
    def memory_stats(self):
        return self.seq.pop(0) if len(self.seq) > 1 else self.seq[0]
    def __str__(self):
        return self.ident


def fake_jax(devices, ready=True):
    calls = {"local_devices": 0}
    m = types.ModuleType("jax")
    def local_devices():
        calls["local_devices"] += 1
        return devices
    m.local_devices = local_devices; m._calls = calls
    xb = types.ModuleType("jax._src.xla_bridge"); xb.backends_are_initialized = lambda: ready
    return m, xb


# ------------------------------------------------------------------------------------------------------------------ the module
def test_module_level_imports_no_backend():
    assert "torch" not in sys.modules or sys.modules["torch"].__name__ == "torch"    # the test process may carry a real torch: never OURS
    src = open(peak.__file__, encoding="utf-8").read()
    assert "import torch" not in src and "import jax" not in src and "from opt_core" not in src   # standard library only; no intra-package import


def test_xla_env_verbatim_and_effective():
    e = {"XLA_PYTHON_CLIENT_PREALLOCATE": "true", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "JAX_TRACEBACK_FILTERING": "off",
         "HOME": "/x", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    v = peak.xla_env(e)
    assert v == {"JAX_TRACEBACK_FILTERING": "off", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}
    assert peak.torch_env(e) == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    eff = peak.xla_effective(v)
    assert (eff["preallocate"], eff["mem_fraction"], eff["allocator"], eff["pool"]) == (True, 0.95, "default", True)
    assert eff["source"] == {"preallocate": "env", "mem_fraction": "XLA_CLIENT_MEM_FRACTION", "allocator": "default"} and "pool" in eff["note"]
    d = peak.xla_effective({})
    assert (d["preallocate"], d["mem_fraction"], d["allocator"], d["pool"], d["source"]["preallocate"]) == (True, 0.75, "default", True, "default")
    n = peak.xla_effective({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
    assert n["pool"] is False and "no preallocated pool" in n["note"]
    assert peak.xla_effective({"XLA_PYTHON_CLIENT_ALLOCATOR": "bfc"})["pool"] is True and peak.xla_effective({"XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async"})["pool"] is False
    p = peak.xla_effective({"XLA_PYTHON_CLIENT_ALLOCATOR": "platform", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5"})
    assert (p["allocator"], p["pool"], p["mem_fraction"], p["source"]["mem_fraction"]) == ("platform", False, 0.5, "XLA_PYTHON_CLIENT_MEM_FRACTION")
    both = peak.xla_effective({"XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5"})          # both names present: the client's precedence, cited
    assert (both["mem_fraction"], both["source"]["mem_fraction"]) == (0.95, "XLA_CLIENT_MEM_FRACTION") and peak.MEM_FRACTION_PRECEDENCE[0] == "XLA_CLIENT_MEM_FRACTION"


def test_expectation_is_checked_never_applied(tmp_path, monkeypatch):
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true"); monkeypatch.delenv("XLA_CLIENT_MEM_FRACTION", raising=False)
    assert peak.check_xla_env({"XLA_PYTHON_CLIENT_PREALLOCATE": "true"}) == []
    c = peak.check_xla_env({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_CLIENT_MEM_FRACTION": "0.95"})
    assert c == ["XLA_CLIENT_MEM_FRACTION: process=None expected='0.95'", "XLA_PYTHON_CLIENT_PREALLOCATE: process='true' expected='false'"]
    monkeypatch.setattr(peak, "_PROBE", None)
    with pytest.raises(peak.PeakRefusal) as ei:
        peak.start(str(tmp_path / "peak_mem.json"), xla_env_expected={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
    assert "never overridden" in str(ei.value) and os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true" and peak.probe() is None
    assert not (tmp_path / "peak_mem.json").exists()
    monkeypatch.delenv(peak.ENV_PATH, raising=False)
    with pytest.raises(peak.PeakRefusal):
        peak.start()                                                  # no path anywhere: refused by name


# ------------------------------------------------------------------------------------------------------------------ the probe
def test_probe_samples_both_families_and_keeps_maxima(tmp_path, monkeypatch):
    t = fake_torch([1_000, 3_000_000_000, 2_000_000_000])
    dev = FakeDev([{"peak_bytes_in_use": 10, "bytes_in_use": 5, "bytes_limit": 80_000_000_000}, {"peak_bytes_in_use": 40_000_000_000, "bytes_in_use": 30_000_000_000, "bytes_limit": 80_000_000_000},
                   {"peak_bytes_in_use": 40_000_000_000, "bytes_in_use": 1_000, "bytes_limit": 80_000_000_000}])
    j, xb = fake_jax([dev])
    monkeypatch.setitem(sys.modules, "torch", t); monkeypatch.setitem(sys.modules, "jax", j); monkeypatch.setitem(sys.modules, "jax._src.xla_bridge", xb)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true"); monkeypatch.setenv("XLA_CLIENT_MEM_FRACTION", "0.95")
    p = peak.Probe(str(tmp_path / "peak_mem.json"), hz=1.0, flush_s=0.2, series=100)   # hz below the floor is raised to the floor
    assert p.hz == peak.HZ_FLOOR
    p.start(); time.sleep(1.6)
    rec = p.stop()
    assert rec["schema"] == peak.SCHEMA and rec["samples"] >= 3 and rec["stopped"] is True and rec["error"] is None and rec["n_errors"] == 0
    assert rec["torch_max_allocated_bytes"] == 3_000_000_000 and rec["torch_max_allocated_gb"] == 3.0 and rec["torch_max_reserved_gb"] == 5.0
    assert rec["jax_peak_bytes_in_use"] == 40_000_000_000 and rec["jax_peak_bytes_in_use_gb"] == 40.0 and rec["jax_bytes_in_use_max"] == 30_000_000_000 and rec["jax_bytes_limit_gb"] == 80.0
    assert (rec["peak_alloc_gb"], rec["peak_alloc_kind"], rec["device"], rec["device_name"]) == (40.0, "jax_peak_bytes_in_use", "cuda:0", "GPU-X")
    assert rec["smi_high_water_gb"] is None and rec["window"][0] <= rec["window"][1] and rec["xla_env"]["XLA_CLIENT_MEM_FRACTION"] == "0.95"
    assert rec["xla_effective"]["applies"] is True and rec["xla_effective"]["pool"] is True                # a JAX backend was created in this process: the pool verdict applies
    assert rec["backend"] == {"torch_loaded": True, "torch_cuda_ready_at": rec["backend"]["torch_cuda_ready_at"], "jax_loaded": True, "jax_backend_ready_at": rec["backend"]["jax_backend_ready_at"]}
    assert rec["backend"]["torch_cuda_ready_at"] >= rec["window"][0] - 0.001 and rec["instrument"]["sha256"] == hashlib.sha256(open(peak.__file__, "rb").read()).hexdigest()
    assert (rec["executable"], rec["prefix"], rec["forked_from"]) == (sys.executable, sys.prefix, None)                      # the record names its interpreter
    assert (rec["loader"], rec["env_source"], rec["env_changed_before_backend"], rec["xla_env_at_start"] == rec["xla_env"]) == ("api", "backend_ready", False, True)
    assert rec["hz_effective"] is not None and 1.0 <= rec["hz_effective"] <= 2 * peak.HZ_FLOOR + 1      # the observed cadence beside the nominal one
    kinds = sorted(d["kind"] for d in rec["devices"]); assert kinds == ["jax", "torch"]
    on_disk = json.load(open(tmp_path / "peak_mem.json")); assert on_disk["peak_alloc_gb"] == 40.0 and on_disk["stopped"] is True
    rows = open(tmp_path / "peak_mem.samples.csv").read().splitlines(); assert rows[0] == "epoch,jax_bytes_in_use,torch_allocated_bytes" and len(rows) >= 4
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]               # every write atomic: no temp file left behind


def test_a_torch_only_process_records_no_pool_verdict(tmp_path, monkeypatch):
    """M1: the JAX client defaults (preallocate true, fraction 0.75) describe nothing in a process that never created a JAX backend — a
    torch-only record says {applies: false, pool: false}, so no reader can print 'the preallocated pool' on a torch engine's row."""
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true"); monkeypatch.setenv("XLA_CLIENT_MEM_FRACTION", "0.95")   # even with the JAX names exported
    t = fake_torch([3_000_000_000]); monkeypatch.setitem(sys.modules, "torch", t); monkeypatch.delitem(sys.modules, "jax", raising=False); monkeypatch.delitem(sys.modules, "jax._src.xla_bridge", raising=False)
    p = peak.Probe(str(tmp_path / "peak_mem.json"), hz=4.0); p.start(); time.sleep(0.5); rec = p.stop()
    assert rec["peak_alloc_kind"] == "torch_max_allocated" and rec["backend"]["jax_backend_ready_at"] is None
    assert rec["xla_effective"] == peak.NO_JAX_EFFECTIVE and rec["xla_effective"]["pool"] is False and rec["xla_effective"]["applies"] is False
    assert rec["xla_env"] == {"XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}       # the env stays verbatim; only the verdict is withheld
    assert peak.xla_effective({}, jax_backend=False)["pool"] is False and peak.xla_effective({})["pool"] is True   # the defaults resolve to a pool ONLY for a JAX process
    # the pass reduce: a torch-only pass carries no pool verdict; a mixed pass takes the XLA fields from the record holding the JAX peak
    torch_rec = {**_rec(1, 0, 1, torch_alloc=3_000_000_000, xla={"XLA_PYTHON_CLIENT_PREALLOCATE": "true"}), "xla_effective": peak.NO_JAX_EFFECTIVE, "backend": {"jax_backend_ready_at": None}}
    d = peak.reduce([torch_rec]); assert (d["xla_effective"]["applies"], d["xla_effective"]["pool"], d["jax_records"]) == (False, False, 0)
    jax_rec = {**_rec(2, 0, 1, jax_peak=50_000_000_000, xla={"XLA_CLIENT_MEM_FRACTION": "0.95"}), "backend": {"jax_backend_ready_at": 0.5}}
    m = peak.reduce([torch_rec, jax_rec]); assert (m["peak_alloc_kind"], m["xla_effective"]["applies"], m["xla_effective"]["pool"], m["xla_effective"]["mem_fraction"], m["jax_records"], m["xla_env"]) == ("jax_peak_bytes_in_use", True, True, 0.95, 1, {"XLA_CLIENT_MEM_FRACTION": "0.95"})


def test_env_is_resnapshotted_when_the_backend_comes_up(tmp_path, monkeypatch):
    """A lever that exports an allocator variable in-process before its first device call: the record's env is the one the backend was
    created under (the backend-ready snapshot), the interpreter-start snapshot beside it, the change named."""
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true"); monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    state = {"ready": False}
    t = fake_torch([7_000_000_000]); t.cuda.is_initialized = lambda: state["ready"]
    monkeypatch.setitem(sys.modules, "torch", t); monkeypatch.delitem(sys.modules, "jax", raising=False)
    p = peak.Probe(str(tmp_path / "peak_mem.json"), hz=4.0); p.start(); time.sleep(0.4)
    assert p.record()["env_source"] == "start" and p.record()["xla_effective"]["applies"] is False                # no backend yet: the start snapshot, said; no pool verdict without a JAX backend
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false"); monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # the lever exports, then the backend comes up
    state["ready"] = True; time.sleep(0.6); rec = p.stop()
    assert (rec["env_source"], rec["env_changed_before_backend"], rec["xla_env"], rec["xla_env_at_start"]) == ("backend_ready", True, {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, {"XLA_PYTHON_CLIENT_PREALLOCATE": "true"})
    assert rec["xla_effective"]["applies"] is False and rec["xla_effective"]["pool"] is False and rec["torch_env"] == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"} and rec["torch_env_at_start"] == {} and rec["env_ready_at"] >= rec["window"][0]
    assert rec["torch_max_allocated_gb"] == 7.0


def test_a_backend_that_is_not_ready_is_never_touched(tmp_path, monkeypatch):
    t = fake_torch([1], initialized=False); j, xb = fake_jax([FakeDev([{"peak_bytes_in_use": 1}])], ready=False)
    monkeypatch.setitem(sys.modules, "torch", t); monkeypatch.setitem(sys.modules, "jax", j); monkeypatch.setitem(sys.modules, "jax._src.xla_bridge", xb)
    p = peak.Probe(str(tmp_path / "peak_mem.json"), hz=4.0); p.start(); time.sleep(0.6); rec = p.stop()
    assert t._calls["device_count"] == 0 and j._calls["local_devices"] == 0            # never initialised by us: no counter read before the process's own init
    assert rec["peak_alloc_gb"] is None and rec["peak_alloc_kind"] is None and rec["devices"] == [] and rec["samples"] >= 2 and rec["n_errors"] == 0
    assert rec["backend"]["torch_cuda_ready_at"] is None and rec["backend"]["jax_backend_ready_at"] is None and rec["env_source"] == "start"


def test_a_failing_counter_is_recorded_never_raised(tmp_path, monkeypatch):
    t = fake_torch([1]); t.cuda.max_memory_allocated = lambda i: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "torch", t)
    p = peak.Probe(str(tmp_path / "peak_mem.json"), hz=4.0); p.start(); time.sleep(0.5); rec = p.stop()
    assert rec["n_errors"] >= 1 and "boom" in rec["error"] and rec["stopped"] is True


def test_start_is_one_probe_per_process(tmp_path, monkeypatch):
    monkeypatch.setattr(peak, "_PROBE", None); monkeypatch.delenv(peak.ENV_PER_PROCESS, raising=False)
    a = peak.start(str(tmp_path / "peak_mem.json"), hz=4.0); b = peak.start(str(tmp_path / "other.json"))
    assert a is b and a.path == str(tmp_path / "peak_mem.json") and (tmp_path / "peak_mem.json").exists()   # the markers are on disk at once
    rec = peak.stop(); assert rec["stopped"] is True and peak.stop()["stopped"] is True
    monkeypatch.setattr(peak, "_PROBE", None); monkeypatch.setenv(peak.ENV_PATH, str(tmp_path / "hook" / "peak_mem.json"))
    c = peak.start(); peak.stop()                                                                             # the env route is per process by default: <stem>.<pid>-<start ms>.json
    assert os.path.basename(c.path).startswith(f"peak_mem.{os.getpid()}-") and c.path.endswith(".json") and os.path.dirname(c.path) == str(tmp_path / "hook")
    assert peak.per_process_path("/x/peak_mem.json", pid=5, t_start=1234.5678) == "/x/peak_mem.5-1234567.json"
    monkeypatch.setattr(peak, "_PROBE", None)


# ------------------------------------------------------------------------------------------------------------------ the reduce
def _rec(pid, t0, t1, torch_alloc=None, jax_peak=None, xla=None):
    r = {"schema": peak.SCHEMA, "pid": pid, "window": [t0, t1], "samples": 3, "hz": 2.0, "hz_effective": 1.9, "loader": "pth", "env_source": "backend_ready", "env_changed_before_backend": False,
         "stopped": True, "xla_env": xla or {}, "torch_env": {}, "devices": [], "instrument": {"file": "peak.py", "sha256": "abc"}, "error": None, "n_errors": 0, "device": "cuda:0", "device_name": "G"}
    if torch_alloc is not None:
        r.update({"torch_max_allocated_bytes": torch_alloc, "torch_max_reserved_bytes": torch_alloc + 1, "devices": [{"device": "cuda:0", "kind": "torch"}]})
    if jax_peak is not None:
        r.update({"jax_peak_bytes_in_use": jax_peak, "jax_bytes_in_use_max": jax_peak - 1, "jax_bytes_limit": 80_000_000_000, "devices": [{"device": "cuda:0", "kind": "jax"}], "backend": {"jax_backend_ready_at": t0 + 0.5}})
    return r


def test_reduce_max_union_window_and_smi_inside_the_window_only():
    recs = [_rec(1, 100.0, 110.0, torch_alloc=2_000_000_000, xla={"XLA_FLAGS": "a"}), _rec(2, 105.0, 130.0, torch_alloc=7_000_000_000, xla={"XLA_FLAGS": "a"})]
    smi = [(99.0, 70_000), (100.5, 1_000), (120.0, 9_537), (130.0, 2_000), (131.0, 80_000)]              # MiB; the samples outside [100, 130] never count
    d = peak.reduce(recs, smi=smi)
    assert (d["window"], d["records"], d["pids"], d["samples"]) == ([100.0, 130.0], 2, [1, 2], 6)
    assert (d["peak_alloc_gb"], d["peak_alloc_kind"], d["torch_max_allocated_bytes"], d["torch_max_reserved_gb"]) == (7.0, "torch_max_allocated", 7_000_000_000, 7.0)
    assert (d["smi_high_water_mib"], d["smi_samples_in_window"], d["smi_high_water_gb"]) == (9_537, 3, 10.0)
    assert d["xla_env_conflict"] is None and d["instruments"] == ["abc"] and d["unstopped"] == [] and len(d["devices"]) == 2 and d["devices"][1]["pid"] == 2
    assert (d["loaders"], d["env_source"], d["hz_effective"]) == (["pth"], "backend_ready", 1.9)
    m = peak.reduce([_rec(1, 0, 1, torch_alloc=1, xla={"XLA_FLAGS": "a"}), _rec(2, 0, 1, jax_peak=50, xla={})])
    assert (m["peak_alloc_kind"], m["peak_alloc_gb"], m["jax_bytes_limit_gb"]) == ("jax_peak_bytes_in_use", 0.0, 80.0) and m["xla_env_conflict"] == ['{"XLA_FLAGS": "a"}', "{}"]
    assert (m["xla_env"], m["xla_effective"]["applies"], m["jax_records"]) == ({}, True, 1) and d["xla_effective"]["applies"] is False and d["jax_records"] == 0   # torch-only pass: no pool verdict
    with pytest.raises(peak.PeakRefusal):
        peak.reduce([])


def test_reduce_dir_and_cli(tmp_path, capsys):
    for r in (_rec(11, 10.0, 20.0, jax_peak=30_000_000_000), _rec(12, 15.0, 25.0, jax_peak=45_000_000_000)):
        peak._write_atomic(str(tmp_path / f"peak_mem.{r['pid']}-{int(r['window'][0] * 1000)}.json"), r)
    (tmp_path / "peak_mem.12-15000.samples.csv").write_text("epoch,jax_bytes_in_use,torch_allocated_bytes\n15.5,1,\n")
    (tmp_path / "pass_smi.csv").write_text("9.0,4,0,77683,50.1,1980\n12.0,4,0,77683,50.1,1980\n[N/A],x\n26.0,4,0,81000,50.1,1980\n")
    with pytest.raises(peak.PeakRefusal):
        peak.reduce_dir(str(tmp_path), stem="nothing")                                       # no record under that stem: refused by name
    assert peak.main(["reduce", str(tmp_path), "--smi", str(tmp_path / "pass_smi.csv")]) == 0
    out = capsys.readouterr().out; assert "peak_alloc_gb=45.0 (jax_peak_bytes_in_use)" in out and "smi_high_water_gb=81.457" in out
    d = json.load(open(tmp_path / "peak_mem.json"))
    assert d["files"] == ["peak_mem.11-10000.json", "peak_mem.12-15000.json"] and d["window"] == [10.0, 25.0] and d["smi_samples_in_window"] == 1 and d["smi_high_water_mib"] == 77683.0 and d["packed"] is None
    assert peak.main(["show", str(tmp_path / "peak_mem.json")]) == 0 and "pool=True" in capsys.readouterr().out
    assert peak.main(["reduce", str(tmp_path), "--pack"]) == 0 and "packed=peak_mem.records.tar.gz" in capsys.readouterr().out        # born-packed: one record + one archive, nothing loose
    import tarfile
    assert sorted(os.listdir(tmp_path)) == ["pass_smi.csv", "peak_mem.json", "peak_mem.records.tar.gz"]
    with tarfile.open(tmp_path / "peak_mem.records.tar.gz") as tf: entries = sorted(tf.getnames())
    assert entries == ["peak_mem.11-10000.json", "peak_mem.12-15000.json", "peak_mem.12-15000.samples.csv"] and json.load(open(tmp_path / "peak_mem.json"))["packed_entries"] == entries
    assert peak.main(["reduce", str(tmp_path)]) == 2                                                                                # the loose records are gone: refused by name, the pass record stands
    os.makedirs(tmp_path / "no_records"); assert peak.main(["reduce", str(tmp_path / "no_records")]) == 2 and "refused" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------------------------ the hook (a stock-style process)
CHILD = ("import sys, time, json; time.sleep(0.4); pr = sys.modules['peak'].probe() if 'peak' in sys.modules else None; "
         "print(json.dumps({'peak': 'peak' in sys.modules, 'opt_core': [m for m in sys.modules if m == 'opt_core' or m.startswith('opt_core.')], "
         "'path': pr.path if pr else None, 'file': sys.modules['peak'].__file__ if 'peak' in sys.modules else None}))")


def _child(env_extra, tmp_path, pythonpath):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH", "PYTHONUSERBASE", "PYTHONNOUSERSITE")}
    env.update({"PYTHONPATH": pythonpath, peak.ENV_HZ: "4", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", **env_extra})
    r = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1]), r.stderr


def test_hook_carries_this_file_and_a_process_started_through_it_writes_its_record_at_exit(tmp_path):
    hook = tmp_path / "hook"; out = tmp_path / "out"
    h = peak.write_hook(str(hook))
    src = open(peak.__file__, "rb").read()
    assert open(hook / "peak.py", "rb").read() == src and h["sha256"] == hashlib.sha256(src).hexdigest()
    assert (hook / "peak.sha256").read_text().split()[0] == h["sha256"] and (hook / "sitecustomize.py").read_text() == peak.SITECUSTOMIZE and (hook / peak.PTH_NAME).read_text() == peak.PTH
    empty = tmp_path / "empty"; os.makedirs(empty)
    baseline, _ = _child({}, tmp_path, str(empty))                                                  # opt_core.* this interpreter imports at startup by itself (a kit's autoload .pth beside the core)
    got, err = _child({peak.ENV_PATH: str(out / "peak_mem.json")}, tmp_path, str(hook))
    assert got["peak"] is True and got["file"] == str(hook / "peak.py") and "REFUSED" not in err         # under its own name, never opt_core.*
    assert set(got["opt_core"]) <= set(baseline["opt_core"]), sorted(set(got["opt_core"]) - set(baseline["opt_core"]))   # the hook imports no opt_core.* of its own
    rec = json.load(open(got["path"]))
    assert rec["stopped"] is True and rec["samples"] >= 1 and rec["xla_env"] == {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"} and rec["xla_effective"]["pool"] is False and rec["loader"] == "sitecustomize"
    assert rec["instrument"] == {"file": str(hook / "peak.py"), "sha256": h["sha256"]} and rec["peak_alloc_gb"] is None and rec["backend"]["torch_loaded"] is False
    assert os.path.basename(got["path"]).startswith(f"peak_mem.") and got["path"].endswith(".json") and got["path"] != str(out / "peak_mem.json")
    p, d = peak.reduce_dir(str(out)); assert d["records"] == 1 and d["files"] == [os.path.basename(got["path"])] and os.path.exists(p) and d["loaders"] == ["sitecustomize"]
    got2, _ = _child({}, tmp_path, str(hook))                                                       # no output path named: the loader does nothing (no probe, no file)
    assert got2["peak"] is True and got2["path"] is None


def test_the_pth_loader_form_and_the_sha_gate(tmp_path):
    """The .pth form: the same two statements from site-packages at site initialisation — for a process whose sitecustomize slot a recipe
    owns; here the hook dir carries NO sitecustomize (a stand-in for a shadowed one) and the .pth sits in a user site dir. A carried copy
    that differs from its peak.sha256 is refused at boot: one stderr line, no record, the process runs on."""
    probe = subprocess.run([sys.executable, "-c", "import site; print(site.ENABLE_USER_SITE)"], capture_output=True, text=True, timeout=60,
                           env={k: v for k, v in os.environ.items() if k not in ("PYTHONNOUSERSITE", "PYTHONUSERBASE")})
    if probe.stdout.strip() != "True":                                   # a venv interpreter never processes user-site .pth files: the form is unreachable here
        pytest.skip("this interpreter does not process user-site .pth files (site.ENABLE_USER_SITE=%s: a venv, -s, or PYTHONNOUSERSITE) — the .pth "
                    "loader form is exercised on an interpreter with a user site" % probe.stdout.strip())
    hook = tmp_path / "hook"; out = tmp_path / "out"; h = peak.write_hook(str(hook)); os.remove(hook / "sitecustomize.py")
    base = tmp_path / "userbase"; site_dir = base / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "site-packages"
    os.makedirs(site_dir); shutil.copy(hook / peak.PTH_NAME, site_dir / peak.PTH_NAME)
    got, err = _child({peak.ENV_PATH: str(out / "peak_mem.json"), "PYTHONUSERBASE": str(base)}, tmp_path, str(hook))
    assert got["peak"] is True and got["path"] is not None and "REFUSED" not in err, err
    rec = json.load(open(got["path"])); assert rec["loader"] == "pth" and rec["stopped"] is True and rec["instrument"]["sha256"] == h["sha256"]
    r = subprocess.run([sys.executable, "-I", "-c", "import sys; print('ran', 'peak' in sys.modules)"], env={**os.environ, "PYTHONUSERBASE": str(base), "PYTHONPATH": str(hook), peak.ENV_PATH: str(out / "isolated" / "peak_mem.json")},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "ran False" in r.stdout and not (out / "isolated").exists()   # `python -I`: user site + PYTHONPATH ignored → the loader never runs (if the .pth were processed, the guard's one line would show, never a traceback)
    with open(hook / "peak.py", "a", encoding="utf-8") as fh: fh.write("# a byte the sha does not name\n")
    got3, err3 = _child({peak.ENV_PATH: str(out / "again" / "peak_mem.json"), "PYTHONUSERBASE": str(base)}, tmp_path, str(hook))
    assert "[peak] REFUSED" in err3 and "sha256" in err3 and got3["path"] is None and not (out / "again").exists()   # refused by name; no record; the process ran on


# ------------------------------------------------------------------------------------------------------------------ where the probe rode (two interpreters)
def test_reduce_names_a_pass_whose_records_never_saw_a_backend():
    """Two interpreters (a launcher venv running a kit's driver and helpers, a model venv running the model process): records from the
    driver side only, none from the model process — a .pth fires only in the interpreter whose site-packages holds it. The reduce names
    that by name instead of a silent None."""
    d = _rec(1, 0.0, 300.0); d.update({"argv": ["/work/tree/kit/opt/kit_opt/__main__.py", "pred"], "executable": "/work/venv/bin/python", "backend": {"torch_loaded": False, "torch_cuda_ready_at": None, "jax_loaded": False, "jax_backend_ready_at": None}})
    e = _rec(2, 1.0, 3.0); e.update({"argv": ["/work/tree/kit/stock/check_pins.py"], "executable": "/work/venv/bin/python", "backend": {"torch_loaded": False, "torch_cuda_ready_at": None, "jax_loaded": False, "jax_backend_ready_at": None}})
    doc = peak.reduce([d, e])
    assert doc["peak_alloc_gb"] is None and doc["backend_records"] == 0 and doc["executables"] == ["/work/venv/bin/python"]
    assert doc["error"].startswith("no recorded process loaded torch or jax: the probe rode __main__.py, check_pins.py under /work/venv/bin/python") and "install" in doc["error"]
    f = _rec(3, 0.0, 1.0, torch_alloc=5); f["backend"] = {"torch_loaded": True, "torch_cuda_ready_at": 0.5, "jax_loaded": False, "jax_backend_ready_at": None}
    ok = peak.reduce([d, f]); assert ok["error"] is None and ok["backend_records"] == 1


def test_reduce_verb_exits_3_on_a_refused_pass_and_still_writes_the_document(tmp_path, capsys):
    d = _rec(1, 0.0, 300.0); d.update({"argv": ["driver.py"], "executable": "/h/bin/python", "backend": {"torch_loaded": False, "torch_cuda_ready_at": None, "jax_loaded": False, "jax_backend_ready_at": None}})
    peak._write_atomic(str(tmp_path / "peak_mem.1-1.json"), d)
    assert peak.main(["reduce", str(tmp_path)]) == 3
    out, err = capsys.readouterr(); assert "[peak] REFUSED" in err and "the probe rode driver.py under /h/bin/python" in err and "peak_alloc_gb=None" in out
    doc = json.load(open(tmp_path / "peak_mem.json")); assert doc["backend_records"] == 0 and doc["error"]


def _venv(path):
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(path)], check=True, capture_output=True, timeout=180)
    return str(path / "bin" / "python")


def test_install_puts_the_pth_into_each_model_interpreter_and_proves_it(tmp_path):
    """Two interpreters (a launcher venv /work/venv and a model venv /model_venv): the .pth installed into ONE
    fires only there; `install` puts it into each named interpreter and proves the loader with a run that leaves a record; a spliced
    site-packages (sys.path.insert at runtime) never fires its .pth."""
    hook = tmp_path / "hook"; h = peak.write_hook(str(hook))
    harness_py, model_py = _venv(tmp_path / "venv_harness"), _venv(tmp_path / "venv_model")
    rows = peak.install_pth(str(hook), [harness_py], prove=True)
    assert rows[0]["error"] is None and rows[0]["proved"] is True and rows[0]["loader"] == "pth" and rows[0]["executable"] == harness_py, rows
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH")}
    env.update({"PYTHONPATH": str(hook), peak.ENV_PATH: str(tmp_path / "out" / "peak_mem.json")})
    # the model interpreter: no .pth in ITS site-packages → no record, even with the hook dir on PYTHONPATH... unless its sitecustomize slot is free — it is here, so
    # the sitecustomize form fires instead (a kit-owned slot would not); prove the .pth mechanism itself by removing the sitecustomize loader
    os.remove(hook / "sitecustomize.py")
    r = subprocess.run([model_py, "-c", "import sys; print('peak' in sys.modules)"], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "False" and not (tmp_path / "out").exists()                # the .pth in venv_harness does nothing for venv_model
    rows = peak.install_pth(str(hook), [model_py, str(tmp_path / "nowhere" / "python")], prove=True)
    assert rows[0]["proved"] is True and rows[0]["loader"] == "pth" and rows[0]["executable"] == model_py and rows[0]["pth"].startswith(str(tmp_path / "venv_model"))
    assert rows[1]["proved"] is None and rows[1]["error"] and "purelib query failed" in rows[1]["error"] or "No such file" in (rows[1]["error"] or "")
    r = subprocess.run([model_py, "-c", "import sys; print('peak' in sys.modules)"], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "True" and peak.per_process_files(str(tmp_path / "out"))                # now it fires there
    rec = peak.read(os.path.join(tmp_path / "out", peak.per_process_files(str(tmp_path / "out"))[0])); assert rec["executable"] == model_py and rec["loader"] == "pth"
    assert peak.main(["install", str(hook), harness_py]) == 0 and peak.main(["install", str(hook), str(tmp_path / "nowhere" / "python")]) == 3


def test_a_forked_child_restarts_the_sampling_with_its_own_record(tmp_path, monkeypatch):
    """multiprocessing 'fork' / os.fork: the child inherits the probe object but no thread — the sampling restarts in the child with its
    own per-process record (forked_from = the parent), its own window and maxima; the parent's record is untouched."""
    monkeypatch.setattr(peak, "_PROBE", None); monkeypatch.setattr(peak, "_FORK_HOOKED", False)
    code = f"""
import os, sys, time, json
sys.path.insert(0, {os.path.dirname(peak.__file__)!r})
import peak, types
t = types.ModuleType("torch"); cu = types.SimpleNamespace(_init=False, _peak=0)
cu.is_initialized = lambda: cu._init; cu.device_count = lambda: 1; cu.get_device_name = lambda i: "STANDIN"
cu.max_memory_allocated = lambda i: cu._peak; cu.max_memory_reserved = lambda i: cu._peak + 7; t.cuda = cu; sys.modules["torch"] = t
os.environ["{peak.ENV_PATH}"] = {str(tmp_path / "out" / "peak_mem.json")!r}; os.environ["{peak.ENV_HZ}"] = "8"; os.environ["{peak.ENV_FLUSH}"] = "0.3"
p = peak.start()
pid = os.fork()
if pid == 0:
    cu._init = True; cu._peak = 4_000_000_000; time.sleep(1.0); os._exit(0)      # the worker: initialises 'cuda', allocates, leaves through os._exit (no atexit); the periodic write is its record
os.waitpid(pid, 0); time.sleep(0.3)
print(json.dumps({{"parent": os.getpid(), "child": pid, "parent_path": p.path}}))
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env={k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)})
    assert r.returncode == 0, r.stderr[-800:]
    got = json.loads(r.stdout.strip().splitlines()[-1]); out = tmp_path / "out"
    files = peak.per_process_files(str(out)); recs = {peak.read(str(out / f))["pid"]: peak.read(str(out / f)) for f in files}
    assert set(recs) == {got["parent"], got["child"]}, files
    par, ch = recs[got["parent"]], recs[got["child"]]
    assert par["forked_from"] is None and par["peak_alloc_gb"] is None and par["stopped"] is True                       # the parent never initialised 'cuda'
    assert ch["forked_from"] == got["parent"] and ch["peak_alloc_gb"] == 4.0 and ch["peak_alloc_kind"] == "torch_max_allocated" and ch["samples"] >= 2
    assert ch["stopped"] is False and ch["window"][0] >= par["window"][0]                                              # left through os._exit: the periodic writes are its record
    doc = peak.reduce(list(recs.values())); assert doc["peak_alloc_gb"] == 4.0 and doc["backend_records"] == 1 and doc["unstopped"] == [got["child"]]


def test_a_failed_fork_restart_says_so(tmp_path, monkeypatch, capsys):
    """A forked child whose sampling restart fails (here: a file sits where the record directory was) leaves one stderr line, never silence."""
    monkeypatch.setattr(peak, "_PROBE", None); monkeypatch.setattr(peak, "_FORK_HOOKED", False)
    code = f"""
import os, sys, time, shutil
sys.path.insert(0, {os.path.dirname(peak.__file__)!r})
import peak
os.environ["{peak.ENV_PATH}"] = {str(tmp_path / "gone" / "peak_mem.json")!r}; os.environ["{peak.ENV_HZ}"] = "8"
p = peak.start()
shutil.rmtree({str(tmp_path / "gone")!r}); open({str(tmp_path / "gone")!r}, "w").close()   # a file now sits where the record directory was: the child's first write cannot land
pid = os.fork()
if pid == 0:
    time.sleep(0.2); os._exit(0)
os.waitpid(pid, 0); print("parent done")
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env={k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)})
    assert r.returncode == 0 and "parent done" in r.stdout, r.stderr[-500:]
    assert "[peak] REFUSED: fork restart in pid" in r.stderr and "no record for this child" in r.stderr, r.stderr[-500:]


def test_a_failed_final_write_at_exit_says_so(tmp_path):
    """The record directory replaced by a file before the interpreter exits: the at-exit write cannot land — one stderr line, the process exits 0."""
    code = f"""
import os, sys, time, shutil
sys.path.insert(0, {os.path.dirname(peak.__file__)!r})
import peak
os.environ["{peak.ENV_PATH}"] = {str(tmp_path / "gone" / "peak_mem.json")!r}; os.environ["{peak.ENV_HZ}"] = "8"
p = peak.start(); time.sleep(0.3)
shutil.rmtree({str(tmp_path / "gone")!r}); open({str(tmp_path / "gone")!r}, "w").close()
print("done")
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env={k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)})
    assert r.returncode == 0 and "done" in r.stdout and "[peak] REFUSED: at-exit write of" in r.stderr and "the last periodic write stands" in r.stderr, r.stderr[-500:]


# ------------------------------------------------------------------------------------------------------------------ a five-process record set (one process reached a torch backend, four loaded none) — the reduce never tracebacks
def _record(pid, t0, t1, alloc=None, reserved=None, samples=1):
    """One per-process record with the schema's fields, built here: a process that reached a torch backend when `alloc` is given, else one that loaded no backend."""
    dev = [{"device": "cuda:0", "kind": "torch", "max_allocated_bytes": alloc, "max_reserved_bytes": reserved, "name": "GPU-X", "samples": samples}] if alloc is not None else []
    ready = t0 + 8.0 if dev else None
    return {"schema": peak.SCHEMA, "pid": pid, "argv": ["-c"], "executable": "/usr/local/bin/python", "prefix": "/usr/local", "loader": "pth", "forked_from": None,
            "window": [t0, t1], "samples": samples, "hz": 2.0, "hz_effective": round((samples - 1) / (t1 - t0), 3) if samples > 1 else None,
            "device": "cuda:0" if dev else None, "device_name": "GPU-X" if dev else None, "devices": dev, "peak_alloc_gb": peak._gb(alloc), "peak_alloc_kind": "torch_max_allocated" if dev else None,
            "torch_max_allocated_bytes": alloc, "torch_max_allocated_gb": peak._gb(alloc), "torch_max_reserved_bytes": reserved, "torch_max_reserved_gb": peak._gb(reserved),
            "jax_peak_bytes_in_use": None, "jax_peak_bytes_in_use_gb": None, "jax_bytes_in_use_max": None, "jax_bytes_in_use_max_gb": None, "jax_bytes_limit": None, "jax_bytes_limit_gb": None,
            "smi_high_water_gb": None, "smi_samples_in_window": None, "xla_env": {}, "xla_env_at_start": {}, "xla_effective": peak.xla_effective({}, jax_backend=False),
            "torch_env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"} if dev else {}, "torch_env_at_start": {}, "env_source": "backend_ready" if dev else "start", "env_ready_at": ready,
            "env_changed_before_backend": bool(dev), "backend": {"torch_loaded": bool(dev), "torch_cuda_ready_at": ready, "jax_loaded": False, "jax_backend_ready_at": None},
            "instrument": {"file": "hook/peak.py", "sha256": "0" * 64}, "error": None, "n_errors": 0, "stopped": True}


LADDER_REP = [_record(101, 1000.0, 1460.0, alloc=75_872_761_344, reserved=79_601_598_464, samples=908)] + [_record(102 + i, 999.5 + 0.1 * i, 999.55 + 0.1 * i) for i in range(4)]


def _ladder_dir(tmp_path):
    d = tmp_path / "pass1-LINE_A"; d.mkdir()
    for r in LADDER_REP: peak._write_atomic(str(d / f"peak_mem.{r['pid']}-{int(r['window'][0] * 1000)}.json"), r)
    return d


def test_reduce_names_a_missing_sampler_csv_instead_of_a_traceback(tmp_path, capsys):
    """The ladder canary's on-box reduce named a sampler csv that was not there: the reduce must still write the pass record (the allocator
    side is complete), name the csv absent (`smi_error`), and exit 0 — a traceback left every rep without a record."""
    d = _ladder_dir(tmp_path)
    assert peak.main(["reduce", str(d), "--smi", str(d / "smi.csv")]) == 0
    out, err = capsys.readouterr(); assert "Traceback" not in err and "smi_error=" in out and "peak_alloc_gb=75.873 (torch_max_allocated)" in out
    doc = json.load(open(d / "peak_mem.json"))
    assert (doc["records"], doc["backend_records"], doc["peak_alloc_gb"], doc["peak_alloc_kind"], doc["smi_high_water_gb"], doc["error"]) == (5, 1, 75.873, "torch_max_allocated", None, None)
    assert doc["smi_error"].startswith(str(d / "smi.csv")) and doc["unreadable"] == [] and doc["malformed"] == [] and doc["executables"] == ["/usr/local/bin/python"]
    (d / "smi.csv").write_text("%s,4,0,70000,50.1,1980\n" % (LADDER_REP[0]["window"][0] + 1))                     # the csv present: the smi side fills in, no error
    assert peak.main(["reduce", str(d), "--smi", str(d / "smi.csv")]) == 0
    doc = json.load(open(d / "peak_mem.json")); assert doc["smi_error"] is None and doc["smi_high_water_mib"] == 70000.0


def test_reduce_names_unreadable_and_malformed_records_and_reduces_the_rest(tmp_path, capsys):
    d = _ladder_dir(tmp_path)
    (d / "peak_mem.999-1.json").write_text("{not json")                                                           # a truncated record (a killed process mid-write)
    peak._write_atomic(str(d / "peak_mem.998-1.json"), {"schema": peak.SCHEMA, "pid": 998})                        # a record without a window
    assert peak.main(["reduce", str(d)]) == 0
    out, _ = capsys.readouterr(); assert "unreadable=['peak_mem.999-1.json']" in out and "malformed=['peak_mem.998-1.json']" in out
    doc = json.load(open(d / "peak_mem.json"))
    assert doc["records"] == 5 and doc["peak_alloc_gb"] == 75.873 and [u["file"] for u in doc["unreadable"]] == ["peak_mem.999-1.json"] and [m["file"] for m in doc["malformed"]] == ["peak_mem.998-1.json"]
    for f in os.listdir(d):
        if f.startswith("peak_mem.") and f not in ("peak_mem.999-1.json", "peak_mem.998-1.json", "peak_mem.json"): os.remove(d / f)
    assert peak.main(["reduce", str(d)]) == 2                                                                       # nothing readable: refused by name, exit 2
    assert "none readable" in capsys.readouterr().err


def test_xla_effective_names_the_two_mem_fraction_names_conflict():
    """Both MEM_FRACTION names set: the CUDA plugin does not initialise on the JAX pinned stack (CPU fallback) — never a pool verdict."""
    from opt_core.mem import peak
    e = peak.xla_effective({"XLA_CLIENT_MEM_FRACTION": "0.9", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5"})
    assert e["conflict"] is True and e["pool"] is False and "both set" in e["note"] and e["source"]["mem_fraction"] == "XLA_CLIENT_MEM_FRACTION"
    one = peak.xla_effective({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5"})
    assert one["conflict"] is False and one["mem_fraction"] == 0.5 and one["pool"] is True
    assert peak.xla_effective({}, jax_backend=False)["conflict"] is False
