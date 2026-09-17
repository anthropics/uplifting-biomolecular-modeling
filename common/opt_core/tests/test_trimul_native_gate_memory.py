"""kernels.trimul native provider: the install-time byte gate's device-memory scope.  The gate (face.check -> vectors.replay / multiweight_gate)
regenerates closed-form inputs and serves ~40 small cases; unbounded it held three full [n, n, c_z] int64 temporaries per input, every case's
workspaces in one shared cache and the allocator's freed blocks afterwards (~2.2 GiB one-shot with a model resident).  gate_memory_scope bounds it:
row-slab input generation (<= GATE_CHUNK_BYTES of int64 temporaries), a per-case cache, release on exit; verdicts unchanged (same inputs byte for
byte -- the replay digest-checks them -- same kernels, same expected bytes); the verification stamp keyed by (payload digest, package, versions,
device cc + name, driver, CUDA, torch, binding, cases, python) lets later processes on the same stack skip the replay entirely."""
import json
import os

import pytest

from opt_core.kernels.trimul import native as NV


def _vectors():
    try:
        return NV.vectors_module()
    except NV.Unavailable as e:
        pytest.skip("payload face not importable here: %s" % e)


CASES = [
    {"id": "a", "c_z": 128, "c_hidden": 128, "n": 67, "batch": 1, "direction": "outgoing", "form": "bf16", "residual": True, "mask": "tail3", "variant": "fast"},
    {"id": "b", "c_z": 64, "c_hidden": 128, "n": 45, "batch": 1, "direction": "incoming", "form": "f32z", "residual": False, "mask": "none", "variant": "fast"},
    {"id": "c", "c_z": 32, "c_hidden": 64, "n": 21, "batch": 2, "direction": "outgoing", "form": "bf16", "residual": True, "mask": "tail3", "variant": "exact"},
    {"id": "d", "c_z": 384, "c_hidden": 64, "n": 19, "batch": 1, "direction": "outgoing", "form": "f32z", "residual": True, "mask": "tail3", "variant": "fast"},
]


def test_chunked_closed_form_inputs_are_byte_identical_and_slab_bounded():
    torch = pytest.importorskip("torch")
    V = _vectors()
    chunk = 64 * 1024                                                        # tiny slabs: many chunks per case, still identical bytes
    gen = NV.chunked_closed_form_inputs(V, chunk_bytes=chunk)
    before = NV.gate_memory_stats()["max_slab_bytes"]
    for case in CASES:
        for salt in (0, 2):
            z0, m0, w0 = V.closed_form_inputs(case, "cpu", weight_salt=salt)
            z1, m1, w1 = gen(case, "cpu", weight_salt=salt)
            assert z0.dtype == z1.dtype and z0.shape == z1.shape and torch.equal(z0, z1), case["id"]
            assert (m0 is None and m1 is None) or torch.equal(m0, m1)
            assert set(w0) == set(w1) and all(w0[k].dtype == w1[k].dtype and torch.equal(w0[k], w1[k]) for k in w0)
            assert V.input_digests(z0, m0, w0) == V.input_digests(z1, m1, w1)          # what the replay checks against the recorded digests
    n_max = max(c["n"] * c["c_z"] for c in CASES)
    assert NV.gate_memory_stats()["max_slab_bytes"] <= max(chunk // 4, 8 * n_max) + 8 * n_max      # one slab of int64 (x ~4 temporaries <= chunk)
    # the real gate's largest case under the default chunk: slabs of (128 MiB / 4) at most
    big = {"id": "big", "c_z": 384, "c_hidden": 384, "n": 403, "batch": 1, "direction": "outgoing", "form": "bf16", "residual": True, "mask": "tail3", "variant": "fast"}
    rows = max(1, NV.GATE_CHUNK_BYTES // (big["n"] * big["c_z"] * 8 * 4))
    assert rows * big["n"] * big["c_z"] * 8 <= NV.GATE_CHUNK_BYTES // 4 < 3 * big["n"] ** 2 * big["c_z"] * 8      # vs the unchunked 3 x 0.5 GB int64 temporaries


class _FakeFace(object):
    """Stands in for the payload face on a host without a device: check() runs a miniature 'gate' through the vectors module's (shimmed) names."""
    class Refusal(Exception):
        def __init__(self, kind, detail=""):
            Exception.__init__(self, kind); self.kind = kind; self.detail = detail

    def __init__(self, V, cls):
        self.V = V; self.cls = cls; self.calls = []; self.seen = {}
        self.__name__ = V.__name__.rsplit(".", 1)[0] + ".face"

    def check(self, device=None, gate=True, binding=None, cases="gate", probe=False, verbose=False):
        self.calls.append({"gate": gate, "cases": cases})
        V = self.V
        self.seen["chunked_during_check"] = bool(getattr(V.closed_form_inputs, "_chunked", False))
        if gate:
            cache = {}
            served = []
            orig = self.seen.setdefault("orig_serve", None)
            for case in CASES[:3]:
                z, m, w = V.closed_form_inputs(case, "cpu")
                for rep in range(3):                                         # a multiweight-style interleave of ONE case: the shared cache must survive it
                    NV_SERVE(V, self, case, z, m, w, cache)
                    cache[("ws", case["id"], rep)] = z                       # what a served call leaves in the cache (workspaces / packs)
                    served.append((case["id"], len(cache)))
            self.seen["served"] = served
        return {"device_class": self.cls, "arch": "sm_90a", "status": "ok"}


def NV_SERVE(V, F, case, z, m, w, cache):
    return V._serve_case(F, case, z, m, w, cache)


def test_install_runs_the_gate_inside_the_memory_scope_and_holds_nothing_afterwards(monkeypatch):
    pytest.importorskip("torch")
    V = _vectors()
    cls = (NV.device_classes() or ["h100"])[0]
    fake = _FakeFace(V, cls)
    orig_cfi, orig_serve = V.closed_form_inputs, V._serve_case
    monkeypatch.setattr(V, "_serve_case", lambda F, case, z, m, w, cache: ("served", case["id"]))    # the payload's serve needs a device; the scope wraps whatever is bound
    monkeypatch.setattr(NV, "face", lambda: fake)
    monkeypatch.setitem(NV._STATE, "report", {})
    s0 = NV.gate_memory_stats()
    rep = NV.install(device=None, gate=True, force_gate=True)
    s1 = NV.gate_memory_stats()
    assert fake.calls == [{"gate": True, "cases": "gate"}] and rep["gate_ran"] is True and rep["verdict_stamp"] == "disabled:no_device_facts"
    assert fake.seen["chunked_during_check"] is True                         # inputs regenerated in slabs while the gate ran
    assert V.closed_form_inputs is orig_cfi and not getattr(V.closed_form_inputs, "_chunked", False)   # shims removed afterwards
    assert s1["scopes"] == s0["scopes"] + 1 and s1["released"] == s0["released"] + 1
    # per-case cache: the shared cache was emptied when the case changed (twice for three cases), never within one case's interleave
    served = fake.seen["served"]
    assert s1["cache_evictions"] - s0["cache_evictions"] == 3 + 3           # case a's 3 entries dropped at b's first call, b's 3 at c's
    assert [n for cid, n in served if cid == "a"] == [1, 2, 3] and [n for cid, n in served if cid == "c"] == [1, 2, 3]
    assert "gate_mem" in rep and rep["gate_mem"]["scopes"] == s1["scopes"]


def test_a_verification_stamp_for_this_stack_device_and_digest_skips_the_replay(monkeypatch, tmp_path):
    """The verdict is cached per (payload SHA256SUMS digest, package path, versions, device cc + name, driver, CUDA, torch, binding, cases, python):
    an intact stamp -> check(gate=False) (digests + cubin loads + load check only, no vector replay); any digest / stack change -> another key ->
    the gate runs again.  In-process the report is memoised per (device, gate, cases, binding)."""
    pytest.importorskip("torch")
    V = _vectors()
    cls = (NV.device_classes() or ["h100"])[0]
    facts = {"schema": NV.STAMP_SCHEMA, "sums_sha256": "ab" * 32, "pkg": "/x", "active_pkg": NV.ACTIVE_PKG, "payload_version": "9.9.9", "cc": "9.0",
             "device_name": "TEST H100", "driver_version": 1, "cuda_version": "13.0", "torch": "2.13.0", "binding": "auto", "cases": "gate", "python": "3.11"}
    monkeypatch.setattr(NV, "stamp_facts", lambda device=None, binding=None, cases="gate", digests=None: dict(facts, cases=str(cases)))
    monkeypatch.setattr(NV, "stamp_dirs", lambda: [("test", str(tmp_path))])
    monkeypatch.delenv(NV.STAMP_ENV, raising=False)
    monkeypatch.setattr(V, "_serve_case", lambda F, case, z, m, w, cache: ("served", case["id"]))
    fake = _FakeFace(V, cls)
    monkeypatch.setattr(NV, "face", lambda: fake)
    monkeypatch.setitem(NV._STATE, "report", {})
    rep1 = NV.install(device=None, gate=True)                                # no stamp yet: the gate runs and a stamp is written
    assert fake.calls[-1]["gate"] is True and rep1["verdict_stamp"] == "written" and os.path.isfile(rep1["verdict_stamp_path"])
    assert NV.install(device=None, gate=True) is rep1 and len(fake.calls) == 1          # same process: memoised, no second check at all
    NV._STATE["report"].clear()                                              # a later process on the same stack (fresh interpreter state, stamp on disk)
    rep2 = NV.install(device=None, gate=True)
    assert fake.calls[-1]["gate"] is False and rep2["verdict_stamp"] == "hit" and rep2["gate_ran"] is False       # no replay
    NV._STATE["report"].clear()                                              # a changed payload digest (or torch / driver / device): another key -> the gate runs again
    monkeypatch.setattr(NV, "stamp_facts", lambda device=None, binding=None, cases="gate", digests=None: dict(facts, sums_sha256="cd" * 32, cases=str(cases)))
    rep3 = NV.install(device=None, gate=True)
    assert fake.calls[-1]["gate"] is True and rep3["verdict_stamp"] == "written" and rep3["verdict_stamp_path"] != rep1["verdict_stamp_path"]


def test_gate_peak_on_a_device_is_bounded_and_nothing_is_held_afterwards():
    """Device check (skipped without CUDA): with a 4 GiB tensor resident, a forced gate replay's peak allocation above the resident level stays under
    the per-case bound and the allocator holds no gate block afterwards."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    resident = torch.empty(4 << 30, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated(); base_res = torch.cuda.memory_reserved()
    NV._STATE["report"].clear()
    rep = NV.install(gate=True, force_gate=True)
    peak = torch.cuda.max_memory_allocated() - base_alloc
    held = torch.cuda.memory_reserved() - base_res
    assert rep["gate_ran"] is True
    assert peak < (1280 << 20), peak                                         # the largest gate case ((384,384) x 403 tokens) alone needs ~1 GiB of inputs + workspaces + output
    assert held <= (64 << 20), held                                          # freed blocks returned to the driver on exit
    del resident


def test_stamp_directory_resolution_order_prefers_the_image_state_dir_and_uses_tmp_last(monkeypatch, tmp_path):
    """$OPT_CORE_VERDICT_DIR, when set, is the ONLY stamp directory (<dir>/trimul_native); unset, the order is the kits' JIT / warm-cache root
    (<MODEL_OPT_JIT_ROOT>[/<MODEL_OPT_STACK_KEY>]/trimul_native, the directory warm images bake their caches into) -> $XDG_CACHE_HOME -> ~/.cache ->
    the per-uid temp dir LAST (a stamp there does not survive a container start, so a bake must never depend on it)."""
    state = tmp_path / "image_state"; jit = tmp_path / "jit"; xdg = tmp_path / "xdg"; home = tmp_path / "home"
    for d in (state, jit, xdg, home):
        d.mkdir()
    monkeypatch.setenv("HOME", str(home)); monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(jit)); monkeypatch.setenv("MODEL_OPT_STACK_KEY", "stackA")
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.setenv(NV.STAMP_ENV, str(state))
    assert NV.stamp_dirs() == [("env", os.path.join(str(state), "trimul_native"))] and NV.stamp_dir() == os.path.join(str(state), "trimul_native")
    monkeypatch.setenv(NV.STAMP_ENV, "0")
    assert NV.stamp_dirs() == [] and NV.stamp_dir() is None                  # off: the gate runs in every process, nothing read or written
    monkeypatch.delenv(NV.STAMP_ENV, raising=False)
    kinds = [k for k, _d in NV.stamp_dirs()]
    assert kinds[0] == "triattn_root" and NV.stamp_dirs()[0][1] == os.path.join(str(jit), "stackA", "trimul_native"), NV.stamp_dirs()
    assert kinds.index("xdg") < kinds.index("home") and kinds[-1] in ("tmp", "home") and (kinds[-1] == "tmp" or "tmp" not in kinds)
    assert NV.stamp_dir() == os.path.join(str(jit), "stackA", "trimul_native")   # the first writable = the image's JIT / state root, never tmp while that exists


def test_a_bake_with_OPT_CORE_VERDICT_DIR_ships_a_stamp_that_a_fresh_process_on_the_image_honours(monkeypatch, tmp_path):
    """The image recipe: run the gate ONCE at bake time on the target device class with $OPT_CORE_VERDICT_DIR=<image state dir>
    (python -m opt_core.kernels.trimul.native stamp); the stamp lands in <dir>/trimul_native/gate-<key>.json inside the image; a process started
    from the image with the same variable finds it and skips the replay (digests + cubin loads + load check still run)."""
    pytest.importorskip("torch")
    V = _vectors()
    cls = (NV.device_classes() or ["h100"])[0]
    state = tmp_path / "opt_core_state"
    monkeypatch.setenv(NV.STAMP_ENV, str(state))
    facts = {"schema": NV.STAMP_SCHEMA, "sums_sha256": "ef" * 32, "pkg": "/img/opt_core", "active_pkg": NV.ACTIVE_PKG, "payload_version": "9.9.9", "cc": "9.0",
             "device_name": "TEST H100", "driver_version": 1, "cuda_version": "13.0", "torch": "2.13.0", "binding": "auto", "cases": "gate", "python": "3.11"}
    monkeypatch.setattr(NV, "stamp_facts", lambda device=None, binding=None, cases="gate", digests=None: dict(facts, cases=str(cases)))
    monkeypatch.setattr(V, "_serve_case", lambda F, case, z, m, w, cache: ("served", case["id"]))
    fake = _FakeFace(V, cls)
    monkeypatch.setattr(NV, "face", lambda: fake)
    monkeypatch.setitem(NV._STATE, "report", {})
    bake = NV.install(device=None, gate=True)                                # the bake step
    assert fake.calls[-1]["gate"] is True and bake["verdict_stamp"] == "written" and bake["verdict_stamp_dir_kind"] == "env"
    assert bake["verdict_stamp_path"].startswith(os.path.join(str(state), "trimul_native") + os.sep) and os.path.isfile(bake["verdict_stamp_path"])
    NV._STATE["report"].clear()                                              # a container started from the image: fresh interpreter, same variable, stamp on disk
    run = NV.install(device=None, gate=True)
    assert fake.calls[-1]["gate"] is False and run["verdict_stamp"] == "hit" and run["gate_ran"] is False and run["verdict_stamp_path"] == bake["verdict_stamp_path"]
