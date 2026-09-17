"""kits/v1_25 at 0 GPU: the numerics read-back, the one apply line, the batch-keyed dispatcher and its sighting rule, the JIT accounting,
the device-class gate, remove(), the documented helper route."""
import ast
import numpy as np
import importlib
import inspect
import re
import os, sys
import tempfile

import pytest
import torch

v25 = importlib.import_module("engines.flashzoi.kits.v1_25")
W = importlib.import_module("engines.flashzoi.kits.v1_25._wrap")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_numerics_class_read_back_from_torch_switches(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True); monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.set_float32_matmul_precision("highest")
    d = W.read_numerics_class(); assert d["class"] == "torch-default" and d["cudnn_allow_tf32"] and not d["matmul_allow_tf32"]
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    assert W.read_numerics_class()["class"] == "tf32-off"
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    assert W.read_numerics_class()["class"] == "tf32-on"
    with torch.autocast("cpu", dtype=torch.bfloat16):
        c = W.read_numerics_class()["class"]
    assert c.startswith("tf32-on") and "autocast" in c
    assert "pinned to the stock" in d["kit_decision"]


class _Runner:
    """A stub of the constructed runner's fields the line and the dispatcher read."""

    def __init__(self):
        self.arm = "v1.25"; self.device_name = "NVIDIA H100 80GB HBM3"; self.levers = frozenset({"graph", "pinned", "stage1", "sites", "precast"})
        self.numerics = {"class": "torch-default", "cudnn_allow_tf32": True, "matmul_allow_tf32": False, "float32_matmul_precision": "highest"}
        self.capture_s_by_batch = {}; self.capture_refused = {}; self.batch = 1
        self.capture = {}; self.stack = None; self.graph_keys = {}; self.graph = None
        self._pkg = "engines.flashzoi.kits.v1_25"
        self.apply_s = {"pool_slots": 4, "pool_s": 0.0, "pool_lazy": True, "jit_cache": "no shipped cache for class h100", "warmup_total_s": 0.61,
                        "patch_s": 0.3, "warmup_forward_s": 0.31, "warmup_batch": 1, "capture_s": 0.0, "flags_at_rest": "unchanged (read back == snapshot at entry)"}
        self.graphs = {}; self.counts = {"graph_replays": 0, "eager_fused_calls": 0, "lazy_captures": 0, "stage1_calls": 0, "predict_calls": 0}
        self.model = torch.nn.Module(); self.model.human_head = torch.nn.Conv1d(8, 5, 1)             # the head a graph is keyed by (graph_head_key)
        self._graph_pool = None; self._eager_fn = lambda xt: ("eager", int(xt.shape[0]))


class _FakeGraph:
    def __init__(self, b):
        self.x = torch.zeros((b, 4, 8)); self.replays = 0

    def __call__(self, x):
        self.replays += 1; return ("graph", int(x.shape[0]))


def test_the_documented_apply_signature():
    sig = inspect.signature(v25.KitRunner.__init__)
    assert list(sig.parameters) == ["self", "model", "levers", "device", "batch", "pool_size", "numerics"]      # kit.KitRunner(model): no switches beyond the levers and the mode's one knob
    assert sig.parameters["numerics"].default == "tf32" and sig.parameters["batch"].default == 1 and sig.parameters["pool_size"].default is None


def test_apply_line_is_one_line_with_every_field():
    r = _Runner(); line = W.compose_apply_line(r)
    assert "\n" not in line and line.startswith("[flashzoi kit v1.25] GPU NVIDIA H100 80GB HBM3 (device class: pinned) | numerics: torch-default")
    for field in ("levers ON: graph, pinned, precast, sites, stage1", "| act dtype fp16 |", "apply 0.61 s = patch 0.30 + warm-up forward@batch 1 0.31",
                  "graphs: one CUDA-graph capture per batch shape at its first call, replay from then on; pinned pool 4 slots at the first lease",
                  "batch bound: 15 windows per dispatch", "jit: no shipped cache", "compile(): 0 calls", "flags at rest: unchanged", "helper: not routed", "skipped: none"):
        assert field in line, field
    r.capture_refused[8] = "capture refused at batch 8: RuntimeError: x — eager fused path (same kernels, no graph)"
    r.apply_s["warmup_skipped"] = "done by runner #1 in this process (the JIT cache + kernels are process-wide; this runner's own first-touch is paid by its first call)"
    line2 = W.compose_apply_line(r)
    assert "graph@batch 8: capture refused" in line2 and "skipped: warm-up forward: done by runner #1" in line2
    r.device_class = {"device_name": r.device_name, "pinned": False, "record": "/kit/class_records/a100.json"}
    assert "(device class: by record a100.json) | numerics:" in W.compose_apply_line(r)                # the A100 form the activation table matches


def test_dispatcher_captures_at_the_first_call_replays_after_rekeys_on_a_new_head_refused_stays_eager_and_oom_propagates(monkeypatch):
    r = _Runner()
    monkeypatch.setattr(W, "graph_cache_pins", lambda m: {}); monkeypatch.setattr(W, "graph_pin_refs", lambda rr: None)
    calls = []

    def fake_capture(runner, fz, b, device):
        calls.append(b); runner.counts["stage1_calls"] += 4                     # the capture's 3 warm-ups + 1 captured forward launch stage-1
        if b == 16:
            raise RuntimeError("CUDA out of memory (fake)")                     # an out-of-memory error during the capture PROPAGATES (never the eager fallback)
        if b >= 8:
            raise RuntimeError("capture failed (fake, not a memory error)")     # any other capture failure: named, the batch stays on the eager fused path
        return _FakeGraph(b)
    monkeypatch.setattr(W, "_capture_graph", fake_capture)
    d = W._GraphDispatch(r, fz=None, device="cpu")
    x1 = torch.zeros((1, 4, 8))
    assert d(x1) == ("graph", 1) and calls == [1] and r.counts["lazy_captures"] == 1 and r.counts["graph_replays"] == 1  # the FIRST call of a shape: captured + replayed, no eager call
    assert r.counts["stage1_calls"] == 0 and r.capture["stage1_calls"] == 4 and r.capture["captures"] == 1               # capture-time launches bookkept apart
    assert d(x1) == ("graph", 1) and r.counts["graph_replays"] == 2 and r.graph is r.graphs[1] and calls == [1] and r.counts["eager_fused_calls"] == 0
    g_old = r.graphs[1]
    r.model.human_head = torch.nn.Conv1d(8, 3, 1)                                                                            # upstream's set_track_subset installs a NEW head: the graph captured with the old one must not replay
    assert d(x1) == ("graph", 1) and calls == [1, 1] and r.graphs[1] is not g_old and r.counts["lazy_captures"] == 2 and r.graph_keys[1] == W.graph_head_key(r.model)
    assert d(x1) == ("graph", 1) and calls == [1, 1]                                                                       # the re-keyed graph replays from then on
    x8 = torch.zeros((8, 4, 8))
    for _ in range(3):
        assert d(x8) == ("eager", 8)
    assert calls == [1, 1, 8] and "capture refused at batch 8" in r.capture_refused[8] and r.counts["eager_fused_calls"] == 3   # a refused batch: named once, eager, not retried every call
    x16 = torch.zeros((16, 4, 8))
    with pytest.raises(RuntimeError, match="out of memory"):
        d(x16)                                                                                                              # the first call's capture hits an OOM: it propagates
    assert 16 not in r.capture_refused and 16 not in r.graphs and calls == [1, 1, 8, 16]
    st = d.status(); assert set(st) == {"graph_replays", "graph_shapes", "capture_s", "capture_refused"} and st["capture_refused"] and d.replays == r.graphs[1].replays


def test_no_capture_heuristic_in_the_kit():
    """The graph rule is structural (a shape's first call captures), never a call-count / break-even heuristic: no sighting rule in PINS or the wrapper."""
    assert "sighting_rule" not in v25.PINS
    src = inspect.getsource(W)
    assert "sighting" not in src and "break_even" not in src and "first call" in inspect.getsource(W._GraphDispatch).lower()


def test_every_documented_surface_is_served_nothing_refused():
    """serve_surfaces installs forward (every argument form) + get_embs_after_crop through the runner and the module-move wrappers that detach first;
    predict / predict_gene_count / set_track_subset stay upstream's own code; no REFUSED / NotImplemented surface anywhere in the wrapper."""
    import types
    src = inspect.getsource(W)
    assert "refuse_unsupported_surfaces" not in src and "REFUSED_SURFACES" not in src and "is not supported through the kit" not in src
    seen = {}
    runner = types.SimpleNamespace(forward_any=lambda x, **kw: ("forward_any", kw), embs_after_crop=lambda x: ("embs", tuple(x.shape)), detach=lambda reason: seen.setdefault("detach", reason), close=lambda: {})
    m = torch.nn.Module(); m.human_head = torch.nn.Conv1d(8, 5, 1)
    served = W.serve_surfaces(m, runner)
    assert served[:2] == ["forward", "get_embs_after_crop"] and "to() -> detach" in served and "half() -> detach" in served
    x = torch.zeros((2, 4, 8))
    assert m(x, is_human=False, return_embeddings=True) == ("forward_any", {"is_human": False, "data_parallel_training": False, "return_embeddings": True})
    assert m.get_embs_after_crop(x) == ("embs", (2, 4, 8)) and m._flashzoi_kit_runner is runner
    assert not hasattr(type(m), "set_track_subset") or "set_track_subset" not in m.__dict__                              # upstream's own methods are never shadowed
    m.float()                                                                                                              # a move: detach first (named), then the module's own method
    assert seen["detach"] == "model.float() called"


def test_apply_items_from_bytes():
    src = inspect.getsource(W)
    assert "_PROCESS_WARMED" in src and "done by runner #" in src                                   # the warm-up forward once per process
    assert "def route_documented_helper" in src and "stock.__code__ = new_code" in src and 'self.apply_s["helper_route"]' in src   # the documented helper routed by the code swap + stamped
    assert 'numerics: str = "tf32"' in src and "torch.backends.cuda.matmul.allow_tf32 = bool(_tf[0])" in src           # the knob, per-call scope: the head GEMM's TF32 = cudnn.allow_tf32 at rest
    body = src[src.index("        def predict_tensor(self, xt):"):src.index("        def predict(self, x")]
    assert body.index("_tf = (torch.backends.cudnn.allow_tf32") < body.index("out = self.fn(xt)") < body.index("torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = _tf")
    assert v25.IMPORT_WALL["import_s"] > 0 and "jit_started_at" in v25.IMPORT_WALL
    assert "os.environ" not in re.sub(r'os\.environ(\.get\(|\[)"TRITON_CACHE_DIR"', "", src)          # the wrapper reads no environment switch (TRITON_CACHE_DIR is Triton's own variable)


def test_class_record_resolves_from_the_kit_dir(tmp_path, monkeypatch):
    """class_records/*.json in the kit dir, keyed by the kit's fz_exact digest; no environment override."""
    pins = {"device_names": ["NVIDIA H100 80GB HBM3"], "fz_exact_sha256": "abc"}
    assert W.assert_device_class(pins, "NVIDIA H100 80GB HBM3")["pinned"] is True
    dc = W.assert_device_class(pins, "NVIDIA A100 80GB PCIe", (8, 0), 81920)                    # the dir's record names sm_80 for ANOTHER digest: served unpinned, the record named in the reason
    assert dc["pinned"] is False and dc["record"] is None and "a100.json" in dc["unpinned"]
    src = inspect.getsource(W.assert_device_class)
    assert "class_records" in src and "os.environ" not in src


def test_helper_route_is_the_direct_landing_with_the_staging_fallback():
    """The documented helper's host side has ONE form: each fold's device slice lands straight in its slot of a pooled page-locked result
    ('direct'); the pinned-staging path is its fallback (a pool overflow or a host-slicing index), never a switch."""
    from engines.flashzoi import predict_tracks_fast as PF
    sig = inspect.signature(PF.predict_tracks_fast)
    assert list(sig.parameters) == ["models", "sequence_one_hot", "slices", "copy_threads", "host_order"] and sig.parameters["host_order"].default == "stock" and PF.HOST_ORDERS == ("stock", "C")
    assert hasattr(PF, "LAST_CALL") and PF.DIRECT_POOL_KIND == "registered"
    src = open(os.path.join(HERE, "_wrap.py")).read()
    assert "return PF.predict_tracks_fast(models, sequence_one_hot, slices)" in src                  # upstream's name keeps the stock's order (bytes AND strides); the CLI asks for C order
    with pytest.raises(ValueError): PF.predict_tracks_fast([], torch.zeros(4, 8), slice(None), host_order="F")
    n = 7611                                                                                        # the device index forms: a slice / a gather (normalised) / host kinds
    kind, idx = PF._device_index([5, -1, 7000], n); assert kind == "gather" and list(idx) == [5, n - 1, 7000]
    assert PF._device_index(slice(0, 89), n) == ("slice", slice(0, 89)) and PF._device_index(3, n)[0] == "host" and PF._device_index(slice(None, None, -1), n)[0] == "host"


def test_effective_flags_of_the_graph_lever():
    eff = W.effective_flags_from({"graph"}, {"predict_calls": 2, "graph_replays": 0, "eager_fused_calls": 2}, {"graph": 1}, {"graph": 1})
    assert eff["graph"] is True                                                                                             # a job of eager (variant-surface) calls only = effective by construction
    assert W.effective_flags_from({"graph"}, {"predict_calls": 2, "graph_replays": 0, "eager_fused_calls": 1}, {"graph": 1}, {"graph": 1})["graph"] is False


def test_batch_check_accepts_other_batches_under_the_dispatcher_and_refuses_empty():
    r = _Runner(); r.graph = object()
    assert W._predict_tensor_checks(r, torch.zeros((3, 4, W.lane.SEQ_LEN), dtype=torch.float32))
    with pytest.raises(W.lane.PinDrift):
        W._predict_tensor_checks(r, torch.zeros((0, 4, W.lane.SEQ_LEN), dtype=torch.float32))
    with pytest.raises(W.lane.PinDrift):
        W._predict_tensor_checks(r, torch.zeros((1, 4, 10), dtype=torch.float32))


def test_no_flag_residue_is_read_back_and_refused(monkeypatch):
    snap = W.numerics_flags_snapshot(); assert set(snap) == set(W.NUMERICS_FLAGS)
    W.assert_no_flag_residue(snap, "apply()")                                   # nothing moved -> silent
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", not snap["cudnn_allow_tf32"])
    with pytest.raises(W.lane.PinDrift, match="flag residue"):
        W.assert_no_flag_residue(snap, "predict_tensor()")


def test_kit_bytes_write_no_global_numerics_flag_at_rest():
    """From bytes: no assignment to a process-global numerics switch outside (a) fz_exact's never-called set_det() helper and (b) the
    numerics='tf32' knob's PER-CALL scope in predict_tensor — set under `if self.numerics_knob == "tf32"` and restored in the same call's
    `finally` (the read-back after every call proves nothing moved at rest)."""
    import re
    for f in ("__init__.py",):
        src = open(os.path.join(HERE, f)).read()
        assert not re.search(r"torch\.backends\.[a-z.]*(allow_tf32|benchmark|deterministic)\s*=[^=]", src), f
        assert "set_float32_matmul_precision(" not in src and "use_deterministic_algorithms(" not in src, f
    src = open(os.path.join(HERE, "_wrap.py")).read()
    writes = [m.start() for m in re.finditer(r"torch\.backends\.[a-z.]*(allow_tf32|benchmark|deterministic)\s*=[^=]", src)]
    body = src[src.index("        def predict_tensor(self, xt):"):src.index("        def _chunks(self, x, call):")]
    lo, hi = src.index("        def predict_tensor(self, xt):"), src.index("        def _chunks(self, x, call):")
    assert writes and all(lo < p < hi for p in writes), [src[p - 40:p + 60] for p in writes if not (lo < p < hi)]   # every write lives inside predict_tensor / _eager_call (per call)
    assert body.count('if self.numerics_knob == "tf32":') == 2 and body.count("finally:") == 2                     # each sets the pair for ITS call and restores it in the same call's finally
    assert "set_float32_matmul_precision(" not in src and "use_deterministic_algorithms(" not in src
    fz = open(os.path.join(HERE, "fz_exact.py")).read()
    assert "set_det" not in fz and "allow_tf32 =" not in fz and "use_deterministic_algorithms" not in fz   # the kernel module writes no global numerics flag
    assert not re.search(r"^\s*set_det\(\)", fz, re.M)


def test_no_env_write_at_import(monkeypatch):
    """Importing the kit writes no FZ_* env var (the dtype is fz_exact's own default; the line names the env state)."""
    src = open(os.path.join(HERE, "__init__.py")).read() + open(os.path.join(HERE, "_wrap.py")).read()
    import re
    assert not re.search(r"os\.environ\.setdefault\(\s*[\"']FZ_", src) and not re.search(r"os\.environ\[[\"']FZ_[A-Z_]+[\"']\]\s*=", src)
    r = _Runner(); line = W.compose_apply_line(r); assert "act dtype fp16" in line


def test_per_call_flag_read_back_is_after_vs_before_the_call(monkeypatch):
    """A torch.autocast scope inside the route once tripped the residue check: the read-back compares the flags AFTER the
    call with those BEFORE it (the caller's context is the caller's), never with apply's entry snapshot."""
    src = inspect.getsource(W)
    body = src[src.index("        def predict_tensor(self, xt):"):src.index("        def ", src.index("        def predict_tensor(self, xt):") + 10)]
    assert "_flags_before = numerics_flags_snapshot()" in body and 'assert_no_flag_residue(_flags_before, "predict_tensor()")' in body
    assert 'assert_no_flag_residue(self._flags_at_entry, "predict_tensor()")' not in body
    with torch.autocast("cpu", dtype=torch.bfloat16):
        snap = W.numerics_flags_snapshot(); W.assert_no_flag_residue(snap, "inside a caller's autocast")      # no residue relative to the call's own start


def test_every_lever_effective_on_the_pure_eager_path_no_capture_at_all():
    """v1_25's one rule fix: a job shorter than K sightings never captures — every stamped lever must still read EFFECTIVE from the eager path's
    own counters."""
    counts = {"predict_calls": 106, "graph_replays": 0, "eager_fused_calls": 106, "stage1_calls": 106, "leases": 106, "releases": 106, "predict_host_calls": 106, "lazy_captures": 0}
    for p in ("fz_forwards", "fz_decoder_fused", "fz_ln_fused", "fz_transformer_fused", "fz_skip_fused", "fz_decoder_fused2", "fz_relu_epilogue", "fz_stage1_mma", "fz_tower_nhwc", "fz_crop_aligned", "fz_head_baddbmm", "fz_bias_fold", "fz_batch_units",
              "site_fused_site_nhwc", "site_bias_to_nchw", "site_bn_gelu_nchw", "site_ln_fused_add", "site_relu_epilogue", "site_up2_add", "site_head_gemm_softplus"):
        counts[p] = 106
    tags = {"graph": 1, "pinned": 2, "precast": 48, "stage1": 3, "sites": (True, "nhwc"), "crop": "aligned", "head": "baddbmm", "decoder_fused": 4, "transformer_fused": 5, "ln_fused": 6, "relu_epilogue": 7, "skip_fused": 8, "decoder_fused2": 9}
    eff = W.effective_flags_from(set(v25.LEVERS), counts, tags, dict(tags))
    assert eff["stage1"] is True and eff["graph"] is True, eff
    assert all(eff[lv] for lv in v25.LEVERS), {k: v for k, v in eff.items() if not v}


def test_helper_router_code_swap_reaches_a_name_bound_before_apply(monkeypatch):
    """The stock predict_tracks function OBJECT gets the kit body by __code__ swap — a name bound BEFORE apply (the notebook's
    `from ... import predict_tracks`) runs the pinned path over kit-attached models and the stock body otherwise; restored by unroute."""
    import types, sys as _s
    fake = types.ModuleType("borzoi_pytorch.pytorch_borzoi_helpers"); fake.__dict__["calls"] = []
    exec("def predict_tracks(models, sequence_one_hot, slices):\n    calls.append('stock'); return 'stock'\n", fake.__dict__)   # closure-free, the stock's parameter names
    stock = fake.predict_tracks
    pkg = types.ModuleType("borzoi_pytorch"); pkg.pytorch_borzoi_helpers = fake
    monkeypatch.setitem(_s.modules, "borzoi_pytorch", pkg); monkeypatch.setitem(_s.modules, "borzoi_pytorch.pytorch_borzoi_helpers", fake)
    W._HELPER_STOCK.clear()
    bound_before = stock                                                        # the notebook's binding BEFORE apply
    line = W.route_documented_helper(); assert "code-swap on the bound object" in line and fake.predict_tracks is stock   # the SAME object
    m_stock = types.SimpleNamespace(); assert bound_before([m_stock], None, None) == "stock" and fake.calls == ["stock"]   # a stock model -> the stock body
    m_kit = types.SimpleNamespace(_flashzoi_kit_runner=object())
    monkeypatch.setattr(W, "predict_tracks", lambda models, x, s, **kw: "kit")
    fake.__dict__["_flashzoi_kit_predict_tracks"] = W.predict_tracks           # the injected name (the module's dict is what the swapped code resolves)
    assert bound_before([m_kit], None, None) == "kit"                         # the name bound before apply runs the kit body
    assert W.route_documented_helper().endswith("(already applied in this process)")
    assert W.unroute_documented_helper() and stock.__code__ is W._HELPER_STOCK["code"] and bound_before([m_kit], None, None) == "stock"


def test_user_path_imports_nothing_outside_the_kit():
    """The kit's USER PATH (import -> apply -> forward) imports no module outside the kit: after a stub apply in a fresh interpreter
    sys.modules holds none of the excluded roots."""
    import subprocess, sys as _s, re as _re, os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); proj = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    assert os.path.isdir(os.path.join(proj, "engines", "flashzoi", "kits")), proj
    env = dict(os.environ, FZ_KIT_BG="off", KMP_AFFINITY="disabled", OMP_PROC_BIND="false"); env.pop("PYTHONPATH", None)
    r = subprocess.run([_s.executable, "-X", "importtime", "-c", "import sys; sys.path.insert(0, %r); import numpy, torch; import engines.flashzoi.kits.v1_25" % proj], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]
    mods = [m.group(1) for m in _re.finditer(r"import time:\s+\d+ \|\s+\d+ \|\s*(\S+)", r.stderr)]
    bad = [m for m in mods if _re.match(r"(engines\.core|engines\.flashzoi\.lane|engines\.borzoi\.(?!__init__)|compare(\.|$)|abeval|orchestration|panels)", m)]
    assert bad == [], bad
    from engines.flashzoi.kits.v1_25 import _pins as P
    assert (P.SEQ_LEN, P.OUTPUT_SHAPE, P.AUTOCAST_DTYPE) == (524288, (7611, 6144), "float16")


def test_ensure_pool_allocates_lazily_at_the_first_lease():
    """_ensure_pool builds the runner's own lease pool on the first lease (never inside apply, no background preparation): the stamps
    say so and the pool is the kit's OutputLeasePool over the pinned output shape."""
    src = open(os.path.join(HERE, "_wrap.py")).read()
    i = src.index("        def _ensure_pool(self) -> None:"); body = src[i:src.index("\n        def ", i + 10)]
    assert "pool_take" not in body and body.count("OutputLeasePool(") == 1 and 'self.apply_s["pool_first_lease"] = True' in body
    made = []
    class _FZ:
        @staticmethod
        def OutputLeasePool(template, n):
            made.append((tuple(template.shape), str(template.dtype), n)); return ("pool", n)
    ns = {"torch": torch, "time": __import__("time"), "sys": sys, "lane": W.lane}
    exec("class _R:\n" + "\n".join("    " + l[4:] if l.startswith("        ") else l for l in body.splitlines()), ns)
    r = ns["_R"](); r.pool = None; r._pool_n = 4; r.apply_s = {}; r.applied = {}; r.attach_tags = {}; r.device = "cpu"; r.fz = _FZ
    r._ensure_pool()
    assert made == [(tuple(W.lane.OUTPUT_SHAPE), "torch.float32", 4)] and r.pool == ("pool", 4) and r.applied["pinned_slots"] == 4 and r.apply_s["pool_first_lease"] is True
    init_src = open(os.path.join(HERE, "__init__.py")).read()
    assert "_bg_pool" not in init_src and "pool_take" not in init_src              # no background pool: nothing is allocated on the device before a lease


def test_remove_restores_the_model_and_the_helper_code(monkeypatch):
    """The documented remove — a stub model mutated the way apply does (precast to fp16 via new .data tensors; the LayerNorm swap;
    forward + the attach mark) is restored (the original objects) by KitRunner.close(); kit.remove(runners) restores the helper's __code__
    on the bound object and detaches the JIT hook; the printed line carries no FAIL word; a second remove is idempotent."""
    import types, sys as _s
    class _Attn(torch.nn.Module):
        def __init__(self): super().__init__(); self.inner_cross_attn = torch.nn.Identity()
    class _Tr(torch.nn.Module):
        def __init__(self): super().__init__(); self.ln = torch.nn.LayerNorm(1536); self.blk = _Attn(); self.lin = torch.nn.Linear(4, 4)
    class _M(torch.nn.Module):
        def __init__(self): super().__init__(); self.conv = torch.nn.Conv1d(4, 4, 3); self.transformer = _Tr(); self.human_head = torch.nn.Linear(4, 4)
    m = _M(); w0, b0, lin0, ln0, attn0 = m.conv.weight.data, m.conv.bias.data, m.transformer.lin.weight.data, m.transformer.ln, m.transformer.blk.inner_cross_attn
    r = types.SimpleNamespace(model=m, _orig={"params": {"conv": (m.conv, w0, b0), "transformer.lin": (m.transformer.lin, lin0, m.transformer.lin.bias.data)}, "layernorms": {("transformer", "ln"): (m.transformer, ln0)}},
                              stack=object(), fn=object(), graphs={1: object()}, _graph_pool=None, pool=object(), leased={}, _flags_at_entry=W.numerics_flags_snapshot())
    # the apply-like mutation
    m.conv.weight.data = w0.to(torch.float16); m.conv.bias.data = b0.to(torch.float16); m.transformer.lin.weight.data = lin0.to(torch.float16)
    r._model_dict_before = dict(m.__dict__)
    m.transformer.ln = torch.nn.Identity(); m.forward = types.MethodType(lambda self, x: x, m); m._flashzoi_kit_runner = r
    m.get_embs_after_crop = types.MethodType(lambda self, x: x, m)                      # a served surface the kit installs on the instance
    m.human_head_bak = torch.nn.Linear(4, 4)                                              # what upstream's own set_track_subset adds AFTER attach: not the kit's, survives close()
    out = v25.KitRunner.close(r)
    assert out["forward_restored"] and out["layernorms_restored"] and out["params_restored"] and out["stack_graphs_pool_released"] and out["flags_at_rest"] == "unchanged", out
    assert "get_embs_after_crop" not in m.__dict__ and "_flashzoi_kit_runner" not in m.__dict__ and not out["instance_attrs_added_by_kit_remaining"] and "human_head_bak" in m._modules
    assert m.conv.weight.data.data_ptr() == w0.data_ptr() and m.conv.weight.dtype == torch.float32 and m.transformer.ln is ln0 and "forward" not in m.__dict__
    # the helper swap + the hook, restored by kit.remove
    fake = types.ModuleType("borzoi_pytorch.pytorch_borzoi_helpers"); fake.__dict__["calls"] = []
    exec("def predict_tracks(models, sequence_one_hot, slices):\n    calls.append('stock'); return 'stock'\n", fake.__dict__)
    stock = fake.predict_tracks; pkg = types.ModuleType("borzoi_pytorch"); pkg.pytorch_borzoi_helpers = fake
    monkeypatch.setitem(_s.modules, "borzoi_pytorch", pkg); monkeypatch.setitem(_s.modules, "borzoi_pytorch.pytorch_borzoi_helpers", fake)
    W._HELPER_STOCK.clear(); W.route_documented_helper(); assert getattr(stock, "_flashzoi_kit_code_swapped", False)
    r.close = lambda: v25.KitRunner.close(r)
    res = W.remove([r], quiet=True)
    assert not res["fail"] and "FAIL" not in res["line"] and res["checks"]["helper_restored"] and stock.__code__ is W._HELPER_STOCK["code"], res["line"]
    assert stock([types.SimpleNamespace()], None, None) == "stock"
    res2 = W.remove([r], quiet=True); assert not res2["fail"]                                    # idempotent
    assert callable(getattr(v25, "remove", None)) and hasattr(v25, "_unhook_triton_compile")


def test_v1_25_batch_bound_from_the_constants_and_the_chunked_dispatch():
    """KIT_MAX_BATCH = min(the one flat int32 launch's bound, upstream's own maximum) = min(2**31 // (512 * SEQ_LEN//2) = 16, 15) = 15: the
    first fused tower site (608 x SEQ_LEN//2 per window, past 2**31 elements at 14 windows) indexes per window from a 64-bit base and no longer
    bounds the batch; a batch above 15 is dispatched in chunks of 15 through the same predict_tensor (the outputs concatenated in order;
    counted); at the bound no chunking. The filters the kernels are written for are asserted on the attached model (assert_batch_bound_dims)."""
    from engines.flashzoi.kits.v1_25 import _pins as lane
    assert (lane.STEM_CHANNELS, lane.TOWER1_CHANNELS, lane.STOCK_MAX_BATCH) == (512, 608, 15)
    assert W.kit_max_batch() == 15 and W.KIT_MAX_BATCH == 15 and W.kit_max_batch(524288, 512, 15) == 15
    assert W.kit_max_batch(524288, 512, 64) == 16 and (16 * 262144 * 512 - 1) <= 2 ** 31 - 1           # the stage-1 store alone: 16 windows fit its int32 offsets
    assert (14 * 262144 * 608 - 1) > 2 ** 31 - 1                                                        # the first site's whole-batch tensor passes 2**31 at 14: hence the 64-bit window base
    fz = open(os.path.join(HERE, "fz_exact.py")).read()
    site = fz[fz.index("def k_pool_bn_gelu_nhwc("):fz.index("@triton.jit", fz.index("def k_pool_bn_gelu_nhwc("))]
    assert "tl.program_id(1).to(tl.int64)" in site and "Xn = X + n64 * n_win * 2" in site and "tl.store(Yn + offs" in site
    assert "k_pool_bn_gelu_nhwc[(triton.cdiv(n_win, 1024), N)]" in fz                                    # one grid row per window
    src = open(os.path.join(HERE, "_wrap.py")).read()
    body = src[src.index("        def predict_tensor(self, xt):"):src.index("        def predict(self, x: np.ndarray")]
    assert "if xt.shape[0] > KIT_MAX_BATCH:" in body and "torch.split(xt, KIT_MAX_BATCH, dim=0)" in body and "torch.cat(outs, dim=0)" in body
    assert body.index("if xt.shape[0] > KIT_MAX_BATCH:") < body.index('self.counts["predict_calls"] += 1')                 # the chunk decision BEFORE the call is counted
    assert "batch bound: {KIT_MAX_BATCH} windows per dispatch" in src.split("def compose_apply_line")[1]
    # the dispatch itself on a stub: the recursion goes through predict_tensor (bound the real method to a stub runner)
    seen = []
    class R:
        counts = {"predict_calls": 0}; device = "cpu"
        numerics_knob = "tf32"; hook_ns = 0.0; hook_n = 0; warmup_deferred = None
        def fn(self, xt): seen.append(int(xt.shape[0])); return torch.zeros(xt.shape[0], *lane.OUTPUT_SHAPE, dtype=torch.float32)
    # a minimal predict_tensor: the chunk branch is taken from the real source; the per-chunk path replaced by the stub's fn (no CUDA on this host)
    code = "def pt(self, xt):\n" + "\n".join(l[4:] for l in body.splitlines()[1:] if l.strip())
    code = code.split("            xt = xt.to(device=self.device, dtype=torch.float32)")[0] if False else code
    ns = {"KIT_MAX_BATCH": 13, "torch": torch, "assert_no_flag_residue": lambda a, b: None, "numerics_flags_snapshot": lambda: {}, "_predict_tensor_checks": lambda r, x: True, "assert_graph_cache_pins": lambda r: None,
          "time": __import__("time"), "sys": sys, "pkg": __name__, "lane": lane, "getattr": getattr}
    # take ONLY the chunk branch + a stub tail (the real tail needs CUDA): assemble from the source lines up to and including the chunk branch
    lines = body.splitlines(); i0 = next(i for i, l in enumerate(lines) if "if xt.shape[0] > KIT_MAX_BATCH:" in l); i1 = next(i for i, l in enumerate(lines) if 'self.counts["predict_calls"] += 1' in l and i > i0)
    chunk_src = "def pt(self, xt):\n    _flags_before = {}\n" + "\n".join("    " + l[12:] for l in lines[i0:i1]) + "\n    self.counts['predict_calls'] += 1\n    return self.fn(xt)\n"
    chunk_src = chunk_src.replace("self.predict_tensor(piece)", "pt(self, piece)")
    exec(chunk_src, ns); pt = ns["pt"]
    r = R(); out = pt(r, torch.zeros(32, 4, lane.SEQ_LEN, dtype=torch.uint8)); assert seen == [13, 13, 6] and out.shape[0] == 32 and r.counts["chunked_calls"] == 1 and r.counts["chunks_dispatched"] == 3 and r.counts["predict_calls"] == 3
    seen.clear(); r = R(); r.counts = {"predict_calls": 0}; out = pt(r, torch.zeros(16, 4, lane.SEQ_LEN, dtype=torch.uint8)); assert seen == [13, 3] and out.shape[0] == 16      # 16 (the stage-1 store's own bound) is two dispatches, never one
    seen.clear(); r = R(); r.counts = {"predict_calls": 0}; out = pt(r, torch.zeros(14, 4, lane.SEQ_LEN, dtype=torch.uint8)); assert seen == [13, 1] and out.shape[0] == 14
    seen.clear(); r = R(); r.counts = {"predict_calls": 0}; out = pt(r, torch.zeros(13, 4, lane.SEQ_LEN, dtype=torch.uint8)); assert seen == [13] and "chunked_calls" not in r.counts
    # the order: chunk outputs concatenated in input order (distinct fills per chunk)
    seen.clear(); r = R(); r.counts = {"predict_calls": 0}
    r.fn = lambda xt: torch.full((xt.shape[0], 1, 1), float(xt.shape[0]) + float(xt[0, 0, 0]))
    x = torch.zeros(20, 4, lane.SEQ_LEN, dtype=torch.float32); x[13:, 0, 0] = 100.0
    out = pt(r, x); assert out.shape[0] == 20 and float(out[0, 0, 0]) == 13.0 and float(out[19, 0, 0]) == 107.0
    # the filters the bound is cut for, asserted on the attached model: pass-through without the modules, checked on the pinned dims, PinDrift otherwise
    from types import SimpleNamespace as NS
    fake = lambda stem, t1: NS(conv_dna=NS(conv_layer=NS(out_channels=stem)), res_tower=[NS(conv_layer=NS(out_channels=t1))])
    assert W.assert_batch_bound_dims(object()) == {"checked": False}
    assert W.assert_batch_bound_dims(fake(512, 608)) == {"checked": True, "stem_channels": 512, "tower1_channels": 608, "max_batch": 15}
    for dims in ((512, 736), (256, 608)):
        try: W.assert_batch_bound_dims(fake(*dims)); raise AssertionError(f"dims {dims} not refused")
        except lane.PinDrift as e: assert "index arithmetic (15 windows per dispatch)" in str(e)
    assert "self.bound_assertion = assert_batch_bound_dims(model)" in src      # asserted at attach (KitRunner.__init__)


def test_v1_25_predict_tensor_counts_device_side_outputs():
    """predict_tensor (device outputs; no lease) is counted apart from predict (host outputs; one lease per row)."""
    src = open(os.path.join(HERE, "_wrap.py")).read(); body = src[src.index("        def predict_tensor(self, xt):"):src.index("        def predict(self, x: np.ndarray")]
    assert 'self.counts["predict_device_calls"] = self.counts.get("predict_device_calls", 0) + 1' in body


def test_import_starts_nothing_and_the_first_runner_starts_jit_accounting(monkeypatch, tmp_path):
    """Nothing runs at import (no thread, no hook, no cache read): JIT_CACHE_STATUS says `not started`. jit_start() — called by the FIRST
    KitRunner — attaches the compile hook, reads the effective Triton cache dir (TRITON_CACHE_DIR, else ~/.triton/cache) and words the
    apply line's `jit:` clause; a second call is a no-op."""
    src = open(os.path.join(HERE, "__init__.py")).read()
    tree = ast.parse(src)
    names = {n.func.id for n in ast.walk(ast.Module(body=[st for st in tree.body if not isinstance(st, (ast.FunctionDef, ast.ClassDef))], type_ignores=[]))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "jit_start" not in names and "_hook_triton_compile" not in names and "Thread" not in src, names       # never called at module level (import time); no thread anywhere
    monkeypatch.setitem(v25._JIT, "started", False); monkeypatch.setattr(v25, "JIT_CACHE_STATUS", "not started: test"); monkeypatch.setattr(v25, "JIT_CACHE_DIR", None)
    (tmp_path / "k1").mkdir(); (tmp_path / "k1" / "a.cubin").write_bytes(b"x"); (tmp_path / "k1" / "a.json").write_text("{}")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    assert v25.triton_cache_dir() == str(tmp_path)
    assert v25.jit_start("test") is True and v25.jit_start("again") is False
    assert v25.JIT_CACHE_DIR == str(tmp_path) and v25.IMPORT_WALL["jit_started_at"] == "test"
    assert v25.JIT_CACHE_STATUS.startswith(f"Triton cache {tmp_path} (2 files at attach; kernels compile on first launch and persist there); compile hook")
    assert v25._unhook_triton_compile() is True
    monkeypatch.delenv("TRITON_CACHE_DIR")
    assert v25.triton_cache_dir() == os.path.join(os.path.expanduser("~"), ".triton", "cache")


def test_the_kit_ships_no_compiled_artefact_and_writes_no_cache_of_its_own():
    """The release tree carries kernel SOURCE only: no jit_cache/ dir, no .cubin/.so/.ptx under the kit; the kit names TRITON_CACHE_DIR only to READ the
    effective dir for the apply line (Triton itself writes there)."""
    kit = os.path.dirname(HERE.rstrip("/"))                                    # engines/flashzoi/kits
    bad = [os.path.join(r, f) for r, _, fs in os.walk(kit) for f in fs if f.endswith((".cubin", ".so", ".ptx", ".hsaco")) or "__grp__" in f]
    assert not os.path.isdir(os.path.join(HERE, "jit_cache")) and bad == [], bad
    init = open(os.path.join(HERE, "__init__.py")).read(); wrap = open(os.path.join(HERE, "_wrap.py")).read()
    assert "TRITON_CACHE_DIR" in init and 'os.environ["TRITON_CACHE_DIR"] =' not in init + wrap and "putenv" not in init + wrap
