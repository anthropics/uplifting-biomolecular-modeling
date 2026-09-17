"""Lever prefetch (prefetch.py + sidecar.py): the next item featurized by one forked helper from the fold process's RNG state; item 0
featurized twice and compared; the parent adopts the helper's post-featurization RNG state; named step-asides. CPU: a stand-in engine
module with upstream's loop shape (DataLoader over InferenceInput-like specs, batch_size=1, identity collate) and a pipeline that draws
from python / numpy (/ torch when importable) so byte identity is a real RNG-stream property, checked leaf by leaf against a stock run."""
import os
import random
import re
import sys
import time
import types

FORWARD_S = 0.05

import pytest

np = pytest.importorskip("numpy")


def _platform_refuses_side_processes():
    """The lever forks a side process and talks over multiprocessing pipes; a sandbox that refuses either cannot run these tests."""
    import multiprocessing as mp
    try:
        ctx = mp.get_context("fork")
        r, w = ctx.Pipe(duplex=False)
        p = ctx.Process(target=w.send_bytes, args=(b"ok",))
        p.start(); p.join(10)
        ok = r.poll(5) and r.recv_bytes() == b"ok"
        r.close(); w.close()
        return None if ok else "side process gave no answer"
    except Exception as e:                                  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


_REFUSED = _platform_refuses_side_processes()
pytestmark = pytest.mark.skipif(_REFUSED is not None, reason=f"platform refuses fork/pipe side processes: {_REFUSED}")

from .. import prefetch as pf
from .. import sidecar as sc


class Spec:                                                  # InferenceInput's shape: example_id + to_pipeline_input()
    def __init__(self, example_id, n):
        self.example_id, self.n = example_id, n

    def to_pipeline_input(self):
        return {"example_id": self.example_id, "atom_array": np.arange(self.n, dtype=np.float32), "chain_info": {"A": {"n": self.n}}}


class Pipeline:                                              # draws from every CPU generator an atomworks pipeline can touch
    def __init__(self):
        self.calls = 0

    def __call__(self, pin):
        self.calls += 1
        torch = pf._torch()
        n = int(pin["atom_array"].shape[0])
        feats = {"py": random.random(), "np": np.random.rand(n, 3).astype(np.float32), "perm": np.random.permutation(n),
                 "strided": np.arange(2 * n, dtype=np.float64).reshape(n, 2)[:, ::-1]}
        if torch is not None:
            feats["t"] = torch.rand(n, 4)
            feats["tt"] = torch.arange(6).reshape(2, 3).t()  # a non-contiguous tensor: strides must survive the pipe
        return {"example_id": pin["example_id"], "feats": feats, "atom_array": pin["atom_array"] * 2, "pid": None}


class _T:                                                    # an atomworks-Transform-like callable: data dict in, data dict out
    def __call__(self, data):
        return self.forward(dict(data))


class TParse(_T):
    def forward(self, d):
        n = int(d["atom_array"].shape[0]); d["np"] = np.random.rand(n, 3).astype(np.float32); d["perm"] = np.random.permutation(n); return d


class TPy(_T):
    def forward(self, d):
        d["py"] = random.random(); return d


class TNoise(_T):                                            # extra_xforms' shape: torch.randn_like of the coordinates to be noised (torch CPU)
    def forward(self, d):
        t = pf._torch()
        d["noise"] = t.randn(int(d["atom_array"].shape[0]), 3) if t is not None else None; return d


class TPost(_T):
    def forward(self, d):
        d["post"] = np.asarray(d["np"]).sum() + (0.0 if d.get("noise") is None else float(d["noise"].sum())); return d


class ComposeLike:                                           # atomworks Compose's surface the lever uses: .transforms + __call__(data, _stop_before=)
    def __init__(self):
        self.transforms = [TParse(), TPy(), TNoise(), TPost()]
        self.calls = 0

    def __call__(self, data, rng_state_dict=None, _stop_before=None):
        self.calls += 1
        for idx, t in enumerate(self.transforms):
            if _stop_before is not None and idx == _stop_before:
                break
            data = t(data)
        return data


class PipelineNP(Pipeline):                                  # a featurization that draws from numpy ONLY (atomworks' generators)
    def __call__(self, pin):
        self.calls += 1
        n = int(pin["atom_array"].shape[0])
        return {"example_id": pin["example_id"], "feats": {"np": np.random.rand(n, 3).astype(np.float32), "perm": np.random.permutation(n)},
                "atom_array": pin["atom_array"] * 2}


class Loader:                                                # the DataLoader shape run() uses: dataset + batch_size=1 + identity collate -> [spec]
    def __init__(self, dataset=None, batch_size=1, collate_fn=None, **kw):
        self.dataset = list(dataset)

    def __iter__(self):
        return iter([[x] for x in self.dataset])

    def __len__(self):
        return len(self.dataset)


def make_engine_module(between_items=None, pipeline_cls=None):
    """A stand-in rf3.inference_engines.rf3: RF3InferenceEngine with _construct_pipeline / run in upstream's shape."""
    m = types.ModuleType("rf3.inference_engines.rf3")
    m.DataLoader = Loader

    class RF3InferenceEngine:
        def __init__(self):
            self.pipeline = None

        def _construct_pipeline(self, cfg=None):
            self.transform_overrides = {"diffusion_batch_size": 5, "n_recycles": 10}   # the real engine's dict (the shadow learner's D hint)
            self.pipeline = (pipeline_cls or Pipeline)()

        def initialize(self):
            self._construct_pipeline({})

        def run(self, inputs):
            if self.pipeline is None:
                self.initialize()
            loader = m.DataLoader(dataset=inputs, batch_size=1, collate_fn=lambda x: x)
            outs = []
            for batch_idx, input_spec in enumerate(loader):
                input_spec = input_spec[0]
                po = self.pipeline(input_spec.to_pipeline_input())
                time.sleep(FORWARD_S)                        # the forward's seconds (the helper speculates the next item meanwhile)
                if between_items:
                    between_items()                          # a "forward" that draws from the CPU generators
                outs.append((po, pf.rng_digest(pf.rng_get())))
            return outs

    m.RF3InferenceEngine = RF3InferenceEngine
    return m


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    t = pf._torch()
    if t is not None:
        t.manual_seed(s)


def reset_lever():
    pf.STATE.update({"armed": False, "installed": False, "on": False, "reason": None, "off_reason": None, "conflict": None, "n_gpu": 1,
                     "helper_pid": None, "forked_before_cuda": None, "threads_at_fork": None, "first_input": None, "first_input_path": None,
                     "split": None, "trace_top": None, "trace_rng": None})
    pf.CENSUS.clear(); pf.ITEMS.clear(); pf._ORIG.clear()
    for p in list(pf._PROXIES):
        if p.car is not None:
            p.car.stop()
    pf._PROXIES.clear()


@pytest.fixture()
def lever():
    reset_lever()
    yield pf
    m = sys.modules.get("rf3.inference_engines.rf3")
    pf.disable(m) if m is not None else None
    reset_lever()
    sc.stop_all()


def specs(n=5):
    return [Spec(f"item{i}", 6 + i) for i in range(n)]


def test_rng_roundtrip_and_compare_words():
    seed_all(3)
    s = pf.rng_get()
    random.random(); np.random.rand(3)
    assert not pf.rng_eq(s, pf.rng_get())
    pf.rng_set(s)
    assert pf.rng_eq(s, pf.rng_get()) and len(pf.rng_digest(s)) == 10
    a = {"x": np.array([1.0, np.nan]), "l": [1, "s", (2.0,)], "d": {"k": np.arange(4).reshape(2, 2)[:, ::-1]}}
    b = {"x": np.array([1.0, np.nan]), "l": [1, "s", (2.0,)], "d": {"k": np.arange(4).reshape(2, 2)[:, ::-1].copy()}}
    assert pf.compare(a, b) is None                                    # NaN bit patterns equal; a strided vs contiguous array with equal bytes in logical order
    b["d"]["k"][0, 0] = 9
    assert pf.compare(a, b) == ".d.k:bytes"
    assert pf.compare({"a": 1}, {"b": 1}).startswith(":keys")
    assert pf.compare([1, 2], [1, 2, 3]) == ":len(2!=3)"
    assert pf.compare(np.zeros(2, np.float32), np.zeros(2, np.float64)).startswith(":array")


def test_prefetch_is_byte_identical_to_stock_and_overlaps(lever, monkeypatch):
    """Stock run vs prefetch run from the same seed: every item's pipeline output equal leaf by leaf, the parent's RNG state after each
    item equal to stock's; item 0 compared (first_input=same), items 1.. served by the helper speculatively (respec=0)."""
    m = make_engine_module()
    monkeypatch.setitem(sys.modules, m.__name__, m)
    seed_all(42)
    stock = m.RF3InferenceEngine().run(specs(5))
    rep = {"levers": ["graph", "prefetch"], "n_gpu": 1, "levers_applied": []}
    lever.arm(rep, lambda trig, cb, r: pytest.fail("module already imported: no watch expected"))
    assert rep["prefetch"]["installed"] and "prefetch" in rep["levers_applied"] and m.DataLoader.__name__ == "DataLoader"
    seed_all(42)
    eng = m.RF3InferenceEngine()
    got = eng.run(specs(5))
    assert isinstance(eng.pipeline, pf.Proxy) and pf.STATE["on"] and pf.STATE["helper_pid"] and pf.STATE["forked_before_cuda"] is True
    for (a, da), (b, db) in zip(stock, got):
        assert pf.compare(a, b) is None
        assert da == db                                                 # the fold process's RNG state after each item: as stock leaves it
    c = pf.census()
    assert c["first_input"] == "same" and c["helper"] == 4 and c["parent"] == {"first": 1} and c["speculated"] == 4 and c["respec"] == 0
    assert eng.pipeline.real.calls == 1                                 # the fold process featurized item 0 only
    toks = dict(pf.lever_tokens())
    assert toks["helper"] == 4 and toks["parent"] == 1 and toks["first_input"] == "same" and toks["forked_before_cuda"] is True
    # a second run() on the same engine: a new order, the helper keeps serving (first_input is per process: not repeated)
    seed_all(7)
    stock2 = make_engine_module().RF3InferenceEngine().run(specs(3))
    seed_all(7)
    got2 = eng.run(specs(3))
    assert all(pf.compare(a, b) is None and da == db for (a, da), (b, db) in zip(stock2, got2))
    assert pf.census()["helper"] == 7


def test_a_forward_that_draws_cpu_rng_is_still_identical_by_respec(lever, monkeypatch):
    """When the fold process draws from the CPU generators between items (a forward that consumes CPU RNG), the helper's speculative
    state no longer matches: it re-featurizes from the parent's state (respec) — outputs and states stay stock's, overlap is lost by name."""
    draw = lambda: (random.random(), np.random.rand(2))
    m = make_engine_module(between_items=draw)
    monkeypatch.setitem(sys.modules, m.__name__, m)
    seed_all(5)
    stock = m.RF3InferenceEngine().run(specs(4))
    lever.arm({"levers": ["prefetch"], "n_gpu": 1}, lambda *a: None)
    seed_all(5)
    got = m.RF3InferenceEngine().run(specs(4))
    assert all(pf.compare(a, b) is None and da == db for (a, da), (b, db) in zip(stock, got))
    c = pf.census()
    assert c["helper"] == 3 and c["respec"] == 3 and c["speculated"] == 0 and c["first_input"] == "same"


def test_compose_split_head_in_helper_tail_here(lever, monkeypatch):
    """RF3's real shape: an atomworks Compose whose late transform draws torch CPU (randn_like of the coordinates to be noised) while the
    forward ALSO draws torch CPU (the sampler's per-step rotations). Item 0 runs transform by transform in the helper (trace: the first
    torch-drawing transform = the split), the split is checked once on item 0 (head from the helper + tail here == the in-process output),
    and every later item's head (the numpy-drawing transforms) is speculated one ahead while the tail runs here from this process's own
    torch state: outputs and all three generator states after every item equal a stock run's; speculated on every item, respec=0."""
    torch = pytest.importorskip("torch")
    def forward_draws():                                     # the sampler's per-step CPU pattern, T=3 steps of batch D=5
        for _ in range(3):
            pf.shadow_step(5)
    m = make_engine_module(between_items=forward_draws, pipeline_cls=ComposeLike)
    monkeypatch.setitem(sys.modules, m.__name__, m)
    seed_all(11)
    stock = make_engine_module(between_items=forward_draws, pipeline_cls=ComposeLike).RF3InferenceEngine().run(specs(6))
    lever.arm({"levers": ["prefetch"], "n_gpu": 1}, lambda *a: None)
    seed_all(11)
    got = m.RF3InferenceEngine().run(specs(6))
    assert all(pf.compare(a, b) is None and da == db for (a, da), (b, db) in zip(stock, got))
    c = pf.census()
    assert c["first_input"] == "same" and c["by_key"].get("split_check:same") == 1, (c, pf.STATE.get("split"))
    assert c["helper"] == 5 and c["respec"] == 0 and c["speculated"] == 5, c
    # item 1 by the split (head in the helper, tail here); the forward's draw pattern learnt at its hand-off; items 2.. whole, from the
    # predicted state. The learnt (T, D) is any pair whose draws advance the generator exactly as the forward's 3 steps x batch 5 do (with the
    # engine's D hint it is 3x5; without one an equivalent factorisation such as 15x1 is as valid) — assert the invariant, not the literal.
    shadow_keys = {k: v for k, v in c["by_key"].items() if k.startswith("shadow:")}
    learnt = [k for k in shadow_keys if re.fullmatch(r"shadow:\d+x\d+", k)]
    assert c["by_key"].get("served:helper:head") == 1, c
    assert len(learnt) == 1 and "not_found" not in " ".join(shadow_keys) and "miss" not in " ".join(shadow_keys), shadow_keys
    assert shadow_keys[learnt[0]] == c["helper"] == 5, (shadow_keys, c["helper"])   # learnt at item 1's hand-off (before its answer): every helper-served item names it
    T_, D_ = map(int, learnt[0][len("shadow:"):].split("x"))
    assert T_ * D_ == 15, learnt                              # the same number of rand / normal draws as 3 steps of batch 5
    assert pf.STATE["split"].startswith("on:2/4:TNoise"), pf.STATE["split"]


def test_compose_without_torch_transform_speculates_whole(lever, monkeypatch):
    """A Compose whose transforms draw numpy / python only: no split (split_before None), whole-pipeline speculation stands on every item
    when the forward draws torch only (or nothing)."""
    class NoTorch(ComposeLike):
        def __init__(self):
            super().__init__(); self.transforms = [TParse(), TPy(), TPost()]
    t_ = pf._torch()
    fwd = (lambda: t_.rand(3)) if t_ is not None else None
    m = make_engine_module(between_items=fwd, pipeline_cls=NoTorch)
    monkeypatch.setitem(sys.modules, m.__name__, m)
    seed_all(2)
    stock = make_engine_module(between_items=fwd, pipeline_cls=NoTorch).RF3InferenceEngine().run(specs(5))
    lever.arm({"levers": ["prefetch"], "n_gpu": 1}, lambda *a: None)
    seed_all(2)
    got = m.RF3InferenceEngine().run(specs(5))
    assert all(pf.compare(a, b) is None and da == db for (a, da), (b, db) in zip(stock, got))
    c = pf.census()
    assert c["helper"] == 4 and c["respec"] == 0 and c["speculated"] == 4 and c["by_key"].get("served:helper:head") is None, c
    assert pf.STATE["split"] is None and c["first_input"] == "same"


def test_named_step_asides(lever, monkeypatch):
    m = make_engine_module()
    monkeypatch.setitem(sys.modules, m.__name__, m)
    lever.arm({"levers": ["prefetch"], "n_gpu": 1}, lambda *a: None)
    seed_all(1)
    eng = m.RF3InferenceEngine()
    eng.run(specs(1))                                                   # one item: nothing to overlap
    assert pf.census()["parent"] == {"n_items=1": 1} and pf.census()["helper"] == 0
    pf.disable(m); reset_lever()
    rep = pf.arm({"levers": ["prefetch"], "n_gpu": 2}, lambda *a: None)  # the row-sharded line: declined by name at activation (conflict:n_gpu), nothing installed
    eng = m.RF3InferenceEngine()
    seed_all(1)
    stock = make_engine_module().RF3InferenceEngine().run(specs(3))
    seed_all(1)
    got = eng.run(specs(3))
    assert pf.STATE["helper_pid"] is None and pf.STATE["conflict"] == "n_gpu" and rep["prefetch"]["conflict"] == "n_gpu" and pf.census()["parent"] == {} and pf.census()["helper"] == 0
    assert m.RF3InferenceEngine._construct_pipeline.__name__ == "_construct_pipeline"   # the engine is untouched
    assert all(pf.compare(a, b) is None for (a, _), (b, _) in zip(stock, got))
    pf.disable(m); reset_lever()
    assert pf.arm({"levers": ["graph"], "n_gpu": 1}, lambda *a: None) == {"levers": ["graph"], "n_gpu": 1}   # a row without the lever: untouched
    assert not pf.STATE["armed"] and m.RF3InferenceEngine._construct_pipeline.__name__ == "_construct_pipeline"


def test_helper_error_falls_back_to_the_stock_statement(lever, monkeypatch):
    """A featurization that raises in the helper (here: only outside the fold process, on item 2) is re-run in the fold process from
    the state it captured: outputs stock's, counted parent:helper_error, the helper keeps serving the items after."""
    parent = os.getpid()
    m = make_engine_module()
    real_call = Pipeline.__call__

    def flaky(self, pin):
        if pin["example_id"] == "item2" and os.getpid() != parent:
            raise RuntimeError("boom in helper")
        return real_call(self, pin)

    monkeypatch.setattr(Pipeline, "__call__", flaky)
    monkeypatch.setitem(sys.modules, m.__name__, m)
    seed_all(9)
    stock = make_engine_module().RF3InferenceEngine().run(specs(5))
    lever.arm({"levers": ["prefetch"], "n_gpu": 1}, lambda *a: None)
    seed_all(9)
    got = m.RF3InferenceEngine().run(specs(5))
    assert all(pf.compare(a, b) is None and da == db for (a, da), (b, db) in zip(stock, got))
    c = pf.census()
    assert c["parent"] == {"first": 1, "helper_error": 1} and c["helper"] == 3 and c["helper_errors"] == 1


def test_disjoint_generators_keep_the_speculation(lever, monkeypatch):
    """The featurization draws numpy only; the 'forward' between items draws python random (and torch CPU when importable): the helper's
    one-ahead run stands on every item (speculated, respec=0) because the parent's numpy state is the speculation's start state, and the
    parent's OTHER generators are left as the forward put them — every item's output and the (py, np, torch) states after each item equal
    a stock run's."""
    def forward_draws():
        random.random()
        t_ = pf._torch()
        if t_ is not None:
            t_.rand(3)
    m = make_engine_module(between_items=forward_draws, pipeline_cls=PipelineNP)
    monkeypatch.setitem(sys.modules, m.__name__, m)
    seed_all(5)
    stock = make_engine_module(between_items=forward_draws, pipeline_cls=PipelineNP).RF3InferenceEngine().run(specs(6))
    lever.arm({"levers": ["prefetch"], "n_gpu": 1}, lambda *a: None)
    seed_all(5)
    got = m.RF3InferenceEngine().run(specs(6))
    assert all(pf.compare(a, b) is None and da == db for (a, da), (b, db) in zip(stock, got))
    c = pf.census()
    assert c["helper"] == 5 and c["speculated"] == 5 and c["respec"] == 0 and c["by_key"].get("rng_used:np") == 5, c


def test_registry_words():
    """The lever's tokens and describe() keys the exit tally reads (report.py: prefetch block)."""
    d = pf.describe()
    assert {"on", "installed", "reason", "off_reason", "helper_pid", "forked_before_cuda", "census", "items", "rule"} <= set(d)
    assert [k for k, _ in pf.lever_tokens(d)][:4] == ["helper", "parent", "speculated", "respec"]


def test_transport_keeps_arrays_and_tensor_views_exact():
    """sidecar's pickler: big ndarrays / CPU tensors through POSIX shm (or by value where shm is refused), small ones inline; a strided
    tensor view arrives as the same view (offset/size/stride) of an equal storage; no fd passing, no sockets."""
    big = np.arange(70000, dtype=np.float32).reshape(700, 100)
    obj = {"big": big, "small": np.arange(6).reshape(2, 3)[:, ::-1], "s": "x", "nested": [big[:1].copy(), (1, 2.5)]}
    torch = pf._torch()
    if torch is not None:
        base = torch.arange(40000, dtype=torch.float64).reshape(200, 200)
        obj["view"] = base[3:50, ::2]                        # non-contiguous view with a storage offset
        obj["bf16"] = torch.ones(5, dtype=torch.bfloat16)
        obj["empty"] = torch.empty(0)
    out = sc.loads(sc.dumps(obj))
    assert pf.compare(obj, out) is None
    if torch is not None:
        v = out["view"]
        assert v.storage_offset() == obj["view"].storage_offset() and v.stride() == obj["view"].stride() and not v.is_contiguous()
    assert sc.transport_word() in ("shm",) or sc.transport_word().startswith("bytes(")
    sc.sweep()


def test_transport_leaves_the_resource_tracker_quiet(tmp_path):
    """A process that sends and receives shm-backed arrays exits with NOTHING on stderr from multiprocessing's resource tracker (no
    KeyError traceback from a double unregister, no leaked-block warning): register / unregister are balanced per process."""
    import subprocess
    code = (
        "import sys, numpy as np\n"
        f"sys.path[:0] = {[os.path.dirname(os.path.dirname(os.path.abspath(pf.__file__)))]!r}\n"
        "from rosettafold3_opt import sidecar as sc\n"
        "big = np.arange(300000, dtype=np.float32)\n"
        "for _ in range(3):\n"
        "    out = sc.loads(sc.dumps({'a': big, 'b': big[:5]}))\n"
        "    assert (out['a'] == big).all()\n"
        "sc.sweep()\n"
        "print(sc.transport_word())\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ROSETTAFOLD3_OPT_", "MODEL_OPT_", "PYTEST_"))}   # a bare interpreter: no kit activation words from the test session
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stderr
    assert "Traceback" not in r.stderr and "resource_tracker" not in r.stderr and "leaked" not in r.stderr, r.stderr
    assert r.stdout.strip() == "shm" or r.stdout.strip().startswith("bytes(")
