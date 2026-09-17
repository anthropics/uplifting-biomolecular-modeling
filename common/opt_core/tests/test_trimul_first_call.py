"""kernels.trimul first-call / per-call host path: the stack-chunk trampoline (_pystack), the no-import stack word, the weight-pack cache keys
(no id(), no version counter: inference_mode-safe), preload(), the output-dtype contract and every row under inference_mode / no_grad /
autocast (GPU part: skipped without CUDA)."""
import os
import sys

import pytest

from opt_core.kernels import trimul as T
from opt_core.kernels.trimul import _pystack as PS


# ----------------------------------------------------------------------------------------------------------------- pure (no framework)
def test_padded_call_is_transparent():
    assert PS.padded_call(lambda a, b=2: (a, b), 1, b=3) == (1, 3)
    with pytest.raises(ZeroDivisionError):
        PS.padded_call(lambda: 1 / 0)
    tr = PS._trampoline()
    assert tr.__code__.co_nlocals >= PS.PAD_SLOTS                      # the frame that gives every callee one large stack chunk
    assert PS.import_padded("json") is sys.modules["json"]
    us = PS.call_cost_here(200)
    assert isinstance(us, float) and us >= 0.0

    def deep(d):
        return deep(d - 1) if d else PS.padded_call(sum, [1, 2, 3])
    assert deep(300) == 6                                              # callable from any depth


def test_pystack_is_standard_library_only():
    import ast
    import opt_core._pystack as TOP
    assert PS.padded_call is TOP.padded_call and PS.import_padded is TOP.import_padded          # kernels/trimul/_pystack re-exports opt_core._pystack
    tree = ast.parse(open(TOP.__file__, encoding="utf-8").read())
    top = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top.add((node.module or "").split(".")[0])
    assert top <= {"sys", "time", "importlib"}, top


def test_stack_word_facts_import_no_model_library(monkeypatch):
    had = "cuequivariance_torch" in sys.modules
    T._CUEQ_FACTS.clear()
    T.cueq_present()
    T.cueq_version()
    assert ("cuequivariance_torch" in sys.modules) == had                  # asking never imports the package
    T._CUEQ_FACTS.clear()
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda dist: "0.11.1" if dist == "cuequivariance-torch" else (_ for _ in ()).throw(md.PackageNotFoundError(dist)))
    if not had:
        assert T.cueq_version() == "0.11.1"
    T._CUEQ_FACTS.clear()
    monkeypatch.setattr(md, "version", lambda dist: (_ for _ in ()).throw(md.PackageNotFoundError(dist)))
    if not had:
        assert T.cueq_version() == "?"
    T._CUEQ_FACTS.clear()


def test_preload_never_raises_and_names_every_row():
    assert set(T.PRELOAD_MODULES) == set(T.ROW_NAMES)
    assert T.preload(rows=("torch_math",)) == {}
    got = T.preload(rows=("v4", "cueq"))                                  # no torch here -> 'error:...' words; cuequivariance absent -> 'absent'; never an exception
    for m, v in got.items():
        assert isinstance(v, float) or v == "absent" or str(v).startswith("error:"), (m, v)
    if not T.cueq_present():
        assert got.get("cuequivariance_torch") == "absent"


def test_warm_imports_is_idempotent_and_never_raises(monkeypatch):
    import opt_core
    from opt_core import warm as WM
    assert opt_core.warm_imports is WM.warm_imports and opt_core.auto_warm is WM.auto_warm     # the package's lazy exports
    WM.REPORT.clear(); WM._DONE.clear(); del WM.ORIGINS[:]
    monkeypatch.setenv("OPT_CORE_NO_WARM_IMPORTS", "1")
    rep = WM.warm_imports(libraries=("json", "no_such_library_xyz"))
    assert rep["json"] == "present" and rep["no_such_library_xyz"] == "skipped:env"           # env word: nothing imported, decided by name
    monkeypatch.delenv("OPT_CORE_NO_WARM_IMPORTS", raising=False); monkeypatch.delenv("OPT_CORE_NO_STOCK_PRELOAD", raising=False)
    WM._DONE.discard("no_such_library_xyz")
    assert WM.warm_imports(libraries=("no_such_library_xyz",))["no_such_library_xyz"] == "absent"
    w = WM.warm_imports(libraries=("email.mime.text",), origin="test")["email.mime.text"]
    assert w == "present" or isinstance(w, float) or (w == "declined:torch_not_loaded" and "torch" not in sys.modules), w   # no torch on this interpreter: nothing else is imported, by rule
    n = len(WM.REPORT)
    WM.warm_imports(libraries=("email.mime.text", "json"))                                    # idempotent: decided libraries are not re-decided
    assert len(WM.REPORT) in (n, n + 1) and WM.REPORT["json"] == "present" and WM.ORIGINS[-2:] == ["test", "kit"]
    rep = WM.auto_warm("unit")                                                                 # the faces' guard: DEFAULT_LIBRARIES decided once
    assert set(WM.DEFAULT_LIBRARIES) <= set(rep)
    k = len(WM.ORIGINS); WM.auto_warm("unit2"); assert len(WM.ORIGINS) == k                    # already decided -> no second pass
    for v in rep.values():
        assert isinstance(v, float) or v in ("present", "absent", "skipped:env", "declined:torch_not_loaded", "torch_not_loaded") or str(v).startswith(("error:", "unavailable:")), v
    WM.REPORT.clear(); WM._DONE.clear(); del WM.ORIGINS[:]


def test_every_face_warms_on_its_first_serving_call():
    import ast, os
    here = os.path.dirname(T.__file__)
    for face, entry in (("triattn", "triangle_attention"), ("apb", "pair_bias_attention"), ("apb", "pair_bias_planes"), ("ln", "layer_norm"), ("transition", "transition")):
        src = open(os.path.join(os.path.dirname(here), face, "__init__.py"), encoding="utf-8").read()
        fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == entry)
        body = ast.unparse(fn)[:1200]
        assert "auto_warm(" in body and "_WARMED" in body, (face, entry)
    assert "stock_preload_once()" in open(T.__file__, encoding="utf-8").read()


def test_stock_preload_once_never_raises_and_runs_once(monkeypatch):
    from opt_core import warm as WM
    T.STOCK_PRELOAD.clear(); WM.REPORT.clear(); WM._DONE.clear()
    monkeypatch.setenv(T.STOCK_PRELOAD_ENV, "1")
    had = "cuequivariance_torch" in sys.modules
    got = dict(T.stock_preload_once())
    assert got.get("cuequivariance_torch") == ("present" if had else "skipped:env"), got
    assert T.stock_preload_once() is T.STOCK_PRELOAD                        # once per process: the second call is the recorded word
    T.STOCK_PRELOAD.clear(); WM.REPORT.clear(); WM._DONE.clear()
    monkeypatch.delenv(T.STOCK_PRELOAD_ENV, raising=False); monkeypatch.delenv("OPT_CORE_NO_WARM_IMPORTS", raising=False)
    if not had and not T.cueq_present():
        assert T.stock_preload_once()["cuequivariance_torch"] == "absent"   # nothing to import on this stack: a dictionary word, no exception
        assert "cuequivariance_torch" not in sys.modules
    T.STOCK_PRELOAD.clear(); WM.REPORT.clear(); WM._DONE.clear(); del WM.ORIGINS[:]


# ----------------------------------------------------------------------------------------------------------------- torch (CPU is enough)
def test_weight_keys_are_data_identity_and_inference_mode_safe():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    W = {k: torch.randn(8, 8, generator=g) for k in T.WEIGHT_KEYS}
    k1 = T.weights_key(W)
    assert T.weights_key({k: v.detach() for k, v in W.items()}) == k1      # a re-wrapped parameter is the same weights
    W2 = dict(W, w_ap=W["w_ap"].clone())
    assert T.weights_key(W2) != k1                                          # another tensor is not
    built = []
    cache = {}
    for _ in range(3):
        T._packed(cache, "t", {k: v.detach() for k, v in W.items()}, lambda: built.append(1) or "pack")
    assert built == [1] and any(isinstance(k, tuple) and k[0] == "t" for k in cache)
    with torch.inference_mode():
        Wi = {k: torch.randn(8, 8) for k in T.WEIGHT_KEYS}                 # inference tensors: no version counter
        T.weights_key(Wi)
        T._packed({}, "t", Wi, lambda: "pack")
        from opt_core.kernels import trimul_esm_shapes as S
        S.pack_key(Wi)                                                      # was: weights[k]._version -> RuntimeError under inference_mode


# ----------------------------------------------------------------------------------------------------------------- GPU: every row's contract
ROWS_C128 = ("v4", "tmk3_exact", "tmk3_fast", "esm_v5_fwd", "esm_v5_fwd:f32in", "esm_v61", "esm_shapes", "esm_shapes:f32in", "esm_k1ptr", "esm_k1ptr:f32in", "cueq", "torch_math")
ROWS_C256 = ("v4", "tx_sm90a", "esm_v5_fwd", "esm_shapes", "esm_k1ptr", "torch_math")


def _cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    return torch


def _weights(torch, C, H, dev, seed=3):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)                              # noqa: E731
    W = {"ln_in_w": 1 + 0.1 * r(C), "ln_in_b": 0.1 * r(C), "w_ag": r(H, C) * C ** -0.5, "w_ap": r(H, C) * C ** -0.5, "w_bg": r(H, C) * C ** -0.5,
         "w_bp": r(H, C) * C ** -0.5, "ln_out_w": 1 + 0.1 * r(H), "ln_out_b": 0.1 * r(H), "w_o": r(C, H) * H ** -0.5, "w_og": r(C, C) * C ** -0.5}
    return {k: v.to(dev) for k, v in W.items()}


def _serve(torch, row, z, W, **kw):
    try:
        return T.triangle_multiplication(z, None, direction="outgoing", weights=W, word=row, residual=False, cache={}, **kw)
    except T.Refusal as e:                                                   # a row this box cannot serve (ABI / library / capability) refuses BY NAME
        return e


@pytest.mark.parametrize("C,rows", [(128, ROWS_C128), (256, ROWS_C256)])
def test_rows_return_the_call_dtype_and_serve_under_inference_mode(C, rows):
    torch = _cuda()
    dev = torch.device("cuda", 0)
    N = 256
    W = _weights(torch, C, C, dev)
    g = torch.Generator().manual_seed(11)
    z32 = torch.randn(N, N, C, generator=g).to(dev)
    z16 = z32.to(torch.bfloat16)
    served = {}
    for row in rows:
        for zdt, z in (("bf16", z16), ("fp32", z32)):
            with torch.no_grad():
                a = _serve(torch, row, z, W)
            with torch.inference_mode():
                Wi = {k: v.detach().clone() for k, v in W.items()}             # inference tensors as weights too
                b = _serve(torch, row, z.clone(), Wi)
            if isinstance(a, T.Refusal):
                assert isinstance(b, T.Refusal) and b.kind == a.kind, (row, zdt, a, b)
                served[(row, zdt)] = "refused:%s" % a.kind
                continue
            assert not isinstance(b, T.Refusal), (row, zdt, b)
            assert a.dtype == z.dtype and b.dtype == z.dtype, (row, zdt, a.dtype, b.dtype)   # OUTPUT DTYPE CONTRACT: the call's dtype
            assert a.shape == z.shape and torch.isfinite(a.float()).all()
            assert torch.equal(a, b), (row, zdt)                              # inference_mode changes nothing
            served[(row, zdt)] = "ok"
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):        # fp32-resident z under bf16 autocast: the fused rows read a bf16 z -> bf16 out
            c = _serve(torch, row, z32, W)
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            d = _serve(torch, row, z32.clone(), {k: v.clone() for k, v in W.items()})
        if not isinstance(c, T.Refusal):
            assert not isinstance(d, T.Refusal), (row, d)
            assert c.dtype == (torch.float32 if row == "torch_math" else torch.bfloat16), (row, c.dtype)
            assert torch.equal(c, d), row
            served[(row, "f32z")] = "ok"
    must = {"v4", "torch_math"} | ({"tmk3_fast"} if C == 128 and torch.cuda.get_device_capability(dev) == (9, 0) else set())
    for row in must:
        assert served.get((row, "bf16")) == "ok", (row, served)
    print("served:", served)


def test_tmk3_batch_writes_slices_not_a_stack_copy():
    torch = _cuda()
    dev = torch.device("cuda", 0)
    C, N = 128, 200
    W = _weights(torch, C, C, dev)
    g = torch.Generator().manual_seed(5)
    zb = torch.randn(2, N, N, C, generator=g).to(dev).to(torch.bfloat16)
    for row in ("tmk3_fast", "tmk3_exact"):
        with torch.no_grad():
            full = _serve(torch, row, zb, W)
            if isinstance(full, T.Refusal):
                continue
            one0 = _serve(torch, row, zb[0], W)
            one1 = _serve(torch, row, zb[1:2], W)
        assert full.shape == zb.shape and one1.shape == (1, N, N, C)
        assert torch.equal(full[0], one0) and torch.equal(full[1], one1[0]), row


def test_first_call_of_a_class_goes_through_the_trampoline(monkeypatch):
    torch = _cuda()
    dev = torch.device("cuda", 0)
    C, N = 128, 160
    W = _weights(torch, C, C, dev)
    z = torch.randn(N, N, C, generator=torch.Generator().manual_seed(2)).to(dev).to(torch.bfloat16)
    calls = []
    real = PS.padded_call
    monkeypatch.setattr(PS, "padded_call", lambda fn, *a, **k: calls.append(getattr(fn, "__name__", "?")) or real(fn, *a, **k))
    cache = {}
    with torch.no_grad():
        for _ in range(3):
            T.triangle_multiplication(z, None, direction="outgoing", weights=W, word="torch_math", residual=False, cache=cache)
    assert calls.count("_dispatch") == 1                                     # once per (row, device, dtype, widths) class and cache, then direct

def test_torch_precedes_every_cuda_extension_in_warm_imports(monkeypatch):
    """cu13 hazard: cuequivariance_ops_torch imported before torch cannot resolve libnvrtc.so.13 and stays broken for the process -> torch FIRST, by rule."""
    from opt_core import warm as WM
    import opt_core._pystack as TOP
    libs = list(WM.DEFAULT_LIBRARIES)
    assert libs[0] == "torch"
    assert all(libs.index("torch") < i for i, n in enumerate(libs) if n.startswith("cuequivariance")), libs
    order = []
    import importlib, importlib.util
    real_import, real_find = importlib.import_module, importlib.util.find_spec
    fake = set()

    def rec_import(name, *a, **k):
        order.append(name)
        if name in fake:
            m = type(sys)("fake_" + name); sys.modules[name] = m; return m
        return real_import(name, *a, **k)
    monkeypatch.setattr(importlib, "import_module", rec_import)
    had_torch = "torch" in sys.modules
    if not had_torch:                                                         # no torch on this interpreter: pretend it is installed so the ORDER is observable
        fake.update({"torch", "cuequivariance_ops_torch"})
        monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a, **k: object() if n in fake else real_find(n, *a, **k))
        monkeypatch.setattr(WM, "_present", lambda n: n in fake or real_find(n) is not None)
    else:
        fake.add("cuequivariance_ops_torch") if "cuequivariance_ops_torch" not in sys.modules else None
        monkeypatch.setattr(WM, "_present", lambda n: n in fake or n in sys.modules or real_find(n) is not None)
    WM.REPORT.clear(); WM._DONE.clear(); del WM.ORIGINS[:]
    monkeypatch.delenv("OPT_CORE_NO_WARM_IMPORTS", raising=False); monkeypatch.delenv("OPT_CORE_NO_STOCK_PRELOAD", raising=False)
    try:
        WM.warm_imports(libraries=("cuequivariance_ops_torch",))              # the caller did NOT list torch: it is imported first anyway
        if not had_torch:
            assert order[:2] == ["torch", "cuequivariance_ops_torch"], order
            assert isinstance(WM.REPORT.get("torch"), float) and isinstance(WM.REPORT.get("cuequivariance_ops_torch"), float), WM.REPORT
        else:
            assert WM.REPORT.get("torch") == "present" and (not order or order[0] != "cuequivariance_ops_torch" or "torch" in sys.modules)
        # import_padded alone (no warm_imports) orders too
        for n in ("torch", "cuequivariance_ops_torch"):
            if n in fake:
                sys.modules.pop(n, None)
        del order[:]
        if not had_torch:
            TOP.import_padded("cuequivariance_ops_torch")
            assert order[:2] == ["torch", "cuequivariance_ops_torch"], order
    finally:
        for n in fake:
            sys.modules.pop(n, None)
        WM.REPORT.clear(); WM._DONE.clear(); del WM.ORIGINS[:]

def _fake_or_real_package(tmp_path, monkeypatch, name):
    """A stand-in package ``name`` (with one sub-module ``_sub``) on sys.path when the real library is not installed here; the real one otherwise."""
    import importlib, importlib.util
    if importlib.util.find_spec(name) is None:
        pkg = tmp_path / name; pkg.mkdir()
        (pkg / "__init__.py").write_text("from . import _sub\nVALUE = 1\n"); (pkg / "_sub.py").write_text("X = 2\n")
        monkeypatch.syspath_prepend(str(tmp_path)); importlib.invalidate_caches()
        return "fake"
    return "real"


def test_warm_imports_never_poisons_the_process(tmp_path, monkeypatch):
    """cu13 blocker: a library whose import FAILS inside warm_imports (libnvrtc.so.13 not resolvable ...) must leave NO trace in sys.modules (the half
    module, its sub-modules), be recorded 'unavailable:<Exc>: <msg>', not stop the other libraries, and import cleanly later once the cause is gone."""
    import importlib, importlib.abc, importlib.machinery
    from opt_core import warm as WM
    import opt_core._pystack as TOP
    LIB = "cuequivariance_ops_torch"
    for k in [k for k in sys.modules if k == LIB or k.startswith(LIB + ".")]:
        monkeypatch.delitem(sys.modules, k)
    kind = _fake_or_real_package(tmp_path, monkeypatch, LIB)
    have_torch = importlib.util.find_spec("torch") is not None
    if not have_torch:                                                          # no torch here: a stand-in so the rule 'torch first' has something to load
        (tmp_path / "torch").mkdir(exist_ok=True); (tmp_path / "torch" / "__init__.py").write_text("Y = 3\n")
        monkeypatch.syspath_prepend(str(tmp_path)); importlib.invalidate_caches(); monkeypatch.delitem(sys.modules, "torch", raising=False)

    class Poison(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        """While armed, every attempt at LIB executes like the real failure -- a sub-module lands in sys.modules, then the extension load raises
        (``fail_times``: how many attempts fail before the loader behaves; None = until disarmed)."""
        armed = True
        fail_times = None
        attempts = 0
        def find_spec(self, name, path, target=None):
            if name == LIB and Poison.armed and (Poison.fail_times is None or Poison.attempts < Poison.fail_times):
                return importlib.machinery.ModuleSpec(name, self, is_package=True)
            return None
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            Poison.attempts += 1
            sys.modules[LIB + "._dragged_in"] = type(sys)(LIB + "._dragged_in")     # what a half-run package body leaves behind
            sys.modules.setdefault("cuequivariance_ops", type(sys)("cuequivariance_ops"))  # ... and the degraded sibling package a pre-torch import leaves
            raise ImportError("libcue_ops.so: libnvrtc.so.13: cannot open shared object file: No such file or directory")
    finder = Poison()
    sys.meta_path.insert(0, finder)
    WM.REPORT.clear(); WM._DONE.clear(); WM._SAID.clear(); del WM.ORIGINS[:]
    monkeypatch.delenv("OPT_CORE_NO_WARM_IMPORTS", raising=False); monkeypatch.delenv("OPT_CORE_NO_STOCK_PRELOAD", raising=False)
    try:
        had_sibling = "cuequivariance_ops" in sys.modules
        rep = dict(WM.warm_imports(libraries=(LIB, "json")))                     # must not raise (fails on the attempt AND on the one purge+retry)
        assert str(rep[LIB]).startswith("unavailable:ImportError: libcue_ops.so: libnvrtc.so.13"), rep
        assert Poison.attempts == 2, Poison.attempts                               # one attempt + ONE retry after the family purge, no more
        assert rep["json"] == "present" and (isinstance(rep["torch"], float) or rep["torch"] == "present"), rep   # the next library still handled; torch first
        assert not [k for k in sys.modules if k == LIB or k.startswith(LIB + ".")], "rolled back: no half module, no dragged-in sub-module"
        assert had_sibling or "cuequivariance_ops" not in sys.modules             # the degraded sibling the failure planted is gone too
        assert LIB not in WM._DONE                                                 # re-attempted by a later call
        sys.meta_path.remove(finder); finder = None                                # the cause fixed (real world: torch's loader now provides libnvrtc)
        importlib.invalidate_caches()
        m = importlib.import_module(LIB)                                           # a plain import in the same process now SUCCEEDS
        assert LIB in sys.modules and m is sys.modules[LIB]
        rep2 = dict(WM.warm_imports(libraries=(LIB,)))
        assert rep2[LIB] == "present", rep2
        # a failure that clears after ONE attempt (the pre-torch poison case): warm_imports recovers inside the same call
        for k in [k for k in sys.modules if k == LIB or k.startswith(LIB + ".")]:
            del sys.modules[k]
        WM.REPORT.clear(); WM._DONE.clear(); WM._SAID.clear()
        finder = Poison(); Poison.armed = True; Poison.fail_times = 1; Poison.attempts = 0; sys.meta_path.insert(0, finder)
        rep3 = dict(WM.warm_imports(libraries=(LIB,)))
        assert isinstance(rep3[LIB], float) and "retried once" in rep3[LIB + ":recovered"], rep3
        sys.meta_path.remove(finder); finder = None; Poison.fail_times = None
        # guarded_import alone: same rollback, exception propagates for callers that refuse by name
        for k in [k for k in sys.modules if k == LIB or k.startswith(LIB + ".")]:
            del sys.modules[k]
        finder = Poison(); Poison.armed = True; Poison.attempts = 0; sys.meta_path.insert(0, finder)
        with pytest.raises(ImportError):
            TOP.guarded_import(LIB)
        assert not [k for k in sys.modules if k == LIB or k.startswith(LIB + ".")]
        sys.meta_path.remove(finder); finder = None; importlib.invalidate_caches()
        sys.modules[LIB + "._orphan"] = type(sys)("o")                             # an orphan left by SOMEONE ELSE's failed attempt is swept before ours
        assert TOP.guarded_import(LIB) is sys.modules[LIB] and LIB + "._orphan" not in sys.modules
    finally:
        if finder is not None and finder in sys.meta_path:
            sys.meta_path.remove(finder)
        if kind == "fake" or not have_torch:
            for k in [k for k in sys.modules if k == LIB or k.startswith(LIB + ".") or (not have_torch and k == "torch")]:
                sys.modules.pop(k, None)
        WM.REPORT.clear(); WM._DONE.clear(); WM._SAID.clear(); del WM.ORIGINS[:]


def test_warm_imports_declines_everything_when_torch_cannot_load(monkeypatch):
    import importlib
    from opt_core import warm as WM
    import opt_core._pystack as TOP
    WM.REPORT.clear(); WM._DONE.clear(); WM._SAID.clear(); del WM.ORIGINS[:]
    monkeypatch.delenv("OPT_CORE_NO_WARM_IMPORTS", raising=False); monkeypatch.delenv("OPT_CORE_NO_STOCK_PRELOAD", raising=False)
    had_torch = sys.modules.pop("torch", None)
    attempted = []
    real = TOP.guarded_import

    def fake_guarded(name):
        attempted.append(name)
        if name == "torch":
            raise ImportError("libtorch_cuda.so: cannot open shared object file")
        return real(name)
    monkeypatch.setattr(TOP, "guarded_import", fake_guarded)
    monkeypatch.setattr(WM, "_present", lambda n: True)                          # everything 'installed': only the torch failure decides
    try:
        rep = dict(WM.warm_imports(libraries=("cuequivariance_ops_torch", "cuequivariance_torch")))
        assert rep["warm_imports"] == "torch_not_loaded", rep
        assert str(rep["torch"]).startswith("unavailable:ImportError: libtorch_cuda.so"), rep
        assert rep["cuequivariance_ops_torch"] == "declined:torch_not_loaded" and rep["cuequivariance_torch"] == "declined:torch_not_loaded", rep
        assert attempted == ["torch"], attempted                                   # NOTHING else was imported
    finally:
        if had_torch is not None:
            sys.modules["torch"] = had_torch
        WM.REPORT.clear(); WM._DONE.clear(); WM._SAID.clear(); del WM.ORIGINS[:]

