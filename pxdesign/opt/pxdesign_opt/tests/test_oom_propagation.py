"""OOM propagates on the served paths (the house rule: a served-path handler that reroutes around a lever or kernel launch re-raises
out-of-memory first, `opt_core.oom.is_oom` being the one classifier). This engine's served paths — `design` / the env route / mode `off`, the
package and the carried lever (`opt/forward/hoist/pxd_xattempt/hoist.py`) — hold NO
broad handler that reroutes: the hook around `InferenceRunner.load_checkpoint` catches `ActivationError` alone, so an out-of-memory raised
inside the lever's `install(model)` reaches the caller of the outermost entry (`cli.main(["design", …])`) as itself; a partial application
(the non-OOM failure with a route) still takes its route — the verb's NOT ACTIVE refusal, exit status 3. Every broad `except` of the tree's
served and carried Python is locked below by (file, function, clause, first statement) with its class, so a change to any of them, or a
new one, is a decision made here: SERVED-OTHER = a gate / probe / telemetry / cleanup / re-raise whose try-body launches no lever or kernel;
NOT-HOUSE-PATH = code no mode of this tree reaches (the package's own tests; the build backend's generated guard text)."""
import ast
import json
import os
import re
import sys
import types

import pytest

from . import _stubs
from pxdesign_opt import cli

TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_is_oom_is_the_cores_one_classifier_and_the_kit_spells_it_once():
    """`opt_core.oom.is_oom` is importable from the pinned core and classifies torch's class (by name, no GPU), the CUDA message, the host's
    MemoryError and a wrapped OOM; the kit defines no classifier of its own and imports it under no other spelling."""
    from opt_core.oom import is_oom
    assert is_oom(_stubs.OutOfMemoryError("CUDA out of memory (mock)")) and is_oom(MemoryError()) and is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    try:
        try:
            raise _stubs.OutOfMemoryError("CUDA out of memory (mock)")
        except RuntimeError as inner:
            raise ValueError("wrapped") from inner
    except ValueError as wrapped:
        assert is_oom(wrapped)
    assert not is_oom(RuntimeError("cuBLAS error")) and not is_oom(ValueError("x")) and not is_oom(KeyError("out"))
    offenders = []
    for root, _dirs, files in os.walk(os.path.join(TREE, "opt")):
        for fn in files:
            if fn.endswith(".py"):
                p = os.path.join(root, fn); text = open(p, encoding="utf-8").read()
                for i, line in enumerate(text.splitlines(), 1):
                    if re.search(r"\bdef\s+_?is_oom\b", line) or (re.search(r"\bis_oom\b", line) and re.search(r"^\s*(from|import)\b", line) and line.strip() != "from opt_core.oom import is_oom"):
                        offenders.append(f"{os.path.relpath(p, TREE)}:{i}: {line.strip()}")
    assert offenders == [], offenders                      # one classifier (the core's), one spelling (`from opt_core.oom import is_oom`)


def _served_design_box(stack, monkeypatch, tmp_path):
    """Everything `design --mode exact` touches before and after the lever, on the stub stack: the box, the weights and CCD files, the required
    environment, upstream's `main()` collaborators (`get_configs` … `convert_to_bioassembly_dict`, `runner/cli.build_argv`). Returns argv."""
    _stubs.install_torch(); _stubs.install_upstream(); _stubs.fake_box(monkeypatch, stack)
    oom = _stubs.install_oom_class()
    pins = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
    ckpt = _stubs.stage_weights(tmp_path, monkeypatch, pins)
    monkeypatch.setenv("LAYERNORM_TYPE", "fast_layernorm")                       # the card's export (the kit modes export the same word themselves)
    out = tmp_path / "out"; tasks = tmp_path / "tasks.json"; tasks.write_text("[]")
    I = sys.modules["pxdesign.runner.inference"]                                # upstream's module on the stub stack: main()'s collaborators, minimal
    def get_configs(argv):
        kv = dict(zip(argv[0::2], argv[1::2]))
        return types.SimpleNamespace(dump_dir=kv["--dump_dir"], input_json_path=kv["--input_json_path"], sample_diffusion=types.SimpleNamespace(N_sample=1))
    for name, fn in {"get_configs": get_configs, "process_input_file": lambda p, out_dir=None: p, "download_inference_cache": lambda cf: None,
                     "DIST_WRAPPER": types.SimpleNamespace(rank=0), "save_config": lambda cf, p: None, "convert_to_bioassembly_dict": lambda x, d: None}.items():
        monkeypatch.setattr(I, name, fn, raising=False)
    rcli = types.ModuleType("pxdesign.runner.cli")
    rcli.build_argv = lambda common, extra: [a for k, v in common.items() for a in (f"--{k}", str(v))] + list(extra)   # upstream's flattening, minimal
    monkeypatch.setitem(sys.modules, "pxdesign.runner.cli", rcli)
    return oom, ["design", "--mode", "exact", "--tasks", str(tasks), "--out_dir", str(out), "--ckpt_dir", str(ckpt), "--seeds", "101", "--N_sample", "1"]


def test_an_oom_inside_the_lever_reaches_the_caller_of_the_served_entry(fresh_stack, monkeypatch, capsys, tmp_path):
    """`cli.main(["design", "--mode", "exact", …])` in-process: the lever's `install(model)` (the call the hook wraps, `stack.apply_to`) raises
    `torch.cuda.OutOfMemoryError("CUDA out of memory (mock)")` — the stub stack's class of that name, constructible without a GPU — and the
    outermost entry RAISES it: no NOT ACTIVE reroute, no stock fall-through, no exit code in its place."""
    from opt_core.oom import is_oom
    stack = fresh_stack
    oom, argv = _served_design_box(stack, monkeypatch, tmp_path)
    hoist = stack._kit_module()
    calls = []
    def install(model=None):
        calls.append(model); raise oom("CUDA out of memory (mock)")
    monkeypatch.setattr(hoist, "install", install)
    with pytest.raises(oom) as e:
        cli.main(argv)
    err = capsys.readouterr().err
    assert is_oom(e.value) and str(e.value) == "CUDA out of memory (mock)" and len(calls) == 1, err
    assert "APPLIED" not in err and "NOT ACTIVE" not in err and "fall through to stock" not in err, err     # nothing rerouted, nothing recorded as a fallback
    assert stack.status()["applications"] == []                                                          # the application never completed


def test_a_partial_application_still_takes_its_route(fresh_stack, monkeypatch, capsys, tmp_path):
    """The non-OOM failure keeps its route: `install(model)` returning without rebinding a family is an incomplete application — the hook names the
    families (APPLIED fallbacks=), prints NOT ACTIVE and raises ActivationError, and `design` exits 3 (a mode is all of its levers); nothing propagates
    as a traceback and the line is printed once."""
    stack = fresh_stack
    _oom, argv = _served_design_box(stack, monkeypatch, tmp_path)
    hoist = stack._kit_module()
    monkeypatch.setattr(hoist, "install", lambda model=None: None)               # installs nothing: every family falls through to stock
    rc = cli.main(argv)
    err = capsys.readouterr().err
    assert rc == cli.EXIT_NOT_ACTIVE == 3, (rc, err)
    assert "NOT ACTIVE: lever application incomplete" in err and "fallbacks=h1,h2,h3,h4,h5" in err and err.count("NOT ACTIVE") == 1, err


# ------------------------------------------------------------------------------------------------ the broad-handler lock
# (file relative to the tree, enclosing function, except clause, first statement of the handler) -> class. A served REROUTE would be listed as
# such and its handler would start with `if is_oom(e): raise`; this tree has none.
TWICE = {("opt/pxdesign_opt/stack.py", "gpu_probe", "except Exception as e", "g['probe'] = ...")}      # two handlers of this shape in one function (torch import; device query)
SERVED_OTHER = {
    ("opt/pxdesign_opt/stack.py", "pins_report", "except Exception as e", "return {...}"): "report: check_pins failing = reported NOT pinned by the check verb (imports + version metadata; no lever, no gate)",
    ("opt/pxdesign_opt/stack.py", "gpu_probe", "except Exception as e", "g['probe'] = ..."): "probe: torch import / CUDA device query for the GPU gate; a failed probe refuses activation (no lever)",
    ("opt/pxdesign_opt/stack.py", "find_spec", "except Exception", "spec = None"): "import machinery: another finder's find_spec failing while the hook looks for the runner module (no lever)",
    ("opt/pxdesign_opt/stack.py", "_disarm_autoload", "except Exception", "return"): "cleanup: removing the autoload finder once the package is active",
    ("opt/pxdesign_opt/stock_infer.py", "run", "except BaseException as e", "rc = 1"): "re-raise: the stock caller records the error in its proof and re-raises (mode off: no lever in the process)",
    ("opt/pxdesign_opt/report.py", "exit_tally", "except Exception as e", "return ..."): "telemetry: the EXIT tally line when the lever module's stats() fails, at interpreter exit",
    ("opt/pxdesign_opt/stack.py", "sampler_prepares", "except Exception", "return ..."): "census: the lever module's prepare count READ for the exit census; unreadable = None, which the census prices as no sampling call through the levers (NOT ACTIVE by name; no lever in the try body)",
    ("opt/pxdesign_opt/stack.py", "_exit_verdict_hook", "except Exception as e", "_report.log()"): "gate: the PXDESIGN_OPT route's exit verdict at interpreter exit; a verdict that cannot be read is named NOT ACTIVE and forces exit 3 (fail-closed; no lever in the try body — the run is over)",
    ("opt/pxdesign_opt/precision.py", "attest", "except Exception as e", "return {...}"): "telemetry: the stock numerics policy READ against the live switches after checkpoint load; a torch without the precision getters records `precision_unreadable` (nothing set, no lever)",
    ("opt/pxdesign_opt/precision.py", "exit_check", "except Exception as e", "return {...}"): "gate: the planned numerics policy READ against the live switches at exit; unreadable getters record `precision_unreadable`, which the design verb's exit gate names as NOT ACTIVE (no lever in the try body)",
    ("opt/pxdesign_opt/precision.py", "lever_line", "except Exception", "live = ..."): "telemetry: the live TF32 words for the tf32 LEVER line at exit; unreadable getters print live=unreadable (the gate above decides the exit)",
    ("opt/pxdesign_opt/sdedup.py", "apply", "except Exception", "consulted = False"): "gate: the hook's source guard (inspect.getsource of the carried lever's f_forward); unreadable source = hook not bound and the lever falls back by name (no lever or kernel in the try body)",
    ("stock/check_pins.py", "stack_report", "except Exception", "pass"): "telemetry: cuDNN version for the stack report of the pins check",
    ("opt/pxdesign_opt/weights.py", "fetch", "except Exception as e", "print()"): "install step: upstream's downloader (or its import) failing = the weights step fails by name, exit 1, nothing deleted (run.sh install --weights; not a run route, no lever)",
}
NOT_HOUSE_PATH = {
    ("opt/pxdesign_opt/tests/*", "*", "*", "*"): "the package's own tests",
    ("opt/_build_backend.py", "*", "*", "*"): "the build backend's generated .pth guard text (a string: NOT ACTIVE + os._exit at interpreter start, fail-closed)",
}
BROAD = {"Exception", "BaseException", "RuntimeError"}


def _first_stmt(src_lines, handler) -> str:
    """A short rendering of the handler's first statement: `name = ...`, `return ...`/`return {...}`/`return False`, `pass`, `raise`, a call, `if …: ...`, `try: ...`."""
    st = handler.body[0]
    if isinstance(st, ast.Pass): return "pass"
    if isinstance(st, ast.Raise): return "raise"
    if isinstance(st, ast.Return):
        v = st.value
        return "return" if v is None else ("return False" if isinstance(v, ast.Constant) and v.value is False else ("return {...}" if isinstance(v, ast.Dict) else "return ..."))
    if isinstance(st, ast.Assign):
        t = st.targets[0]; v = st.value
        name = ast.unparse(t).replace('"', "'")
        simple = isinstance(v, ast.Constant) and (v.value is None or isinstance(v.value, (bool, int)) or (isinstance(v.value, str) and len(v.value) <= 3))
        return f"{name} = " + (repr(v.value) if simple else "...")
    if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call): return ast.unparse(st.value.func) + "()"
    if isinstance(st, ast.If): return f"if {ast.unparse(st.test)}: ..."
    if isinstance(st, ast.Try): return "try: ..."
    return type(st).__name__


def _broad_handlers(tree_root):
    """Every broad except handler (bare, Exception, BaseException, or a clause naming RuntimeError) in the tree's own and carried Python under opt/
    plus stock/check_pins.py (the check verb's pins report): [(relpath, function, clause, first statement)]. upstream's checkout under stock/src is not the tree's."""
    found = []
    files = [os.path.join(tree_root, "stock", "check_pins.py")]
    for root, _dirs, names in os.walk(os.path.join(tree_root, "opt")):
        files += [os.path.join(root, n) for n in names if n.endswith(".py")]
    for path in sorted(files):
        rel = os.path.relpath(path, tree_root)
        src = open(path, encoding="utf-8").read()
        mod = ast.parse(src)
        owner = {}
        for node in ast.walk(mod):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for sub in ast.walk(node):
                    if sub is not node:
                        owner.setdefault(id(sub), node.name)
        for node in ast.walk(mod):                                   # `owner`: the OUTERMOST enclosing function (ast.walk is breadth-first; setdefault keeps the first)
            if isinstance(node, ast.ExceptHandler):
                t = node.type
                names = set() if t is None else ({t.id} if isinstance(t, ast.Name) else {e.id for e in getattr(t, "elts", []) if isinstance(e, ast.Name)})
                if t is None or names & BROAD:
                    clause = "except" + ("" if t is None else " " + ast.unparse(t)) + (f" as {node.name}" if node.name else "")
                    found.append((rel, owner.get(id(node), "<module>"), clause, _first_stmt(src.splitlines(), node)))
    return found


def _classify(site):
    rel = site[0]
    if site in SERVED_OTHER: return "SERVED-OTHER"
    for key in NOT_HOUSE_PATH:
        if key == site or (key[1:] == ("*", "*", "*") and (key[0] == rel or (key[0].endswith("/*") and rel.startswith(key[0][:-1])))):
            return "NOT-HOUSE-PATH"
    return None


def test_every_broad_handler_of_the_tree_is_classified_and_none_reroutes_on_a_served_path():
    """The lock: each broad handler found by the AST scan is one of the SERVED-OTHER sites (exactly those: no more, no fewer) or lives in a
    NOT-HOUSE-PATH file; an unclassified handler — a new served `except Exception`, or an edit that moves one into another function or changes
    its first statement — fails here until it is classified (a served reroute must start with `if is_oom(e): raise`)."""
    from collections import Counter
    sites = _broad_handlers(TREE)
    unclassified = [s for s in sites if _classify(s) is None]
    assert unclassified == [], unclassified
    served_found = Counter(s for s in sites if _classify(s) == "SERVED-OTHER")
    expected = Counter({k: (2 if k in TWICE else 1) for k in SERVED_OTHER})
    assert served_found == expected, (served_found - expected, expected - served_found)
    n_served_other, n_dev = sum(served_found.values()), sum(1 for s in sites if _classify(s) == "NOT-HOUSE-PATH")
    assert (n_served_other, n_dev) == (15, 1), (n_served_other, n_dev)               # 0 served reroutes, 15 served-other, 1 not-house-path
    for path in ("opt/forward/hoist/pxd_xattempt/hoist.py", "opt/pxdesign_opt/cli.py", "opt/pxdesign_opt/infer_loop.py"):
        assert not any(s[0] == path for s in sites), path                             # the lever, the verbs, the in-process route: no broad handler at all
