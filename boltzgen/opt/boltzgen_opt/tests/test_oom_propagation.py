"""Out-of-memory propagates on the served paths: a broad `except` that reroutes around a lever (a fallback / stock / 'lever unavailable,
continue without it' route) re-raises an out-of-memory FIRST — `if is_oom(e): raise`, `opt_core.oom.is_oom`, the core's one classifier — so an
OOM is the caller's to see, never a NOT ACTIVE report, never a stock run in its place. The package's one such handler is `stack.activate`'s
kit-import loop (a kit module that raises while it installs); proven here through the served in-process entry `boltzgen_opt.enable()` with a
mocked `torch.cuda.OutOfMemoryError`, beside the unchanged route of any other exception (an inactive report naming the failed import; strict:
`ActivationError`). Every other broad handler on a served path — the package's probes and the carried kits' modules a mode imports — is held
to the table below by an AST census (file, line, whether its first statement is the re-raise), each with the reason it is or is not a lever
reroute that can originate an OOM; a handler added, moved or reclassified fails here until the table says so."""
import ast
import os
import unittest

from opt_core.oom import is_oom

from boltzgen_opt import modes, stack

from . import _stubs

HOME = stack.opt_home()
PKG = os.path.dirname(os.path.abspath(stack.__file__))

# (line, first statement is `if is_oom(e): raise`, class) — class: REROUTE = wraps an allocating lever/kernel call and continues without it;
# OTHER = cannot originate an OOM (import / attribute re-point / parse / probe / telemetry) or is cleanup / a named refusal, out of the rule
PACKAGE_HANDLERS = {
    "_autoload.py": [(52, False, "OTHER probe of the other finders on sys.meta_path"), (199, False, "OTHER named refusal inside site (exit 3); a raise there is swallowed by site.py")],
    "census.py": [(166, False, "OTHER census hook: probe of the other finders on sys.meta_path"), (288, True, "REROUTE census install: a library that fails to import is the `absent` word (a finding of the run's record; a kit mode's child refuses on it before the step) — OOM re-raised first"), (337, False, "OTHER census: distribution-name probe for the ops build tag"), (353, False, "OTHER census: the library's capability-flag probe"), (363, False, "OTHER census: device capability probe"), (398, False, "OTHER census: autocast dtype probe while naming a reference call's rule"), (415, False, "OTHER census: naming a reference call's rule from shapes (no allocation); an unparsed shape is named, not guessed"), (437, False, "OTHER census: metadata copy onto the pass-through wrapper"), (452, False, "OTHER census: the library's forward-backend name for a served call (shape predicates, no allocation)"), (475, False, "OTHER census: metadata copy onto the pass-through wrapper"), (483, False, "OTHER census: shape read for a word"), (530, False, "OTHER census: metadata copy onto the BATCH timer wrapper"), (543, False, "OTHER census stack words: TF32-flag read at a timed call"), (632, False, "OTHER census stack words: TorchDynamo counter read (named `unknown:` on failure)"), (639, False, "OTHER census stack words: one fact probe (named `unknown:` on failure)"), (665, False, "OTHER census PEAK line: allocator-maxima reset probe (no device / no context: nothing to reset)"), (685, False, "OTHER census PEAK line: allocator-maxima read (report-only; a process without CUDA prints no PEAK line)"), (712, False, "OTHER census: the exit line never prints nothing — a census that cannot format itself says so")],
    "stack.py": [(161, False, "OTHER package metadata probe"), (179, False, "OTHER nvidia-smi probe"), (187, False, "OTHER nvidia-smi parse"), (256, False, "OTHER torch version.py parse"), (388, False, "OTHER import probe before arming the instance counter"), (597, True, "REROUTE kit-import loop of activate(): a failed install is a NOT ACTIVE report — OOM re-raised first"), (648, True, "REROUTE exit verdict: a kit module's gate() that raises is itself the finding (the lever counts as fallen back) — OOM re-raised first"), (684, True, "REROUTE step gate: upstream's task class not importable in this process (no model step can run here) — OOM re-raised first"), (738, False, "OTHER exit verdict at interpreter exit: a verdict that raised is a named refusal (exit 3), never a pass — nothing is left to re-raise into")],
    "weights.py": [(103, False, "OTHER install --weights: the downloader import, named"), (118, False, "OTHER install --weights: upstream's transfer error relayed by name (CPU step, nothing of the model is loaded)")],
}
SERVED_CARRIED_DIRS = [stack.kit_src(k)[len(HOME) + 1:] for k in modes.KITS]   # the src dirs the modes put on PYTHONPATH: every carried kit (add-on, partner, size / fast / host levers)
CARRIED_HANDLERS = {
    "forward/xattempt_addon/src/xa_fastinit.py": [(59, False, "OTHER num_workers probe (unserved_reason: a probe error is itself the reason the lever cannot serve)"),
                                                  (131, False, "OTHER coverage proof over parameter names and shapes (no allocation) -> replay of the stock inits"),
                                                  (161, False, "OTHER install: attribute re-point")],
    "forward/xattempt_addon/src/xa_hoist.py": [(152, False, "OTHER tensor version attribute probe"),
                                               (190, True, "REROUTE mask build/refresh (allocates) -> lever disabled, stock masks"),
                                               (227, True, "REROUTE mask build inside the wrapped DiffusionModule.forward (allocates) -> lever disabled"),
                                               (410, True, "REROUTE mask refresh after a graph bind (allocates) -> lever disabled"),
                                               (426, False, "OTHER install: attribute re-points")],
    "forward/xattempt_addon/src/xa_run.py": [(56, False, "OTHER cleanup between steps")],
    "forward/fast_inference/src/bg_graph_patch.py": [(161, True, "REROUTE CUDA-graph capture (allocates the pool) -> predraw mode")],
    "forward/fast_levers/src/fl_levers.py": [(321, True, "REROUTE dit_fused: the fused DiT layer call (_layer_fused, allocates) -> the lever disabled by name, the unfused statements serve (partial run)"),
                                             (389, True, "REROUTE attn_bf16: the bf16 attention call (allocates casts, the aligned mask buffer, the kernel's workspace) -> the lever disabled by name, the stock fp32 call serves"),
                                             (469, True, "REROUTE dit_fused: the kernel import probe (no triton / no CUDA toolchain) -> off by name, OOM re-raised first"),
                                             (605, False, "OTHER census: the exit lines never print nothing — a census that cannot format itself says so")],
    "forward/fast_inference/src/bg_hook.py": [(19, False, "OTHER telemetry flush"), (63, False, "OTHER kernel call counters: import + re-point"),
                                             (68, False, "OTHER cuda synchronize for timing"), (125, False, "OTHER memory telemetry"),
                                             (138, False, "OTHER telemetry: graph stats read"), (148, False, "OTHER per-step telemetry record"),
                                             (153, False, "OTHER hook install at import (imports, re-points, the numpy-stream seed fix): a failure leaves seed_fix unproven -> the census names it, exit 3")],
    "forward/fast_inference/src/bg_inproc.py": [(29, False, "OTHER cleanup: autocast cache"), (43, False, "OTHER cleanup: graph release")],
    "forward/fast_inference/src/sitecustomize.py": [(3, False, "OTHER interpreter-start hook loader: site.py swallows a raise here; activation refuses by name without bg_hook")],
    "host/writer_levers/src/hl_levers.py": [(102, False, "OTHER writer-process priority probe (os.nice)"), (107, False, "OTHER writer-process thread-count probe"),
                                            (139, False, "OTHER install: spawned-worker probe (the lever lives in the design process only)"),
                                            (163, False, "OTHER design-count read for the tally (a shape attribute of a tensor already on the host; no allocation)"),
                                            (187, False, "OTHER drain at the predict loop's end: the census of the failed write printed, then the failure re-raised by name — nothing continues without the lever"),
                                            (243, False, "OTHER drain at interpreter exit after a loop that ended mid-way: the failure is logged into the tally and the GATE-FAIL line; the core's guard sets the exit status"),
                                            (256, False, "OTHER census: the exit report never masks the process's own status")],
}


def broad_handlers(path):
    """[(lineno, first_statement_is_the_oom_reraise)] of every broad handler in a module: bare `except:`, `except Exception` /
    `except BaseException`, or a tuple naming one of those or RuntimeError."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            broad = True
        else:
            elts = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            names = [(e.attr if isinstance(e, ast.Attribute) else getattr(e, "id", None)) for e in elts]
            broad = any(n in ("Exception", "BaseException") for n in names) or (isinstance(node.type, ast.Tuple) and "RuntimeError" in names)
        if not broad:
            continue
        first = node.body[0]
        reraise = (isinstance(first, ast.If) and isinstance(first.test, ast.Call) and getattr(first.test.func, "id", None) == "is_oom"
                   and [type(s) for s in first.body] == [ast.Raise] and first.body[0].exc is None)
        out.append((node.lineno, reraise))
    return sorted(out)


class TestOomPropagates(unittest.TestCase):
    def test_the_classifier_is_the_cores_one(self):
        self.assertIs(stack.is_oom, is_oom)
        self.assertEqual(is_oom.__module__, "opt_core.oom")

    def test_enable_raises_an_oom_from_a_kit_install_instead_of_reporting_not_active(self):
        """Through the served in-process entry: `boltzgen_opt.enable('big')` on this tree with stub upstream; the mode's own module raises a
        mocked torch OOM while it installs → the OOM propagates out of enable(); a plain RuntimeError there is the unchanged route (an
        inactive report naming the failed import; strict=True: ActivationError)."""
        site = _stubs.materialize(self._tmp())
        code = r"""
import importlib, json, sys
import torch
import boltzgen_opt
from boltzgen_opt import stack
OOM = getattr(torch.cuda, "OutOfMemoryError", None) or torch.OutOfMemoryError
KIND, STRICT = sys.argv[1], sys.argv[2] == "strict"
real = importlib.import_module
def raising(name, *a, **k):
    if name == "sz_levers":                                            # the mode's own lever module, imported (and installing) last in activate()'s loop
        raise (OOM("CUDA out of memory (mock)") if KIND == "oom" else RuntimeError("mock install failure"))
    return real(name, *a, **k)
stack.importlib.import_module = raising
try:
    rep = boltzgen_opt.enable("big", strict=STRICT)
    rec = {"returned": True, "active": rep["active"], "reason": rep.get("reason")}
except BaseException as e:                                              # the test's own probe: which exception left enable()
    rec = {"returned": False, "exc": type(e).__name__, "is_oom": __import__("opt_core.oom", fromlist=["is_oom"]).is_oom(e), "msg": str(e)[:120]}
print("RECORD " + json.dumps(rec))
"""
        for kind in ("oom", "other"):
            for strict in ("strict", "lenient"):                                       # one fresh interpreter per case: an activation attempt leaves kit modules loaded
                rc, out, err = _stubs.run_py(code, _stubs.clean_env(site), args=[kind, strict])
                self.assertEqual(rc, 0, err[-2000:])
                rec = next(__import__("json").loads(ln[7:]) for ln in out.splitlines() if ln.startswith("RECORD "))
                if kind == "oom":
                    self.assertEqual((rec["returned"], rec["exc"], rec["is_oom"]), (False, "OutOfMemoryError", True), (strict, rec))
                    self.assertNotIn("NOT ACTIVE", err)                               # no report line stands in for the OOM
                elif strict == "lenient":
                    self.assertEqual((rec["returned"], rec["active"]), (True, False), rec)
                    self.assertTrue(rec["reason"].startswith("kit import 'sz_levers' failed after ['bg_hook', 'xa_fastinit', 'xa_hoist']: RuntimeError('mock install failure')"), rec)
                else:
                    self.assertEqual((rec["returned"], rec["exc"], rec["is_oom"]), (False, "ActivationError", False), rec)

    def test_a_carried_levers_fallback_reraises_an_oom_through_the_activated_model_call(self):
        """Through the served entry and a carried lever: `enable('exact')` installs the add-on's mask hoist on `DiffusionModule.forward`
        (stub upstream); the hoist's mask build raising a mocked torch OOM inside the wrapped forward propagates out of the model call —
        the lever's own fallback (`[xa_hoist] disabled …`, stock masks) is taken for any other exception, unchanged."""
        site = _stubs.materialize(self._tmp())
        code = r"""
import json, sys
import torch
import boltzgen_opt
OOM = getattr(torch.cuda, "OutOfMemoryError", None) or torch.OutOfMemoryError
KIND = sys.argv[1]
rep = boltzgen_opt.enable("exact", strict=True)
import xa_hoist
from boltzgen.model.modules.diffusion import DiffusionModule
assert getattr(DiffusionModule.forward, "_xa_hoist", False), "the hoist is installed on the stub DiffusionModule"
def raising(*a, **k):
    raise (OOM("CUDA out of memory (mock)") if KIND == "oom" else RuntimeError("mock mask build failure"))
xa_hoist._get_entry = raising
m = DiffusionModule().eval()
rec = {"active": rep["active"], "enabled_before": xa_hoist.STATS["enabled"]}
try:
    with torch.no_grad():
        out = m(None, None, None, None, {}, {}, 1)
    rec.update(returned=True, out=repr(out))
except BaseException as e:
    rec.update(returned=False, exc=type(e).__name__, is_oom=__import__("opt_core.oom", fromlist=["is_oom"]).is_oom(e))
rec["enabled_after"] = xa_hoist.STATS["enabled"]
print("RECORD " + json.dumps(rec))
"""
        for kind in ("oom", "other"):
            rc, out, err = _stubs.run_py(code, _stubs.clean_env(site), args=[kind])
            self.assertEqual(rc, 0, err[-2000:])
            rec = next(__import__("json").loads(ln[7:]) for ln in out.splitlines() if ln.startswith("RECORD "))
            self.assertEqual((rec["active"], rec["enabled_before"]), (True, True), rec)
            if kind == "oom":
                self.assertEqual((rec["returned"], rec["exc"], rec["is_oom"], rec["enabled_after"]), (False, "OutOfMemoryError", True, True), rec)
                self.assertNotIn("[xa_hoist] DISABLED", err)
            else:
                self.assertEqual((rec["returned"], rec["out"], rec["enabled_after"]), (True, "None", False), rec)   # the stub's stock forward ran; the lever disabled itself by name
                self.assertIn("[xa_hoist] DISABLED: build failed: RuntimeError('mock mask build failure')", err)

    def test_every_broad_handler_on_a_served_path_is_accounted(self):
        """The package's modules and the carried modules a mode imports: every broad handler is in the table with its line, and exactly the
        ones marked re-raise an OOM as their first statement."""
        have = {f: broad_handlers(os.path.join(PKG, f)) for f in sorted(os.listdir(PKG)) if f.endswith(".py")}
        have = {f: v for f, v in have.items() if v}
        self.assertEqual(have, {f: [(ln, rr) for ln, rr, _ in rows] for f, rows in PACKAGE_HANDLERS.items()})
        have = {}
        for d in SERVED_CARRIED_DIRS:
            for f in sorted(os.listdir(os.path.join(HOME, d))):
                if f.endswith(".py"):
                    rows = broad_handlers(os.path.join(HOME, d, f))
                    if rows:
                        have[f"{d}/{f}"] = rows
        self.assertEqual(have, {f: [(ln, rr) for ln, rr, _ in rows] for f, rows in CARRIED_HANDLERS.items()})
        for table in (PACKAGE_HANDLERS, CARRIED_HANDLERS):
            for f, rows in table.items():
                for ln, rr, why in rows:
                    self.assertTrue(why.startswith(("REROUTE ", "OTHER ")), (f, ln, why))
                    if why.startswith("OTHER "):
                        self.assertFalse(rr, (f, ln))                                     # the rule is applied where it means something, nowhere else

    def test_the_house_levers_catch_no_broad_exception_but_the_named_ones(self):
        """big's own lever module has no broad handler at all: its OOM trace catches the OOM class itself inside Boltz.forward, prints its
        `[sz] {"event": "oom", …}` evidence line and re-raises the same exception to upstream's own handler — evidence, never a reroute.
        fast's module has exactly the ones in the table (each attention/DiT lever's disable-by-name reroute, OOM re-raised first; the exit
        census); the host levers' module has probes, a drain that re-raises by name, and its exit report — none continues without a lever."""
        self.assertEqual(broad_handlers(os.path.join(stack.kit_src(modes.KIT_SIZE_LEVERS), "sz_levers.py")), [])
        self.assertEqual(broad_handlers(os.path.join(stack.kit_src(modes.KIT_FAST_LEVERS), "fl_levers.py")),
                         [(ln, rr) for ln, rr, _ in CARRIED_HANDLERS["forward/fast_levers/src/fl_levers.py"]])
        hl = broad_handlers(os.path.join(stack.kit_src(modes.KIT_HOST_LEVERS), "hl_levers.py"))
        self.assertEqual(hl, [(ln, rr) for ln, rr, _ in CARRIED_HANDLERS["host/writer_levers/src/hl_levers.py"]])
        self.assertFalse(any(rr for _, rr in hl))                                         # no lever reroute in it: nothing there serves a stock path in a lever's place

    def _tmp(self):
        import shutil, tempfile
        d = tempfile.mkdtemp(prefix="bgopt_oom_")
        self.addCleanup(shutil.rmtree, d, True)
        return d


if __name__ == "__main__":
    unittest.main()
