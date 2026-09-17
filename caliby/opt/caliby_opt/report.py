"""Observability for caliby_opt: the activation line and the exit tally (both on stderr, prefixed ``[caliby-opt]``).

* activation — one line when a mode is resolved and gated in a process:
  ``[caliby-opt] ACTIVE mode=fast variant=single tier=2 row=X source=modes.py:single/serial tree=exact gpu=<name> stack=<key>
  [target_gpu=<MODEL_OPT_TARGET_GPU> [MISMATCH: <note>]] switches=CALIBY_FAST_SAMPLER=2 ...`` or ``[caliby-opt] NOT ACTIVE: <reason>``
  (the caller exits 3), or for stock
  ``[caliby-opt] STOCK mode=off variant=<variant> tree=stock digest=<16> switches=none``.
* partial — the design process's exit line when a lever of the row could not run at call time (``activate.partial_exit``; one formatter,
  ``partial_not_active_line``): ``[caliby-opt] NOT ACTIVE: <mode> ran short of its lever set — <lever>: <the kit's record>, ...; the
  outputs on disk are not a <mode> run; exit 3``. A mode is all of its levers: it refuses by name rather than run a subset.
* gate — a lever's DECLARED input gate met at call time (``registry.LEVERS[...].gates``; ``note_gate``, one formatter ``lever_gate_line`` over
  the core's ``opt_core.report.lever_line``): ``[caliby-opt] LEVER name=<switch> state=skipped reason=<gate> impl=<module> origin=kit
  <evidence k=v>`` — the call takes upstream's own lines for that step with the same outputs (stock arithmetic), so it is said and recorded
  (``tally()["gates"]``, the manifest's ``lever_gates``) and never counted as a fallback: the exit code is unchanged. A reason the registry
  does not declare for the lever is refused (``ValueError``), never printed.
* exit — at interpreter exit, what the kit's own modules left readable in this process: which lever modules were loaded and from
  where (``tree=exact``: every loaded lever module is its kit file by the import hook; ``stock``: the installed upstream files;
  ``mixed``: neither — overlay.exit_state), the fused
  LCP kernel's state (``chroma.layers.complexity._X_LCP_DISABLED_REASON``: None = the kernel is defined in this process; a repr = triton or
  the kernel did not import, or a served call could not build / launch it — that call raised ``XLcpKernelError``), the kit's own fallback lines counted per lever on this process's stderr (``FALLBACK_MARKERS``: the ``stderr`` probes of
  registry.LEVERS — ``[CALIBY_X_CLEAN=...] ... -> serial fallback``, ``[CALIBY_X_SPARSE_EXACT] ... -> kit path for this call``,
  ``[CALIBY_X_MULTISEQ] ... -> serial fallback``, ``[CALIBY_FAST_SAMPLER] ... -> stock sampler for this call``), the
  switches in effect, and ``partial=<levers>|none`` — the levers of the row that fell back (``activate.completion``).
  ``register_exit_tally()`` is called at activation, before any kit module is imported; when no lever module was ever loaded the tally
  says so (never silent). It also registers the exit gate of the environment route (``_exit_gate``, run after the EXIT line has printed):
  in a process whose exit code no caller of this package owns (``CALIBY_OPT=<mode>`` in a host script; the design children call
  ``exit_owned()`` and decide their own code), a partial activation or a mixed lever-module set (``activate.modules_wrong``) ends the
  process NOT ACTIVE by name — its line(s), then exit 3 through the core's forced exit — by the same two rules the design verb applies;
  there is no opt-out (``CALIBY_OPT=off`` runs stock).
* census — the tag lines of ``CENSUS_LINES`` (``[caliby-opt] <TAG> k=v k=v ...``, the tokens in the tuple's order, no spaces inside a
  value; clocks are ``time.time()`` seconds printed %.6f so concurrent processes merge on one axis, GiB %.3f; ``route`` = the mode word):
  ``STACK`` — the process's stack facts, read never set, printed once per design process by its caller (design_run / stock_design)
  after activation, before the writer runs. ``OUTPUTS_WRITTEN`` — once per design process, by its caller after the writer's own last
  write (``outputs_written``: the clock, and the writer's input / design counts from its timing.json), followed by ``PEAK`` item=PASS —
  torch's running maxima (max_memory_allocated / max_memory_reserved, GiB, never reset; ``torch.cuda.synchronize()`` precedes the clock
  read once CUDA is initialised). ``ENV-CLEAN`` / ``DESIGNS`` — the parent's lines (cli.py): the switch names stripped from the caller's
  environment, the design count against the request. Per-unit timing is not this package's: nothing inside upstream's loops is stamped.
The prefix, the exit table and the exit-tally registration are the shared core's (``opt_core.report``); the sentences are this kit's
(``CENSUS_LINES`` is the one table of the tag lines' tokens; log readers match these bytes).
"""
from __future__ import annotations

import atexit
import os
import platform
import sys
import time
from typing import Dict, List, Optional

from opt_core import report as _core

from . import registry, stack

TAG = stack.TAG                                                        # "caliby-opt": the one spelling (stack.py), shared with the core pin gate's line and the .pth guard
PREFIX = _core.prefix(TAG)                                             # "[caliby-opt]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = _core.EXIT_OK, _core.EXIT_FAIL, _core.EXIT_USAGE, _core.EXIT_NOT_ACTIVE
FALLBACK_MARKERS: Dict[str, str] = registry.stderr_probes()           # switch -> the kit's own line prefix (pinned to the kit sources by the tests)
DECLARED_GATES: Dict[str, tuple] = registry.declared_gates()          # switch -> the gate reasons its LEVER state=skipped line may carry (registry.Lever.gates)
_TALLY = {"registered": False, "mode": None, "tree_state": None, "fallback_lines": {}, "gates": {}, "stderr_wrapped": False, "exit_owned": False}
# The census tag lines: tag word -> its tokens in print order. Every formatter below emits exactly these tokens (census_line); a reader
# builds its patterns from this table.
CENSUS_LINES = {
    "PEAK": ("item", "alloc_gib", "reserved_gib", "route", "variant", "pid"),
    "OUTPUTS_WRITTEN": ("route", "variant", "t", "n_items", "n_designs", "pid"),
    "STACK": ("torch", "cuda", "triton", "python", "gpu", "cc", "tf32_matmul", "cudnn_tf32", "cudnn_deterministic", "cudnn_benchmark", "alloc_conf",
              "autocast", "compile", "accelerators", "image", "pip_freeze_sha256", "route", "variant", "pid"),
    "ENV-CLEAN": ("stripped", "route", "variant"),
    "DESIGNS": ("designs_written", "designs_expected", "verdict"),
}
WRITER_TIMING = "timing.json"                                           # the writer's own last write (xcaliby_design.py): n_inputs, n_designs, its clocks


class _CountingStderr:
    """A stderr proxy that counts the kit's fallback lines per lever and passes everything through."""

    def __init__(self, inner):
        self._inner = inner

    def write(self, s):
        try:
            for line in str(s).splitlines():
                if PREFIX in line:
                    continue
                for lever, marker in FALLBACK_MARKERS.items():
                    if marker in line:
                        _TALLY["fallback_lines"][lever] = _TALLY["fallback_lines"].get(lever, 0) + 1
        except Exception:                                              # noqa: BLE001 — counting must never break stderr
            pass
        return self._inner.write(s)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def activation_line(rep: dict) -> str:
    if rep.get("would_refuse") or rep.get("reason"):
        return f"{PREFIX} NOT ACTIVE: {rep.get('reason')}"
    if rep.get("mode") == "off":
        return (f"{PREFIX} STOCK mode=off variant={rep.get('variant')} tree={rep.get('tree_state')} "
                f"digest={(rep.get('tree_digest') or '')[:16]} switches=none")
    if not rep.get("active"):
        return f"{PREFIX} NOT ACTIVE: {rep.get('reason')}"
    gpu = rep.get("gpu") or {}
    return (f"{PREFIX} ACTIVE mode={rep.get('mode')} variant={rep.get('variant')} tier={rep.get('tier')} row={rep.get('row_label')} source={rep.get('row_source')} "
            f"tree={rep.get('tree_state')} gpu={gpu.get('name')} sm={gpu.get('sm')} stack={rep.get('stack_key')}{target_gpu_part(rep)} "
            f"switches={' '.join(f'{k}={v}' for k, v in (rep.get('switches') or {}).items())}{notes_part(rep)}")


def notes_part(rep: dict) -> str:
    """`` notes=<note>; <note>`` at the end of the ACTIVE / CHECK line when the kit has environment facts to name (activate.environment_notes:
    no C compiler, a torch / triton other than the tested stack) — said, never acted on; absent when there is nothing to say."""
    notes = rep.get("notes") or []
    return (" notes=" + "; ".join(notes)) if notes else ""


def target_gpu_part(rep: dict) -> str:
    """`` target_gpu=<name> [MISMATCH: <note>]`` when MODEL_OPT_TARGET_GPU is set (report only; never a refusal)."""
    tg = rep.get("target_gpu")
    if not tg:
        return ""
    note = rep.get("target_gpu_note")
    return f" target_gpu={tg}" + (f" MISMATCH: {note}" if note else "")


def partial_detail(levers: List[str], reasons: Optional[Dict[str, str]] = None) -> str:
    """``<lever>: <the kit's record>, ...`` for the levers of the row that fell back (``activate.completion``: ``levers_fallback`` in the
    row's order, ``fallback_reasons``), the lever names first."""
    reasons = reasons or {}
    return ", ".join(f"{k}: {reasons[k]}" if reasons.get(k) else k for k in levers)


def partial_not_active_line(mode: str, detail: str, code: int) -> str:
    """The exit line of a run in which a lever of the row could not run, one grammar for every entry point (``design``, ``warm``):
    ``[caliby-opt] NOT ACTIVE: <mode> ran short of its lever set — <detail>; the outputs on disk are not a <mode> run; exit <code>``."""
    return f"{PREFIX} NOT ACTIVE: {mode} ran short of its lever set — {detail}; the outputs on disk are not a {mode} run; exit {code}"


def modules_not_active_line(done: dict, code: int) -> str:
    """``[caliby-opt] NOT ACTIVE: the lever modules loaded in this process were not the <arm> arm's files (tree=<state>: <modules>); outputs
    are not this arm's; exit <code>`` — the line of the fail-closed module rule (``activate.modules_wrong``), said by the design children
    (``activate.modules_exit``, which appends the writer's rc) and by the environment route's exit gate (which appends its remedy)."""
    names = ", ".join(done.get("modules_wrong") or []) or "see opt_manifest.json overlay"
    return (f"{PREFIX} NOT ACTIVE: the lever modules loaded in this process were not the {done.get('modules_expected')} arm's files "
            f"(tree={done.get('modules_state')}: {names}); outputs are not this arm's; exit {code}")


def lever_gate_line(lever: str, reason: str, **evidence) -> str:
    """``[caliby-opt] LEVER name=<switch> state=skipped reason=<gate> impl=<the replaced module> origin=kit <evidence k=v ...>`` — the core's
    per-lever line (``opt_core.report.lever_line``, its grammar) for a DECLARED input gate of the lever met at call time
    (``registry.LEVERS[lever].gates``): the call takes upstream's own lines for that step with the same outputs; only the step's speed-up is
    idle. ``impl`` is the replaced module's name. A lever or reason the registry does not declare raises ``ValueError`` (an undeclared gate is
    a defect of the caller, never a line)."""
    if reason not in DECLARED_GATES.get(lever, ()):
        raise ValueError(f"{lever}: gate reason {reason!r} is not declared in registry.LEVERS[{lever!r}].gates {DECLARED_GATES.get(lever, ())}")
    impl = os.path.splitext(os.path.basename(registry.LEVERS[lever].file))[0]
    return _core.lever_line(TAG, lever, "skipped", reason=reason, impl=impl, origin="kit", **evidence)


def note_gate(lever: str, reason: str, **evidence) -> str:
    """Say the lever's gate line once for this call and count it (``tally()["gates"][lever][reason]``, the manifest's ``lever_gates``); the
    exit code is untouched — a declared gate is not a fallback. Returns the line."""
    text = lever_gate_line(lever, reason, **evidence)
    per = _TALLY["gates"].setdefault(lever, {})
    per[reason] = per.get(reason, 0) + 1
    say(text)
    return text


def check_line(rep: dict) -> str:
    verdict = "would refuse: " + str(rep.get("reason")) if rep.get("would_refuse") else "would activate"
    return (f"{PREFIX} CHECK mode={rep.get('mode')} variant={rep.get('variant')} tier={rep.get('tier')} row={rep.get('row_label')} tree={rep.get('tree_state')} "
            f"gpu={(rep.get('gpu') or {}).get('name')} stack={rep.get('stack_key')} compiler={rep.get('compiler')}{target_gpu_part(rep)}{notes_part(rep)} -> {verdict}")


FRESH = "\n"                                                           # every report line opens a fresh line: progress bars (tqdm's pending ``\r`` segments on the same stream) never prefix it


def fresh(line: str) -> str:
    """``line`` as this package prints it: a line break first, so a reader splitting the stream on ``\n`` and ``\r`` finds the line
    starting at ``[caliby-opt]`` whatever a progress bar left pending on the stream."""
    return FRESH + line


def say(line: str) -> None:
    """The one home of every ``[caliby-opt]`` line this package prints on stderr (activation, census, refusals, the parent's lines):
    a fresh line, the line, a line break, flushed."""
    sys.stderr.write(fresh(line) + "\n")
    sys.stderr.flush()


# ----------------------------------------------------------------------------------------------------------------- census lines
def _token(v) -> str:
    s = str(v)
    if not s or any(c.isspace() for c in s):
        raise ValueError(f"census token value {v!r} is empty or holds whitespace")
    return s


def census_line(tag: str, **values) -> str:
    """``[caliby-opt] <TAG> k=v ...`` with exactly the tokens of ``CENSUS_LINES[tag]`` in that order (a missing or extra key raises)."""
    keys = CENSUS_LINES[tag]
    if set(values) != set(keys):
        raise ValueError(f"{tag}: tokens {sorted(values)} != CENSUS_LINES {sorted(keys)}")
    return f"{PREFIX} {tag} " + " ".join(f"{k}={_token(values[k])}" for k in keys)


def route_variant() -> tuple:
    """(route word, variant) of this process: the mode (off | fast | exact); ``unknown`` before activation."""
    from . import activate                                             # activate imports this module: resolved at call time
    rep = activate.status()
    if "mode" not in rep:
        return "unknown", "unknown"
    return rep.get("mode") or "unknown", rep.get("variant") or "unknown"


def _torch():
    """torch if this process imported it (the report lines never import it themselves)."""
    return sys.modules.get("torch")


def _cuda_live() -> bool:
    t = _torch()
    return bool(t is not None and t.cuda.is_available() and t.cuda.is_initialized())


def clock() -> float:
    """``time.time()`` after ``torch.cuda.synchronize()`` when CUDA is initialised in this process."""
    if _cuda_live():
        _torch().cuda.synchronize()
    return time.time()


def peak_line(item: str) -> str:
    t = _torch()
    alloc = reserved = 0.0
    if _cuda_live():
        alloc, reserved = t.cuda.max_memory_allocated() / 2**30, t.cuda.max_memory_reserved() / 2**30
    route, variant = route_variant()
    return census_line("PEAK", item=item, alloc_gib=f"{alloc:.3f}", reserved_gib=f"{reserved:.3f}", route=route, variant=variant, pid=os.getpid())


def stack_line() -> str:
    """The process's stack facts, read never set: versions, the GPU, torch's TF32 / cuDNN flags as they stand, the allocator config,
    an environment label (``image``: $MODEL_OPT_ENV_LABEL, else ``unknown``; never read back), the package-set digest (activation's ``pip_freeze_sha256``, stack.PIP_FREEZE_RULE). autocast / compile / accelerators
    are constants of this tree: no route enters an autocast region, compiles a module or imports an optional accelerator package."""
    from . import activate
    rep = activate.status()
    t = _torch()
    torch_v = getattr(t, "__version__", None) or "unknown"
    cuda_v = (getattr(getattr(t, "version", None), "cuda", None) if t is not None else None) or "unknown"
    triton = sys.modules.get("triton")
    if triton is not None:
        triton_v = getattr(triton, "__version__", "unknown")
    else:
        from importlib import metadata as _md
        try:
            triton_v = _md.version("triton")
        except _md.PackageNotFoundError:
            triton_v = "absent"
    gpu, cc = "none", "none"
    if t is not None and t.cuda.is_available():
        gpu = t.cuda.get_device_name(0).replace(" ", "_")
        cc = "%d.%d" % tuple(t.cuda.get_device_capability(0))
    flags = {"tf32_matmul": "unknown", "cudnn_tf32": "unknown", "cudnn_deterministic": "unknown", "cudnn_benchmark": "unknown"}
    if t is not None:
        flags = {"tf32_matmul": t.backends.cuda.matmul.allow_tf32, "cudnn_tf32": t.backends.cudnn.allow_tf32,
                 "cudnn_deterministic": t.backends.cudnn.deterministic, "cudnn_benchmark": t.backends.cudnn.benchmark}
    sha = rep.get("pip_freeze_sha256") or stack.pip_freeze_sha256()[0] or "unknown"
    route, variant = route_variant()
    return census_line("STACK", torch=torch_v, cuda=cuda_v, triton=triton_v, python=platform.python_version(), gpu=gpu, cc=cc,
                       alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or "unset", autocast="never", compile="never", accelerators="none",
                       image="_".join((os.environ.get(stack.ENV_LABEL) or "").split()) or "unknown", pip_freeze_sha256=sha, route=route, variant=variant, pid=os.getpid(),
                       **{k: str(v) for k, v in flags.items()})


def outputs_written(out_dir: Optional[str] = None) -> None:
    """OUTPUTS_WRITTEN + PEAK item=PASS: once per design process, called by the process's caller (design_run / stock_design) after the
    writer's own last write — seq_des_outputs.csv and timing.json; ``n_items`` / ``n_designs`` are the writer's counts (``WRITER_TIMING``:
    n_inputs, n_designs; 0 when the file is absent)."""
    n_items = n_designs = 0
    if out_dir:
        try:
            import json
            with open(os.path.join(out_dir, WRITER_TIMING), "r", encoding="utf-8") as fh:
                tj = json.load(fh)
            n_items, n_designs = int(tj.get("n_inputs") or 0), int(tj.get("n_designs") or 0)
        except (OSError, ValueError, TypeError):
            pass
    route, variant = route_variant()
    say(census_line("OUTPUTS_WRITTEN", route=route, variant=variant, t=f"{clock():.6f}", n_items=n_items, n_designs=n_designs, pid=os.getpid()))
    say(peak_line("PASS"))


def env_clean_line(stripped: List[str], route: str, variant: str) -> str:
    """The kit and package switch names the parent stripped from the caller's environment before starting the design child (``none``
    when the shell carried none)."""
    return census_line("ENV-CLEAN", stripped=",".join(stripped) or "none", route=route, variant=variant)


def designs_line(written: int, expected: int) -> str:
    """The design count against the request (the parent's one count: cli.designs_written), verdict ok | incomplete."""
    return census_line("DESIGNS", designs_written=int(written), designs_expected=int(expected), verdict="ok" if int(written) >= int(expected) else "incomplete")


def register_exit_tally(mode: Optional[str], tree_state: Optional[str] = None) -> None:
    if _TALLY["registered"]:
        return
    _TALLY["registered"], _TALLY["mode"], _TALLY["tree_state"] = True, mode, tree_state
    if not _TALLY["stderr_wrapped"] and not isinstance(sys.stderr, _CountingStderr):
        sys.stderr = _CountingStderr(sys.stderr)
        _TALLY["stderr_wrapped"] = True
    atexit.register(_exit_gate)                                        # registered BEFORE the EXIT tally: atexit runs last-registered first, so the gate runs after the EXIT line has printed
    _core.register_exit_tally(TAG, exit_tally_line_fresh)             # the whole line is this kit's grammar; printed once at exit, on a fresh line


def exit_owned() -> None:
    """The caller of this process owns its exit code — the design children (``design_run`` / ``stock_design``: ``activate.partial_exit`` and
    ``activate.modules_exit`` decide it and the parent reports it); the environment route's exit gate (``_exit_gate``) then does nothing."""
    _TALLY["exit_owned"] = True


ENV_ROUTE_REMEDY = f"; {stack.ENV_MODE}=off runs stock"                  # the environment route's one remedy, appended to its NOT ACTIVE lines (no opt-out on that route)


def env_route_not_active_lines(mode: Optional[str], done: dict) -> List[str]:
    """The environment route's verdict lines at interpreter exit for ``activate.completion()``'s record, each ending ``…; exit 3;
    CALIBY_OPT=off runs stock``: the partial line every entry point prints (``partial_not_active_line``) when a lever of the row could not
    run at call time, and the module rule's line (``modules_not_active_line``, the predicate ``activate.modules_wrong`` the design children
    apply) when the lever modules loaded were not the mode's files (``tree=mixed``). Empty = the run was the mode's run."""
    from . import activate                                             # activate imports this module: resolved at call time
    lines = []
    if done.get("partial"):
        detail = partial_detail(done.get("levers_fallback") or [], done.get("fallback_reasons"))
        lines.append(partial_not_active_line(str(mode), detail, EXIT_NOT_ACTIVE) + ENV_ROUTE_REMEDY)
    if activate.modules_wrong(done):
        lines.append(modules_not_active_line(done, EXIT_NOT_ACTIVE) + ENV_ROUTE_REMEDY)
    return lines


def _exit_gate() -> None:
    """At interpreter exit, after the EXIT line: on the environment route (``CALIBY_OPT=<mode>`` in a host script, or ``caliby_opt.enable()``
    in one — no caller of this package owns the exit code, ``exit_owned()`` was never called) a run that was not the mode's run — a lever of
    the row could not run at call time (``completion()["partial"]``), or the lever modules loaded were not the mode's files
    (``activate.modules_wrong``: ``tree=mixed``) — ends NOT ACTIVE by name whatever the host was about to exit with:
    ``env_route_not_active_lines``, then ``os._exit(EXIT_NOT_ACTIVE)`` through the core's ``forced_exit`` (streams flushed; every EXIT tally
    has printed by then). Exit hooks the host registered BEFORE activation do not run in that case (atexit is last-in-first-out and the
    forced exit ends the interpreter): results must be on disk before interpreter exit. The same two rules the design children apply
    (``activate.partial_exit`` / ``activate.modules_exit``); nothing lets a subset or a mixed module set pass under the mode's name."""
    if _TALLY["exit_owned"] or not _TALLY["registered"]:
        return
    from . import activate                                             # activate imports this module: resolved at call time
    lines = env_route_not_active_lines(_TALLY["mode"], activate.completion())
    if not lines:
        return
    for text in lines:
        say(text)
    _core.forced_exit(EXIT_NOT_ACTIVE)


def lcp_state() -> str:
    mod = sys.modules.get("chroma.layers.complexity")
    if mod is None:
        return "not_loaded"
    reason = getattr(mod, "_X_LCP_DISABLED_REASON", "n/a")
    if reason == "n/a":
        return "stock_module"                                          # the upstream file: no such attribute
    return "available" if reason is None else f"disabled({reason!r})"


def multiseq_state() -> str:
    """``multiseq=`` at exit: ``not_loaded`` | ``stock_module`` (the potts module in this process is not the kit's) |
    ``<calls>calls/<sequences>seqs[,serial:<n>]`` — what X010 served in this process (`multiseq.census_word`)."""
    mod = sys.modules.get("caliby.model.seq_denoiser.denoisers.seq_design.potts")
    if mod is None:
        return "not_loaded"
    if not hasattr(mod, "sample_potts_multiseq"):
        return "stock_module"
    from . import multiseq
    return multiseq.census_word()


def cif_writers_state() -> str:
    """``cif_writers=`` at exit: ``not_loaded`` | ``stock_module`` (the seq_des_utils module in this process is not the kit's) | ``unused``
    (its writer never ran here: the ensemble route writes elsewhere) | ``sync`` (`CALIBY_X_BG_CIF` unset: the stock writes) |
    ``<mode>:<workers>w/<chunks>chunks/<files>files[,failed:<n>]`` — what X003/X011 did in this process (`bg_writers.census_word`)."""
    mod = sys.modules.get("caliby.eval.eval_utils.seq_des_utils")
    if mod is None:
        return "not_loaded"
    if not hasattr(mod, "_x_write_cifs"):
        return "stock_module"
    from . import bg_writers
    return bg_writers.census_word()


def fallback_lines_part() -> str:
    """``<lever>:<count> ...`` for the kit fallback lines counted so far, or ``none``."""
    return " ".join(f"{k}:{n}" for k, n in _TALLY["fallback_lines"].items()) or "none"


def partial_part() -> str:
    """``partial=<levers of the row that fell back>|none`` from ``activate.completion`` (the record on the env route)."""
    from . import activate                                             # activate imports this module: resolved at call time
    done = activate.completion()
    return f"partial={','.join(done['levers_fallback']) or 'none'}"


def exit_tree_state() -> str:
    """``tree=`` at exit: what the loaded lever modules actually are (overlay.exit_state) — ``exact`` | ``stock`` | ``mixed``;
    ``unknown`` before any activation."""
    if not _TALLY["registered"]:
        return "unknown"
    from . import overlay
    return overlay.exit_state(_TALLY["mode"])


def exit_tally_line(pid: Optional[int] = None) -> str:
    pid = os.getpid() if pid is None else pid
    loaded = stack.lever_modules_loaded()
    sw = stack.kit_switches_in()
    tree_state = exit_tree_state()
    if not loaded:
        return (f"{PREFIX} EXIT pid={pid} mode={_TALLY['mode']} tree={tree_state} touched_modules_loaded=0 (no module the kits replace was loaded in this process) "
                f"fallback_lines={fallback_lines_part()} {partial_part()} switches={' '.join(f'{k}={v}' for k, v in sw.items()) or 'none'}")
    return (f"{PREFIX} EXIT pid={pid} mode={_TALLY['mode']} tree={tree_state} touched_modules_loaded={len(loaded)} lcp_kernel={lcp_state()} "
            f"multiseq={multiseq_state()} cif_writers={cif_writers_state()} "
            f"fallback_lines={fallback_lines_part()} {partial_part()} switches={' '.join(f'{k}={v}' for k, v in sw.items()) or 'none'}")


def exit_tally_line_fresh() -> str:
    """The EXIT line as printed at interpreter exit (the core writes it verbatim): on a fresh line, like every other report line."""
    return fresh(exit_tally_line())


def tally() -> Dict[str, object]:
    return {"mode": _TALLY["mode"], "tree_state": exit_tree_state(), "touched_modules_loaded": stack.lever_modules_loaded(), "lcp_kernel": lcp_state(),
            "multiseq": multiseq_state(), "cif_writers": cif_writers_state(),
            "fallback_lines": dict(_TALLY["fallback_lines"]), "gates": {k: dict(v) for k, v in _TALLY["gates"].items()},
            "switches": stack.kit_switches_in()}
