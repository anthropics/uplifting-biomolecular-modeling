"""Activation: resolve and gate a mode in this process, prove the installed tree is the pinned upstream, load the levers, export the row, report.

The levers are the kits' files (``opt/forward/xattempt_addon/fast/*``); ``exact`` loads them by import hook under the upstream module
names (overlay.py) — nothing is written into site-packages, which holds the pinned upstream tree in every mode. What ``enable`` does,
once per process, before the model package is imported:

  0. the core pin: ``stack.core_gate()`` — statement one of every entry, so it has already run when activation starts, before anything
     of the core was imported — refuses an absent or older ``opt_core`` by name (exit 3; a newer core passes, the pin is a floor); called
     again here it is the same producer, and its facts (the ``[tool.opt_core]`` pin of ``opt/pyproject.toml`` and the installed
     ``opt_core``'s own version) are the report's core block;
  1. resolves the row (modes.resolve) and gates the box: kit directories and pins present, a variant named, the variant's weights
     under $MODEL_PARAMS_DIR, a CUDA GPU for a kit mode, the pinned upstream distributions installed (stock/check_pins.py); what the
     kit cannot be sure of — the GPU named by MODEL_OPT_TARGET_GPU vs the box's, no C compiler (the fused LCP kernel is Triton,
     JIT-compiled at first call), a torch / triton other than the tested stack — is named on the activation line and never refused;
  2. proves the installed tree is the pinned upstream tree, in both modes: every touched file at its upstream digest
     (``stack.tree_state() == "stock"``) and the kit's tree digest equal to stock/PINS.json "tree_digest_upstream" — anything
     else is the ``NOT STOCK`` refusal naming the files and the way back (reinstall the pinned upstream: tree.REINSTALL);
  3. for ``exact``: proves every kit file of the row is present (each is tracked in this tree; nothing is digest-compared) and installs
     the overlay (overlay.install); applies the late-activation rule — refused by name once any module the kits replace is
     imported (``stack.LEVER_MODULES``; ``import caliby`` alone imports ``caliby.api``), because an import hook no longer reaches a
     loaded module;
  4. exports the row's switches into this process's environment: the row is exported whole or not at all — a switch already present
     with another value, or any kit switch outside the row, is a refusal; for off asserts that no kit or package switch is present;
  5. prints the activation line, registers the exit tally, keeps the report (``status()``).

Repeated calls return the first report; a different mode or variant in the same process is refused. ``check(mode, variant)`` is
the same resolution as a dry run: gates and reports (the overlay files proven, not installed), exports nothing, imports nothing
upstream; ``env`` lets the caller gate the environment a child process will see instead of this process's (cli.py's ``check``
does, for both modes).

The report: ``mode``, ``variant``, ``active``, ``row_label``, ``row_source``, ``switches``, ``levers_applied`` (the
switches of the row at a non-inert value), ``levers_fallback`` / ``fallback_reasons`` / ``levers_unavailable`` / ``partial`` (empty /
False at activation — the kits report a fallback at call time; design_run.py completes them at exit from the exit tally, ``completion``,
and a run in which a lever could not run ends NOT ACTIVE by name, ``partial_exit``; ``lever_gates``: the declared input gates met at
call time, never partial), ``notes`` (environment facts named on the
ACTIVE line and never acted on: ``environment_notes``), ``tree_state`` (the lever-module
set of this process: ``exact`` | ``stock``), ``site_packages_state`` / ``tree_files`` / ``tree_digest`` (the installed tree, proven
``stock`` at the pinned digest), ``overlay`` (module -> kit file and digest), ``opt_core`` (the installed core's facts, ``ok``, ``pinned``: the core pin gate's), ``gpu``,
``target_gpu`` / ``target_gpu_note``, ``stack_key``, ``compiler``, ``package_version``, ``upstream`` / ``caliby_version``, ``reason``.
"""
from __future__ import annotations

import os
from typing import Optional

from . import ActivationError, __version__, modes, overlay, registry, report as _report, stack, tree

_REPORT: Optional[dict] = None
INERT_VALUES = ("0", "")                                              # a switch at this value applies no lever (the kits' own convention)


def status() -> dict:
    return dict(_REPORT) if _REPORT is not None else {"active": False, "reason": "caliby_opt.enable() has not run in this process"}


def _effective_variant(variant: Optional[str]) -> Optional[str]:
    env_v = (os.environ.get(stack.ENV_VARIANT) or "").strip().lower() or None
    v = (variant or "").strip().lower() or None
    if v and env_v and v != env_v:
        raise ValueError(f"variant {v} disagrees with {stack.ENV_VARIANT}={env_v}; unset one")
    return v or env_v or modes.DEFAULT_VARIANT


def levers_applied(switches: dict) -> list:
    """The row's switches at a non-inert value, in the row's order."""
    return [k for k, v in (switches or {}).items() if v not in INERT_VALUES]


def target_gpu_note(target: Optional[str], gpu: Optional[dict]) -> Optional[str]:
    """One line comparing MODEL_OPT_TARGET_GPU with the box's GPU (name, compute capability, memory); None when they agree or when
    either side is unknown. Reported, never a refusal."""
    g = gpu or {}
    if not target or not g.get("name"):
        return None
    if target.lower() in g["name"].lower():
        return None
    return (f"MODEL_OPT_TARGET_GPU={target} but the GPU is {g['name']} (sm {g.get('sm')}, {g.get('memory_gb')} GB): not the GPU this "
            f"configuration was tuned on; the levers engage as they are (the fused LCP kernel JIT-compiles for this device, or the mode refuses by name)")


def environment_notes(rep: dict, pins: Optional[dict] = None) -> list:
    """The environment facts a kit mode names on its activation line without acting on them (a lever engages wherever its mechanism
    applies; uncertainty about the box is said, never a reason to disengage or refuse): no C compiler on PATH (the fused LCP kernel then
    builds from the Triton cache, or the mode refuses by name at its first call), and a torch / triton other than the kit's tested stack
    (stock/PINS.json ``pinned_stack``). The GPU's name against MODEL_OPT_TARGET_GPU is the separate ``MISMATCH:`` note."""
    notes = []
    if "CALIBY_X_LCP" in (rep.get("levers_applied") or []) and not rep.get("compiler"):
        notes.append("no C compiler on PATH: CALIBY_X_LCP builds from the Triton cache or the mode refuses by name at the kernel's first call")
    want = ((pins or {}).get("pinned_stack") or {})
    torch_v = (rep.get("gpu") or {}).get("torch")
    if torch_v and want.get("torch") and torch_v != want["torch"]:
        notes.append(f"torch {torch_v} is not the kit's tested {want['torch']} (stock/PINS.json pinned_stack): levers engage as they are")
    triton_v = stack.dist_version("triton")
    if torch_v and want.get("triton") and triton_v != want["triton"]:
        notes.append(f"triton {triton_v or 'absent'} is not the kit's tested {want['triton']}: CALIBY_X_LCP builds its kernel with it or the mode refuses by name")
    return notes


def _base(mode: str, variant: Optional[str], trigger: Optional[str] = None) -> dict:
    up = stack.upstream_versions()
    return {"active": False, "mode": mode, "variant": variant, "tier": None, "row_label": None, "row_source": None, "switches": {}, "row_alternatives": [],
            "levers_applied": [], "levers_fallback": [], "levers_unavailable": [], "partial": False, "lever_gates": {},
            "tree_state": None, "site_packages_state": None, "tree_digest": None, "overlay": {}, "opt_core": None,
            "gpu": {"name": None, "sm": None}, "stack_key": None, "compiler": None,
            "package_version": __version__, "trigger": trigger, "upstream": up, "caliby_version": (up.get("caliby") or {}).get("version"),
            "target_gpu": os.environ.get(stack.ENV_TARGET_GPU) or None, "target_gpu_note": None, "notes": [], "reason": None}


def resolve_report(mode: str, variant: Optional[str] = None, clean_workers: Optional[int] = None, *,
                   dry_run: bool = False, trigger: Optional[str] = None, need_gpu: bool = True, env: Optional[dict] = None,
                   ckpt: Optional[str] = None) -> dict:
    """The full report for (mode, variant) on this box. ``dry_run``: gate and describe, touch nothing. ``env``: the environment to
    gate (default: this process's). ``ckpt``: the run's sequence-design checkpoint — any name or path upstream's ``load_model`` takes
    (the weights gate requires the file only for the checkpoints stock/PINS.json names; default upstream's, PINS weights.weight_set). Sets ``reason``
    and ``would_refuse`` instead of raising."""
    variant = _effective_variant(variant)
    env = os.environ if env is None else env
    rep = _base(mode, variant, trigger)
    rep["would_refuse"] = False

    def refuse(why: str) -> dict:
        rep["reason"], rep["would_refuse"] = why, True
        return rep

    why = modes.mode_refusal(mode, variant)                            # the mode table's one refusal: an unknown name, exact on single
    if why:
        return refuse(why)
    rep["tier"] = modes.TIER_OF.get(mode)
    core = stack.core_gate()                                           # gate 0 ran at the entry (its refusal is exit 3 there); here it is the same producer: the report's core block
    rep["opt_core"] = dict(core["installed"], ok=True, pinned=core["pinned"])
    try:
        for k in (stack.KIT_PARTNER, stack.KIT_ADDON):
            stack.kit_dir(k)
    except FileNotFoundError as e:
        return refuse(str(e))
    if not os.path.isfile(stack.pins_path()):
        return refuse(f"stock pins not found at {stack.pins_path()}")
    p = stack.pins()
    try:
        res = modes.resolve(mode, variant, clean_workers)
    except ValueError as e:
        return refuse(str(e))
    rep["row_label"] = res.label
    rep["row_source"] = res.source
    rep["switches"] = dict(res.env)
    rep["clean_workers"] = res.clean_workers
    rep["row_alternatives"] = [f"{k}: {modes.ALTERNATIVES[k]}" for k in res.alternatives]
    rep["lever_classes"] = registry.classes(res.env.keys())
    rep["levers_applied"] = levers_applied(res.env)
    # pins
    bad, detail = stack.check_pins_detail()
    rep["pins"] = detail
    if bad:
        return refuse("upstream not at its pin: " + "; ".join(bad))
    if variant == "ensemble32" and detail.get("protpardelle", {}).get("source") == "not installed":
        return refuse("protpardelle is not installed (the ensemble32 variant generates conformers with it; stock/PINS.json upstream.protpardelle)")
    # weights: the variant's files, the run's sequence-design checkpoint among them
    rep["ckpt"] = ckpt or p["weights"]["weight_set"]
    why, wdetail = stack.weights_gate(variant, p, env, ckpt=rep["ckpt"])
    rep["weights"] = wdetail
    if why:
        return refuse(why)
    # the installed tree: the pinned upstream in both modes (its digest is proven by enable(), a subprocess away)
    files = stack.installed_files()
    state = stack.tree_state(files)
    rep["site_packages_state"] = state
    rep["tree_files"] = {n: f["state"] for n, f in files.items()}
    if state != "stock":
        return refuse(f"NOT STOCK: the installed tree is not the pinned upstream tree (tree state {state!r}); {tree.REINSTALL} "
                      f"(a kit mode loads the kits' files by import hook over it and installs no file); files: {rep['tree_files']}")
    rep["tree_state"] = "exact" if modes.is_kit_mode(mode) else "stock"  # the lever-module set this process imports (a kit mode loads the kits' files: the "exact" tree state)
    # late activation: an import hook no longer reaches a loaded module
    loaded = stack.lever_modules_loaded()
    rep["lever_modules_loaded"] = loaded
    if loaded and not dry_run:
        return refuse(f"late activation: {loaded[0]} is already imported in this process (import caliby alone imports caliby.api); "
                      f"the levers are the kits' files loaded by import hook, so enable() must run before the model package is imported")
    ov = overlay.plan(files, mode)                                     # what the import hook serves: the kit rows' files (a kit mode); nothing on off
    rep["overlay"] = {m: {"file": e["file"], "path": os.path.relpath(e["path"], stack.tree_home()), "sha256": e["sha256"]} for m, e in ov.items()}
    bad_files = overlay.missing(ov)
    if bad_files:
        return refuse("overlay files not proven (missing at their planned path): " + "; ".join(bad_files))
    # box
    rep["compiler"] = stack.compiler()
    if need_gpu or not dry_run:
        gpu = stack.gpu_info()
        rep["gpu"] = gpu
        rep["stack_key"] = stack.stack_key(gpu)
        rep["target_gpu_note"] = target_gpu_note(rep["target_gpu"], gpu)
    if modes.is_kit_mode(mode):
        if not rep["gpu"].get("available", False) and (need_gpu or not dry_run):
            return refuse(f"{mode} needs a CUDA GPU: {rep['gpu'].get('reason')}")
        rep["notes"] = environment_notes(rep, p)                       # named on the ACTIVE / CHECK line, never a refusal
        conflicts = {k: env[k] for k, v in res.env.items() if k in env and env[k] != v}
        if conflicts:
            return refuse(f"switches already set to other values than the row's: {conflicts} (the row is exported whole; unset them)")
        extra = {k: v for k, v in stack.kit_switches_in(env).items() if k not in res.env}
        rep["switches_extra_in_env"] = extra
        if extra:
            return refuse(f"kit switches outside the row in the environment: {extra} (the row is the whole composition; unset them)")
    else:
        proof = stack.stock_env_proof(env)
        rep["stock_env_proof"] = proof
        if not proof["clean"]:
            return refuse(f"stock with kit or package switches in the environment: {proof['present']} (unset them)")
    return rep


def enable(mode: str, variant: Optional[str] = None, *, clean_workers: Optional[int] = None,
           strict: bool = False, trigger: Optional[str] = None, ckpt: Optional[str] = None) -> dict:
    """Activate ``mode`` for ``variant`` in this process (see the module docstring). Returns the report; raises ActivationError
    when it cannot (and always prints the NOT ACTIVE line first)."""
    global _REPORT
    mode = (mode or "").strip().lower()
    variant = _effective_variant(variant)
    if _REPORT is not None:
        if _REPORT.get("mode") == mode and _REPORT.get("variant") == variant:
            return dict(_REPORT)
        why = f"already activated as mode={_REPORT.get('mode')} variant={_REPORT.get('variant')} in this process; {mode}/{variant} refused"
        _report.say(f"{_report.PREFIX} NOT ACTIVE: {why}")
        raise ActivationError(why)
    rep = resolve_report(mode, variant, clean_workers, dry_run=False, trigger=trigger, ckpt=ckpt)

    def refused(why: str):
        rep["reason"], rep["would_refuse"], rep["active"] = why, True, False
        _report.say(_report.activation_line(rep))
        return ActivationError(why)

    if rep.get("would_refuse"):
        rep["active"] = False
        _REPORT = rep
        _report.say(_report.activation_line(rep))
        raise ActivationError(rep["reason"])
    rep["tree_digest"], _ = tree.digest()                              # the kit's own digest over the installed trees, both modes
    want = stack.pins()["tree_digest_upstream"]["sha256"]
    if rep["tree_digest"] != want:
        _REPORT = rep
        raise refused(f"NOT STOCK: tree digest {rep['tree_digest'][:16]} != pinned {want[:16]}; {tree.REINSTALL}")
    if mode == "off":
        rep["active"] = False                                          # stock: no lever applied and no finder installed — the installed upstream files as they are
    else:
        try:                                                           # a kit mode: the kit rows' files by import hook
            overlay.install(overlay.plan(mode=mode))
        except overlay.OverlayError as e:
            _REPORT = rep
            raise refused(str(e))
        for k, v in rep["switches"].items():
            os.environ[k] = v
        rep["active"] = True
    rep["pip_freeze_sha256"], rep["pip_freeze_lines"] = stack.pip_freeze_sha256()   # the STACK line's digest of this interpreter's package set (stack.PIP_FREEZE_RULE), read once here
    _REPORT = rep
    _report.register_exit_tally(mode, rep.get("tree_state"))
    _report.say(_report.activation_line(rep))
    return dict(rep)


def check(mode: str, variant: Optional[str] = None, clean_workers: Optional[int] = None, need_gpu: bool = True, *,
          env: Optional[dict] = None, ckpt: Optional[str] = None) -> dict:
    """Dry run: resolve and gate on this box, apply nothing. ``need_gpu=False`` skips the torch import (CPU-only tests); ``env``
    gates that environment instead of this process's."""
    rep = resolve_report(mode, variant, clean_workers, dry_run=True, need_gpu=need_gpu, env=env, ckpt=ckpt)
    if modes.is_kit_mode(mode) and not rep.get("would_refuse") and stack.lever_modules_loaded():
        rep["note"] = "a lever module is already imported in this process: enable() here would be refused (late activation)"
    return rep


def completion(rep: Optional[dict] = None) -> dict:
    """What the kits' modules left readable at the end of the process, in the report's own keys: the levers of the row that fell
    back to their stock lines at call time, each by its own stderr line counted per lever (the ``stderr`` probes,
    report.FALLBACK_MARKERS), plus the fused LCP kernel when its module holds a failure record (``_X_LCP_DISABLED_REASON``, the
    ``module_attr`` probe: triton / the kernel did not import, or a served call could not build / launch it — the stock lines served
    from then on, by name) — as ``levers_fallback`` (the row's order) with ``fallback_reasons``
    (lever -> the kit's record) and ``partial``. A lever at an inert value (not in ``levers_applied``) cannot fall back. ``lever_gates``
    ({lever: {gate: calls}}) is the other record: the DECLARED input gates met at call time (``report.note_gate`` — upstream's lines for
    that step, the same outputs), which never count toward ``partial``. Read by design_run.py before it completes ``opt_manifest.json``
    and decides the exit, by the EXIT line (``partial=``) and by the environment route's exit gate (``report._exit_gate``)."""
    t = _report.tally()
    rep = rep if rep is not None else (_REPORT or {})
    applied = levers_applied(rep.get("switches") or {})
    reasons = {}
    lcp = str(t.get("lcp_kernel") or "")
    if "CALIBY_X_LCP" in applied and lcp.startswith("disabled"):
        reasons["CALIBY_X_LCP"] = lcp
    for lever, n in (t.get("fallback_lines") or {}).items():
        if lever in applied and n > 0:
            reasons[lever] = f"{n} kit fallback line(s): {_report.FALLBACK_MARKERS.get(lever)}"
    fallback = [k for k in applied if k in reasons]
    mode = rep.get("mode") or "off"
    want = "exact" if modes.is_kit_mode(mode) else "stock"
    state = overlay.exit_state(mode)                                   # what the loaded lever modules actually are: the arm's files, or not
    got = overlay.loaded()                                             # the installed finder's plan: the kit rows (a kit mode); nothing on off
    wrong = sorted(n for n, e in got.items() if not e["ok"])           # a planned module whose code came from another file
    if not modes.is_kit_mode(mode):
        wrong += overlay.foreign_lever_modules(exclude=set(got))       # off: a lever module that is not the installed upstream file
    return {"levers_fallback": fallback, "fallback_reasons": reasons, "partial": bool(fallback), "lever_gates": dict(t.get("gates") or {}),
            "exit_tally": t, "modules_state": state, "modules_expected": want, "modules_wrong": wrong}


def modules_wrong(done: dict) -> bool:
    """The fail-closed predicate on the lever modules of a process (``completion``'s record): True when any module the kits replace was
    not this arm's file (``modules_state`` != ``modules_expected``: an ``exact`` process whose lever module came from the installed tree,
    an ``off`` process that loaded a kit file). One predicate for both routes — the design children (``modules_exit``) and the environment
    route's exit gate (``report._exit_gate``)."""
    state = done.get("modules_state")
    return state is not None and state != done.get("modules_expected")


def modules_exit(rc: int, done: dict, not_active_code: int) -> int:
    """The fail-closed rule on the lever modules of a design process, read after its writer returned: when ``modules_wrong(done)`` the run
    is NOT this arm's run — announced by name (``report.modules_not_active_line``) and exited ``not_active_code`` whatever ``rc`` says (a
    module set is not a lever fallback; nothing allows it)."""
    if not modules_wrong(done):
        return rc
    _report.say(_report.modules_not_active_line(done, not_active_code) + f" (writer rc={rc})")
    return not_active_code


def partial_exit(rc: int, done: dict, mode: str, not_active_code: int) -> int:
    """The exit rule of a design process after its writer returned ``rc`` when a lever of the row could not run at call time
    (``done["partial"]``: the kits' own ``-> serial fallback`` / ``-> kit path`` lines counted on stderr, or the fused LCP kernel's
    failure record). A mode is all of its levers: a run in which one of them stepped aside is not a run of the mode, so the process
    ends NOT ACTIVE by name — ``[caliby-opt] NOT ACTIVE: <mode> ran short of its lever set — <lever>: <record>, ...; the outputs on disk
    are not a <mode> run; exit <code>`` — whatever ``rc`` says. The record is the manifest's (``partial``, ``levers_fallback``)."""
    if not done.get("partial"):
        return rc
    _report.say(_report.partial_not_active_line(mode, _report.partial_detail(done["levers_fallback"], done.get("fallback_reasons")), not_active_code))
    return not_active_code
