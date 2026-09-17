"""The kit's per-step levers (registry route `install`), installed in THIS process after the stack is imported and before anything is traced
or a model is loaded — the ONE installer every caller uses (the driver's `--levers` flag through `tools/recipe.py install_levers`,
`mosaic_opt.enable` / the `.pth` hook via stack.activate):

    from mosaic_opt import levers
    levers.MODES                                   # the served mode words (a tier word appears only once a lever is wired into it)
    levers.env_required("fast")                    # {VAR: value} the mode's levers need exported BEFORE the interpreter starts (declared, never set here)
    info = levers.install("fast")                  # the mode's per-step levers, once per process (a second install() raises LeverError)
    info = levers.install("exact", plus="E1")      # a development request: the mode's levers plus named ones — labelled `exact+E1`, never `fast`
    info = levers.install("big", P5="tri64+pf8") # a per-lever setting: the lever's configure(SPEC) (the driver spells it `--levers big+P5=tri64+pf8`)
    levers.levers_off()                            # the ablation switch: MODEL_OPT_LEVERS_OFF=K1,F8 in the environment → ("K1", "F8"), skipped by name by install() (label `fast-K1-F8`)
    levers.installed()                             # {"mode", "label", "levers": [ids], "lines": [LEVER …], "effective": {id: describe()}, "env_required": {VAR: value}}

A mode IS its lever set (modes.KIT_MODES): `install(mode)` applies `modes.install_levers_of(mode)` in row order and prints, for EVERY
per-step lever of the registry, exactly one LEVER line per process (`opt_core.report.lever_line`, the shared core's grammar: `state=on` with the
lever's own `describe()` facts, or `state=off reason=mode:<label>`) — the arm log accounts for all of them. The qol levers (P1: environment;
P2/P3: driver flags) are not this module's: they do not touch the step. A lever module (registry ``module``: `mosaic.fast.<stem>` — a kit file, obtained through the recipe's ONE door
`tools/recipe.py kit_module(stem)` so every caller holds the same module object — or `opt_core.…`, a core module) exposes the uniform API
``ENV_REQUIRED: dict`` (module level; {} when none) · ``configure(spec_or_None)`` (None = the default setting) · ``install()`` · ``uninstall()`` ·
``describe() -> dict`` (flat: every value one SCALAR token — str / int / float / bool / None, no blank inside, never a container — in EVERY state the driver
reads it: before install, after configure, and after traffic, when a lever's counters and shape ledgers are populated (the post-phase record `manifest_record()`
re-reads describe() then);
never the line's own slots name/state/tag/reason; ``impl`` / ``origin`` default to the
registry entry's ``module`` / ``origin`` — a kernel lever names its kernel@version and origin itself) and,
optionally, ``emit_line(tag)`` / ``gate()`` (a kernel lever's census line and fail-closed gate, run by ``finalize()`` after the phases), imports without initialising the JAX backend, refuses by name before anything is traced, and
clears what it caches on ``uninstall()``. This installer never guesses around a refusal (LeverError names the lever and the reason; the
driver records `lever_refused`, the CLI exits 3) and never runs stock's step under a tier word.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Dict, List, Optional, Tuple

from opt_core import report as _core_report

from . import modes
from .registry import INSTALL, LEVERS, LEVERS_FLAG

TAG = "mosaic-opt"
PLUS = "plus"                                                     # the settings key for levers beyond the mode's set: install(mode, plus="ID[=SPEC],…") — the driver flag's grammar
DESCRIBE_KEYS = ("impl", "origin")                                # what a lever's describe() must carry for its LEVER line (opt_core.report.lever_line's pinned fields)
RESERVED_KEYS = ("name", "state", "tag", "reason")                # the LEVER line's own slots: never keys of a lever's describe()
ENV_LEVERS_OFF = "MODEL_OPT_LEVERS_OFF"                            # the ablation switch, the model-optimization family's one spelling: `MODEL_OPT_LEVERS_OFF=<id>[,<id>…]` — the named levers are skipped BY NAME (levers_off)
_EMPTY = {"mode": None, "label": None, "levers": [], "lines": [], "effective": {}, "env_required": {}, "specs": {}, "levers_off": []}
_STATE: dict = dict(_EMPTY)


class LeverError(RuntimeError):
    """A lever could not be installed (unknown id, not a per-step lever, no module, a required variable not exported, the module's
    own named refusal, a second install() in one process): the arm refuses by name (driver: status error `lever_refused`; CLI / hook: NOT
    ACTIVE, exit 3)."""


def parse_ids(text: Optional[str]) -> List[Tuple[str, Optional[str]]]:
    """`ID[=SPEC][,ID[=SPEC]...]` (the `plus` setting: what follows the WORD and its `+` in the driver's flag) -> [(id, spec or None)] in the given order. An unknown id,
    an id that is not a per-step lever (route env / flag: switched by the environment or its own driver flag) or a repeated id is a
    LeverError naming it."""
    out: List[Tuple[str, Optional[str]]] = []
    for word in (w.strip() for w in (text or "").split(",")):
        if not word:
            continue
        lid, eq, spec = word.partition("=")
        if lid not in LEVERS:
            raise LeverError(f"lever_unknown: {lid!r} ({LEVERS_FLAG} names registry levers: {', '.join(LEVERS)})")
        if LEVERS[lid].route != "install":
            raise LeverError(f"lever_not_per_step: {lid} is switched by its {LEVERS[lid].route} ({LEVERS[lid].flag or 'environment'}), not by {LEVERS_FLAG} "
                             f"(the per-step levers: {', '.join(INSTALL) or 'none'})")
        if any(l == lid for l, _ in out):
            raise LeverError(f"lever_repeated: {lid} named twice in {text!r}")
        out.append((lid, spec if eq else None))
    return out


def levers_off(environ: Optional[dict] = None) -> Tuple[str, ...]:
    """The ablation switch: `MODEL_OPT_LEVERS_OFF=<id>[,<id>…]` names levers of the registry to skip BY NAME in this process — every lever stays
    individually switchable without a new mode word. Returns the ids in registry order (empty when unset / blank); an id the registry does
    not carry is refused by name (`levers_off_unknown`), never ignored. A per-step lever named here gets `LEVER … state=off
    reason=levers_off:MODEL_OPT_LEVERS_OFF` from `install()` and the label `<mode>-<ID>…`; a row lever (P1: environment, P2 / P3: driver
    flags) is dropped from the row by the resolver (`modes.resolve(levers_off=…)`), which reads this same function. The variable rides the
    arm's environment (the `MODEL_OPT` prefix is not a stock-stripped one) and means nothing to stock."""
    env = os.environ if environ is None else environ
    text = (env.get(ENV_LEVERS_OFF) or "").replace("+", ",")
    ids = [t.strip() for t in text.split(",") if t.strip()]
    unknown = [i for i in ids if i not in LEVERS]
    if unknown:
        raise LeverError(f"levers_off_unknown: {ENV_LEVERS_OFF}={env.get(ENV_LEVERS_OFF)!r} names no lever of this kit: {','.join(unknown)} (levers: {', '.join(LEVERS)})")
    return tuple(l for l in LEVERS if l in ids)


def check_needs(wanted: List[str], off: List[str]) -> None:
    """A lever whose registry ``needs`` names a lever not in the request is refused by name (`lever_needs`) — F8 rides F6's served call:
    switching F6 off while keeping F8 would mis-compose silently otherwise."""
    for lid in wanted:
        missing = [n for n in LEVERS[lid].needs if n not in wanted]
        if missing:
            why = f"switched off by {ENV_LEVERS_OFF}" if any(m in off for m in missing) else "not in this request"
            raise LeverError(f"lever_needs: {lid} needs {','.join(missing)} on in the same process ({why}); switch {lid} off with it "
                             f"({ENV_LEVERS_OFF}={','.join(sorted(set(off) | {lid} | set(missing)))}) or keep {','.join(missing)}")


def plan(mode: Optional[str], **settings) -> dict:
    """What `install(mode, **settings)` would apply, without importing or applying anything: {"mode", "label", "levers": [ids in order],
    "specs": {id: spec|None}, "levers_off": [ids]}: the mode's levers at the settings the mode pins (`modes.specs_of`; None = the module's default setting). `settings`:
    ``plus="ID[=SPEC],…"`` (levers beyond the mode's set) and ``<ID>="SPEC"`` per lever (its configure(SPEC)) — a development request: every lever
    added or setting changed by the caller is spelled in the label (`<mode>+<ID>[=SPEC]…`), so a changed composition never wears the bare mode word;
    the ablation switch (`levers_off()`: `MODEL_OPT_LEVERS_OFF`) removes the named per-step levers from the request and spells them `-<ID>` in the
    label. Refusals: a word outside MODES (a tier word with no lever wired is named as such), `plus` on `off` (stock is no lever applied), a
    setting for a lever the request does not install, a lever whose ``needs`` the request leaves out (`lever_needs`)."""
    word = (mode or "").strip().lower()
    if word not in modes.MODES:
        raise LeverError(modes.unknown_mode_message(mode if mode is not None else ""))
    settings = dict(settings)
    pairs = parse_ids(settings.pop(PLUS, None))
    own = list(modes.install_levers_of(word))
    extra = [lid for lid, _ in pairs if lid not in own]
    if word == "off" and pairs:
        raise LeverError(f"mode 'off' is stock (no lever applied): {PLUS}={','.join(l for l, _ in pairs)} needs a kit mode (exact, or the tier the lever belongs to)")
    switched_off = levers_off()
    off = [lid for lid in own + extra if lid in switched_off]                      # the per-step levers of this request the ablation switch names (row levers are the resolver's)
    wanted = [lid for lid in own + extra if lid not in off]
    check_needs(wanted, off)
    pinned = modes.specs_of(word)                                                  # the settings the mode pins for its own levers (KIT_MODES[mode]["specs"]); None = the module's default setting
    specs: Dict[str, Optional[str]] = {lid: pinned.get(lid) for lid in wanted}
    asked: Dict[str, Optional[str]] = {}                                           # what the CALLER set (plus ID=SPEC, <ID>=SPEC): part of the label, so a changed setting never wears the bare word
    for lid, spec in pairs:
        if spec is not None:
            specs[lid] = asked[lid] = spec
    for k, v in settings.items():
        if k not in LEVERS:
            raise LeverError(f"lever_setting_unknown: {k}={v!r} (settings are {PLUS}=ID[=SPEC],… or <lever id>=SPEC; levers: {', '.join(LEVERS)})")
        if k not in specs:
            why = f"switched off by {ENV_LEVERS_OFF}" if k in off else f"a lever this request does not install ({', '.join(wanted) or 'no per-step lever'})"
            raise LeverError(f"lever_setting_unused: {k}={v!r} for {why}")
        specs[k] = asked[k] = None if v is None else str(v)
    marks = [l + (f"={asked[l]}" if asked.get(l) is not None else "") for l in wanted if l in extra or (asked.get(l) is not None and asked[l] != pinned.get(l))]
    return {"mode": word, "label": word + "".join("+" + m for m in marks) + "".join("-" + l for l in off), "levers": wanted, "specs": specs, "levers_off": off}


KIT_PREFIX = "mosaic.fast."                                       # registry `module` values under the kit's sub-package resolve through the recipe's ONE door (recipe.kit_module)


def _module(lid: str):
    """The lever's module object: a kit module (`mosaic.fast.<stem>`) through `tools/recipe.py kit_module(stem)` — the one door, so the
    driver, this installer and any tool hold the SAME object — or a core module (`opt_core.…`) by import."""
    lv = LEVERS[lid]
    if not lv.module:
        raise LeverError(f"lever_no_module: {lid} has no module on this kit line (registry.py; CHANGES.md \"What is not wired\")")
    try:
        if lv.module.startswith(KIT_PREFIX):
            from . import recipe as _recipe
            mod, rec = _recipe.module().kit_module(lv.module[len(KIT_PREFIX):])
            if rec["source"] != KIT_PREFIX.rstrip("."):                      # a per-step lever runs from the installed package's copy (placed by run.sh install / the image), never a tools/ or source-tree file
                raise LeverError(f"lever_module_not_installed: {lid}: {lv.module} is not carried by the installed mosaic (loaded from {rec['source']}: {rec['file']}); "
                                 f"install the kit's lever files into site-packages/mosaic/fast/ (bash run.sh install, or python -m mosaic_opt.leverfiles) — an image whose installed copy lacks or differs from the kit's file is not this kit line's")
            return mod
        return importlib.import_module(lv.module)
    except LeverError:
        raise
    except Exception as e:  # noqa: BLE001  an import failure of the lever's own module (or the door's named refusal), named
        raise LeverError(f"lever_import_failed: {lid} ({lv.module}): {type(e).__name__}: {e}") from e


def _env_of(lid: str, mod) -> Dict[str, str]:
    env = getattr(mod, "ENV_REQUIRED", None)
    if not isinstance(env, dict):
        raise LeverError(f"lever_env_undeclared: {lid} ({LEVERS[lid].module}) has no module-level ENV_REQUIRED dict ({{}} when it needs none)")
    return {str(k): str(v) for k, v in env.items()}


def env_required(mode: Optional[str], **settings) -> Dict[str, str]:
    """{VAR: value} the request's levers declare (their module-level ENV_REQUIRED, in lever order): what must be exported before the
    interpreter starts (the row carries exactly these assignments). Two levers
    declaring different values for one variable is a LeverError naming both."""
    p = plan(mode, **settings)
    out: Dict[str, str] = {}
    owner: Dict[str, str] = {}
    for lid in p["levers"]:
        for k, v in _env_of(lid, _module(lid)).items():
            if k in out and out[k] != v:
                raise LeverError(f"lever_env_conflict: {k} = {out[k]!r} ({owner[k]}) vs {v!r} ({lid})")
            out[k], owner[k] = v, lid
    return out


SCALARS = (str, int, float, bool, type(None))                     # a describe() value is a scalar: a container (list / tuple / dict / set) is never one token, whatever it holds today


def _token_ok(v) -> bool:
    if not isinstance(v, SCALARS):
        return False
    t = str(v)
    return bool(t) and not any(c.isspace() for c in t)


def _describe(lid: str, mod) -> dict:
    """The lever's own facts for its LEVER line and the manifest: describe() must return a dict carrying `impl` and `origin` (origin one of
    opt_core.report.LEVER_ORIGINS), every key and value a single non-blank SCALAR token — str / int / float / bool / None, never a container (the shared LEVER grammar: `k=v` fields split on whitespace; a list that is empty before install() and worded after it is the same defect either way)."""
    try:
        facts = mod.describe()
    except Exception as e:  # noqa: BLE001  the lever's describe() failing is a refusal, named
        raise LeverError(f"lever_describe_failed: {lid}: {type(e).__name__}: {e}") from e
    if not isinstance(facts, dict):
        raise LeverError(f"lever_describe_incomplete: {lid}: describe() returned {type(facts).__name__}, not a dict")
    facts = dict(facts)
    defaults = {"impl": LEVERS[lid].module, "origin": LEVERS[lid].origin}   # a lever whose implementation IS its module says nothing: the registry names it (a kernel lever names its kernel@version and origin itself)
    missing = [k for k in DESCRIBE_KEYS if k not in facts and not defaults[k]]
    if missing:
        raise LeverError(f"lever_describe_incomplete: {lid}: neither describe() nor the registry gives {', '.join(missing)}")
    for k in DESCRIBE_KEYS:
        facts.setdefault(k, defaults[k])
    reserved = [k for k in RESERVED_KEYS if k in facts]
    if reserved:
        raise LeverError(f"lever_evidence_invalid: {lid}: describe() carries the LEVER line's own slots {', '.join(reserved)} (the installer names the lever and its state)")
    origins = tuple(getattr(_core_report, "LEVER_ORIGINS", ("core", "kit")))
    if facts["origin"] not in origins:
        raise LeverError(f"lever_evidence_invalid: {lid}: origin={facts['origin']!r} is not one of {origins}")
    bad = [k for k, v in facts.items() if not _token_ok(k) or not _token_ok(v)]
    if bad:
        raise LeverError(f"lever_evidence_invalid: {lid}: describe() keys and values must be single non-blank tokens: {', '.join(f'{k}={facts[k]!r}' for k in bad)}")
    return dict(facts)


def off_line(lid: str, reason: str) -> str:
    lv = LEVERS[lid]
    return _core_report.lever_line(TAG, lid, "off", reason=reason, impl=lv.module or lv.kit_file, origin=lv.origin)


def on_line(lid: str, facts: dict) -> str:
    facts = dict(facts)
    impl, origin = facts.pop("impl"), facts.pop("origin")
    strategy = facts.pop("strategy", None)
    return _core_report.lever_line(TAG, lid, "on", impl=impl, origin=origin, strategy=strategy, **{k: _token(v) for k, v in facts.items()})


def _token(v) -> str:
    """A LEVER-line value is one token (opt_core.report): blanks inside a lever's fact become `_`."""
    s = "none" if v is None else str(v)
    return "_".join(s.split()) or "''"


def preconditions() -> Optional[str]:
    """The installer's own late-install rule, whichever caller (the driver, enable(), a program importing the package): per-step levers
    rebind what the model build and the traced step read, so they go in before any Boltz2 instance exists. Returns the refusal words or
    None. (The environment a lever declares is checked by value in install(); exporting it before the interpreter starts is the caller's
    process setup.)"""
    from . import stack as _stack
    chk = _stack.register_instance_counter()                          # upstream Boltz2 instances: counted from now on, the existing ones found once (sys.modules only: nothing of upstream is imported here)
    if chk["n"]:
        return f"lever_after_model: {chk['n']} Boltz2 instance(s) already exist in this process ({chk['method']}); per-step levers go in before the model is built"
    return None


def install(mode: Optional[str], **settings) -> dict:
    """Install the per-step levers of `mode` (plus the requested ones) in THIS process — whichever caller (the driver's `--levers` flag,
    `enable()`, the `.pth` hook): after the stack is imported, before any model is loaded and before anything is traced. Every variable of `env_required` must already be in the environment
    with its value (declared by the levers, exported by the row or the caller — never set here: the JAX backend reads them at initialisation).
    Levers install in the request's order (the mode's row order, then the extra ones), each `install()` then `configure(spec)`; every per-step lever of the registry gets exactly one
    LEVER line (on | off). Returns `installed()`. Atomic: any refusal or failure uninstalls what this call installed and raises LeverError by
    name, leaving the process as it was — a second install() raises `lever_already_installed` only after a COMPLETE first one."""
    if _STATE["mode"] is not None:
        raise LeverError(f"lever_already_installed: this process installed {_STATE['label']} ({','.join(_STATE['levers']) or 'no per-step lever'}); "
                         f"per-step levers are installed once per process, before tracing")
    p = plan(mode, **settings)
    late = preconditions()
    if late:
        raise LeverError(late)
    need = env_required(mode, **settings)
    wrong = [f"{k}={v!r} (found {os.environ.get(k)!r})" for k, v in need.items() if os.environ.get(k) != v]
    if wrong:
        raise LeverError("lever_env_required: export before the interpreter starts: " + "; ".join(wrong))
    lines: List[str] = []
    effective: dict = {}
    done: List[str] = []
    try:
        for lid in p["levers"]:                                                                  # the request's order: the mode's row order, then the extra ones
            mod = _module(lid)
            try:
                mod.install()                                                                  # the module's entry points rebound (nothing changes numerically yet)
                done.append(lid)
                mod.configure(p["specs"][lid])                                                 # None = the lever's default setting; a refusal here is rolled back with the install
            except LeverError:
                raise
            except Exception as e:  # noqa: BLE001  the lever's own refusal or failure, named; an out-of-memory is re-raised as itself below
                if _is_oom(e):
                    raise
                raise LeverError(f"lever_refused: {lid}: {type(e).__name__}: {e}") from e
            effective[lid] = _describe(lid, mod)
            lines.append(_core_report.emit(on_line(lid, effective[lid])))
        for lid in INSTALL:                                                                      # every other per-step lever of the registry: one `off` line each — the ablation switch's by its name, the rest by the mode's label
            if lid not in p["levers"]:
                reason = f"levers_off:{ENV_LEVERS_OFF}" if lid in p["levers_off"] else f"mode:{p['label']}"
                lines.append(_core_report.emit(off_line(lid, reason)))
    except Exception as e:                                                                       # atomic: whatever failed (a named refusal, the LEVER grammar's ValueError, an OSError on emit), nothing stays half-installed
        failed = _rollback(done)
        if _is_oom(e) and not failed:
            raise
        if isinstance(e, LeverError) and not failed:
            raise
        words = str(e) if isinstance(e, LeverError) else f"lever_evidence_invalid: {type(e).__name__}: {e}"
        if failed:
            words += "; rollback_incomplete: " + "; ".join(failed)
        raise LeverError(words) from e
    _STATE.update({"mode": p["mode"], "label": p["label"], "levers": [l for l in p["levers"]], "lines": lines, "effective": effective,
                   "env_required": need, "specs": dict(p["specs"]), "levers_off": list(p["levers_off"])})
    return installed()


def _rollback(done: List[str]) -> List[str]:
    """Uninstall `done` in reverse; returns the NAMED failures (empty = clean)."""
    failed: List[str] = []
    for lid in reversed(done):
        try:
            _module(lid).uninstall()
        except Exception as ex:  # noqa: BLE001  a rollback failure is named in the raised LeverError, never swallowed
            failed.append(f"{lid}: {type(ex).__name__}: {ex}")
    return failed


def _is_oom(e: BaseException) -> bool:
    """The core's out-of-memory classifier (opt_core.oom.is_oom): an OOM reaches the caller as itself, never as a lever refusal."""
    from opt_core.oom import is_oom
    return bool(is_oom(e))


def installed() -> dict:
    """This process's per-step lever state: {"mode", "label" (`<mode>`, `<mode>+<ID>…`, `<mode>-<ID>…`), "levers": [ids installed, in order], "lines": [the
    LEVER lines printed], "effective": {id: describe() at install}, "env_required": {VAR: value}, "specs": {id: spec|None}, "levers_off": [the
    per-step levers the ablation switch skipped]}; before install(): mode None and empty fields. `effective` re-read live: `effective_now()`."""
    return {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in _STATE.items()}


def effective_now() -> dict:
    """{id: describe()} re-read from the installed lever modules now (counters a lever keeps while the step runs)."""
    return {lid: _describe(lid, _module(lid)) for lid in _STATE["levers"]}


def manifest_record() -> dict:
    """The driver's `manifest["levers"]`: {<id>: {"state": "on", "requested": the row's spec request (None = the default setting), "spec": the
    lever's own canonical effective spec (its describe(); the request when the lever declares none), **describe() now}} — `levers.<id>.state`
    is the registry probe of a per-step lever (registry.probe_key)."""
    now = effective_now()
    out = {}
    for lid in _STATE["levers"]:
        requested = _STATE["specs"].get(lid)
        out[lid] = dict(now[lid], state="on", requested=requested, spec=now[lid].get("spec", requested))
    return out


def finalize(tag: str = "") -> dict:
    """After every phase ran (the driver before it writes results.json; any other caller at its end): each installed lever's facts re-read
    (`describe()` — the served / fallback counts a kernel lever keeps while the step runs), its own census line printed when the module has one
    (`emit_line(tag)`: kernel levers print the core Ledger's `served= fallback= fallback_by=` line), and its fail-closed gate run when the module
    has one (`gate()`: an undeclared fallback reason, a kernel error or zero served calls raises) — a raise is `LeverError("lever_gate: <id>: …")`
    and the arm fails by name. Returns `manifest_record()` (re-read now: installed is not the same as ran)."""
    for lid in _STATE["levers"]:
        mod = _module(lid)
        if hasattr(mod, "emit_line"):
            mod.emit_line(tag)
        if hasattr(mod, "gate"):
            try:
                mod.gate()
            except Exception as e:  # noqa: BLE001  the lever's own fail-closed gate, named
                raise LeverError(f"lever_gate: {lid}: {type(e).__name__}: {e}") from e
    return manifest_record()


def uninstall() -> list:
    """Uninstall every lever this process installed (reverse order; each module clears what it cached) and forget the request; returns the
    ids removed. For tests and tools that build several arms in one interpreter — a design process never calls it."""
    removed = []
    for lid in reversed(list(_STATE["levers"])):
        _module(lid).uninstall()
        removed.append(lid)
    reset_for_tests()
    return removed


def __getattr__(name):                                            # PEP 562: `levers.MODES` is modes.MODES itself — the served mode words have one home
    if name == "MODES":
        return modes.MODES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def reset_for_tests() -> None:
    _STATE.clear()
    _STATE.update({k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in _EMPTY.items()})
