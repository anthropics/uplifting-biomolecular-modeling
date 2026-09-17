"""The kit route's forward levers, installed in THIS process before any model is built — the ONE installer every in-process caller uses
(the kit arm kit_launch.py, `colabdesign_opt.enable` / the `.pth` hook via stack.activate, an external probe):

    from colabdesign_opt import levers
    info = levers.install("fast")                 # idempotent for the same mode word, a different word is refused

A mode IS its lever set, derived from the registry (registry.LEVERS, in registry order; modes.levers_of): `exact` installs the exact-class
levers (compilecache, parcompile, hoist_prev), `fast` every registered lever not replaced by another of its set (today compilecache,
parcompile, hoist_prev, nosub, nosub_fn, trimul, triatt, opm_fold, ln, proj, transition, txla — pallas is replaced by triatt and restored by
`fast-no-triatt`). Every lever of the mode prints exactly one LEVER line per process in the kit arm (on / skipped with its reason) — the arm
log accounts for all of them (evidence.py). The mode word's subtractive form `<mode>-no-<lever>` (modes.resolve) installs the mode's set
without the named levers and prints each dropped lever's line here, at install, as `state=off reason=ablated`.

Doctrine (the kit rule): a kit accepts everything stock accepts — a lever that cannot engage HERE steps aside BY NAME, never silently and never
as a refusal of the mode: its install raising one of the module's declared REFUSALS (no GPU lowering for a Pallas kernel, an unwritable
compile-cache directory, a jax backend initialised before the cache could attach, an upstream body off the pin a source patch expects) prints
`[colabdesign-opt] LEVER name=<lever> state=skipped reason=cannot_run detail=<Class:text> source=install` and the mode continues with the
rest of its set; a lever that composes on an attention-kernel lever of this kit (NEEDS_ATTENTION_KERNEL: proj rebinds the kernel-served
Attention, txla bridges triatt's op) steps aside as `reason=no_attention_kernel` when no such lever is installed in
the run (dropped by the word, or stepped aside itself). What stays a refusal (LeverError, exit 3) is configuration, never a lever: an unknown
mode word, a stock-route word on the kit route, a second mode word in one process.
Each lever is a module of the registry's protocol (registry.py): install / installed / uninstall / off_line / evidence, REFUSALS.
"""
from __future__ import annotations

import importlib
import os
import re
from typing import Dict, Iterable, Optional

from opt_core import report as _core_report

from . import modes, registry
from .names import LEVER_LOWERCACHE, KIT_ROUTE_MODES, LEVER_COMPILECACHE, LEVER_LN, LEVER_PALLAS, LEVER_PROJ, LEVER_TRIATT, LEVER_TRIMUL, TAG, LEVER_TXLA, LEVER_TRANSITION

_STATE: dict = {"mode": None, "installed": [], "skipped": {}, "info": {}}
ABLATED_REASON = "ablated"                         # the reason word on a dropped lever's state=off line (evidence.ABLATED_REASON reads the same word)
REPLACED_REASON = "replaced"                       # … on a registered lever that another lever of the mode supersedes (registry.py Supersession)
MODE_REASON = "mode"                               # … on a registered lever outside the mode's numerics class (a precision lever under an exact mode)
CANNOT_RUN = "cannot_run"                          # the reason word of a lever that stepped aside at install: its module raised one of its REFUSALS here (state=skipped)
NO_ATTENTION_KERNEL = "no_attention_kernel"        # … of a lever that composes on an attention-kernel lever when none is installed in the run (state=skipped)
NO_GPU = "no_gpu"                                  # … the dry-run routes foretell for the levers that need the GPU backend when no card is visible (stack.activate, nothing installed)
STEP_ASIDE_REASONS = (CANNOT_RUN, NO_ATTENTION_KERNEL, NO_GPU)   # `state=skipped reason=<one of these>` = the lever stepped aside by name (evidence.py: state `skipped`, exit 0) — distinct from a size / memory gate's `reason=gated`
ATTENTION_KERNEL_LEVERS = (LEVER_PALLAS, LEVER_TRIATT)    # the registered levers whose module declares ATTENTION_FWD_KERNELS (tests/test_levers.py holds the two equal)
NEEDS_ATTENTION_KERNEL = (LEVER_PROJ, LEVER_TXLA)   # levers that engage only beside an attention-kernel lever of this kit installed in the same run (proj rebinds the served Attention; txla bridges triatt's op)
NEEDS_GPU = (LEVER_COMPILECACHE, LEVER_LOWERCACHE, LEVER_TRIMUL, LEVER_PALLAS, LEVER_TRIATT, LEVER_LN, LEVER_PROJ, LEVER_TRANSITION, LEVER_TXLA)   # levers whose install needs the jax GPU backend (the Pallas kernels lower on it only; the compile cache is keyed by the GPU product) — registry `requires`
ATTENTION_FWD_KERNELS_ATTR = "ATTENTION_FWD_KERNELS"
DETAIL_MAX = 200                                   # a skipped line's detail= token: the refusal's class and text, blank-free, cut here


class LeverError(RuntimeError):
    """A configuration the kit route cannot serve — an unknown mode word, a stock-route word, a second mode word in one process: the arm
    refuses by name (exit 3). Never raised for a LEVER that cannot engage here: that lever steps aside by name (install: state=skipped)."""


def module_of(lever: str):
    """The lever's module (registry.LEVERS[lever].module), imported now."""
    return importlib.import_module(registry.LEVERS[lever].module)


def attention_kernel_levers() -> tuple:
    """The registered levers whose module declares attention-forward kernels (`<module>.ATTENTION_FWD_KERNELS`: pallas, triatt) — what the
    NEEDS_ATTENTION_KERNEL levers compose on. A literal (ATTENTION_KERNEL_LEVERS, held equal to the modules' declarations by
    tests/test_levers.py) so the dry-run routes name it without importing a lever module (jax)."""
    return tuple(l for l in registry.ORDER if l in ATTENTION_KERNEL_LEVERS)


def detail_token(text: str) -> str:
    """A LEVER line value carries no blank (opt_core.report.lever_line): the refusal's text with runs of whitespace as `_`, `=` as `:`."""
    t = re.sub(r"\s+", "_", str(text).strip()).replace("=", ":")
    return (t[:DETAIL_MAX - 3] + "...") if len(t) > DETAIL_MAX else (t or "none")


def skipped_line(lever: str, reason: str, detail: str) -> str:
    """The ONE line of a lever that stepped aside in this process: `LEVER name=<line name> state=skipped reason=<reason> impl=… origin=…
    [numerics=…] detail=<token> source=install pid=…` — impl / origin / numerics read off the module's own `off_line` so the line names the
    same implementation its on / off lines do."""
    mod = module_of(lever)
    kv = dict(re.findall(r"(\S+?)=(\S+)", mod.off_line(ABLATED_REASON)))
    extra = [("numerics", kv["numerics"])] if "numerics" in kv else []
    return _core_report.lever_line(TAG, registry.LEVERS[lever].line_name, "skipped", *extra, ("detail", detail_token(detail)), ("source", "install"), ("pid", os.getpid()),
                                   reason=reason, impl=kv.get("impl") or registry.LEVERS[lever].module.rsplit(".", 1)[-1] + "@kit", origin=kv.get("origin") or "kit")


def foretell(levers_of_run: Iterable[str], gpu_visible: bool) -> Dict[str, str]:
    """What the dry-run routes (check, the design parent) can say WITHOUT installing, in registry order: with no GPU visible the NEEDS_GPU
    levers of the run will step aside in the arm (reason no_gpu here; the arm's own lines say cannot_run with the backend's words), and with
    them every NEEDS_ATTENTION_KERNEL lever left without an attention kernel. {} when a card is visible: the arm's LEVER lines (EVIDENCE
    `skipped=`, `skipped: <lever>: <why>`) are then the only record of a lever that stepped aside."""
    run = [l for l in levers_of_run]
    gpu_less = {l for l in run if l in NEEDS_GPU and not gpu_visible}
    kernels = [l for l in attention_kernel_levers() if l in run and l not in gpu_less]
    out: Dict[str, str] = {}
    for l in run:
        if l in gpu_less:
            out[l] = f"{NO_GPU}: no GPU visible (nvidia-smi) — {registry.LEVERS[l].requires.split(';')[0].strip()}"
        elif l in NEEDS_ATTENTION_KERNEL and not kernels:
            out[l] = f"{NO_ATTENTION_KERNEL}: needs an attention-kernel lever of this kit ({'|'.join(attention_kernel_levers()) or 'none registered'}) in the same run; none will be installed"
    return out


def install(mode: Optional[str]) -> dict:
    """Install the levers of `mode` (a kit-route mode word) in this process, registry order; returns {"mode", "levers" (the word's resolved
    set), "levers_installed" (what engaged), "skipped" ({lever: "<reason>: <detail>"} — the levers that stepped aside by name), "ablated",
    "restored", <lever>: its install record | None}. LeverError only for configuration (see the module doc)."""
    try:
        res = modes.resolve(mode)
    except ValueError as e:
        raise LeverError(str(e)) from None
    if res.route != "kit" or (res.base or res.mode) not in KIT_ROUTE_MODES:
        raise LeverError(f"mode {res.mode!r} runs stock in a subprocess (stock_launch.py); the kit route installs levers only for {'|'.join(KIT_ROUTE_MODES)}")
    if _STATE["mode"] is not None:
        if _STATE["mode"] != res.mode:
            raise LeverError(f"mode {res.mode!r} requested but this process already installed the levers of mode {_STATE['mode']!r}")
        return dict(_STATE["info"])
    installed, skipped, info = [], {}, {"mode": res.mode, "levers": list(res.levers), **{l: None for l in registry.ORDER}}
    kernel_levers = attention_kernel_levers()
    for lever in registry.ORDER:
        if lever in res.ablated:
            _core_report.emit(module_of(lever).off_line(ABLATED_REASON))         # the dropped lever's ONE line of this process: state=off reason=ablated
            continue
        if lever not in res.levers:                                              # registered, outside this run's set: replaced by a lever OF THE SET that supersedes it, or not of this mode's class
            why = REPLACED_REASON if registry.replaced_by(lever, tuple(res.levers)) else MODE_REASON   # (a replaced lever whose replacer is dropped is IN the set: restored, installed below, named on every line)
            _core_report.emit(module_of(lever).off_line(why))
            continue
        mod = module_of(lever)
        if lever in NEEDS_ATTENTION_KERNEL and not any(k in installed for k in kernel_levers):     # registry order puts the attention-kernel levers first: installed or stepped aside by now
            why = f"needs an attention-kernel lever of this kit ({'|'.join(kernel_levers) or 'none registered'}) installed in the same run; none is (installed={','.join(installed) or 'none'})"
            _core_report.emit(skipped_line(lever, NO_ATTENTION_KERNEL, why)); skipped[lever] = f"{NO_ATTENTION_KERNEL}: {why}"   # steps aside by name: the attention it composes on is stock's here
            continue
        try:
            info[lever] = mod.install()
        except tuple(getattr(mod, "REFUSALS", ())) as e:                          # the lever's named cannot-run HERE: it steps aside by name (one skipped line) and the mode continues with the rest of its set
            why = f"{type(e).__name__}: {e}"
            _core_report.emit(skipped_line(lever, CANNOT_RUN, why)); skipped[lever] = f"{CANNOT_RUN}: {why}"
            continue
        installed.append(lever)
    info["levers_installed"] = installed
    info["skipped"] = dict(skipped)
    if res.ablated:
        info["ablated"] = list(res.ablated)
    if getattr(res, "restored", ()):
        info["restored"] = list(res.restored)
    _STATE.update({"mode": res.mode, "installed": installed, "skipped": dict(skipped), "info": info})
    return dict(info)


def uninstall() -> list:
    """Restore the upstream objects every installed lever replaced (reverse registry order); returns the levers uninstalled. For a process that
    A/Bs lever sets itself; the package's own routes never call it (one mode word per process). NOTE: a model built under lever set A keeps
    A's traced programs — (un)installing patches the classes future models are built from, never a built model or a compiled executable: rebuild
    the model (`mk_afdesign_model` + `prep_inputs`) after every install / uninstall, or an in-process A/B measures the same programs twice."""
    done = []
    for lever in reversed(list(_STATE["installed"])):
        module_of(lever).uninstall(); done.append(lever)
    _STATE.update({"mode": None, "installed": [], "skipped": {}, "info": {}})
    return done


def installed() -> list:
    return list(_STATE["installed"])


def skipped() -> dict:
    """{lever: "<reason>: <detail>"} of the levers that stepped aside at this process's install (empty before install / after uninstall)."""
    return dict(_STATE["skipped"])


def evidence() -> dict:
    """Live lever state of this process, per registered lever: {"installed": bool, ...the module's evidence()} (nosub's builds, pallas's Ledger fields)."""
    return {"mode": _STATE["mode"], "levers_installed": list(_STATE["installed"]), "skipped": dict(_STATE["skipped"]),
            **{l: {"installed": module_of(l).installed(), **module_of(l).evidence()} for l in registry.ORDER}}


def reset_for_tests() -> None:
    _STATE.update({"mode": None, "installed": [], "skipped": {}, "info": {}})
