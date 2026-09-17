"""The package's printed lines — one grammar; every reader of these lines derives its expectations from this source.

  ACTIVE    printed once per process at activation (enable / the autoload hook), before any model is built; a stack pin off its version
            (torch / triton / flash_attn / transformer_engine) is named at its end — `drift=<dist>:<installed>(pin_<pinned>),…` — never refused
  APPLIED   printed once per model instance when the levers of the mode have been applied to it (the kits' own lines precede it)
  DRY-RUN   `check`: the same resolution with nothing applied
  NOT ACTIVE  the mode could not be activated (the reason names the gate: an unknown mode word, the stock package absent or off its pinned
            upstream commit, no visible GPU, a usage-order conflict), or a lever of the mode's set was not applied to the loaded model
            (levers_fallback: its kit cannot run here — a compile / launch / import failure, an unsupported shape or dtype, its own reason) — a mode
            is all of its levers or it refuses: `NOT ACTIVE: partial activation — <detail>; a mode is all of its levers: the load is refused`,
            printed after the APPLIED line (whose `partial=` names the levers), <detail> = each lever with its kit record's reason
            (partial_refused_line). Environment drift (versions off pin, an unrecorded shape, a cache miss) is never this line: it is named below
  KERNELS   printed once per built model, before its first call:
            `[esmc-opt <mode>] KERNELS route=<stock|exact> via=kit regime=<b1|batched> stack=<name>
            flash_attn=<word> rotary=<word> te=<word> xformers=<word>` — the accelerators the BUILT model engages, read off its bound objects
            (kernels.py; the words: engaged:<impl>@<version> | off-by-rule:<rule>(…) | fallback:<impl>:<why> | off-by-route:<impl>:<why> |
            absent | present:<version>)
  KERNELS SHORT  the built model engages less than the pinned stack expects (kernels.expected): every differing accelerator named with
            its expected word — `KERNELS SHORT <route context>: <accelerator>=<observed> (expected <word>); … (named; the run proceeds on
            what the model engages)`; printed right after the KERNELS line, never an exit
  EXIT      the process tally at exit (the kits' own counters, read from their modules)
Every line starts with PREFIX; field order is fixed (key=value, space separated); values never contain spaces except the reason.
"""
from __future__ import annotations

import atexit
import os
import sys

PREFIX = "[esmc-opt]"
TALLY_MAX_FIELDS = 48


MODE_OFF_ESCAPE = "`ESMC_OPT=off` runs the stock SDK without the kit"   # named on every refusal line of a kit mode: a mode fails closed, running without the kit is the user's explicit choice

def _join(items) -> str:
    items = list(items or ())
    return ",".join(str(x) for x in items) if items else "none"


def gpu_label(gpu) -> str:
    if not gpu:
        return "none"
    name = str(gpu.get("name", "?")).replace(" ", "_")
    sm = str(gpu.get("sm", "?"))
    return f"{name}({sm})" if sm.startswith("sm") else f"{name}(sm{sm})"


def _head(rep: dict) -> str:
    return f"mode={rep.get('mode')} variant={rep.get('variant')} regime={rep.get('regime')}"


def _versions(rep: dict) -> str:
    up = rep.get("upstream") or {}
    return f"esm={up.get('esm')} torch={up.get('torch')} flash_attn={up.get('flash_attn')} transformer_engine={up.get('transformer_engine')}"


def activation_line(rep: dict | None) -> str:
    rep = rep or {}
    mode = rep.get("mode")
    if rep.get("active"):
        return (f"{PREFIX} ACTIVE {_head(rep)} levers={_join(rep.get('levers'))} levers_out={_join(rep.get('levers_out'))} "
                f"{_versions(rep)} gpu={gpu_label(rep.get('gpu'))} package={rep.get('package_version')}{_levers_on(rep)}")
    if rep.get("dry_run"):
        return (f"{PREFIX} DRY-RUN {_head(rep)} levers={_join(rep.get('levers'))} levers_out={_join(rep.get('levers_out'))} "
                f"{_versions(rep)} gpu={gpu_label(rep.get('gpu'))} package={rep.get('package_version')}{_levers_on(rep)}"
                + (f" would_refuse={rep.get('reason')}" if rep.get("reason") else ""))
    reason = rep.get("reason") or "enable() has not run"
    return f"{PREFIX} NOT ACTIVE: {reason}" + (f" ({_head(rep)})" if mode else "") + (f"; {MODE_OFF_ESCAPE}" if mode not in (None, "off") else "")   # a refusal in a kit mode names the one-flag escape on the same line (the stock modes' own NOT ACTIVE lines are not refusals)


def applied_line(rep: dict, rec: dict) -> str:
    return (f"{PREFIX} APPLIED model#{rec.get('model_index')} variant={rec.get('variant') or rep.get('variant')} regime={rep.get('regime')} "
            f"levers_applied={_join(rec.get('levers_applied'))} levers_fallback={_join(rec.get('levers_fallback'))} "
            f"t_apply={float(rec.get('t_apply_s') or 0.0):.2f}" + (f" partial={_join(rec['partial'])}" if rec.get("partial") else ""))


def partial_detail(partial, kit_records=None) -> str:
    """``<detail>`` of the partial lines: the levers of the mode not applied in full (stack.partial_levers, in order), each followed by
    its kit record's own reason when the record names one (``refused`` — the kit's refusal of its lever; ``fallback`` — an applied lever
    minus a part)."""
    recs = kit_records or {}
    out = []
    for name in partial or ():
        rec = recs.get(name.split("[", 1)[0]) or {}
        why = rec.get("refused") or rec.get("fallback")
        out.append(f"{name}: {why}" if why else name)
    return ", ".join(out) if out else "none"


def partial_refused_line(detail: str) -> str:
    """The refusal line of a partial activation (stack.apply_to raises PartialActivation from ``ESMC.from_pretrained`` right after it):
    ``NOT ACTIVE: partial activation — <detail>; a mode is all of its levers: the load is refused; <the mode-off escape>``."""
    return f"{PREFIX} NOT ACTIVE: partial activation — {detail}; a mode is all of its levers: the load is refused; {MODE_OFF_ESCAPE}"



def kernels_line(rec: dict) -> str:
    """The accelerator proof line of one built model (kernels.prove_once): the caller's mode in the tag, the route word, the serving
    process, the pass's regime, the pinned stack, one word per accelerator in
    modes.ACCELERATORS order."""
    from . import modes
    words = rec.get("words") or {}
    return (f"[esmc-opt {rec.get('mode')}] KERNELS route={rec.get('route')} via={rec.get('via')} regime={rec.get('regime')} stack={rec.get('stack_token')} "
            + " ".join(f"{n}={words.get(n, 'unread')}" for n in modes.ACCELERATORS))


def kernels_short_reason(rec: dict) -> str:
    diffs = rec.get("differences") or []
    body = "; ".join(f"{d['accelerator']}={d['observed']} (expected {d['expected']})" for d in diffs) or "none"
    warn = rec.get("upstream_warnings") or []
    return (f"route={rec.get('route')} via={rec.get('via')} regime={rec.get('regime')} stack={rec.get('stack_token')}: {body}"
            + (f" | upstream warned: {' / '.join(str(w)[:120] for w in warn)}" if warn else ""))


def kernels_short_line(rec: dict) -> str:
    """The accelerator proof's naming line when the built model engages less than the pinned stack: `KERNELS SHORT <route context>:
    <accelerator>=<observed> (expected <word>); … (named; the run proceeds on what the model engages)` — an environment short of the pins
    is stated on the line, never a refusal."""
    return f"{PREFIX} KERNELS SHORT {kernels_short_reason(rec)} (named; the run proceeds on what the model engages)"





def log_activation(rep: dict | None, stream=None) -> str:
    line = activation_line(rep)
    print(line, file=stream or sys.stdout, flush=True)
    return line


# ---------------------------------------------------------------------------------------------- the exit tally (the kits' own counters)
def _counters(mod, attr: str = "COUNTERS"):
    c = getattr(mod, attr, None)
    return dict(c) if isinstance(c, dict) else None


def memory_stats() -> dict | None:
    """Every kit lever module already imported in this process -> its COUNTERS dict (nothing is imported here)."""
    from . import registry
    out = {}
    for name in registry.LEVER_ORDER:
        modname = registry.lever_module(name)
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        c = _counters(mod)
        if c is not None:
            out[name] = c
        patch = sys.modules.get(modname + "._patch")
        if patch is not None and _counters(patch) is not None:
            out[name + "._patch"] = _counters(patch)
    return out or None


def tally_fields(stats: dict) -> list:
    out = []
    for mod, items in stats.items():
        items = list(items.items())
        out.append(f"{mod}={{" + ",".join(f"{k}={v}" for k, v in items[:TALLY_MAX_FIELDS // 4]) + "}")
    return out


def _drift(rep: dict) -> str:
    """`` drift=<dist>:<installed>(pin_<pinned>),…`` when a stack pin (torch, triton, flash_attn, transformer_engine) is off its version or
    absent (stack.version_gate) — environment drift NAMED on the activation line, never a refusal; nothing when the box is at the pins."""
    d = rep.get("drift") or {}
    return (" drift=" + ",".join(f"{n}:{v}" for n, v in d.items())) if d else ""


def _levers_on(rep: dict) -> str:
    """The package's own facts on the line: drift."""
    return _drift(rep)


def exit_tally_line(pid: int | None = None) -> str:
    pid = os.getpid() if pid is None else pid
    stats = memory_stats()
    return (f"{PREFIX} EXIT pid={pid} source=memory " + " ".join(tally_fields(stats)) if stats
            else f"{PREFIX} EXIT pid={pid} no lever counters: the kits' lever modules were never loaded in this process")


_REGISTERED = {"done": False}


def _print_exit_tally() -> None:
    try:
        line = exit_tally_line()
    except Exception as e:  # noqa: BLE001 — the tally never masks the exit
        line = f"{PREFIX} EXIT tally failed: {e!r}"
    print(line, flush=True)


def register_exit_tally() -> None:
    if not _REGISTERED["done"]:
        atexit.register(_print_exit_tally)
        _REGISTERED["done"] = True
