"""Upstream's triangle-kernel flags run as stated — the kit's triangle levers of that site step aside BY NAME.

`opendde pred --triatt_kernel <k>` / `--trimul_kernel <k>` (runner/batch_inference.py:749-760; `auto` | `cuequivariance` | `torch`) are
upstream's own selectors and every route accepts them (`off` included): the stock caller hands them to the engine untouched, and the KERNELS
census expects what the flag says for that site (modes.kernel_expectations: kind ``user``, word ``user:<flag>=<value>`` — never a refusal).
Under a kit mode a STATED value other than `auto` is a stock knob (settings.stock_knobs): the line is composed WITHOUT the kit's levers of
that site before anything is exported or installed (modes.line_without, the ablation rule every lever has; registry.STOCK_KNOB_LEVERS names
them), their LEVER rows read ``state=off reason=aside:stock_knob:<flag>=<value>``, the ACTIVE / PRED lines carry ``stock_knobs=<flag>=<value>``,
and the one per-run fact ``ODDE_STOCK_KNOBS=<flag>=<value>[,…]`` reaches the kit process (the ARM add-on's own attention / TriMul sites read it at
install and stay on the stock op: `attention=stock:stock_knob`, `trimul=stock:stock_knob`) and the rank processes of `--n_gpu P`. The rest of the
line is unchanged and the run completes (rc 0, not PARTIAL). Nothing stated, or `auto`: nothing changes — the shipped lines exactly.

  plan(knobs)        cli: the stated knobs of this call ({} = none), before the line resolves.
  compose(line)      modes.resolve: ``line`` without the levers of the stated knobs' sites (dependents follow, modes.LEVER_DEPENDENTS).
  gated_off(line)    {lever: "aside:stock_knob:<flag>=<value>"} for the report (stack.activate: levers_gated_off -> the LEVER rows' reason).
  word()             ``<flag>=<value>[,…]`` for the ACTIVE / PRED lines' ``stock_knobs=`` token and the fact's value; None when no knob is stated.
  export(environ)    the fact into the kit process's environment (cli, after activation; inherited by the rank processes); in_force(environ) reads it.
"""
from __future__ import annotations

import os

ENV = "ODDE_STOCK_KNOBS"                              # the per-run fact (never a switch a line exports or a user sets: cli writes it from the stated flags)
REASON = "aside:stock_knob"                          # the LEVER row's reason head
_ST = {"knobs": {}, "dropped": {}, "line": None}


def _reset() -> None:
    _ST.update(knobs={}, dropped={}, line=None)


def plan(knobs: dict | None) -> dict:
    """Record the stated stock knobs of this call ({flag: value}, settings.stock_knobs) before the line resolves; returns them."""
    _ST.update(knobs=dict(knobs or {}), dropped={}, line=None)
    return dict(_ST["knobs"])


def knobs(environ=None) -> dict:
    """The knobs of this call: the planned ones (cli), else the fact in the environment (a rank process / the kit process's hook)."""
    return dict(_ST["knobs"]) if _ST["knobs"] else in_force(environ)


def in_force(environ=None) -> dict:
    """Parse the fact ``ODDE_STOCK_KNOBS=<flag>=<value>[,…]`` from the environment ({} when absent / empty / malformed tokens skipped)."""
    env = os.environ if environ is None else environ
    out = {}
    for tok in (env.get(ENV) or "").split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            if k.strip() and v.strip():
                out[k.strip()] = v.strip()
    return out


def word(kn: dict | None = None):
    """``<flag>=<value>[,…]`` (settings.KERNEL_KNOBS order) or None when no knob is stated."""
    from . import settings
    kn = knobs() if kn is None else kn
    return ",".join(f"{f}={kn[f]}" for f in settings.KERNEL_KNOBS if f in kn) or None


def levers_for(kn: dict, carried) -> dict:
    """{lever: "aside:stock_knob:<flag>=<value>"} for the registry levers of the stated knobs' sites among ``carried`` (registry.STOCK_KNOB_LEVERS order)."""
    from . import registry
    out = {}
    for f, v in kn.items():
        for lv in registry.STOCK_KNOB_LEVERS.get(f, ()):
            if lv in carried and lv not in out:
                out[lv] = f"{REASON}:{f}={v}"
    return out


def compose(line, environ=None):
    """``line`` without the kit levers of the stated knobs' sites (modes.line_without; the levers that go with them follow); the line itself when
    no knob is stated or under `off` (None). The dropped levers and their reasons are kept for gated_off()."""
    from . import modes
    kn = knobs(environ)
    _ST.update(dropped={}, line=(line.name if line is not None else None))
    if not kn or line is None:
        return line
    named = levers_for(kn, line.levers)
    drop = tuple(lv for lv in named if lv in modes.LEVER_SWITCHES and lv not in modes.LEVER_NOT_SHEDDABLE)
    if not drop:
        return line
    out = modes.line_without(line, drop)
    gone = {x: named.get(x) or f"{REASON}:{word(kn)}(with:{','.join(d for d in drop if x in modes.LEVER_DEPENDENTS.get(d, ()))or 'line'})"
            for x in line.levers if x not in out.levers}
    _ST["dropped"] = gone
    return out


def dropped() -> dict:
    """{lever: reason} this process's resolution left out for the stated knobs ({} when none)."""
    return dict(_ST["dropped"])


def gated_off(line) -> dict:
    """{lever: "aside:stock_knob:<flag>=<value>"} for the levers left out of ``line`` for a stated knob; {} otherwise."""
    if line is None or not _ST["dropped"] or _ST["line"] != line.name:
        return {}
    return {n: w for n, w in _ST["dropped"].items() if n not in line.levers}


def export(environ=None) -> str | None:
    """Write the fact into ``environ`` (os.environ) for the kit process's hook and its rank processes; removes it when no knob is stated."""
    env = os.environ if environ is None else environ
    w = word()
    if w:
        env[ENV] = w
    else:
        env.pop(ENV, None)
    return w
