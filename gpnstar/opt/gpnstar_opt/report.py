"""Observability for gpnstar_opt: every line the package prints, in one place, composed with the core's line builders (opt_core.report:
prefix / line / lever_line / not_active_line / with_words / word) — all on stderr, flushed, prefixed `[gpnstar-opt]`. The grammar is a
contract (the RE_* patterns below hold every line to it).

  [gpnstar-opt] ACTIVE mode=exact levers=<a+b+…> gpu=<name>(smNN) card=<class|untested> mib=<MiB> model=<repo tail>@<rev8> gpn=<commit8> torch=<ver> transformers=<ver> tf32=on|off|unknown [<word> …]
                                                       printed in the model's process when the levers have engaged (the model on the GPU, before
                                                       its first forward). tf32 = the caller's fp32-matmul TF32 setting as read at that moment
                                                       (`gpn star … --tf32` -> on; unknown: torch answers through neither interface): the levers set
                                                       no numerics flag, they follow it. Trailing words
                                                       name UNCERTAINTY the levers engaged under — a card of no tested class `card=untested(<name>,<MiB>,<sm>)`,
                                                       a dependency off its pin `stack=drift(<name>:<have>!=<want>)`, a model outside the pinned table
                                                       `weights=unpinned(<name>)` or without digests to compare `weights=unchecked(<why>)`. NAMED, never a refusal
  [gpnstar-opt] LEVER name=<lever> state=on impl=gpnstar_opt.accel.patches origin=kit        one line per lever, after ACTIVE, in apply order
  [gpnstar-opt] KV route=dedup|unifiedkv|stock|none pairs=<n> rejects=<k> shape=<B>x<L>|all first_ms=<ms|none> reason=kvcheck|small_batch|memory|none compile=on|off
                                                       after the first forward of every new batch shape: the K/V projection route that shape runs, the
                                                       layer-0 reduced-vs-full K/V check's (pairs, rejects) so far, that first forward's wall time, why the
                                                       route (the check chose it | B·L below the de-dup floor | the reduced route did not fit device memory),
                                                       whether a torch.compile'd caller drove it; once more at the process's exit over every shape (shape=all)
  [gpnstar-opt] COMPILE levers=eager(kit-disabled) rest=compiled|eager     once, the first time a forward is entered from inside a torch.compile wrapper
  [gpnstar-opt] NOT ACTIVE: <reason>                   a refusal by name — no CUDA device, the installed gpn not the pinned stock, transformers off
                                                       upstream's pin, the weights off their digests, the model not on the GPU at engage, a target-row
                                                       contract violation, a mode word other than exact. Exit 3 under GPNSTAR_OPT=exact; `enable()` returns it
  [gpnstar-opt] DRY-RUN mode=exact levers=<…> gpu=<…> card=<…> mib=<…> model=none gpn=<…> torch=<…> transformers=<…> [<word> …] would_refuse=<reason|none>
  [gpnstar-opt] REMOVED arm=<n> hooks=<n> models=<k> levers=kept       `disable()`: the wrapper hooks withdrawn, the reporters off; applied levers stay
  [gpnstar-opt] ERROR: <message>                       a forward out of device memory on the stock projections too (the error then propagates)
"""
from __future__ import annotations

import re
from functools import partial
from typing import Mapping, Optional, Sequence

from opt_core import report as core
from opt_core.modes import levers_label

from ._names import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE, LEVERS, MODE, TAG   # noqa: F401

PREFIX = core.prefix(TAG)
line = partial(core.line, PREFIX)
IMPL = "gpnstar_opt.accel.patches"                                  # where every lever's implementation lives
KV_ROUTES = ("dedup", "unifiedkv", "stock")                          # the K/V route words (accel's own: dedup | p3b | stock projections)
KV_REASONS = ("kvcheck", "small_batch", "memory", "none")            # why a shape runs the route it runs: the layer-0 K/V check chose it | B·L below the de-dup floor (unified K/V by rule) | the reduced K/V path did not fit in device memory (stock projections by rule) | nothing to say (exit tally)
HEAD_KEYS = ("gpu", "card", "mib", "model", "gpn", "torch", "transformers")   # the ACTIVE / DRY-RUN fields after mode= levers=, in order (ACTIVE adds tf32=)

RE_ACTIVE = re.compile(r"^\[gpnstar-opt\] ACTIVE mode=(?P<mode>exact) levers=(?P<levers>\S+) gpu=(?P<gpu>\S+) card=(?P<card>\S+) mib=(?P<mib>\S+) model=(?P<model>\S+) gpn=(?P<gpn>\S+) "
                       r"torch=(?P<torch>\S+) transformers=(?P<transformers>\S+) tf32=(?P<tf32>on|off|unknown)(?P<words>( \S+=\S+)*)$")
RE_LEVER = re.compile(r"^\[gpnstar-opt\] LEVER name=(?P<name>\S+) state=(?P<state>on|off|skipped)(?: reason=(?P<reason>\S+))? impl=(?P<impl>\S+) origin=kit$")
RE_KV = re.compile(r"^\[gpnstar-opt\] KV route=(?P<route>dedup|unifiedkv|stock|none) pairs=(?P<pairs>\d+) rejects=(?P<rejects>\d+) shape=(?P<shape>\S+)"
                   r" first_ms=(?P<first_ms>\d+|none) reason=(?P<reason>kvcheck|small_batch|memory|none) compile=(?P<compile>on|off)$")
RE_NOT_ACTIVE = re.compile(r"^\[gpnstar-opt\] NOT ACTIVE: (?P<reason>.+)$")
RE_DRY_RUN = re.compile(r"^\[gpnstar-opt\] DRY-RUN mode=(?P<mode>exact) levers=(?P<levers>\S+) .*would_refuse=(?P<would_refuse>.+)$")
RE_COMPILE = re.compile(r"^\[gpnstar-opt\] COMPILE levers=eager\(kit-disabled\) rest=(?P<rest>compiled|eager)$")
RE_REMOVED = re.compile(r"^\[gpnstar-opt\] REMOVED arm=(?P<arm>\d+) hooks=(?P<hooks>\d+) models=(?P<models>\d+) levers=kept$")
COMPILE_REST = ("compiled", "eager")


def emit(text: str, stream=None) -> str:
    return core.emit(text, stream)


def token(x) -> str:
    """A blank-free token (blanks -> `_`): every value on a line is one token (a line is split on blanks)."""
    s = "none" if x is None else str(x).strip()
    return re.sub(r"\s+", "_", s) or "none"


def gpu_field(gpu: Optional[Mapping]) -> str:
    """`<name>(smNN)` from the probe dict, blank-free (the core's gpu_label over a tokenised name)."""
    if not gpu:
        return "none"
    return core.gpu_label({"name": token(gpu.get("name")), "sm": gpu.get("sm"), "cc": gpu.get("cc")})


def _head(rep: Mapping) -> list:
    gpu = rep.get("gpu") or {}
    return [("mode", MODE), ("levers", levers_label(rep.get("levers") or LEVERS)), ("gpu", gpu_field(gpu)),
            ("card", token(rep.get("card") or ("untested" if gpu else None))), ("mib", token(gpu.get("mib"))), ("model", token(rep.get("model_label"))), ("gpn", token(rep.get("gpn_commit8"))), ("torch", token(rep.get("torch"))),
            ("transformers", token(rep.get("transformers")))]


def tf32_field(v) -> str:
    return "unknown" if v is None else ("on" if v else "off")


def active_line(rep: Mapping) -> str:
    return core.with_words(line("ACTIVE", *_head(rep), ("tf32", tf32_field(rep.get("tf32")))), rep.get("words"))


def dry_run_line(rep: Mapping) -> str:
    wr = rep.get("would_refuse") or "none"
    return core.with_words(line("DRY-RUN", *_head(rep)), rep.get("words")) + f" would_refuse={wr}"


def compile_line(rest: str) -> str:
    """`[gpnstar-opt] COMPILE levers=eager(kit-disabled) rest=<compiled|eager>` -- printed once, the first time a forward of the patched
    model is entered from inside a torch.compile (dynamo) wrapper: the levers and the patched model run eager by design (the model's forward
    is excluded from tracing as one piece); rest = whether dynamo compiled any caller code around the model in this process."""
    return line("COMPILE", levers="eager(kit-disabled)", rest=rest if rest in COMPILE_REST else "eager")


def not_active_line(reason: str, head: Optional[str] = None) -> str:
    return core.not_active_line(TAG, reason, head)


def lever_lines(levers_on: Sequence[str] = LEVERS) -> list:
    """One LEVER line per lever of the kit (LEVERS order), state=on."""
    return [core.lever_line(TAG, name, "on", impl=IMPL, origin="kit") for name in LEVERS if name in levers_on]


def kv_line(route: Optional[str], pairs: int, rejects: int, shape: str = "all", first_ms=None, reason: Optional[str] = None, compile: Optional[bool] = None) -> str:
    """`[gpnstar-opt] KV route=<r> pairs=<n> rejects=<k> shape=<BxL|all> first_ms=<ms|none> reason=<kvcheck|small_batch|memory|none> compile=<on|off>`:
    first_ms = wall time of the FIRST forward of that shape (the layer-0 K/V check and the constant caches are built inside it; later
    forwards of the shape pay neither), compile = whether a dynamo-compiled caller drove the forward (the levers themselves run eager)."""
    ms = "none" if first_ms is None else str(int(round(float(first_ms))))
    return line("KV", route=route if route in KV_ROUTES else "none", pairs=int(pairs), rejects=int(rejects), shape=token(shape), first_ms=ms,
                reason=reason if reason in KV_REASONS else "none", compile="on" if compile else "off")


def removed_line(arm: int, hooks: int, models: int) -> str:
    return line("REMOVED", arm=int(arm), hooks=int(hooks), models=int(models), levers="kept")


def error_line(msg: str) -> str:
    return f"{PREFIX} ERROR: {msg}"
