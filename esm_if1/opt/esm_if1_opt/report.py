"""The banner and verdict lines of the ``design`` / ``check`` command line — everything the parent process prints around a pass (the
driver's own clock lines are ``lines.py``; the stock child's proof line is ``stock_design.env_clean_line``). One grammar, stderr, flushed::

    [esm_if1-opt] NOT ACTIVE: mode off (stock route) (mode=off)          the banner of --mode off: upstream's script, nothing of the kit applies
    [esm_if1-opt] ACTIVE mode=fast route=kit levers_requested=batched_sampling levers_applied=batched_sampling batch_size=<B> seed=<N> det=<0|1> multichain=<0|1> nogpu=<0|1>
    [esm_if1-opt] LEVER name=batched_sampling state=<on|off> [reason=no_batch_ran] impl=esm_if1_opt.batched origin=kit rows_per_forward=<B> batches=<n> rows=<n>   after the kit child, from its ITEM records: on = at least one batched forward ran (batches, rows = what ran); off = the child ended before its first forward
    [esm_if1-opt] NOT ACTIVE: <reason> (mode=<m>)                       the mode cannot run here (exit 3)
    [esm_if1-opt] usage: <what is wrong>                                 a usage error: an unknown mode word, a flag or value the verb refuses (exit 2)
    [esm_if1-opt] NOT APPLIED route=<stock|kit> <flag>=<value> … (<why>)  flags the mode does not consume, named: --seed / --batch_size under off (upstream is unseeded, one sequence per call), --det 1 under either mode
    [esm_if1-opt] WEIGHTS file=<name> torch_home=<path> hub_entry=<linked|kept|present|absent> source=<ESM_IF1_WEIGHTS=<path>[(not a file)]|torch.hub>
    [esm_if1-opt] INVOCATION route=<stock|kit> name=<pass|stem> argv0=<sample_sequences.py|esm_if1_opt.kit_design> launch_ts=<unix s> exit_ts=<unix s> wall_s=<s> rc=<int> cache=<hit|miss|na>
    [esm_if1-opt] EXIT pid=<pid> mode=<m> route=<stock|kit> rc=<rc> inputs=<n> n_fasta=<complete files> designs_written=<records> designs_expected=<records> wall=<s> [incomplete=<k/n>] exit=<code>
    [esm_if1-opt] CHECK <key>=<value> …                                   check: what the tree carries / what the box has
    [esm_if1-opt] DRY-RUN mode=<m> route=<stock|kit> levers=<none|batched_sampling> …  ok=<bool>   check: whether the mode can run here
"""
from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence

from opt_core import report as _core

from . import TAG

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_NOT_ACTIVE = _core.EXIT_OK, _core.EXIT_FAIL, _core.EXIT_USAGE, _core.EXIT_NOT_ACTIVE   # 0 1 2 3: the core's table
PREFIX = _core.prefix(TAG)                                   # "[esm_if1-opt]"
emit = _core.emit
STOCK_CHILD = "esm_if1_opt.stock_design"                      # the proven stock child of --mode off (python -s -m <this> … -- <script> <args>)
KIT_CHILD = "esm_if1_opt.kit_design"                          # the kit child of --mode fast (python -m <this> <driver args>)
LEVER_IMPL = "esm_if1_opt.batched"

INVOCATION_RE = (r"^\[esm_if1-opt\] INVOCATION route=(?P<route>stock|kit) name=(?P<name>\S+) argv0=(?P<argv0>\S+) launch_ts=(?P<launch_ts>\d+\.\d{6}) "
                 r"exit_ts=(?P<exit_ts>\d+\.\d{6}) wall_s=(?P<wall_s>\d+\.\d{3}) rc=(?P<rc>-?\d+) cache=(?P<cache>hit|miss|na)$")
NOT_APPLIED_RE = r"^\[esm_if1-opt\] NOT APPLIED route=(?P<route>stock|kit)(?: (?:seed|batch_size|det)=\S+)+ \(.*\)$"
ACTIVE_RE = (r"^\[esm_if1-opt\] ACTIVE mode=fast route=kit levers_requested=batched_sampling levers_applied=(?P<applied>batched_sampling) "
             r"batch_size=(?P<batch_size>\d+) seed=(?P<seed>-?\d+) det=(?P<det>[01]) multichain=(?P<multichain>[01]) nogpu=(?P<nogpu>[01])$")


def line(kind: str, *pairs, **fields) -> str:
    return _core.line(PREFIX, kind, *pairs, **fields)


def not_active_stock() -> str:
    """The banner of ``--mode off``."""
    return _core.not_active_line(TAG, "mode off (stock route)", head="mode=off")


def active_line(rep: Mapping, *, batch_size: int, seed: int, det: bool, multichain: bool, nogpu: bool) -> str:
    """The banner of ``--mode fast`` from its activation report (``stack.enable``): the lever requested and applied, the pass's inputs and switches."""
    pairs = [("mode", rep["mode"]), ("route", rep["route"]), ("levers_requested", _core.join(rep.get("levers"))), ("levers_applied", _core.join(rep.get("levers_applied"))),
             ("batch_size", int(batch_size)), ("seed", int(seed)), ("det", int(bool(det))), ("multichain", int(bool(multichain))), ("nogpu", int(bool(nogpu)))]
    return line("ACTIVE", *pairs)


NO_BATCH_RAN = "no_batch_ran"                                # the LEVER line's reason when the kit child ended before its first batched forward


def lever_lines(rep: Mapping, *, batch_size: int, records: Sequence[Mapping]) -> list:
    """One LEVER line per lever of the mode (``opt_core.report.lever_line``), composed AFTER the kit child from its evidence records
    (``manifest.read_records``): ``state=on`` with the batches and rows its ITEM records show when at least one batched forward ran,
    ``state=off reason=no_batch_ran`` (batches=0 rows=0) when the child ended before its first forward — what ran, never the plan."""
    items = [r for r in records if r.get("kind") == "ITEM"]
    ran = {"rows_per_forward": int(batch_size), "batches": len(items), "rows": sum(int(r.get("n_seq") or 0) for r in items)}
    return [_core.lever_line(TAG, name, "on", impl=LEVER_IMPL, origin="kit", **ran) if items else
            _core.lever_line(TAG, name, "off", reason=NO_BATCH_RAN, impl=LEVER_IMPL, origin="kit", **ran)
            for name in rep.get("levers_applied") or []]


def refused(reason: str, mode: Optional[str] = None) -> str:
    """``NOT ACTIVE: <reason> (mode=<m>)`` — the mode cannot run on this box."""
    return _core.not_active_line(TAG, reason, head=(f"mode={mode}" if mode else None))


def not_applied(route: str, pairs, why: str) -> str:
    """``NOT APPLIED route=<r> <flag>=<value> … (<why>)`` — flags given that the mode does not consume, named once."""
    return f"{PREFIX} NOT APPLIED route={route} " + " ".join(f"{k}={v}" for k, v in pairs) + f" ({why})"


def weights_line(w: Mapping) -> str:
    return line("WEIGHTS", file=w["file"], torch_home=w["torch_home"], hub_entry=w["hub_entry"], source=w["source"])


def gpu_word(gpu) -> str:
    return _core.gpu_label(gpu).replace(" ", "_")


def invocation_line(route: str, name: str, argv0: str, launch_ts: float, exit_ts: float, rc: int, cache: str) -> str:
    """One child's line (every value one token; ``name`` = ``pass`` for the kit child, the structure's stem for a stock child). ``cache``: ``hit`` = after the child the torch.hub cache entry still resolves to
    ``$ESM_IF1_WEIGHTS`` (nothing was downloaded over it) · ``miss`` = it does not · ``na`` = no ``ESM_IF1_WEIGHTS`` file was given."""
    if cache not in ("hit", "miss", "na"):
        raise ValueError(f"cache word {cache!r} is not hit|miss|na")
    return (f"{PREFIX} INVOCATION route={route} name={str(name).replace(' ', '_')} argv0={argv0} "
            f"launch_ts={launch_ts:.6f} exit_ts={exit_ts:.6f} wall_s={exit_ts - launch_ts:.3f} rc={int(rc)} cache={cache}")


def exit_line(*, mode: str, route: str, rc: int, inputs: int, n_fasta: int, designs_written: int, designs_expected: int, wall: float,
              incomplete: Optional[str], exit_code: int) -> str:
    pairs = [("pid", os.getpid()), ("mode", mode), ("route", route), ("rc", int(rc)), ("inputs", int(inputs)), ("n_fasta", int(n_fasta)),
             ("designs_written", int(designs_written)), ("designs_expected", int(designs_expected)), ("wall", f"{wall:.1f}")]
    if incomplete:
        pairs.append(("incomplete", incomplete))
    pairs.append(("exit", int(exit_code)))
    return line("EXIT", *pairs)


def dry_run_line(**fields) -> str:
    return line("DRY-RUN", *fields.items())


def verdict(rc: int, rep: Optional[Mapping], incomplete: Optional[str]) -> dict:
    """The core's exit rule (``opt_core.report.verdict``; the one lever applies on every ``fast`` pass, so there is no partial state and
    ``allow_partial`` is False): a child's non-zero status stands · status 0 with incomplete outputs -> 1 · else 0."""
    return _core.verdict(EXIT_OK if int(rc) == 0 else EXIT_FAILED, rep, False, incomplete)
