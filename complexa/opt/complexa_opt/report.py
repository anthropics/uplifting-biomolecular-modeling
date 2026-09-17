"""The parent's lines of a run (stderr), composed from the core's grammar (``opt_core.report``) under this kit's tag. In print order, the
stock route (``--mode off``)::

    [complexa-opt] NOT ACTIVE: mode off (stock route) (mode=off)                                     no lever is active on the stock route
    [complexa-opt] WEIGHTS dir=<path> complexa.ckpt=<12 hex> complexa_ae.ckpt=<12 hex> (pinned, bytes only: sha256 not recomputed)   stock/check_pins.py weights_line
    [complexa-opt] ENV-CLEAN ok (<detail>)                                                            stock_design.env_clean_line
    [complexa-opt] NOTE <key>=<value>: <words>                                                        only when something named changes the run's accounting: upstream's results file /
                                                                                                      earlier designs already in <out>, a design count the package cannot read
                                                                                                      (designs_expected=unknown), a lever idle by its declared gate (levers_gated=…, kit route)
    [complexa-opt] INVOCATION route=stock name=<item|none> run=<run> cwd=<dir> t_launch=<unix s> t_exit=<unix s> wall_s=<s> rc=<rc> cmd=<shell-quoted command>
    [complexa-opt] OUTPUTS route=stock item=<item|none> designs_written=<w> designs_expected=<e|unknown> dir=<the run's root> root=<composed|newest>
    [complexa-opt] manifest: <out>/opt_manifest.json
    [complexa-opt] EXIT pid=<pid> mode=off route=stock rc=<rc> designs_written=<w> designs_expected=<e|unknown> wall=<s> [incomplete=<reason>] exit=<code>

and the kit route (``--mode exact|fast|big``): WEIGHTS, then ``HOOK ok python=<interpreter> pth=<site dir>/complexa_opt_autoload.pth
package=<the complexa_opt directory that interpreter resolves> finder=armed mode=<m>`` (the autoload hook proven live in the interpreter
upstream's console script runs), ``ENV-KIT ok (<n> variables removed:
<names>; exported: COMPLEXA_OPT=<m>,COMPLEXA_OPT_RECORD=<dir>; present: <names>)``, the generation process's own ``ACTIVE`` / ``LEVER`` /
``TALLY`` lines in its relayed log output (``activate.py``), INVOCATION and OUTPUTS with ``route=kit``, ``KIT-RECORD processes=<n> mode=<m>
state=<complete|gated|partial|absent> levers_on=<a,b,…> levers_skipped=<none|…> predict_steps=<n> forwards=<n> peak_alloc_gib=<x>
levers_gated=<none|…>`` (the activation records read back: every lever of the set in exactly one list), a ``NOTE levers_gated=<lever>:<reason>:
<words> (<counter>=<n> calls …)`` line per lever idle by its declared input gate (pair_assembly on padded batches — a binder_length range; exit
unaffected), the manifest line and ``EXIT … route=kit … [partial=<levers>] [gated=<lever:reason>] exit=<code>`` — exit 3 when no generation
process activated the mode or a lever of the set never engaged.

``check`` prints the WEIGHTS line and one DRY-RUN line instead: ``[complexa-opt] DRY-RUN mode=<m> route=<r> upstream=<version@commit8|none>
dirty=<n|none> complexa=<path|none> weights=<pinned-bytes|UNPINNED|none> gpu=<name(smNN)|none> target=<T|none> match=<yes|no|none> ok=<True|False>
[reason=<...>]``. Refusals are ONE named line each: ``NOT ACTIVE: <reason>`` (exit 3: a tier refused by name — the words name the escape,
``--mode off`` — or a deployment gate: weights, checkout, the env proof) or ``USAGE refused: <reason>`` (exit 2: an unknown option or mode,
an entry file that is not one item). Upstream's own arguments after ``--`` are never judged here. INVOCATION is printed once per child
process when it exits, by the one statement ``invocation_line``; ``cmd=`` is the exact argv, shell-quoted, so the line replays by hand.
"""
from __future__ import annotations

import functools
import os
import shlex
from typing import Optional, Sequence

from opt_core import report as core

from . import TAG

PREFIX = core.prefix(TAG)
line = functools.partial(core.line, PREFIX)
emit = core.emit
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = core.EXIT_OK, core.EXIT_FAIL, core.EXIT_USAGE, core.EXIT_NOT_ACTIVE

INVOCATION_FMT = PREFIX + " INVOCATION route=%s name=%s run=%s cwd=%s t_launch=%.3f t_exit=%.3f wall_s=%.3f rc=%d cmd=%s"
OUTPUTS_FMT = PREFIX + " OUTPUTS route=%s item=%s designs_written=%d designs_expected=%s dir=%s root=%s"


def _word(v) -> str:
    """A value as one token: None -> ``none``, blanks -> ``_``."""
    return "none" if v is None or str(v) == "" else str(v).replace(" ", "_")


def not_active(mode: str, route: str) -> str:
    """``[complexa-opt] NOT ACTIVE: mode <m> (<route> route) (mode=<m>)`` — the stock route activates nothing."""
    return core.not_active_line(TAG, f"mode {mode} ({route} route)", head=f"mode={mode}")


def refused(reason: str, mode: Optional[str] = None) -> str:
    """``[complexa-opt] NOT ACTIVE: <reason> [(mode=<m>)]`` — a tier refused by name, a failed deployment gate."""
    return core.not_active_line(TAG, reason, head=(f"mode={mode}" if mode else None))


def usage_refused(reason: str) -> str:
    """``[complexa-opt] USAGE refused: <reason>`` — an option or mode the command line does not take, named (exit 2)."""
    return f"{PREFIX} USAGE refused: {reason}"


def note_line(key: str, value, words: str) -> str:
    """``[complexa-opt] NOTE <key>=<value>: <words>`` — something found in <out> before the launch that changes what upstream will do; named, never refused."""
    return line("NOTE", (key, _word(value))) + f": {words}"


def shell(argv: Sequence[str]) -> str:
    """The argv as one shell-quoted string (``shlex.join``): what INVOCATION ``cmd=`` and the manifest's ``command_line`` carry."""
    return " ".join(shlex.quote(str(a)) for a in argv)


def invocation_line(route: str, name, run, cwd: str, t_launch: float, t_exit: float, rc: int, argv: Sequence[str]) -> str:
    return INVOCATION_FMT % (route, _word(name), _word(run), cwd, t_launch, t_exit, t_exit - t_launch, int(rc), shell(argv))


def expected_word(expected) -> str:
    """The designs asked as one token: the count, or ``unknown`` when the caller's nsamples / nrepeat token is not an integer literal the
    package can count with (``settings.values_of`` counted=False; a NOTE line names the token before the launch)."""
    return "unknown" if expected is None else _word(expected)


def outputs_line(route: str, item, written: int, expected, root: str, root_source: str = "composed") -> str:
    return OUTPUTS_FMT % (route, _word(item), int(written), expected_word(expected), root, root_source)


def exit_line(*, mode: str, route: str, rc: int, written: int, expected, wall: float, incomplete: Optional[str], exit_code: int, partial: Sequence[str] = (),
              gated: Sequence[str] = ()) -> str:
    """``… [incomplete=<reason>] [partial=<levers>] [gated=<lever:reason,…>] exit=<code>`` — each bracketed field only when it has something to
    name: ``incomplete`` / ``partial`` price the exit (1 / 3), ``gated`` names a lever idle by its declared input gate (exit unaffected)."""
    pairs = [("pid", os.getpid()), ("mode", mode), ("route", route), ("rc", rc), ("designs_written", written), ("designs_expected", expected_word(expected)), ("wall", f"{wall:.1f}")]
    if incomplete:
        pairs.append(("incomplete", _word(incomplete)))
    if partial:
        pairs.append(("partial", ",".join(partial)))
    if gated:
        pairs.append(("gated", ",".join(gated)))
    pairs.append(("exit", exit_code))
    return line("EXIT", *pairs)


def dry_run_line(**fields) -> str:
    return line("DRY-RUN", *fields.items())


def hook_line(probe: dict, mode: str) -> str:
    """``[complexa-opt] HOOK ok python=<interpreter> pth=<path> finder=armed mode=<m>`` — the autoload hook is live in upstream's interpreter."""
    return line("HOOK ok", ("python", probe.get("python")), ("pth", probe.get("pth") or "none"), ("package", probe.get("package") or "none"), ("finder", "armed" if probe.get("finder_armed") else "absent"), ("mode", mode))


def env_kit_line(removed: dict, exported: dict, present) -> str:
    """``[complexa-opt] ENV-KIT ok (<n> variables removed: <names|none>; exported: K=V,…; present: <names|none>)`` — the kit child's environment, listed."""
    names = removed["variables"] + [f"PYTHONPATH:{e}" for e in removed["pythonpath"]]
    return (f"{PREFIX} ENV-KIT ok ({len(names)} variables removed: {','.join(names) or 'none'}; exported: {','.join(f'{k}={v}' for k, v in exported.items())}; "
            f"present: {','.join(present or []) or 'none'})")


def kit_record_line(summary: dict) -> str:
    """``[complexa-opt] KIT-RECORD processes=<n> mode=<m> state=<complete|gated|partial|absent> levers_on=<…> levers_skipped=<…> predict_steps=<n> forwards=<n> peak_alloc_gib=<x> levers_gated=<…|none>``
    — every lever of the mode's set in exactly one of the three lists (``kit_design.read_records``)."""
    return line("KIT-RECORD", ("processes", summary["processes"]), ("mode", _word(summary.get("mode"))), ("state", summary["state"]),
                ("levers_on", ",".join(summary["levers_on"]) or "none"), ("levers_skipped", ",".join(summary["levers_skipped"]) or "none"),
                ("predict_steps", summary.get("predict_steps", 0)), ("forwards", summary.get("forwards", 0)), ("peak_alloc_gib", _word(summary.get("peak_alloc_gib"))),
                ("levers_gated", ",".join(summary.get("levers_gated") or []) or "none"))
