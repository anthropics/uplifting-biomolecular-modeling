"""e1-opt {score,check,warm} — the command line over the two modes.

* ``score``  — ONE assay, upstream's own flags: ``--parent-path P --mutants-path M --output-path O`` (+ any other ``E1.tools.score``
               option, passed through verbatim: ``--max-batch-tokens``, ``--scoring-method``, ``--context-path``, ``--context-reduction`` …).
               ``--variant 150m|300m|600m`` (or upstream's ``--model-name Profluent-Bio/E1-<size>``: the pinned checkpoints only) names the
               model; ``--model-name`` on the tool's line is always the pinned snapshot directory. ``--mode off`` runs the upstream CLI through
               the stock runner (stock_score.py: every kit variable stripped and proved absent by the subprocess, no kit directory on
               sys.path); ``--mode exact`` (the default) runs the same upstream CLI in a subprocess with the kit armed (kit_score.py: the kit
               applies its lever set at the tool's model construction and prints its KIT / LEVER lines; the KERNELS proof follows). ``--det 1``
               applies the deterministic recipe (det.py) in both modes. The output is the tool's own file at ``--output-path``; the kit writes
               nothing beside it. The last line of every run is ``[e1-opt] EXIT mode=<m> variant=<v> complete=<0|1> kernels_fallback=<none|names>
               rc=<code>``. A lever of the mode's set that cannot run in the tool's process is the mode's refusal by name there (``NOT ACTIVE:
               …``) and the run's exit 3 — never the stock under the kit's name. ``--mode`` / ``--variant`` disagreeing with E1_OPT / E1_VARIANT
               is a usage refusal (exit 2); ``fast`` / ``big``, a stock off its pin, missing weights, no GPU are refused by name (exit 3).
* ``check``  — resolve + gate the mode WITHOUT applying: prints the DRY-RUN line (what enable() would do and why).
* ``warm``   — score the shipped fixture under the mode once (the gates run as in ``score``) and report cache-file counts and wall time (warm.py).

Console entry point ``e1-opt`` (pyproject) and ``python -m e1_opt``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

from . import ActivationError, __version__, accel, det, kit_score, modes, outputs, registry, report, stack, stock_score

from ._names import EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE   # the one exit-code table

USAGE = f"""usage: e1-opt <command> [options]      (e1_opt {__version__})

  score   [--mode off|exact] [--variant V | --model-name Profluent-Bio/E1-V] [--det 0|1]
          --parent-path P --mutants-path M --output-path O   [any other E1.tools.score option, passed through verbatim]
  check   [--variant V] [--mode off|exact]
  warm    [--variant V] [--mode off|exact] [--det 0|1]

modes: {' | '.join(modes.MODES)} (default {modes.DEFAULT_MODE})   variants: 150m | 300m | 600m (E1_VARIANT or --variant)
this host outside the kit's test record (a dependency off its pin, an untested GPU) is named on the ACTIVE line (notes=), never refused; the mode is all of its levers: one that cannot run refuses the run by name
exit codes: 0 ok, 1 failed (no scores.csv), 2 usage (incl. a --mode / --variant that disagrees with E1_OPT / E1_VARIANT), 3 not active (fast / big; E1 not the pinned stock; kit / weights / GPU missing; a lever cannot run)
"""


class CliError(Exception):
    def __init__(self, msg: str, code: int = EXIT_USAGE):
        super().__init__(msg)
        self.code = code


class NotActive(CliError):
    """A refusal by name printed as the NOT ACTIVE line (exit 3)."""

    def __init__(self, msg: str):
        super().__init__(msg, EXIT_NOT_ACTIVE)


class Disagreement(CliError):
    """Two mode / variant sources disagree (the argument vs the environment): a usage refusal (exit 2) printed as the NOT ACTIVE line."""


def _parser(cmd: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"e1-opt {cmd}", add_help=True)
    ap.add_argument("--variant", default=None, help="150m | 300m | 600m (default: E1_VARIANT)")
    ap.add_argument("--mode", default=None, help=f"{' | '.join(modes.MODES)} (default: E1_OPT, else {modes.DEFAULT_MODE})")
    if cmd in ("score", "warm"):
        ap.add_argument("--det", default=None, help="0 | 1: the deterministic recipe (default: E1_OPT_DET, else 0)")
    if cmd == "score":
        ap.add_argument("--model-name", default=None, help="upstream's flag: Profluent-Bio/E1-150m | E1-300m | E1-600m — the pinned checkpoints (the same choice as --variant)")
        ap.add_argument("--parent-path", default=None, help="upstream's flag: the wild-type FASTA (one record)")
        ap.add_argument("--mutants-path", default=None, help="upstream's flag: the variants FASTA (one record per variant, the parent's length)")
        ap.add_argument("--output-path", default=None, help="upstream's flag: the scores.csv to write")
    return ap


def _resolve_det(value) -> int:
    v = value if value is not None else os.environ.get(stack.ENV_DET)
    try:
        lv = det.check_level(v if v is not None else 0)
    except ValueError as e:
        raise CliError(str(e))
    os.environ[stack.ENV_DET] = str(lv)                  # the one switch every module reads (stack.det_level)
    return lv


def _model_name_variant(a) -> None:
    """upstream's --model-name names one of the three pinned checkpoints (`Profluent-Bio/E1-<size>`, or the bare size): the same choice as
    --variant, folded into it; anything else is refused by name (usage, exit 2); both given and disagreeing is a usage refusal too."""
    mn = getattr(a, "model_name", None)
    if not mn:
        return
    pins = stack.load_pins()
    by_repo = {str(w.get("repo", "")).lower(): size for size, w in pins.WEIGHTS.items()}
    key = str(mn).strip().rstrip("/").lower()
    size = by_repo.get(key) or (key if key in pins.WEIGHTS else None) or by_repo.get("profluent-bio/" + key)
    if size is None:
        raise CliError(f"--model-name {mn!r}: the kit serves the pinned checkpoints only — {', '.join(sorted(w.get('repo') for w in pins.WEIGHTS.values()))} (or --variant {'|'.join(pins.WEIGHTS)})")
    v = getattr(a, "variant", None)
    if v and str(v).strip().lower() != size:
        raise Disagreement(f"--model-name {mn} disagrees with --variant {v}; one model per process — drop one of them", EXIT_USAGE)
    a.variant = size


def _mode_arg(a) -> Optional[str]:
    """The effective mode: --mode, else E1_OPT, else the table's default; --mode / --variant disagreeing with E1_OPT / E1_VARIANT = a usage refusal (exit 2)."""
    v = getattr(a, "variant", None)
    env_v = (os.environ.get(stack.ENV_VARIANT) or "").strip().lower() or None
    if v and env_v and env_v != str(v).strip().lower():
        raise Disagreement(f"variant {v} disagrees with {stack.ENV_VARIANT}={env_v}; one variant per process — drop one of them", EXIT_USAGE)
    m = getattr(a, "mode", None)
    if m is None:
        return (os.environ.get(stack.ENV) or "").strip().lower() or modes.DEFAULT_MODE
    try:
        m = modes.check_mode(m)
    except ValueError as e:
        raise NotActive(str(e)) if str(m).strip().lower() in modes.NOT_SHIPPED else CliError(str(e))
    dis = stack._mode_vs_env(m)
    if dis:
        raise Disagreement(dis + "; a run has one mode — drop one of them", EXIT_USAGE)
    return m


# ---------------------------------------------------------------------------------------------------------------------- score
def cmd_score(argv: List[str]) -> int:
    a, passthrough = _parser("score").parse_known_args(argv)                 # every option this parser does not know is upstream's: handed to the tool verbatim
    t_start = time.perf_counter()
    _model_name_variant(a)
    mode = _mode_arg(a)
    lv = _resolve_det(a.det)
    missing = [k for k, x in (("--parent-path", a.parent_path), ("--mutants-path", a.mutants_path), ("--output-path", a.output_path)) if not x]
    if missing:
        raise CliError(f"score takes upstream's --parent-path, --mutants-path and --output-path (missing: {', '.join(missing)})")
    scores = os.path.abspath(a.output_path)
    off = mode == "off"
    rc_all, variant, kernels_line, complete = EXIT_OK, a.variant, None, False
    try:
        rep = stack.activate(mode, a.variant, route="score")
        if not rep.get("active") and rep.get("reason") != report.OFF_REASON:
            rc_all = EXIT_NOT_ACTIVE
            return rc_all
        pins = stack.load_pins()
        v = variant = rep["variant"]
        tool_args = ["--model-name", registry.snapshot_dir(v, pins), "--parent-path", os.path.abspath(a.parent_path), "--mutants-path", os.path.abspath(a.mutants_path),
                     "--output-path", scores, *passthrough]                   # tools/score.py: the pinned snapshot directory, the two FASTA paths, the output path, then upstream's own options as given
        env, unset = stock_score.clean_env(os.environ)                       # both children start from the same environment: every kit variable stripped (the kit's child is armed by its arguments)
        env.update(det.env_for(pins, lv))
        os.makedirs(os.path.dirname(scores), exist_ok=True)
        report.emit(report.ready_line(v, time.perf_counter() - t_start))
        if unset:
            report.emit(report.stripped_line(unset))                         # never silent: every variable removed from the tool's environment, by name
        py, pins_path = sys.executable, stack.pins_path()
        cmd = stock_score.command(py, v, lv, pins_path, tool_args) if off else kit_score.command(py, tool_args, det=lv, variant=v, pins_path=pins_path)
        rc, wall, lines = kit_score.relay(cmd, env, cwd=os.path.dirname(scores))
        lst = outputs.listing(scores)
        kernels_line = next((s for s in lines if report.RE_KERNELS.match(s)), None)
        complete = rc == 0 and lst["present"]
        report.emit(report.score_line(os.path.basename(scores), wall, stock=off))
        rc_all = EXIT_OK if complete else EXIT_FAIL
        why = kit_score.refusal(lines) if not off else None                  # the kit refused BY NAME in the tool's process (its KitRefused / PinDrift line: a lever of the set cannot run there)
        child_said = next((report.RE_NOT_ACTIVE.match(x).group("reason") for x in lines if report.RE_NOT_ACTIVE.match(x)), None)   # or the child's own NOT ACTIVE line (the apply's or the KERNELS proof's refusal by name, rc 3)
        if not off and why is None and child_said is None and rc == 0 and kernels_line is None:
            why = "no KERNELS line from the completed process (the kit's hook did not fire: the tool built no model)"
        if why is not None or child_said is not None or (rc == EXIT_NOT_ACTIVE and not off):
            if child_said is None:                                            # the child's own NOT ACTIVE line was relayed verbatim above; otherwise the run words the kit's refusal (its KitRefused line) itself
                report.emit(report.not_active_line(why or f"the tool's process exited {rc}"))
            rc_all = EXIT_NOT_ACTIVE
        return rc_all
    finally:
        report.emit(report.exit_line(mode, variant, complete=complete, kernels_fallback=_fell_back([kernels_line]), rc=rc_all))


def _fell_back(kernel_lines) -> List[str]:
    """The accelerators whose word on the KERNELS line is not `engaged:…` — what ran fallen back or absent on this host, by name, in
    accel.ACCELERATORS order ([] = every accelerator engaged; a missing line contributes nothing). The EXIT line's `kernels_fallback=`."""
    seen = set()
    for s in kernel_lines:
        m = report.RE_KERNELS.match(s) if s else None
        if m:
            seen.update(n for n in accel.ACCELERATORS if not m.group(n).startswith("engaged:"))
    return [n for n in accel.ACCELERATORS if n in seen]


# ---------------------------------------------------------------------------------------------------------------------- check
def cmd_check(argv: List[str]) -> int:
    a = _parser("check").parse_args(argv)
    mode = _mode_arg(a)
    rep = stack.activate(mode, a.variant, dry_run=True, route="check")
    return EXIT_OK if rep.get("would_refuse") is None else EXIT_NOT_ACTIVE   # a dry run that would refuse exits 3, as the activation it stands for would


# ----------------------------------------------------------------------------------------------------------------------- warm
def cmd_warm(argv: List[str]) -> int:
    from . import warm
    a = _parser("warm").parse_args(argv)
    mode = _mode_arg(a)
    lv = _resolve_det(a.det)
    res = warm.run(mode, a.variant, det=lv)
    if res.get("exit_code") == EXIT_NOT_ACTIVE:
        return EXIT_NOT_ACTIVE
    return EXIT_OK if res.get("status") == "PASS" else EXIT_FAIL


# ---------------------------------------------------------------------------------------------------------------------- entry
COMMANDS = {"score": cmd_score, "check": cmd_check, "warm": cmd_warm}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE, end="")
        return EXIT_OK if argv else EXIT_USAGE
    cmd = argv[0]
    fn = COMMANDS.get(cmd)
    if fn is None:
        print(report.error_line(f"unknown command {cmd!r}") + "\n" + USAGE, end="", file=sys.stderr)
        return EXIT_USAGE
    try:
        return fn(argv[1:])
    except (Disagreement, NotActive) as e:
        report.emit(report.not_active_line(str(e)))
        return e.code
    except CliError as e:
        report.emit(report.error_line(str(e)), stream=sys.stderr)
        return e.code
    except ActivationError:
        return EXIT_NOT_ACTIVE


if __name__ == "__main__":
    sys.exit(main())
