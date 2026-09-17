"""The command-line skeleton of a kit: verbs, the shared arguments, one ``main``.

Contract. A kit registers its verbs (:class:`Verb`: name, help, an ``add_arguments(parser)`` and a ``run(args) -> int``) — typically
``pred``, ``check``, ``warm`` and ``serve`` where a served line exists; the skeleton dispatches and owns one
discipline only: a usage error found while running (:class:`CliError`, plus the kit's own ``usage_errors`` types) prints
``usage: <prog> <verb> ...`` and ``[tag] <verb>: <message>`` and returns ``EXIT_USAGE``. Every other exception propagates with its
traceback (a run log must show the stack; the launcher's error markers match it) — the skeleton never converts one into a line.
:func:`add_mode_arguments` adds the shared switches: ``--mode`` (the kit's table; default = the environment variable, then the table's
default — the one precedence, modes.mode_argument), ``--variant`` when the kit has variants, ``--allow-partial``, ``--det``.
The item-loop helpers are pure: :func:`input_paths` (the ``--json_path`` / ``--input_dir`` inputs of a pass), :func:`item_name` (the file
stem that names an item's output dir, its line and its manifest key), :func:`read_report` (a step's JSON report or None),
:func:`step_reason` (the named reason a step record failed).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from . import report
from .modes import ModeError, ModeTable, mode_argument


class CliError(Exception):
    """A usage error found while running a verb (wrong combination of arguments, a missing input): EXIT_USAGE."""


@dataclass(frozen=True)
class Verb:
    name: str
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], int]


def add_mode_arguments(p: argparse.ArgumentParser, table: ModeTable, *, env: str, variant_env: Optional[str] = None,
                       variants: Sequence[str] = (), det_levels: Sequence[int] = (), environ=None) -> None:
    """``--mode`` (choices = the table; default from ``env`` then the table), ``--variant`` (when ``variants``), ``--allow-partial``,
    ``--det`` (when ``det_levels``). An unknown ``--mode`` value is an argparse usage error (exit 2)."""
    environ = os.environ if environ is None else environ

    def mode_type(value):
        try:
            return table.check(value)
        except ModeError as e:
            raise argparse.ArgumentTypeError(str(e))

    p.add_argument("--mode", type=mode_type, default=None, metavar="|".join(table.modes),
                   help=f"one of {', '.join(table.modes)} (default: ${env} when set, else {table.default})")
    if variants:
        p.add_argument("--variant", choices=list(variants), default=(environ.get(variant_env) or None) if variant_env else None,
                       help=f"one of {', '.join(variants)}" + (f" (default: ${variant_env})" if variant_env else ""))
    p.add_argument("--allow-partial", action="store_true", help="a partial activation is recorded and the run proceeds with its own exit code")
    if det_levels:
        p.add_argument("--det", type=int, choices=list(det_levels), default=0, help="the deterministic recipe level (0 = production numerics)")
    p.set_defaults(_mode_table=table, _mode_env=env)


def resolve_mode(args: argparse.Namespace, environ=None) -> str:
    """The mode of a parsed command line (modes.mode_argument over ``--mode`` and the kit's variable)."""
    environ = os.environ if environ is None else environ
    table: ModeTable = args._mode_table
    return mode_argument(getattr(args, "mode", None), environ.get(args._mode_env), table)


def build_parser(prog: str, description: str, verbs: Sequence[Verb]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=description)
    sub = p.add_subparsers(dest="verb", metavar="|".join(v.name for v in verbs))
    sub.required = True
    for v in verbs:
        sp = sub.add_parser(v.name, help=v.help, description=v.help)
        v.add_arguments(sp)
        sp.set_defaults(_run=v.run)
    return p


def main(argv: Optional[Sequence[str]], *, prog: str, description: str, verbs: Sequence[Verb], tag: str,
         usage_errors: Sequence[type] = (), on_usage_error: Optional[Callable[[argparse.Namespace, BaseException], int]] = None,
         stream=None) -> int:
    """Parse and dispatch (module contract). Returns the verb's exit code; a usage error found while running (:class:`CliError` or one
    of the kit's ``usage_errors`` types) prints ``usage: <prog> <verb> ...`` + ``[tag] <verb>: <message>`` and returns EXIT_USAGE — or,
    when the kit's usage bytes differ, ``on_usage_error(args, exc)`` prints the kit's own text and returns its code; anything else
    propagates. (An unknown ``--mode`` at parse time is argparse's own usage error — exit 2 with argparse's framing around the table's
    ``unknown_message``; a kit whose parse-time line is held byte-for-byte keeps ``--mode`` a plain string and calls ``mode_argument``
    in its verb, so the error surfaces here.)"""
    stream = stream or sys.stderr
    parser = build_parser(prog, description, verbs)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))      # argparse exits 2 on a usage error, 0 on --help
    try:
        return int(args._run(args))
    except (CliError, *tuple(usage_errors)) as e:
        if on_usage_error is not None:
            return int(on_usage_error(args, e))
        print(f"usage: {prog} {args.verb} ...", file=stream)
        print(f"{report.prefix(tag)} {args.verb}: {e}", file=stream, flush=True)
        return report.EXIT_USAGE


# ----------------------------------------------------------------------------------------------------------------- the item loop
def input_paths(args=None, *, json_path=None, input_dir=None, pattern: str = "*.json") -> list:
    """The input files of a pass: every ``json_path`` entry (a list) plus every ``pattern`` match under ``input_dir``, sorted within the
    directory; ``args`` (an argparse namespace with ``json_path`` / ``input_dir``) is the shorthand a kit's ``pred`` verb passes."""
    if args is not None:
        json_path = getattr(args, "json_path", None) if json_path is None else json_path
        input_dir = getattr(args, "input_dir", None) if input_dir is None else input_dir
    paths = list(json_path or [])
    if input_dir:
        paths += sorted(glob.glob(os.path.join(input_dir, pattern)))
    return paths


def item_name(path: str) -> str:
    """The item name of an input file: its stem (the output dir, the ITEM line, the manifest key)."""
    return os.path.splitext(os.path.basename(path))[0]


def read_report(path: str):
    """A step's JSON report as a dict, or None when absent or unreadable (the caller names the consequence)."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def step_reason(step) -> str:
    """The named reason a step record (:func:`opt_core.process.run_step`) failed: its ``error``, else ``step rc=<rc>``."""
    return step.get("error") or f"step rc={step.get('rc')}"
