"""Unified AtlasFold command-line interface."""

import argparse
import sys
from collections.abc import Sequence


def create_parser() -> argparse.ArgumentParser:
    """Create the top-level parser used to select an inference pipeline."""
    parser = argparse.ArgumentParser(description="Run AtlasFold inference.")
    commands = parser.add_subparsers(
        dest="model",
        metavar="{monomer,multimer}",
        required=True,
    )
    commands.add_parser(
        "monomer",
        add_help=False,
        help="Run AtlasFold monomer inference.",
    )
    commands.add_parser(
        "multimer",
        add_help=False,
        help="Run AtlasFold-Multimer inference.",
    )
    return parser


def _parse_command(argv: Sequence[str] | None) -> tuple[str, list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help"}:
        create_parser().parse_args(values)

    command = values[0]
    if command not in {"monomer", "multimer"}:
        create_parser().parse_args([command])

    return command, values[1:]


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected AtlasFold inference pipeline."""
    command, command_argv = _parse_command(argv)
    prog = f"{create_parser().prog} {command}"

    if command == "monomer":
        from atlasfold.cli import monomer

        parser = monomer.create_parser(prog=prog)
        args = parser.parse_args(command_argv)
        monomer.run(args)
        return

    elif command == "multimer":
        from atlasfold.cli import multimer

        parser = multimer.create_parser(prog=prog)
        args = parser.parse_args(command_argv)
        multimer.run(args)
        return

    else:
        raise ValueError(f"Unsupported model: {command}")


if __name__ == "__main__":
    main()
