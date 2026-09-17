"""``python -m boltz2_opt ...`` / ``boltz2-opt ...`` -> :func:`boltz2_opt.cli.main`, behind the kit's core pin gate (``_core_gate.gate``: the importable
``opt_core`` must be the pinned one — absent / mismatched → one NOT ACTIVE line, exit 3) and the package's module-granular core check
(``_autoload.require_core``), both before anything of the core is imported."""
import sys


def main(argv=None) -> int:
    from ._core_gate import gate
    from ._autoload import TAG, require_core
    gate(__file__, TAG)
    require_core()
    from .cli import main as cli_main
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
