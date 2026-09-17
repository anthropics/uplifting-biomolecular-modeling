"""``python -m af2ig_opt ...`` / ``af2ig-opt ...`` (and ``run.sh``, which execs the former) -> :func:`af2ig_opt.cli.main` behind two guards, in this
order, before anything of the command line resolves: (1) the core pin — ``_core_gate.gate`` (the tree's kit_template copy, standard library only)
locates ``opt_core`` without importing it and compares the kit's ``[tool.opt_core]`` pin with the located core's MANIFEST: an absent, older, newer or
edited core is ``[af2ig-opt] NOT ACTIVE: reason=core_missing:opt_core …`` / ``reason=core_mismatch: …`` / ``reason=core_pin_unreadable: …``, exit 3;
(2) the import of the command line itself, which imports the core's sub-modules: a pinned core that lacks one is named
(``[af2ig-opt] NOT ACTIVE: core_missing:<module> …``, exit 3) — never a traceback, never a stock run in its place."""
import sys

from . import TAG

EXIT_INACTIVE = 3                                   # == cli.EXIT_INACTIVE == _core_gate.EXIT_NOT_ACTIVE (cli imports the core at module level, so the constant is restated import-free; tests/test_cli.py locks them)


def main(argv=None):
    from ._core_gate import gate
    gate(__file__, tag=TAG)                          # (1) NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: … -> SystemExit(3)
    try:
        from .cli import main as cli_main           # (2) the pinned core is importable: a sub-module it lacks is named
    except ImportError as e:
        missing = getattr(e, "name", None) or str(e)
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: core_missing:{missing} ({e}); install the tree's core at the kit's pin: pip install -e <tree>/common/opt_core -e <tree>/af2ig/opt\n")
        sys.stderr.flush()
        return EXIT_INACTIVE
    return cli_main(argv) if argv is not None else cli_main()


if __name__ == "__main__":
    sys.exit(main())
