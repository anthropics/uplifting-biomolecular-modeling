"""``python -m opendde_opt ...`` -> :func:`opendde_opt.cli.main`. A shared core (``opt_core``) that cannot be imported is the kit's
NOT ACTIVE exit (3) with the reason ``core_missing:<module>`` — never a traceback, never a stock run under a kit selection."""
import sys


def main(argv=None):
    from ._core_gate import gate
    gate(__file__, "opendde-opt")                            # THE pin gate: an absent / mismatched opt_core is one NOT ACTIVE line, exit 3, before any core import
    from ._producers import refusal                          # then the finer words: a producer module this package imports is absent (rc 3)
    r = refusal()
    if r:
        sys.stderr.write(r + "\n"); sys.stderr.flush()
        return 3
    try:
        from .cli import main as cli_main
    except ImportError as e:                                   # the core (or one of its modules) absent: named, exit 3
        name = getattr(e, "name", None) or str(e)
        if name and (name == "opt_core" or str(name).startswith("opt_core")):
            sys.stderr.write(f"[opendde-opt] NOT ACTIVE reason=core_missing:{name} (pip install -e common/opt_core -e opendde/opt)\n")
            sys.stderr.flush()
            return 3
        raise
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
