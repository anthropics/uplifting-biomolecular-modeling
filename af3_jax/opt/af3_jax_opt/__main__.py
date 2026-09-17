"""``python -m af3_jax_opt ...`` / ``af3-jax-opt ...`` -> :func:`af3_jax_opt.cli.main`; a missing shared core is the kit's NOT ACTIVE exit 3
(``reason=core_missing:<module>``), never a traceback."""
import os
import sys

from ._autoload import core_missing, not_active_core_missing


def _mode_asked(argv) -> str:
    for i, a in enumerate(argv):
        if a == "--mode" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--mode="):
            return a.split("=", 1)[1]
    return os.environ.get("AF3_JAX_OPT", "") or "?"


def main(argv=None) -> int:
    from ._autoload import TAG
    from ._core_gate import gate
    gate(__file__, tag=TAG)                                               # THE pin gate (_core_gate.py): core absent or older than the pin → NOT ACTIVE by name, exit 3 — before any opt_core import
    try:
        from .cli import main as cli_main
    except ImportError as e:
        missing = core_missing(e)
        if missing:
            return not_active_core_missing(missing, _mode_asked(sys.argv[1:] if argv is None else argv))
        raise
    return cli_main(argv) if argv is not None else cli_main()


if __name__ == "__main__":
    sys.exit(main())
