"""``python -m openfold3_opt ...`` -> :func:`openfold3_opt.cli.main`. A missing core (opt_core not importable) is the kit's NOT ACTIVE refusal
(exit 3, reason ``core_missing``), the same words the .pth route prints (``_autoload``), never a traceback."""
import sys

TAG_ = "openfold3-opt"                                     # the kit's line tag (report.TAG; spelled here so nothing is imported before the gate)


def main(argv=None):
    from ._core_gate import gate
    gate(__file__, tag=TAG_)                               # statement one: the core pin gate (kit_template/_core_gate.py) — absent / mismatched core -> NOT ACTIVE, exit 3, before any opt_core import
    from ._autoload import TAG, EXIT_NOT_ACTIVE, core_refusal
    refusal = core_refusal()                               # statement two: the module-granular producer probe (an older core that passed no pin: producer_missing by name)
    if refusal:
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: reason={refusal}\n")
        return EXIT_NOT_ACTIVE
    from .cli import main as cli_main
    return cli_main(argv) if argv is not None else cli_main()


if __name__ == "__main__":
    sys.exit(main())
