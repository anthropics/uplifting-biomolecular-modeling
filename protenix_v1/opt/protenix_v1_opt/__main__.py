"""``python -m protenix_v1_opt ...`` / the ``protenix-v1-opt`` console script -> :func:`protenix_v1_opt.cli.main`. Statement one is the
core pin gate (_core_gate.gate: the ``opt_core`` this interpreter would import, located without importing it, is the one
opt/pyproject.toml [tool.opt_core] pins — else ``NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …``,
exit 3); statement two the producers gate (_producers: every core module this package imports resolves, else
``reason=producer_missing:<modules>``, exit 3); only then the package resolves."""
import sys

TAG = "protenix-v1-opt"          # == report.TAG (the gate runs before report can import)


def main(argv=None):
    from ._core_gate import gate
    gate(__file__, tag=TAG)
    from ._producers import refuse_if_missing
    refuse_if_missing()
    from .cli import main as cli_main
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
