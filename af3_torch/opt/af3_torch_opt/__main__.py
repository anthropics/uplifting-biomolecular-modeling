"""``python -m af3_torch_opt ...`` / ``af3-torch-opt ...`` -> :func:`af3_torch_opt.cli.main`, after the pin gate (``_core_gate.gate``,
the shared core's kit template carried byte-for-byte: the importable ``opt_core`` must be the one ``opt/pyproject.toml [tool.opt_core]``
pins — absent / older / newer / edited is the ``NOT ACTIVE: reason=core_missing:opt_core`` / ``reason=core_mismatch: …`` line and exit 3)
and the producers gate (``_producers.refuse_if_missing``: ``reason=producer_missing:<module,...>``, exit 3) — both before anything of the
package resolves; never a traceback, never a stock run under the mode's name."""
import sys

TAG = "af3-torch-opt"
EXIT_NOT_ACTIVE = 3


def main(argv=None) -> int:
    from ._core_gate import gate
    gate(__file__, tag=TAG)                               # THE pin gate (kit_template/_core_gate.py, carried byte-for-byte): the importable opt_core is the pinned one, else NOT ACTIVE + exit 3
    from ._producers import PREFIX, mode_word, refuse_if_missing
    refuse_if_missing(argv=argv)                          # the finer words: which producer modules an importable-but-incomplete core lacks
    try:
        from .cli import main as cli_main                 # imports the shared core (stack / modes / report)
    except ImportError as e:                              # a producer present by name that fails to import: named, exit 3
        sys.stderr.write(f"{PREFIX} NOT ACTIVE mode={mode_word(argv)} reason=core_missing:{getattr(e, 'name', None) or e}\n")
        sys.stderr.flush()
        return EXIT_NOT_ACTIVE
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
