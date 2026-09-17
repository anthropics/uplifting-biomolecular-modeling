"""``python -m evo2_opt check`` — the dry run: read the environment exactly as ``enable()`` does and print the CHECK / NOT ACTIVE line (exit 0 / 3).
Scoring from the command line is the tree's route driver (``bash run.sh score …`` = ``python route/evo2_route.py …``)."""
import sys

USAGE = "usage: python -m evo2_opt check"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] != ["check"] or len(argv) != 1:
        print(USAGE, file=sys.stderr)
        return 2
    from evo2_opt import activation
    rep = activation.check()
    return 0 if rep["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
