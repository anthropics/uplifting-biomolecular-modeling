"""python -m gpnstar_opt check — the dry run: resolve the kit on this machine (CUDA device, the gpn pin, transformers' pin, the primary
checkpoint's digests when it is staged), print the one line (``[gpnstar-opt] DRY-RUN ... would_refuse=none`` and exit 0, or
``... would_refuse=<reason>`` and exit 3), apply nothing. Console script ``gpnstar-opt``."""
from __future__ import annotations

import sys

from ._names import EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE, TAG

USAGE = "usage: python -m gpnstar_opt check"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE, file=sys.stderr)
        return EXIT_OK if argv else EXIT_USAGE
    if argv != ["check"]:
        print(f"[{TAG}] unknown arguments {argv!r}\n{USAGE}", file=sys.stderr)
        return EXIT_USAGE
    from . import ActivationError, check
    try:
        rep = check()
    except ActivationError:
        return EXIT_NOT_ACTIVE
    return EXIT_OK if rep.get("would_refuse") is None else EXIT_NOT_ACTIVE     # a dry run that would refuse exits 3, as the activation it stands for would


if __name__ == "__main__":
    sys.exit(main())
