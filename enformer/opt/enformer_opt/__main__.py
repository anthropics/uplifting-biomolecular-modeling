"""python -m enformer_opt check [--json] — the dry run: resolve the kit on this machine (device, build, kit files, upstream pin), print the one line
(``[enformer-opt] DRY ...`` and exit 0, or ``[enformer-opt] NOT ACTIVE ...`` and exit 3), apply nothing."""
from __future__ import annotations

import json
import sys

from . import _runtime

USAGE = "usage: python -m enformer_opt check [--json]"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE, file=sys.stderr)
        return 0 if argv else 2
    if argv[0] != "check" or any(a not in ("--json",) for a in argv[1:]):
        print(f"{_runtime.PREFIX} unknown arguments {argv!r}\n{USAGE}", file=sys.stderr)
        return 2
    rep = _runtime.check()
    if "--json" in argv:
        print(json.dumps(rep, indent=1, default=str))
    return 0 if rep.get("active") else 3


if __name__ == "__main__":
    sys.exit(main())
