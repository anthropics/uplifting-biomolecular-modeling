"""_names.py — the package's one switch, its tag, its one mode and lever set, and the exit-code table, in ONE place. Standard library only."""
import os

ENV = "GPNSTAR_OPT"                                      # the one switch: exact | off / unset
TAG = "gpnstar-opt"                                      # the prefix of every line the package prints: [gpnstar-opt]
MODE = "exact"                                           # the one mode
LEVERS = ("devconst", "constcache", "srcgather", "unifiedkv", "dedup", "colattn", "fusedattn", "graph")   # its lever set, in apply order (the ACTIVE line's levers=, one LEVER line each)
ACCEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accel")            # the lever tree (its digest manifest: ACCEL_MANIFEST.json)
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 1, 2, 3   # the exit-code table, in ONE place (== opt_core.report's)
