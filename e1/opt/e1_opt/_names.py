"""_names.py — the package's environment-variable names, forbidden prefixes and kit-tree markers, in ONE place (stack.py, registry.py,
the stock runner imports them; the stock runner runs in a child process and must not re-type them). Standard library only."""
import os

ENV, ENV_VARIANT, ENV_DET = "E1_OPT", "E1_VARIANT", "E1_OPT_DET"
ENV_HOME = "MODEL_OPT"                                   # the e1/ directory (run.sh / configs export it); default: beside this package
ENV_TARGET_GPU = "MODEL_OPT_TARGET_GPU"                  # a config's target class (h100 / h200 / a100 — stack.CARDS): the probed class must match when set
MUST_BE_ABSENT_PREFIXES = ("E1_OPT", "E1_KIT", "E1_VARIANT", "MODEL_OPT")   # what the stock subprocess proves absent (report.ENV_CHECK_PATTERN)
ABSENT_STARRED = ("E1_OPT*", "E1_KIT*", "E1_VARIANT", "MODEL_OPT*")          # the same names as the ENV-CLEAN line prints them (prefixes starred)
DET_VALUES = ("0", "1")                                  # E1_OPT_DET takes 0 or 1 only (one reader: stack.det_level); anything else is refused by name
KIT_MODULE_PREFIXES = ("engines",)                      # the carried tree's top-level package: none of it may be loaded in a stock process
KIT_DIR_MARKERS = (os.path.join("opt", "forward"), os.path.join("engines", "e1"))   # a sys.path entry holding these is a kit directory
PINS_MODULE_NAME = "e1_opt._pins"                        # the name the file-loaded pins module gets in sys.modules (no `engines.*` name is ever bound)
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 1, 2, 3   # the exit-code table, in ONE place (cli.py and the stock runner import it)
EXIT_NOT_CLEAN = EXIT_NOT_ACTIVE                          # the stock runner's unclean environment exits with the not-active code (what would run is not the mode that was asked for)
