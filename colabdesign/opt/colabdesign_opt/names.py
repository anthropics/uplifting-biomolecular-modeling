"""The package's names, defined once (import-free: `os` and `hashlib` only — `_autoload.py` (when a mode is set, and at the trigger) and the arms' proof import this at interpreter
start): the mode names, the carried kit directory and the files the package reads from it, the modules of the kit route, the output file names,
the report prefix, and the one sha256 helper. Every other module imports these; none redefines them.
"""
import os

PREFIX = "[colabdesign-opt]"                                                     # every line the package prints (stderr)
TAG = "colabdesign-opt"                                                          # the same, for opt_core.report (prefix(TAG) == PREFIX)

# mode names (the table itself — levers, numerics class, guarantee — is modes.MODES; `_autoload.py` needs only the names when a mode is set)
MODE_OFF, MODE_EXACT, MODE_FAST = "off", "exact", "fast"
MODE_NAMES = (MODE_OFF, MODE_EXACT, MODE_FAST)                                             # the modes that run (modes.MODES keys), in table order
KIT_ROUTE_MODES = (MODE_EXACT, MODE_FAST)                                                 # the modes that run the design in the kit's own arm process (levers.py)
DEFAULT_MODE = MODE_FAST                                                       # the package default: fast wherever a fast mode ships

# lever ids (registry.LEVERS keys, in registry order; a mode's lever set is derived from the registry: modes.levers_of)
LEVER_COMPILECACHE, LEVER_LOWERCACHE, LEVER_PARCOMPILE, LEVER_HOIST_PREV, LEVER_NOSUB, LEVER_NOSUB_FN, LEVER_TRIMUL, LEVER_PALLAS, LEVER_TRIATT, LEVER_OPM_FOLD, LEVER_LN, LEVER_PROJ, LEVER_TRANSITION, LEVER_TXLA = "compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "pallas", "triatt", "opm_fold", "ln", "proj", "transition", "txla"

# the mode word's subtractive form: `<mode>-no-<lever>[-no-<lever>...]` runs a kit-route mode WITHOUT the named levers of its set (an A/B
# attribution run: tier none, class `ablation`, never one of the table's modes — modes.resolve validates the base and the levers; this split is
# the syntax only, import-free for the .pth hook)
ABLATION_SEP = "-no-"
ABLATION_CLASS = "ablation"                                                    # the ACTIVE line's class= word of such a run


def split_mode_word(word):
    """`fast-no-pallas-no-x` -> ("fast", ("pallas", "x")); a plain word -> (word, ()). Stripped and lowercased; no validation here."""
    w = (word or "").strip().lower()
    if ABLATION_SEP not in w:
        return w, ()
    base, rest = w.split(ABLATION_SEP, 1)
    return base, tuple(rest.split(ABLATION_SEP))

# the vendored tree and the kit route (paths relative to the tree root)
BINDCRAFT_DIR = os.path.join("stock", "src", "bindcraft")                         # the vendored BindCraft tree (stock/PINS.json upstream.bindcraft): the design step this kit drives
EXAMPLE_SETTINGS = os.path.join(BINDCRAFT_DIR, "settings_target", "PDL1.json")   # BindCraft's own example target settings: the case `warm` designs when no --starting-pdb is given (warm.py)
EXAMPLE_DIR = os.path.join(BINDCRAFT_DIR, "example")                            # where that settings file's `starting_pdb` lives in the vendored tree (example/PDL1.pdb)
KIT_MODULES = ("colabdesign_opt.levers", "colabdesign_opt.compilecache_jax", "colabdesign_opt.launchpad_parcompile", "colabdesign_opt.hoist_prev", "colabdesign_opt.nosub", "colabdesign_opt.nosub_fn", "colabdesign_opt.kernels.trimul_fused", "colabdesign_opt.kernels.provider", "colabdesign_opt.pallas", "colabdesign_opt.kernels.triatt_lever", "colabdesign_opt.kernels.layers_opm", "colabdesign_opt.kernels.layers_ln", "colabdesign_opt.kernels.proj_attn", "colabdesign_opt.kernels.layers_transition", "colabdesign_opt.kernels",
               "opt_core.kernels.pallas_attn_serve", "opt_core.jax_design.subbatch_policy", "colabdesign_opt.txla")   # the kit route's modules (none may be loaded in a stock arm)
NEVER_LEVERED = ("stock_design.py", "stock_launch.py", "kit_launch.py")        # the design script and the package's arms: the .pth hook refuses in them

# the outputs (the design script's file set under --out) and the arm's log beside them
DESIGN_PDB, DESIGN_FASTA, TRAJECTORY, RUN_LOG = "design.pdb", "design.fasta", "trajectory.jsonl", "run.log"
OUTPUTS = (DESIGN_PDB, DESIGN_FASTA, TRAJECTORY)                              # the design script's own files: content only (no label, no clock, no path inside)
DIGEST_FILES = OUTPUTS                                                        # the files the `[run]` line and the run record digest (sha256)


def sha256_file(path: str) -> str:
    import hashlib                                                             # inside: the .pth chain imports this module at interpreter start when a mode is set and must load nothing else
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
