"""The kit's autoload hook, run at interpreter start by `openfold3_opt_autoload.pth` (`import openfold3_opt._autoload`).

Import-free until OPENFOLD3_OPT names a mode: the .pth runs in every interpreter on the box, the stock arm's processes included, and those
load nothing of the core. A variable under the package prefix the package does not read (OPENFOLD3_OPT_LINE, a mistyped switch) is refused
here, in every process, with the NOT ACTIVE line and exit 3 — never ignored silently; so is a line's parameter (OPENFOLD3_OPT_N_GPU, the
big/tp line's) under an explicit `off` — a multi-GPU request that stock would run on one card (with a mode named, the same refusal is
the package's: modes.foreign_params at activation; with no selection at all the hook is inert and a selection made on the CLI's argv is
refused by the CLI). Otherwise the core's finder (`opt_core.autoload`) is
installed from this kit's AutoloadSpec: it waits for the first import of the trigger package and, right after that package's own body has
executed, calls `openfold3_opt.enable(mode, strict=True, trigger=<name>)`; when the mode cannot be activated the package prints its NOT
ACTIVE line and the process exits 3 (stock never runs silently under OPENFOLD3_OPT); a selection outside the mode table is refused by the
core at start with the same line form and exit 3. Until then nothing else is imported (no torch, no openfold3); with OPENFOLD3_OPT unset or
"off" no finder is installed at all. A bare `importlib.util.find_spec(<trigger>)` probe leaves the finder armed — it disarms only when the
trigger's body has run.

The trigger is `openfold3` itself: its `__init__` imports gemmi, packaging and deepspeed (none of the modules the add-ons' hooks target), and
every route into the model — the `run_openfold` console script, `openfold3.run_openfold`, library use — passes through it before
`openfold3.core.model.*` or `openfold3.projects.of3_all_atom.model` is first imported, which is where the add-ons' own finders must already
be installed. The finder fires once and removes itself.
"""
import os
import sys

ENV = "OPENFOLD3_OPT"
DECLARED = ("OPENFOLD3_OPT", "OPENFOLD3_OPT_N_GPU", "OPENFOLD3_OPT_HOME", "OPENFOLD3_OPT_GRAPHS_MAX_TOKENS", "OPENFOLD3_OPT_GRAPHS_KEEP", "OPENFOLD3_OPT_REACH_GATE",
            "OPENFOLD3_OPT_CONFHEAD", "OPENFOLD3_OPT_CONFHEAD_CHAIN",
            "OPENFOLD3_OPT_PAIR", "OPENFOLD3_OPT_PAIR_IMPL", "OPENFOLD3_OPT_PAIR_CORE", "OPENFOLD3_OPT_PAIR_LN", "OPENFOLD3_OPT_PAIR_STRICT", "OPENFOLD3_OPT_PAIR_CHAIN",
            "OPENFOLD3_OPT_DIT", "OPENFOLD3_OPT_DIT_MIN_TOKENS", "OPENFOLD3_OPT_DIT_CORE", "OPENFOLD3_OPT_APB_TIER", "OPENFOLD3_OPT_APB_WORD", "OPENFOLD3_OPT_ROLLOUT",
            "OPENFOLD3_OPT_DIT_GLUE", "OPENFOLD3_OPT_DIT_GLUE_MIN_TOKENS", "OPENFOLD3_OPT_DIT_GLUE_CORE", "OPENFOLD3_OPT_DIT_GLUE_ROWS", "OPENFOLD3_OPT_TOKEN_AGG",
            "OPENFOLD3_OPT_APB_TRUNK", "OPENFOLD3_OPT_APB_TRUNK_MIN_TOKENS", "OPENFOLD3_OPT_APB_TRUNK_HIGH_PRECISION", "OPENFOLD3_OPT_APB_TRUNK_SCOPE", "OPENFOLD3_OPT_APB_TRUNK_PRODUCER", "OPENFOLD3_OPT_TEMPL_EMBED",
            "OPENFOLD3_OPT_ATOM_WINDOW", "OPENFOLD3_OPT_ATOM_WINDOW_PRECISION", "OPENFOLD3_OPT_ATOM_WINDOW_INV",
            "OPENFOLD3_OPT_ATOM_HOIST", "OPENFOLD3_OPT_ATOM_HOIST_MAX_GB", "OPENFOLD3_OPT_ATOM_HOIST_INV",
            "OPENFOLD3_OPT_CASTCACHE", "OPENFOLD3_OPT_EXACTLN", "OPENFOLD3_OPT_EXACTLN_WORD", "OPENFOLD3_OPT_LN_PROVIDER", "OPENFOLD3_OPT_LN_TIER", "OPENFOLD3_OPT_LN_WORD", "OPENFOLD3_OPT_APB_HOIST", "OPENFOLD3_OPT_TRUNK_GRAPH", "OPENFOLD3_OPT_TRUNK_GRAPH_NMAX",
            "OPENFOLD3_OPT_POST_RELEASE", "OPENFOLD3_OPT_POST_RELEASE_NTOK", "OPENFOLD3_OPT_POST_RELEASE_MIN_FREE_GB", "OPENFOLD3_OPT_TUNER_GUARD", "OPENFOLD3_OPT_FASTJSON", "OPENFOLD3_OPT_FASTJSON_ROUTE", "OPENFOLD3_OPT_WRITER_OVERLAP", "OPENFOLD3_OPT_WRITER_OVERLAP_ROUTE", "OPENFOLD3_OPT_HOSTFEAT", "OPENFOLD3_OPT_HOSTFEAT_PARTS", "OPENFOLD3_OPT_CKPT_MMAP",
            "OPENFOLD3_OPT_SYNC_HOIST", "OPENFOLD3_OPT_SYNC_HOIST_PARTS",
            "OPENFOLD3_OPT_POSTFWD_MEM", "OPENFOLD3_OPT_POSTFWD_MEM_MIB", "OPENFOLD3_OPT_LOADER_WORKERS", "OPENFOLD3_OPT_LOADER_WORKERS_CAP", "OPENFOLD3_OPT_TRIATT_EXACT", "OPENFOLD3_OPT_TRIMUL_EXACT", "OPENFOLD3_OPT_TRIMUL_EXACT_FORM", "OPENFOLD3_OPT_TRANSITION_EXACT", "OPENFOLD3_OPT_TRIMUL_PROVIDER", "OPENFOLD3_OPT_TRIMUL_WORD", "OPENFOLD3_OPT_TRIMUL_TIER",
            "OPENFOLD3_OPT_RECORDS")   # every variable of the package (README.md §Modes / §Install), the confhead hook's two (modes.ENV_CONFHEAD / CONFHEAD_CHAIN_ENV), the cell levers' (modes.PAIR_ENVS + PAIRFUSED_CHAIN_ENV, DIT_ENVS, ROLLOUT_ENVS: exported by the resolver, inherited by child interpreters) and the tp launcher's per-launch records directory (manifest.RECORDS_ENV: set by tp.launch in every rank's environment, where this gate runs first); another OPENFOLD3_OPT* name is a mistyped switch — tests/test_tp.py locks that every OPENFOLD3_OPT* name a rank inherits is declared here
TRIGGERS = ("openfold3",)
MODES = ("exact", "fast", "big", "off")   # the names of modes.MODES, spelled here so the .pth imports nothing (tests/test_startup.py locks the pair)
LINE_PARAMS = {"OPENFOLD3_OPT_N_GPU": "big/tp"}
REQUIRED_PRODUCER_FILES = (("opt_core.mem.ngpu", ("mem", "ngpu.py")), ("opt_core.arch", ("arch.py",)))   # modes.REQUIRED_PRODUCERS as files under the core package (locked beside)


def core_refusal():
    """THE core probe of every entry route (this hook, `python -m openfold3_opt` / `openfold3-opt`, `openfold3_opt.enable` / `check`,
    `stack.activate`): None when the installed core carries what this package imports; else the NOT ACTIVE reason —
    `core_missing:opt_core (…)` when no core is importable, `producer_missing:<module>[,<module>] — …` when the core is older than the modules
    this package imports (REQUIRED_PRODUCER_FILES, probed by FILE under the core package so the probe itself imports nothing but `opt_core`).
    The version/tree pin proper is `stack.core_pin_gate` (opt_core.gates.core_pin_check) once activation runs."""
    from ._core_gate import gate as _gate
    _gate(__file__, tag=TAG)                            # statement one of the hook once a mode is named: the core pin gate (absent / mismatched core -> NOT ACTIVE, exit 3) before any opt_core import
    try:
        import opt_core as _core
    except ImportError as e:
        return f"core_missing:{getattr(e, 'name', None) or 'opt_core'} ({e}); install the core beside the package: pip install -e <kit>/../common/opt_core -e <kit>/opt"
    root = os.path.dirname(os.path.abspath(getattr(_core, "__file__", "") or ""))
    missing = [m for m, rel in REQUIRED_PRODUCER_FILES if not os.path.isfile(os.path.join(root, *rel))]
    if missing:
        return (f"producer_missing:{','.join(missing)} — this package imports opt_core >= 0.4.0 (the `--n_gpu` axis and the GPU-class registry); "
                f"the installed core at {root} lacks {', '.join(missing)}: install the core this kit's [tool.opt_core] pin names (pip install -e <kit>/../common/opt_core)")
    return None   # a line parameter's variable -> its owning mode/line (modes.LINE_PARAMS, spelled here for the same reason; locked beside)
TAG = "openfold3-opt"                     # report.TAG, spelled here for the same reason (locked beside)
EXIT_NOT_ACTIVE = 3


def spec():
    """This kit's AutoloadSpec for the core's finder (imports the core: called under a set variable, or from the package)."""
    from opt_core.autoload import AutoloadSpec
    return AutoloadSpec(env=ENV, package=__package__, tag=TAG, triggers=TRIGGERS, modes=MODES, exit_not_active=EXIT_NOT_ACTIVE,
                        on_unknown="exit", fold_mode=True)


def install(environ=None):
    """The kit's gate (idempotent): a mistyped switch under the prefix is refused at once with the package's line (SystemExit 3, the
    module-level wrapper turns it into the exit code at .pth time); nothing is installed unless a mode is named; else the core's finder
    from ``spec()`` (an unknown selection: the core's NOT ACTIVE line, SystemExit 3). Returns the finder, or None when nothing is to be done."""
    environ = os.environ if environ is None else environ
    undeclared = sorted(k for k in environ if k.startswith(ENV) and k not in DECLARED)
    if undeclared:                                     # a mistyped switch under the package prefix would be ignored silently: refused by name
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: undeclared variable(s) {','.join(undeclared)} (the package reads {', '.join(DECLARED)})\n")
        sys.exit(EXIT_NOT_ACTIVE)
    mode = (environ.get(ENV) or "").strip().lower()
    if not mode:                                       # no selection on the env route: the hook is inert (a selection on argv is the CLI's, refused there when it does not own a set parameter)
        return None
    if mode == "off":
        foreign = [k for k in LINE_PARAMS if (environ.get(k) or "").strip() and not (k == "OPENFOLD3_OPT_N_GPU" and (environ.get(k) or "").strip() == "1")]   # n_gpu=1 is every route's
        if foreign:                                    # `off` selected with a line's parameter set: the caller asked for N GPUs and stock would run on one — refused, never ignored
            sys.stderr.write(f"[{TAG}] NOT ACTIVE: {', '.join(f'{k}={environ.get(k)}' for k in foreign)} is the {', '.join(LINE_PARAMS[k] for k in foreign)} line's "
                             f"parameter; {ENV}={mode!r} — select the owning mode ({ENV}=<mode>) or unset it\n")
            sys.exit(EXIT_NOT_ACTIVE)
        return None
    try:                                               # the shared core absent (not installed beside the kit, a broken venv): the kit's NOT ACTIVE line, never a traceback or a silent stock run
        from opt_core.autoload import install as core_install
        sp = spec()
    except ImportError as e:
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: reason=core_missing:{getattr(e, 'name', None) or e} — the kit imports the shared core opt_core "
                         f"(pip install -e ../common/opt_core beside the kit, README.md Install); {ENV}={mode!r}\n")
        sys.exit(EXIT_NOT_ACTIVE)
    refusal = core_refusal()                           # an older core beside this package (0.3.x): refused by name in every process, never a traceback at activation
    if refusal:
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: reason={refusal}; {ENV}={mode!r}\n")
        sys.exit(EXIT_NOT_ACTIVE)
    return core_install(sp, environ)


try:
    FINDER = install()
except SystemExit as _e:                               # at interpreter start (the .pth): site.py would turn the exit into a traceback and rc 1 — exit 3 directly
    sys.stderr.flush()
    os._exit(int(_e.code or 0))
