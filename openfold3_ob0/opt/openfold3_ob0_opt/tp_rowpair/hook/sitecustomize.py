# The big/tp line's import hook (this directory FIRST on PYTHONPATH of a rank process; `openfold3_ob0_opt pred --mode big --n_gpu P`
# puts it there). With OF3TP_WORLD > 1 and OF3TP_RANK set it installs openfold3_ob0_opt.tp_rowpair (the adapter onto opt_core.mem.rowpair) the
# moment `openfold3.projects.of3_all_atom.runner` has been imported, and the line's DATA-pipeline patches (tp_rowpair/data.py: the lazy template
# pair features) the moment the template featurizer module has been imported — in EVERY process of the rank that imports it: OpenFold3 0.5.0
# featurizes in forkserver DataLoader workers, fresh interpreters that never import the runner, so the featurizer is rebound where it runs.
# With OF3TP_WORLD absent or 1 it installs NOTHING (the n_gpu=1 rule).
# OF3_DETERMINISTIC=1|warn applies the deterministic recipe (CUBLAS_WORKSPACE_CONFIG=:4096:8 + torch.use_deterministic_algorithms)
# so a stock run and this line share it. The next `sitecustomize.py` later on sys.path is chain-loaded (the kit's other hooks keep working).
import importlib.abc as _abc
import importlib.util as _util
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG_PARENT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))          # .../opt  (makes `import openfold3_ob0_opt` work without an install)
if _PKG_PARENT not in _sys.path:
    _sys.path.insert(1, _PKG_PARENT)
_TARGET = "openfold3.projects.of3_all_atom.runner"                       # the model seams: installed once the runner module exists (the rank's main process)
_DATA_TARGET = "openfold3.core.data.pipelines.featurization.template"      # the data seams: installed wherever the featurizer module is imported (DataLoader workers too)


def _log(m):
    _sys.stderr.write(f"[tp_rowpair hook r{_os.environ.get('OF3TP_RANK', '-')}] {m}\n")
    _sys.stderr.flush()


if _os.environ.get("OF3_DETERMINISTIC") in ("1", "warn"):
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch as _torch
        _torch.use_deterministic_algorithms(True, warn_only=(_os.environ.get("OF3_DETERMINISTIC") == "warn"))
        _log("deterministic algorithms ON")
    except Exception as _e:                                   # noqa: BLE001 — printed, and the run's DET record shows the level reached
        _log(f"deterministic setup failed: {_e!r}")


def _world() -> int:
    try:
        return int((_os.environ.get("OF3TP_WORLD") or "1").strip())
    except ValueError:
        return 1


def _install_data_patches():
    """The line's data-pipeline patches in THIS process (``tp_rowpair.data.install`` through the line's PatchSet; idempotent): the process that
    featurizes — a forkserver DataLoader worker or the rank itself — emits the lazy template placeholders instead of dense dummy pair features."""
    from openfold3_ob0_opt.tp_rowpair import data as _data
    if _data.install(loaded_only=True):                                  # data.PATCHES (this process's record); model.PATCHES stays the rank main process's model-seam record
        _log(f"data patches installed in pid {_os.getpid()} (the template featurizer of this process: lazy template pair features)")


class _TPFinder(_abc.MetaPathFinder):
    """Runs ``openfold3_ob0_opt.tp_rowpair.install()`` once, right after the runner module finishes importing (every OpenFold3 class exists then),
    and the data-pipeline patches once, right after the template featurizer module finishes importing (any process of the rank).
    Stays on sys.meta_path (the kit's activation record lists it: openfold3_ob0_opt.hooks.installed, class name in env.HOOK_FINDER_CLASSES)."""
    _targets = (_TARGET, _DATA_TARGET)

    def __init__(self):
        self.done = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._targets or fullname in self.done:
            return None
        _sys.meta_path.remove(self)
        try:
            spec = _util.find_spec(fullname)
        finally:
            _sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        finder = self

        class _Loader(_abc.Loader):
            def create_module(self, spec_):
                return loader.create_module(spec_) if hasattr(loader, "create_module") else None

            def exec_module(self, module):
                loader.exec_module(module)
                if fullname in finder.done:
                    return
                finder.done.add(fullname)
                if fullname == _DATA_TARGET:
                    _install_data_patches()
                else:
                    import openfold3_ob0_opt.tp_rowpair as _tp
                    names = _tp.install()
                    _log(f"installed ({len(names)} seams): {', '.join(names)}")

        spec.loader = _Loader()
        return spec


if _world() > 1 and (_os.environ.get("OF3TP_RANK") or "").strip():
    _sys.meta_path.insert(0, _TPFinder())
    _log(f"armed: world={_world()} rank={_os.environ.get('OF3TP_RANK')} pid={_os.getpid()} (installs on import of {_TARGET}; data patches on import of {_DATA_TARGET})")

# ---- chain-load the next sitecustomize.py LATER on sys.path than this file (the fast-inference hook's of3_levers etc.). The scan starts AFTER this
# hook's own directory: a hook EARLIER on the path — the cells hook, the tp line's entry, which executes this file through
# OPENFOLD3_OB0_OPT_PAIR_CHAIN — is never re-entered (a scan from the head of sys.path would re-execute the cells hook, which re-executes this file:
# RecursionError at activation and every rank NOT ACTIVE). Directories are taken once, in first-seen order; when this file's directory is not on
# sys.path at all the whole path is scanned (this file excluded).
_me = _os.path.abspath(__file__)
_cands = []
for _d in list(_sys.path):
    _c = _os.path.abspath(_os.path.join(_d or ".", "sitecustomize.py"))
    if _c not in _cands:
        _cands.append(_c)
for _cand in (_cands[_cands.index(_me) + 1:] if _me in _cands else _cands):
    if _os.path.isfile(_cand) and _cand != _me:
        _spec = _util.spec_from_file_location("sitecustomize_chained_" + str(abs(hash(_cand))), _cand)
        if _spec is not None and _spec.loader is not None:
            _mod = _util.module_from_spec(_spec)
            try:
                _spec.loader.exec_module(_mod)
            except Exception as _e:                           # noqa: BLE001 — a broken chained hook is printed, never silently skipped
                _log(f"chained sitecustomize {_cand} failed: {_e!r}")
                raise
        break
