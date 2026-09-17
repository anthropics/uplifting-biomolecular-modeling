# The big/tp line's import hook (this directory FIRST on PYTHONPATH of a rank process; `openfold3_opt pred --mode big --n_gpu P`
# puts it there). With OF3TP_WORLD > 1 and OF3TP_RANK set it installs openfold3_opt.tp_rowpair (the adapter onto opt_core.mem.rowpair) the
# moment `openfold3.projects.of3_all_atom.runner` has been imported; with OF3TP_WORLD absent or 1 it installs NOTHING (the n_gpu=1 rule).
# OF3_DETERMINISTIC=1|warn applies the deterministic recipe (CUBLAS_WORKSPACE_CONFIG=:4096:8 + torch.use_deterministic_algorithms)
# so a stock run and this line share it. The next `sitecustomize.py` later on sys.path is chain-loaded (the kit's other hooks keep working).
import importlib.abc as _abc
import importlib.util as _util
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG_PARENT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))          # .../opt  (makes `import openfold3_opt` work without an install)
if _PKG_PARENT not in _sys.path:
    _sys.path.insert(1, _PKG_PARENT)
_TARGET = "openfold3.projects.of3_all_atom.runner"


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


class _TPFinder(_abc.MetaPathFinder):
    """Runs ``openfold3_opt.tp_rowpair.install()`` once, right after the runner module finishes importing (every OpenFold3 class exists then).
    Stays on sys.meta_path (the kit's activation record lists it: openfold3_opt.hooks.installed, class name in env.HOOK_FINDER_CLASSES)."""
    _targets = (_TARGET,)
    done = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET or self.done:
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
                if not finder.done:
                    finder.done = True
                    import openfold3_opt.tp_rowpair as _tp
                    names = _tp.install()
                    _log(f"installed ({len(names)} seams): {', '.join(names)}")

        spec.loader = _Loader()
        return spec


if _world() > 1 and (_os.environ.get("OF3TP_RANK") or "").strip():
    _sys.meta_path.insert(0, _TPFinder())
    _log(f"armed: world={_world()} rank={_os.environ.get('OF3TP_RANK')} (installs on import of {_TARGET})")

# ---- chain-load the next sitecustomize.py LATER on sys.path than this file (the fast-inference kit's of3_levers etc.). The scan starts AFTER this
# hook's own directory: a hook EARLIER on the path — the cells hook, the tp line's entry, which executes this file through
# OPENFOLD3_OPT_PAIR_CHAIN — is never re-entered (a scan from the head of sys.path re-executed the cells hook, which re-executed this file:
# RecursionError at activation, every rank NOT ACTIVE, `big --n_gpu P` dead). Directories are taken once, in first-seen order; when this
# file's directory is not on sys.path at all the whole path is scanned (this file excluded), as before.
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
