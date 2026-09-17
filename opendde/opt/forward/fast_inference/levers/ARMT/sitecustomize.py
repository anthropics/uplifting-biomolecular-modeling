# OPENDDE_ARM_T_ADDON_v0 sitecustomize: with <ADDON> first on PYTHONPATH, ODDE_ARM_U=1 / ODDE_ARM_Z=1 installs the arm in every python
# process (the stock `opendde pred` CLI or any driver) with no code edits. Inert when the flags are unset.
# It chain-loads the next sitecustomize on the path after its own position (this kit's other shims, or a foreign one) so their hooks keep working.
# The OPENDDE_ARM_T_ADDON tag on the first line is read by the other shims (_replaced_by_served_levers looks for it in the first 400 bytes): keep it there.
from opt_core.oom import is_oom                      # an out-of-memory error is re-raised before any reroute below (the core's one classifier)
import os, sys, importlib.util
# --- chain protocol (every hook shim of this kit carries this same block). site.py imports ONE `sitecustomize` -- the first on
# sys.path.  Within that import cascade each shim runs its own body once (re-entry guard shared by all of them: sys._odde_shim_chain =
# {"live": the cascade is running, "ran": the realpaths executed, in order}) and hands the turn to the NEXT sitecustomize.py on sys.path
# AFTER ITS OWN POSITION, so every shim on the path -- and a foreign one listed after the kit's directories --
# executes exactly once, in path order, with no recursion whatever the order of the directories; a chainer that scans from the head of the
# path and re-imports a shim of this kit mid-cascade gets a no-op.  The shim that opened the cascade closes it, so a later site-style
# import in the same interpreter (a test, an embedded second start-up) runs afresh.  Under ODDE_SERVED_LEVERS=1 the ARM-T add-on's shim
# (levers/ARMT, recognised by the tag on its first line) is passed over: its import-time install route is exactly what the served-levers
# hook replaces (post-runner install of the same arm flags).
_me = os.path.realpath(__file__)
_state = getattr(sys, "_odde_shim_chain", None)
_root = not (isinstance(_state, dict) and _state.get("live"))
if _root:
    _state = sys._odde_shim_chain = {"live": True, "ran": []}
_first = _me not in _state["ran"]
if _first:
    _state["ran"].append(_me)


def _replaced_by_served_levers(f):
    if os.environ.get("ODDE_SERVED_LEVERS", "0") in ("", "0"):
        return False
    try:
        with open(f, "r", errors="ignore") as fh:
            head = fh.read(400)
    except OSError:
        return False
    return "OPENDDE_ARM_T_ADDON" in head or head.lstrip().startswith("# OPENDDE_SERVED_LEVERS_ADDON_v0")


def _chain_next(tag):
    """Execute the next sitecustomize.py on sys.path after this file's own entry (once per file per cascade)."""
    entries = []
    for p in sys.path:
        try:
            entries.append(os.path.realpath(os.path.join(p or os.getcwd(), "sitecustomize.py")))
        except OSError:
            entries.append("")
    idx = entries.index(_me) if _me in entries else -1
    for rf in entries[idx + 1:]:
        if not rf or rf == _me or rf in _state["ran"] or not os.path.isfile(rf) or _replaced_by_served_levers(rf):
            continue
        spec = importlib.util.spec_from_file_location("sitecustomize_chained_by_" + tag, rf)
        try:
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        except Exception as e:  # noqa: BLE001
            if is_oom(e): raise
            print(f"[{tag}] chained sitecustomize {rf} failed: {e!r}", file=sys.stderr)
        if rf not in _state["ran"]:
            _state["ran"].append(rf)                                          # a foreign sitecustomize (no protocol of its own) counted as run
        return


try:
    if _first:
        _chain_next("odde_arm_t")
        if any(os.environ.get(k, "0") not in ("", "0") for k in ("ODDE_ARM_U", "ODDE_ARM_Z")):
            # install lazily: opendde must be importable; patch at first import of opendde.model.triangular.layers via a meta-path hook so that
            # processes which never import opendde pay nothing and import order does not matter.
            import importlib.abc
            class _Hook(importlib.abc.MetaPathFinder):
                _busy = False
                def find_spec(self, name, path=None, target=None):
                    if name != "opendde.model.modules.pairformer" or _Hook._busy: return None
                    _Hook._busy = True
                    try:
                        spec = None
                        for f in sys.meta_path:
                            if f is self or not hasattr(f, "find_spec"): continue
                            spec = f.find_spec(name, path, target)
                            if spec is not None: break
                        if spec is None or spec.loader is None: return spec
                        _orig_exec = spec.loader.exec_module
                        def exec_module(module, _orig_exec=_orig_exec):
                            _orig_exec(module)
                            try:
                                sys.meta_path.remove(self)
                            except ValueError: pass
                            try:
                                import odde_arm_t; odde_arm_t.install_from_env()
                            except Exception as e:  # noqa: BLE001
                                if is_oom(e): raise
                                print(f"[odde_arm_t] install failed -> stock path: {e!r}", file=sys.stderr)
                        spec.loader.exec_module = exec_module
                        return spec
                    finally:
                        _Hook._busy = False
            if "opendde.model.modules.pairformer" in sys.modules:
                import odde_arm_t; odde_arm_t.install_from_env()
            else:
                sys.meta_path.insert(0, _Hook())
finally:
    if _root:
        _state["live"] = False                                                  # the cascade this shim opened is over
