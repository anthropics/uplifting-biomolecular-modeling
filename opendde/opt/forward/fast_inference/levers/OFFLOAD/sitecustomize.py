# ODDE_OFFLOAD sitecustomize: installs the offload patches in every python process that has ODDE_OFFLOAD set and can import
# opendde; then chains to the next sitecustomize on sys.path after its own position (the chain protocol below).
from opt_core.oom import is_oom                      # an out-of-memory error is re-raised before any reroute below (the core's one classifier)
import os, sys, importlib, importlib.util
# ODDE_OFFLOAD_STRICT=1 (default): an install failure (a retired switch set, an unknown token, a bind that does not resolve on the
# installed opendde) ends the interpreter at startup with `NOT ACTIVE` and exit status 3 -- a process never runs STOCK stages under an
# offload banner.  ODDE_OFFLOAD_STRICT=0: one stderr line and the process continues unpatched (for helper interpreters that share the env).
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
        if os.environ.get("ODDE_OFFLOAD"):
            try:
                import importlib
                if importlib.util.find_spec("opendde") is not None:
                    import odde_offload
                    # patch lazily at first import of the model module to keep interpreter start cheap
                    import opendde.model.opendde  # noqa
                    odde_offload.install()
            except Exception as e:
                if is_oom(e): raise
                print(f"[odde_offload] sitecustomize: install failed: {e!r}", file=sys.stderr, flush=True)
                if os.environ.get("ODDE_OFFLOAD_STRICT", "1") != "0":
                    print(f"[odde_offload] NOT ACTIVE: ODDE_OFFLOAD={os.environ.get('ODDE_OFFLOAD')!r} could not be installed ({e}); exit 3 (ODDE_OFFLOAD_STRICT=0 continues unpatched)", file=sys.stderr, flush=True)
                    os._exit(3)     # interpreter startup: site.py would swallow an Exception and SystemExit aborts with status 1; _exit keeps status 3
        # ODDE_TRAJ_EVERY=k: the diffusion-trajectory capture hook (module odde_traj_hook, when it is on the path) — independent of ODDE_OFFLOAD,
        # installed AFTER the offload patches; observe-only; a failure here (the module absent included) never breaks the run.
        if os.environ.get("ODDE_TRAJ_EVERY", "0") not in ("", "0"):
            try:
                if importlib.util.find_spec("opendde") is not None:
                    import odde_traj_hook
                    odde_traj_hook.apply_from_env()
            except Exception as e:
                if is_oom(e): raise
                print(f"[odde_traj_hook] sitecustomize: install failed: {e!r}", file=sys.stderr, flush=True)
        _chain_next("odde_offload")
finally:
    if _root:
        _state["live"] = False                                                  # the cascade this shim opened is over
