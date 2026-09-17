"""The `ckpt_mmap` lever (exact class, in BYTES): the engine reads its checkpoint file
(`<engine>.core.utils.checkpoint_loading_utils.load_checkpoint`: `torch.load(path)` at 0.4.x, `torch.load(path, map_location='cpu',
weights_only=False)` — the checkpoint's CPU-tagged float32 tensors unpickled into fresh host memory in every process) through the
same `torch.load` call with `mmap=True` added: the file's tensor storages are memory-mapped (copy-on-write) instead of copied, the dict the
engine gets holds the same tensors with the same values, dtypes, devices and keys, and the engine's next statements
(`get_state_dict_from_checkpoint`, `load_state_dict`: the copy into the model's parameters) read only the pages they touch.
The engine's own keyword arguments are kept per pinned text (DIGESTS: 0.4.x none — torch's defaults, weights_only included, apply as they apply
to the engine's call; 0.5.x map_location='cpu', weights_only=False); a directory checkpoint (DeepSpeed), a torch that has no `mmap=`, or a file
`torch.load` cannot map (the legacy non-zip format raises) is the engine's own call, counted. The runner's by-name import of the function
(`<engine>.entry_points.experiment_runner.load_checkpoint`) is re-pointed too.
Why exact: the same bytes deserialised by the same reader; every tensor equal (all 4 890 tensors of the 0.5.0 checkpoint compared equal, mmap vs plain, on one H100 host; the unit test compares every tensor of a saved file); nothing
downstream can tell a mapped storage from a copied one except by speed. Not exact: nothing.
Census (exit): `<PREFIX> LEVER name=ckpt_mmap state=<on|off|refused> loads=<n> mmap=<n> fallback=<n> load_s=<seconds inside load_checkpoint>`.
Switch (the kit adapter's `configure(ENV=…)`): <KIT>_CKPT_MMAP=1 arms. Engines: the OF3 code family (0.4.x, 0.5.x); kits bind it."""
from __future__ import annotations

import atexit
import hashlib
import inspect
import os
import sys
import textwrap
import time
from typing import Any, Dict

CONFIGURABLE = ("PREFIX", "ENV", "M_CKPT", "REBIND", "DIGESTS")
PREFIX = "[of3-opt/ckpt_mmap]"
ENV = ""
M_CKPT = ""                                             # <engine>.core.utils.checkpoint_loading_utils
REBIND = ()                                             # modules that import load_checkpoint by name (<engine>.entry_points.experiment_runner)
DIGESTS: Dict[str, str] = {}                            # sha256[:16] of the engine's load_checkpoint text (dedented) -> variant: "v041" | "v050"
KWARGS = {"v041": {}, "v050": {"map_location": "cpu", "weights_only": False}}   # the engine's own torch.load keyword arguments per variant
VALUES = ("1",)
MARK = "__of3opt_ckpt_mmap__"
STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "variant": "", "loads": 0, "mmap": 0, "fallback": 0, "load_s": 0.0}
ORIG: Dict[str, Any] = {}
_ATEXIT = {"registered": False}


def configure(**kw) -> None:
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise TypeError(f"configure: unknown setting {k}")
        globals()[k] = v


def _log(msg: str) -> None:
    try:
        sys.stderr.write(f"{PREFIX} {msg}\n"); sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    v = (environ.get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r}: expected one of {VALUES} or unset")
    return True


def serving() -> bool:
    return bool(STATE["installed"] and STATE["state"] == "on")


def census_line() -> str:
    return (f"{PREFIX} LEVER name=ckpt_mmap state={STATE['state']}{(' reason=' + STATE['reason']) if STATE['reason'] else ''} "
            f"loads={STATE['loads']} mmap={STATE['mmap']} fallback={STATE['fallback']} load_s={STATE['load_s']:.2f}")


def digest(fn) -> str:
    return hashlib.sha256(textwrap.dedent(inspect.getsource(fn)).encode("utf-8")).hexdigest()[:16]


def make_load_checkpoint(orig, variant: str, torch=None):
    kwargs = dict(KWARGS[variant])

    def load_checkpoint(ckpt_path):
        from pathlib import Path
        nonlocal torch
        STATE["loads"] += 1
        t = time.perf_counter()
        try:
            p = Path(ckpt_path)
            if p.is_file():
                if torch is None:
                    import torch as _torch
                    torch = _torch
                try:
                    r = torch.load(p, mmap=True, **kwargs)
                    STATE["mmap"] += 1
                    return r
                except Exception as e:  # noqa: BLE001 — a file torch cannot map (legacy format), a torch without mmap=: the engine's own call
                    STATE["fallback"] += 1
                    _log(f"{p.name}: torch.load(mmap=True) raised {type(e).__name__} — the engine's own torch.load reads it")
            return orig(ckpt_path)
        finally:
            STATE["load_s"] += time.perf_counter() - t
    load_checkpoint.__wrapped__ = orig
    load_checkpoint.__doc__ = orig.__doc__
    setattr(load_checkpoint, MARK, True)
    return load_checkpoint


def patch_module(mod) -> None:
    fn = getattr(mod, "load_checkpoint", None)
    if fn is None or getattr(fn, MARK, False):
        return
    d = digest(fn)
    variant = DIGESTS.get(d)
    if variant is None:
        STATE.update(state="refused", reason=f"digest:load_checkpoint={d}")
        _log(f"REFUSED: {mod.__name__}.load_checkpoint has a text this port does not re-state (sha256[:16] {d}; accepted {sorted(DIGESTS)}) — the engine reads its checkpoint as it does")
        return
    new = make_load_checkpoint(fn, variant)
    ORIG[mod.__name__] = fn
    mod.load_checkpoint = new
    for m2 in REBIND:
        mm = sys.modules.get(m2)
        if mm is not None and getattr(mm, "load_checkpoint", None) is fn:
            mm.load_checkpoint = new
    STATE.update(state="on", reason="", variant=variant)
    _log(f"installed: {mod.__name__}.load_checkpoint reads the checkpoint file through torch.load(mmap=True) ({variant} arguments)")


class _CkptMmapFinder:
    """Meta-path finder: patches the checkpoint-utils module right after its body runs and re-points a by-name importer's global right after
    ITS body runs; delegates the find to the other finders; re-entrant by name (composes with other finders for the same modules)."""

    def __init__(self):
        self._busy = set()

    def find_spec(self, fullname, path=None, target=None):
        if (fullname != M_CKPT and fullname not in REBIND) or fullname in self._busy:
            return None
        self._busy.add(fullname)
        try:
            spec = None
            for finder in sys.meta_path:
                if finder is self or type(finder).__name__ == type(self).__name__:
                    continue
                try:
                    spec = finder.find_spec(fullname, path, target)
                except Exception:  # noqa: BLE001
                    spec = None
                if spec is not None:
                    break
        finally:
            self._busy.discard(fullname)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec, _name=fullname):
            _orig(module)
            try:
                if _name == M_CKPT:
                    patch_module(module)
                else:                                          # a by-name importer: re-point its global if the utils module is patched
                    src = sys.modules.get(M_CKPT)
                    if src is not None and ORIG.get(M_CKPT) is not None and getattr(module, "load_checkpoint", None) is ORIG[M_CKPT]:
                        module.load_checkpoint = src.load_checkpoint
            except Exception as e:  # noqa: BLE001
                STATE.update(state="refused", reason=f"patch_error:{type(e).__name__}")
                _log(f"REFUSED: patching {_name} raised {type(e).__name__}: {e} — the engine reads its checkpoint as it does")
        spec.loader.exec_module = exec_module
        return spec


FINDER = _CkptMmapFinder()


def install(environ=None) -> dict:
    environ = os.environ if environ is None else environ
    if STATE["installed"]:
        return STATE
    try:
        if not requested(environ):
            return STATE
        if not M_CKPT:
            raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_CKPT=) before install")
    except ValueError as e:
        STATE.update(installed=True, state="refused", reason="bad_word:" + str(e).split("=", 1)[0])
        _log(f"REFUSED: {e} — the engine reads its checkpoint as it does")
        if not _ATEXIT["registered"]:
            atexit.register(lambda: sys.stderr.write(census_line() + "\n")); _ATEXIT["registered"] = True
        return STATE
    STATE.update(installed=True, state="armed", reason="")
    mod = sys.modules.get(M_CKPT)
    if mod is not None:
        patch_module(mod)
        for m2 in REBIND:
            mm = sys.modules.get(m2)
            if mm is not None and getattr(mm, "load_checkpoint", None) is ORIG.get(M_CKPT):
                mm.load_checkpoint = mod.load_checkpoint
    if any(n not in sys.modules for n in (M_CKPT,) + tuple(REBIND)):
        sys.meta_path[:] = [f for f in sys.meta_path if f is FINDER or type(f).__name__ != type(FINDER).__name__]
        if FINDER not in sys.meta_path:
            sys.meta_path.insert(0, FINDER)
    if not _ATEXIT["registered"]:
        atexit.register(lambda: sys.stderr.write(census_line() + "\n")); _ATEXIT["registered"] = True
    return STATE


def uninstall() -> None:
    try:
        sys.meta_path.remove(FINDER)
    except ValueError:
        pass
    for mname, orig in list(ORIG.items()):
        mod = sys.modules.get(mname)
        if mod is not None:
            cur = getattr(mod, "load_checkpoint", None)
            mod.load_checkpoint = orig
            for m2 in REBIND:
                mm = sys.modules.get(m2)
                if mm is not None and getattr(mm, "load_checkpoint", None) is cur:
                    mm.load_checkpoint = orig
    ORIG.clear()
    STATE.update(installed=False, state="off", reason="", variant="", loads=0, mmap=0, fallback=0, load_s=0.0)
