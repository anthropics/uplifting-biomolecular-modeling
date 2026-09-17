"""``enable(mode)``: read the environment once (the gate), print the ONE ``[evo2-opt] ACTIVE …`` / ``NOT ACTIVE …`` line, and wrap
``evo2.Evo2.__init__`` so every constructed model gets the kit (``evo2_opt.kit.apply``) and the mode's generation members (``evo2_opt.gen``) as
the constructor returns. Modes: ``exact`` (scores, logits and generated tokens bit-identical to the stock's) and ``fast`` (the same kit and
members — scoring IS exact's — plus speculative sampling as ``generate()``: ``evo2_opt.gen.specdec``).

The gate refuses (``Evo2OptRefused``, exit 3 in the CLI) only what cannot run or is not the stock: no CUDA device; ``evo2`` / ``vtx`` absent or
not the versions ``stock/PINS.json`` pins (another version is another stock). Everything else it reads is NAMED on the line and the kit engages
whole: the GPU (listed: the classes in ``stock/PINS.json gpus``; any other card runs under its own name), Transformer Engine present
(the FP8 routes) or absent (upstream's bf16 projections: the bf16 route), torch / triton / flash-attn / transformer_engine versions against the
stack's pins (``unlisted: torch x != pinned y``)."""
from __future__ import annotations
import contextlib

import os
import sys
from typing import Optional

from evo2_opt import pins as P

PREFIX = "[evo2-opt]"
MODES = ("exact", "fast")
MODE = "exact"                                               # the process's mode once enable() ran (one mode per process)
UPSTREAM = ("evo2", "vtx")                                    # the stock: pinned exactly or refused
NAMED = ("torch", "triton", "flash_attn", "transformer_engine")   # the stack: named when off the pin
_REPORT: Optional[dict] = None
_WRAPPED = {"cls": None, "orig_init": None, "instances": 0, "plain": 0}


class Evo2OptRefused(RuntimeError):
    """The kit cannot run in this process / on this model as constructed; the message is the NOT ACTIVE / KIT REFUSED reason."""


def gpu_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"count": 0, "devices": [], "torch": False}
    if not torch.cuda.is_available():
        return {"count": 0, "devices": [], "torch": True}
    devs = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        cc = torch.cuda.get_device_capability(i)
        devs.append({"index": i, "name": torch.cuda.get_device_name(i), "mib": int(p.total_memory // (1024 * 1024)), "sm": f"sm_{cc[0]}{cc[1]}"})
    return {"count": len(devs), "devices": devs, "torch": True}


def listed_word(dev: dict, doc: dict) -> str:
    """``listed(<class>)`` when the device's compute capability and memory match a ``gpus`` entry (memory within 10 %), else
    ``unlisted(<sm>, <GiB> GiB: not a card stock/PINS.json gpus lists — engaged and named)``."""
    for cls, spec in (doc.get("gpus") or {}).items():
        if spec.get("sm") == dev["sm"] and abs(dev["mib"] - int(spec["memory_mib"])) <= 0.10 * int(spec["memory_mib"]):
            return f"listed({cls})"
    return f"unlisted({dev['sm']}, {round(dev['mib'] / 1024)} GiB: not a class the kit was measured on — engaged and named)"


def gate(doc: Optional[dict] = None) -> dict:
    """Read the environment: {ok, reason, notes[], stack, te, versions, gpu}. ``ok=False`` only for: no CUDA device, evo2/vtx absent or off their pins."""
    doc = doc if doc is not None else P.load()
    out = {"ok": True, "reason": None, "notes": [], "versions": {}, "gpu": None, "te": None, "stack": None}
    for name in UPSTREAM:
        got, want = P.dist_version(name), (doc.get("upstream") or {}).get(name, {}).get("version")
        out["versions"][name] = got
        if got is None:
            return dict(out, ok=False, reason=f"{name} is not installed (the stock is {name} {want}: bash run.sh install)")
        if want and got != want:
            return dict(out, ok=False, reason=f"{name} {got} is installed but the stock is {name} {want} (stock/PINS.json upstream.{name}): another version is another stock")
    info = gpu_info()
    if not info["torch"]:
        return dict(out, ok=False, reason="torch is not importable")
    if info["count"] == 0:
        return dict(out, ok=False, reason="no CUDA device is visible: the kit's levers are CUDA kernels")
    dev = info["devices"][0]
    out["gpu"] = f"{dev['name']}x{info['count']} {listed_word(dev, doc)}"
    te = P.te_version()
    out["te"] = te
    sid = P.stack_of(doc, te is not None); out["stack"] = sid
    spins = ((doc.get("stacks") or {}).get(sid) or {}).get("pins") or {}
    for name in NAMED:
        want = spins.get(name)
        got = te if name == "transformer_engine" else P.dist_version(name)
        out["versions"][name] = got
        if want in (None, "", "absent") or got is None:
            continue
        if str(got).split("+")[0] != str(want).split("+")[0]:
            out["notes"].append(f"unlisted: {name} {got} != pinned {want} (stock/PINS.json stacks.{sid})")
    if te is None:
        out["notes"].append("transformer_engine absent: upstream's bf16 input projections (the 7b's Light install; the 40b needs FP8) — the bf16 route")
    return out


def active_line(g: dict, word: str = "ACTIVE", trigger: Optional[str] = None) -> str:
    v = g["versions"]
    te = f"transformer_engine={g['te']}" if g.get("te") else "transformer_engine=absent"
    notes = ("; " + "; ".join(g["notes"])) if g.get("notes") else ""
    trig = f" ({trigger})" if trigger else ""
    return f"{PREFIX} {word} mode={MODE} evo2={v.get('evo2')} vtx={v.get('vtx')} stack={g.get('stack')} {te} torch={v.get('torch')} gpu={g.get('gpu')} package={_pkg_version()}{notes}{trig}"


def _pkg_version() -> str:
    import evo2_opt
    return getattr(evo2_opt, "__version__", "?")


def _log():
    return sys.stderr


# ----------------------------------------------------------------------------------------------------------------------- the constructor wrap
def _on_constructed(inst, model_name, local_path=None) -> None:
    """Runs as ``Evo2.__init__`` returns: the kit on the model, then the mode's generation members on the instance. A refusal propagates out of
    the constructor as Evo2OptRefused (a mode never serves a subset under its name)."""
    from evo2_opt import kit
    from evo2_opt import gen
    try:
        rec = kit.apply(inst, model_name=model_name, log=_log())
    except kit.KitRefused as e:
        print(f"{PREFIX} KIT REFUSED model={model_name}: {e} — EVO2_OPT={MODE} cannot serve this model as constructed; unset EVO2_OPT to run the stock", file=_log(), flush=True)
        raise Evo2OptRefused(str(e)) from e
    try:
        grec = gen.arm(inst, log=_log(), mode=MODE, local_path=local_path)
    except gen.GenerationRefused as e:
        print(f"{PREFIX} GENERATION REFUSED model={model_name}: {e} — EVO2_OPT={MODE} cannot serve this model as constructed", file=_log(), flush=True)
        raise Evo2OptRefused(str(e)) from e
    if _REPORT is not None:
        _REPORT.setdefault("models", []).append({"model": rec["model"], "route": rec["route"], "levers": rec["levers"], "generation": grec})


def _wrap_class(cls) -> None:
    if _WRAPPED["cls"] is not None:
        return
    orig = cls.__init__

    def __init__(self, model_name="evo2_7b", *args, **kwargs):          # evo2/models.py:27 Evo2.__init__(self, model_name: str = MODEL_NAMES[1], local_path: str = None, ...)
        orig(self, model_name, *args, **kwargs)
        _WRAPPED["instances"] += 1
        if _WRAPPED["plain"]:                                              # a model the kit builds for itself (mode fast's draft): the stock constructor only
            return
        _on_constructed(self, model_name, local_path=kwargs.get("local_path", args[0] if args else None))

    __init__.__wrapped__ = orig
    cls.__init__ = __init__
    _WRAPPED.update(cls=cls, orig_init=orig)


@contextlib.contextmanager
def plain_construction():
    """Inside: a constructed ``Evo2`` gets the stock constructor only (no kit, no members) — the draft model mode fast builds for itself."""
    _WRAPPED["plain"] += 1
    try:
        yield
    finally:
        _WRAPPED["plain"] -= 1


class _ModelsFinder:
    """Wraps ``evo2.models.Evo2`` the moment ``evo2.models`` finishes importing (enable() may run before the package is imported)."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "evo2.models":
            return None
        import importlib.util
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _wrap_class(module.Evo2)
        loader.exec_module = exec_module
        return spec


def _install_wrap() -> str:
    mod = sys.modules.get("evo2.models")
    if mod is not None:
        _wrap_class(mod.Evo2)
        return "evo2.models already imported: Evo2 wrapped now"
    sys.meta_path.insert(0, _ModelsFinder())
    return "Evo2 wrapped at the import of evo2.models"


def _existing_instances() -> int:
    mod = sys.modules.get("evo2.models")
    if mod is None:
        return 0
    import gc
    return sum(1 for o in gc.get_objects() if isinstance(o, mod.Evo2))


# ----------------------------------------------------------------------------------------------------------------------- enable / status
def enable(mode: str = "exact", *, trigger: Optional[str] = None) -> dict:
    """Engage the kit for this process in `mode` (idempotent). Returns the report; raises Evo2OptRefused (after the NOT ACTIVE line) when it cannot run."""
    global _REPORT, MODE
    if mode == "off":
        return {"active": False, "mode": "off", "reason": "off is the stock: nothing installed"}
    if mode not in MODES:
        raise Evo2OptRefused(f"mode {mode!r}: the kit's modes are {' | '.join(MODES)} (off = the stock)")
    if _REPORT is not None and _REPORT.get("active"):
        if _REPORT.get("mode") != mode:
            raise Evo2OptRefused(f"enable({mode!r}) after enable({_REPORT.get('mode')!r}): one mode per process")
        return _REPORT
    env_mode = os.environ.get("EVO2_OPT")
    if env_mode not in (None, "", mode):
        line = f"{PREFIX} NOT ACTIVE: enable({mode!r}) with EVO2_OPT={env_mode!r} in the environment: one mode per process"
        print(line, file=_log(), flush=True); raise Evo2OptRefused(line)
    MODE = mode
    g = gate()
    if not g["ok"]:
        line = f"{PREFIX} NOT ACTIVE: {g['reason']} (mode={MODE})"
        print(line, file=_log(), flush=True)
        _REPORT = dict(g, active=False, line=line)
        raise Evo2OptRefused(g["reason"])
    n = _existing_instances()
    if n:
        line = f"{PREFIX} NOT ACTIVE: {n} Evo2 instance(s) already constructed in this process: enable() (or EVO2_OPT={mode}) goes before the first Evo2(...)"
        print(line, file=_log(), flush=True); raise Evo2OptRefused(line)
    how = _install_wrap()
    line = active_line(g, "ACTIVE", trigger)
    print(line, file=_log(), flush=True)
    _REPORT = dict(g, active=True, mode=MODE, line=line, wrap=how, models=[])
    return _REPORT


def check() -> dict:
    """The dry run of the CLI: the gate and the CHECK line; applies nothing."""
    g = gate()
    line = active_line(g, "CHECK") if g["ok"] else f"{PREFIX} NOT ACTIVE: {g['reason']} (mode={MODE})"
    print(line, file=_log(), flush=True)
    return dict(g, line=line)


def status() -> dict:
    return _REPORT if _REPORT is not None else {"active": False, "reason": "enable() has not run in this process"}
