"""The `loader_workers` lever (exact class): the engine's predict DataLoader forks
`data_module_args.num_workers` featurisation workers (10 by default) however many items the query set holds — one query x one seed still
forks ten interpreters (~5 GB host RSS each at 1200 tokens) of which nine never receive a task. This lever caps the PREDICT
loader's worker count at the item count: `used = max(1, min(items, configured))`.

Why it is exact in bytes: the engine's loader is in-order with a per-mode torch.Generator (one base-seed draw, whatever the worker count),
`worker_init_fn` seeds worker w from (base_seed, w, rank) and task i goes to worker i mod W — with items <= W every item is the first task
of the worker of its own index under either count; with items > W the count is unchanged. Never 0 workers (that would move featurisation
and its RNG draws into the main process). Training / validation loaders are untouched.

Census (exit): `<PREFIX> LEVER name=loader_workers state=<on|off|refused> loaders=<n> configured=<w,…> used=<w,…> items=<n,…>`.
Switches (the kit adapter's `configure(ENV=…, ENV_CAP=…)`): <KIT>_LOADER_WORKERS=1 arms; <KIT>_LOADER_WORKERS_CAP=items (default) | none
(the engine's count as configured on every loader, by name: the ablation switch).
"""
from __future__ import annotations

import atexit
import os
import sys
from typing import Any, Dict, Optional

PREFIX = "[opt_core/of3_post.loader_workers]"
ENV: Optional[str] = None                                # <KIT>_LOADER_WORKERS=1
ENV_CAP: Optional[str] = None                            # <KIT>_LOADER_WORKERS_CAP=items|none
VALUES = ("1",)
CAPS = ("items", "none")
M_DATA: Optional[str] = None                             # ….core.data.framework.data_module (class DataModule, enum DatasetMode)
PREDICT_MODES = ("predict", "prediction", "inference")   # DatasetMode member names the cap applies to
CONFIGURABLE = ("PREFIX", "ENV", "ENV_CAP", "M_DATA", "PREDICT_MODES")

STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "cap": None, "loaders": 0, "configured": [], "used": [], "items": []}


def configure(**kw) -> None:
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        globals()[k] = v


def _log(msg: str) -> None:
    sys.stderr.write(f"{PREFIX} {msg}\n")


def requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    if not ENV:
        return False
    v = (environ.get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    return True


def cap_word(environ=None) -> str:
    environ = os.environ if environ is None else environ
    v = (environ.get(ENV_CAP) or "").strip() if ENV_CAP else ""
    if not v:
        return "items"
    if v not in CAPS:
        raise ValueError(f"{ENV_CAP}={v!r} is not one of {'|'.join(CAPS)}")
    return v


def serving() -> bool:
    return STATE["state"] == "on"


def plan(items: int, configured: int) -> int:
    """The worker count the predict loader gets: max(1, min(items, configured)) — never 0, never above the configured count."""
    configured = int(configured)
    if configured <= 0:
        return configured                                                                  # 0 workers configured: the engine featurises in-process; as configured
    return max(1, min(int(items), configured))


def census_line() -> str:
    j = lambda xs: ",".join(str(int(x)) for x in xs) or "-"                                # noqa: E731
    return (f"{PREFIX} LEVER name=loader_workers state={STATE['state']}" + (f" reason={STATE['reason']}" if STATE["reason"] else "") +
            f" cap={STATE['cap']} loaders={STATE['loaders']} configured={j(STATE['configured'])} used={j(STATE['used'])} items={j(STATE['items'])}")


def _is_predict(mode) -> bool:
    name = getattr(mode, "name", None) or str(mode)
    return str(name).split(".")[-1].lower() in PREDICT_MODES


def _make_generate(orig):
    def generate_dataloader(self, mode, *a, **k):
        if not _is_predict(mode):
            return orig(self, mode, *a, **k)
        try:
            items = len(self.datasets_by_mode[mode])
            configured = int(self.num_workers)
        except Exception as e:  # noqa: BLE001
            _log(f"predict loader: item count unreadable ({type(e).__name__}: {str(e)[:80]}) — the engine's worker count as configured")
            STATE["reason"] = "items_unreadable"
            return orig(self, mode, *a, **k)
        used = plan(items, configured) if STATE["cap"] == "items" else configured
        STATE["loaders"] += 1; STATE["configured"].append(configured); STATE["used"].append(used); STATE["items"].append(items)
        if used == configured:
            return orig(self, mode, *a, **k)
        self.num_workers = used                                                            # the engine reads self.num_workers for this loader; restored right after
        try:
            _log(f"predict loader: {items} item(s), workers {configured} -> {used}")
            return orig(self, mode, *a, **k)
        finally:
            self.num_workers = configured
    generate_dataloader.__wrapped__ = orig
    generate_dataloader._of3opt_loader_workers = True
    return generate_dataloader


def install(environ=None) -> dict:
    """Wrap `DataModule.generate_dataloader` on the engine's data module. Idempotent; an engine without the class / method is refused by
    name; <KIT>_LOADER_WORKERS_CAP=none leaves the engine as it is (state=off reason=cap_none)."""
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    if not M_DATA:
        raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_DATA=) before install")
    cap = cap_word(environ)
    STATE["cap"] = cap                                                                     # none: installed, and every predict loader keeps the engine's count (reason=cap_none — the ablation switch)
    import importlib
    try:
        D = importlib.import_module(M_DATA)
        cls = D.DataModule
        orig = cls.generate_dataloader
        _ = cls.predict_dataloader
    except Exception as e:  # noqa: BLE001
        STATE.update(installed=True, state="refused", reason=f"no_datamodule:{type(e).__name__}")
        _log(f"REFUSED: {M_DATA}.DataModule.generate_dataloader not found ({type(e).__name__}: {str(e)[:120]}) — the engine's loaders as they are")
        atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
        return STATE
    if not getattr(orig, "_of3opt_loader_workers", False):
        cls.generate_dataloader = _make_generate(orig)
    STATE.update(installed=True, state="on", reason="cap_none" if cap == "none" else "")
    _log("installed: the predict DataLoader's worker count is capped at the query set's item count (max(1, min(items, configured)))" if cap == "items" else
         f"installed, {ENV_CAP}=none: every predict loader keeps the engine's configured worker count")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE


def uninstall() -> None:
    import importlib
    D = importlib.import_module(M_DATA)
    g = D.DataModule.generate_dataloader
    if getattr(g, "_of3opt_loader_workers", False):
        D.DataModule.generate_dataloader = g.__wrapped__
    STATE.update(installed=False, state="off", reason="", loaders=0, configured=[], used=[], items=[])
