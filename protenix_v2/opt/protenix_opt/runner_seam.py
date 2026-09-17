"""The runner seam: ONE patch of ``runner.inference.InferenceRunner.__init__`` (the shared core's fail-closed import patch,
``opt_core.autoload.patch_attr_at_import``) that runs an ORDERED list of the package's installers on the runner the stock constructor
just built — ``sampler_fuse`` (the DIT_FUSE kernels), then the sampler attention levers (``apb_levers``: ``dit_attn``, ``atom_attn``),
in the order the levers registered.

Why one patch: the core keeps one patch per site per process and a second factory for the same site re-wraps the ORIGINAL, not the first
wrapper (``AttrPatch``) — two levers patching ``InferenceRunner.__init__`` each on their own would leave only the last one installed. Every
lever that installs on the built runner therefore registers here (``add(name, installer)``) and ``arm()`` patches the site once; the
wrapper runs the stock constructor whole (``init_model`` — where the sampler graph + DiT hoist install —, ``load_checkpoint``,
``init_dumper``), then each installer with the runner. An installer names its own failure on the kit's stream and raises: the run stops
there (no stock fallback), the installers after it do not run.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from . import _core  # noqa: F401  (makes the pinned opt_core importable on the no-install route)
from .report import TAG

TARGET = "runner.inference"
CLASS, METHOD = "InferenceRunner", "__init__"
SITE = f"{CLASS}.{METHOD}"
_INSTALLERS: List[Tuple[str, Callable[[Any], None]]] = []          # (lever or unit name, installer(runner)) in registration order
_STATE: Dict[str, Any] = {"patch": None, "runners": 0, "ran": []}  # the core's AttrPatch; runners constructed through the seam; installers that ran (name per runner)


def add(name: str, installer: Callable[[Any], None]) -> None:
    """Register `installer(runner)` under `name` (idempotent by name: a second registration replaces the callable, keeps the position)."""
    for i, (n, _) in enumerate(_INSTALLERS):
        if n == name:
            _INSTALLERS[i] = (name, installer)
            return
    _INSTALLERS.append((name, installer))


def names() -> List[str]:
    """The registered installer names, in the order they run."""
    return [n for n, _ in _INSTALLERS]


def make_wrapper(orig):
    """The wrapper factory the core's AttrPatch calls with the stock ``InferenceRunner.__init__``: the stock constructor runs whole, then every
    registered installer with the built runner, in order."""

    def __init__(self, *a, **kw):
        orig(self, *a, **kw)
        for name, installer in list(_INSTALLERS):
            installer(self)
            _STATE["ran"].append(name)
        _STATE["runners"] += 1

    return __init__


def arm():
    """Patch the seam now if ``runner.inference`` is imported, else at its import (fail-closed: a missing seam is the kit's NOT ACTIVE line
    and exit 3). One AttrPatch per process (the same factory every call). Returns it."""
    from opt_core.autoload import patch_attr_at_import
    _STATE["patch"] = patch_attr_at_import(TARGET, SITE, make_wrapper, tag=TAG, name="runner_seam")
    return _STATE["patch"]


def patched() -> bool:
    """Whether the seam's patch is installed on the runner class (``runner.inference`` was imported in this process)."""
    p = _STATE["patch"]
    return p is not None and getattr(p, "state", None) == "installed"


def state() -> dict:
    return {"patched": patched(), "installers": names(), "runners": int(_STATE["runners"]), "ran": list(_STATE["ran"])}
