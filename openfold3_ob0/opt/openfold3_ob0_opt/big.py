"""The adapter of the memory mode `big` (`openfold3_ob0_opt.big`, exposing apply() / report()).

The memory INSTALL of this engine lives in the offload port (`opt/forward/offload/of3o/of3_offload.py` `apply_core`, run inside the stock
process by the line's ONE hook chain — confhead > offload > fast_inference on the resident line — when the line's switches are set); the package's
lines (`modes.LINES[("big", <line>)]`) are the switch tables and `modes.resolve` the one resolver. `apply()` is the in-process form of a
line's activation — the same route the CLI gives the stock child through the environment and PYTHONPATH (`cli.py`), reproduced inside a
process that already started (`hooks.run`): exports and unsets go into the environment, the hook directories onto sys.path in the line's
order, the entry hook is executed and the chain installs the finders; the levers install when the model module is imported. `report()` is
the port's own record (`of3_offload.census()`) beside the package's levers record. ONE attach name per line = the first PORT directory in the
line's hook chain (`modes.PORTS`; a package hook ahead of it — the resident line's `confhead`, `modes.PACKAGE_HOOKS` — is the chain's prelude and
executes as the entry hook, chaining into the port's) = the line's entry port
directory: `offload` for the resident composition (one GPU), `tp` for the row-sharded one (`n_gpu` P > 1; its hook executed inside each rank process the CLI
spawned — `pred --mode big --n_gpu P`, tp.py; it resolves only in a rank, `modes.resolve` refuses it elsewhere). The resource decides the composition (modes.line_for); nothing else selects it
whose record is the rank's process-group state (`tp_state`) in-process and the launcher's census in rank 0's manifest. Nothing here changes
a numeric: the adapter routes, the ports install.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional

from . import ActivationError
from . import env as _env
from . import hooks as _hooks
from . import modes
from . import stack as _stack

MODE = "big"


def resolution(n_gpu=None, environ: Optional[Mapping[str, str]] = None, home: Optional[str] = None) -> modes.Resolution:
    """The composition's Resolution against the tree (modes.resolve; resident on one GPU, row-sharded on `n_gpu` P > 1): exports, unsets, hook
    directories, entry hook, conflicts."""
    home = home or _env.tree_home(environ)
    return modes.resolve(MODE, home, environ=environ, n_gpu=n_gpu)


def apply(n_gpu=None, environ: Optional[Mapping[str, str]] = None, home: Optional[str] = None) -> dict:
    """Activate the big composition in THIS process: environment exports/unsets, the hook directories on sys.path, the entry hook executed
    (the chain installs the rest). Refuses (ActivationError) on a caller-preset switch that contradicts the line — never overridden — and
    when the target model module is already imported (the levers install at import; a late apply would be silent)."""
    res = resolution(n_gpu, environ, home)
    if res.conflicts:
        raise ActivationError("big.apply: " + "; ".join(res.conflicts))
    attach = next((h for h in res.hooks if h in modes.PORTS), None)               # the line's entry port: the first PORT directory in the chain (a package hook ahead of it is its prelude)
    if attach is None:                                                            # every big line carries a port; a chain without one is a broken table, refused by name — never a quiet attach
        raise ActivationError(f"no entry port in hook chain: {'>'.join(res.hooks) or '(empty)'} (modes.PORTS = {', '.join(modes.PORTS)})")
    imported = _hooks.targets_imported()
    if imported:
        raise ActivationError(f"big.apply: the hook targets are already imported ({', '.join(imported)}); apply before importing openfold3's model module")
    for k, v in res.exports.items():
        os.environ[k] = v
    for k in res.unsets:
        os.environ.pop(k, None)
    _hooks.put_on_sys_path(res.hook_dirs)
    run = _hooks.run(res.entry_hook, res.hooks[0])
    return {"mode": res.mode, "line": res.line, "attach": attach, "hook_dirs": list(res.hook_dirs), "entry_hook": run["file"], "exports": dict(res.exports),
            "unsets": list(res.unsets), "finders": [f["cls"] for f in _hooks.installed(home)], "tier": res.tier}


def report() -> dict:
    """The record of the line as run: the port's census (applied units, fallbacks, pageable buffers, the confidence path, the pinned pool,
    the confidence head's stock blocks) and the package's levers record (the exit tally's fields)."""
    return {"census": _stack.offload_census(), "tp": _stack.tp_state(), "levers": _stack.levers_record()}
