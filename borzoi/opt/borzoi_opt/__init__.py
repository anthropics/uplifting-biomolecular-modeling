"""borzoi_opt — explicit interface to the Borzoi (TensorFlow, calico/borzoi) inference optimizations: the pipeline_tf kit.

    import borzoi_opt
    borzoi_opt.enable("exact")      # resolve + gate the mode, install the library levers in this process; returns the activation report
    borzoi_opt.status()             # the last report

or, without code changes, `BORZOI_OPT=<mode> <any program>`: for the stock `borzoi_sad.py` the package's .pth hook re-execs the kit's own
entry with the same arguments (the swap is printed); for every other program that imports `baskerville` it installs the kit's forward call
on `SeqNN.__call__` and the LUT one-hot on `dna.dna_1hot` when the library loads (the library route, `ACTIVE … route=hook`); or
`borzoi-opt sad --mode <mode> <the same arguments>` for the documented command.

The kit (`opt/datapath/pipeline_tf`, frozen dir v17) is an ENTRY-SCRIPT SWAP: `v17/borzoi_sad.py` is the stock script's bytes
plus anchored insertions that build the levers around the stock's own loop (the LUT one-hot, the pipelined column-chunked host post and its
writer, the forward traced once and run as one graph from the second call, the copy-free forward return). The forward call and the one-hot
are also LIBRARY levers: the hook (any other program under `BORZOI_OPT`) and `enable()` install those two on the stock library in the running
process (`v17/kitlib/forward.py install_class`, `kitlib/onehot.py install`); the post levers are the entry's alone. Modes (`modes.MODES`): "exact" = the kit in its one composition (the stock's bytes by
construction) — THE DEFAULT (`modes.DEFAULT_MODE`); "off" = the stock script in a proven clean subprocess. `borzoi-opt sad --det 1` exports the reproducibility recipe (`modes.DET_RECIPE`) into the job. Importing the
package loads this module only; `enable` / `status` / `resolve` and the constants load `modes` (with kit, report, registry: standard library
only) on first use; the kit is never imported by the package (only located, checked and exec'd).

The contract: `enable(mode)` -> the activation report (`active`, `mode`, `route`, `entry`, `kit` {name, dir, frozen, ...},
`writer`, `tier`, `env` (the variables the job's environment gains), `kit_env_overrides` (KIT_* switches found in the environment),
`gpu`, `versions`, `package_version`, `reason` when inactive); `status()` -> the last report; `check` (CLI) = the same gate as a dry run.
"""
from __future__ import annotations

__version__ = "0.1.0"
ENV = "BORZOI_OPT"                                               # the mode switch (modes.ENV is this constant; the start-up hook reads it before importing anything else)

__all__ = ["ActivationError", "DEFAULT_MODE", "ENV", "MODES", "enable", "resolve", "status", "__version__"]
_FROM_MODES = ("ActivationError", "DEFAULT_MODE", "MODES", "enable", "resolve", "status")


def __getattr__(name):                                           # PEP 562: the API names resolve `modes` on first use, so `import borzoi_opt` (every interpreter start, through the .pth) loads no other module
    if name in _FROM_MODES:
        from . import modes
        return getattr(modes, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_FROM_MODES))
