"""genie3_opt — the explicit interface to the Genie 3 inference optimizations.

    import genie3_opt
    rep = genie3_opt.enable("exact")          # resolve + gate the mode on this box and arm it for design(); the activation report
    genie3_opt.status()                       # the last report, or {"active": False, "reason": ...} before enable()
    rc, manifest = genie3_opt.run_design("experiment.yaml", "out/", mode="exact", n_sample=24)

Genie 3's levers live in the control flow of the kit's own resident driver (opt/forward/fast_inference/driver/g3fast.py) and cannot
attach to the stock command line's one-process-per-invocation shape, so enable() arms the mode for this process's design() calls
(report `attach=driver`) and design() runs the mode's line in a child process; `off` runs the stock line in a proven-clean subprocess.
`GENIE3_OPT=<mode> genie3 generate …` (the .pth finder, _autoload.py) hands upstream's generate command to the design verb carrying the mode, refuses
any other importer of genie3 under a kit mode by name (exit 3), and stays inert for `off`. Modes: off | exact (the batched capture line: byte-identical to stock at the same batch size under the recipe, det.py) | fast (exact +
the pair transition chunked per design L12 + TF32 matmuls L13 + the shared core's fused TriangleMultiplication kernel L7 + the inductor-compiled core L16, classes 2/3 judged as tier 2);
the default mode is modes.DEFAULT_MODE, a literal value.

The report (`enable()`, `status()`, `check`): `active`, `mode`, `attach`, `tier`, `levers_planned`, `levers_applied` / `levers_fallback` /
`levers_unavailable` / `levers_declined` / `partial` (filled from the driver's own evidence after a design pass), `precision`, `trimul` (the line's
TriangleMultiplication provider: fpf | stock), `gpu` (name, class, key), `stack`, `pins`, `genie3_version`, `package_version`, `reason` when inactive.
"""
from __future__ import annotations

__version__ = "0.8.4"
__all__ = ["enable", "status", "run_design", "check", "ActivationError", "__version__"]


class ActivationError(RuntimeError):
    """enable(strict=True) refused: the message is the named fact."""


def enable(mode=None, *, strict=False, dry_run=False):
    """Resolve, gate and arm `mode` (no mode = GENIE3_OPT, else the default mode) for this process; returns the activation report.
    Statement one is the core pin gate (an absent / mismatched shared core: its NOT ACTIVE line and SystemExit(3))."""
    from ._core import core_gate
    core_gate()
    from . import stack
    return stack.activate(mode, dry_run=dry_run, strict=strict, trigger="enable()")


def status():
    from . import stack
    return stack.status()


def check(mode=None):
    """The activation dry run: the report with `would_refuse`; nothing armed. Statement one is the core pin gate."""
    from ._core import core_gate
    core_gate()
    from . import stack
    return stack.activate(mode, dry_run=True, trigger="check()")


def run_design(request_path, out_dir, mode=None, **kw):
    """`design` from Python: returns (exit code, the manifest dict) — the code is the caller's to raise (this route never sets the host
    process's exit). ``out_dir`` None = the request's own output directory. Keywords as the CLI's: tag, seed, n_sample, selections, batch,
    det_level, verbose, log_dir, num_devices, shard_id, num_shards (upstream's generate flags), python."""
    from ._core import core_gate
    core_gate()
    from . import design
    return design.run(request_path, out_dir, mode, **kw)
