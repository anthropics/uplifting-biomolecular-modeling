"""The seed lever (L1): a `-seed` switch for the driver, argv-only, numerics-neutral at its default.

The carried driver fixes its seeds — the TF feature pipeline's `random_seed=0` (predict_pdb.py:365) and the model call's
`jax.random.PRNGKey(0)` (:528; the precompile path :763) — so stock has no seed space and a seed-spread row cannot exist for it. The
lever is one patch of the kit's driver (`levers/04_seed_switch.diff`, applied by `af2ig-opt pred --seed N` to a COPY of the checkout
under the run's output directory): `-seed N` (int, default 0) feeds both sites. At the default the values are the driver's own
constants — seed 0 through the lever is the unseeded driver, bitwise (a run without `--seed`, or with `--seed 0`, never touches the
checkout and never builds the copy). `--mode off --seed N` is the named supplementary stock arm `af2ig-seeded` (the stock driver at its
defaults plus this one switch); it is not the stock line, whose driver is unseeded.
"""
import os
import shutil
import subprocess
import tempfile

PATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "levers", "04_seed_switch.diff")
DRIVER = "predict_pdb.py"
FLAG = "-seed"


def seeded_dir(af2ig_dir: str) -> dict:
    """Copy the checkout's driver directory to a fresh temporary directory (tempfile.mkdtemp: TMPDIR, never the run's output directory)
    and apply the seed patch there; the caller removes the directory when the child has exited (`remove`), so a seeded run leaves no
    file beyond the run's own outputs. The driver the patch applies to is the checkout's, which the checkout gate has compared
    byte-for-byte with the kit's patched copy before any run (stack.checkout_gate).

    Returns {"dir", "driver"}; raises RuntimeError (the caller exits EXIT_FAIL with the line) when the patch does not apply — a
    partially seeded checkout never runs (and is removed).
    """
    dest = os.path.join(tempfile.mkdtemp(prefix="af2ig_seeded_"), "af2_initial_guess")
    try:
        shutil.copytree(af2ig_dir, dest, symlinks=True)
        return _patch(dest)                                  # the checkout is af2ig_dir's copy; a failure names the patch, the copy is removed below
    except BaseException:
        remove({"dir": dest})
        raise


def remove(seeded: dict) -> None:
    """Delete the seeded copy (its parent temp directory) — the caller's `finally`."""
    shutil.rmtree(os.path.dirname(seeded["dir"]), ignore_errors=True)


def _patch(dest: str) -> dict:
    r = subprocess.run(["patch", "-p2", "-s", "--fuzz=0", "-i", PATCH], cwd=dest, capture_output=True, text=True)   # no fuzz — a diff stale against the driver refuses by name here instead of yielding a garbled driver (offsets are exact-context moves and stay allowed)
    if r.returncode != 0:
        raise RuntimeError(f"seed lever: the patch did not apply to the copy of the checkout: {(r.stdout + r.stderr).strip()[:300]}")
    driver = os.path.join(dest, DRIVER)
    try:                                                     # the seeded driver must compile — a hunk that lands inside a statement (a SyntaxError in the copied driver) refuses here by name, nothing runs
        compile(open(driver, encoding="utf-8").read(), driver, "exec")
    except SyntaxError as e:
        raise RuntimeError(f"seed lever: the seeded copy of the driver does not compile: {e}") from None
    return {"dir": dest, "driver": driver}
