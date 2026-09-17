"""The ``design`` argument surface: the stock driver's own arguments (``batched.driver_arguments`` — upstream ``sample_sequences.py``'s
``pdbfile``, ``--chain``, ``--temperature``, ``--outpath``, ``--num-samples``, ``--multichain-backbone`` / ``--singlechain-backbone`` and
``--nogpu`` with the script's own names and defaults, which the package's tests hold equal to the carried script's source; the driver's
``--input``, ``--out``, ``--seed``, ``--batch_size``) plus the kit's ``--mode``, ``--det`` and ``--upstream-fix <ID>[,<ID>…]`` (default none:
confirmed upstream issues fixed on request only, upstream_fix.py). ``check_values`` resolves the kit flags.
"""
from __future__ import annotations

import argparse
from typing import Dict

from . import batched, modes


class SettingsError(ValueError):
    """--batch_size < 1, --num-samples < 1, or --multichain-backbone without --chain."""


def parser(prog: str = "esm_if1-opt design") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description="ESM-IF1 sequence design: S sampled sequences per input backbone.")
    p.add_argument("--mode", default=None, metavar="M", help=f"{'|'.join(modes.MODES)} (default: {modes.ENV_MODE} from the environment, else {modes.DEFAULT_MODE}); off = stock: upstream's script as shipped; fast = batched sampling")
    p.add_argument("--det", nargs="?", type=int, const=1, default=0, choices=(0, 1), metavar="0|1",
                   help="the kit modes' deterministic-recipe switch (default 0); nothing to switch on this engine: named NOT APPLIED")
    p.add_argument("--upstream-fix", dest="upstream_fix", default=None, metavar="ID[,ID…]",
                   help="apply the named confirmed upstream fixes (README 'Known upstream issues'); default: none — every mode runs upstream as shipped")
    batched.driver_arguments(p)
    return p


def check_values(ns: argparse.Namespace) -> Dict[str, object]:
    """``{det, seed, batch}``. ``SettingsError`` (usage) for --batch_size < 1, --num-samples < 1 or --multichain-backbone without --chain —
    the driver's own refusals; the other sampling values are upstream's to judge."""
    if int(ns.batch) < 1:
        raise SettingsError(f"--batch_size must be >= 1 (got {ns.batch})")
    if int(ns.num_samples) < 1:
        raise SettingsError(f"--num-samples must be >= 1 (got {ns.num_samples})")
    if ns.multichain_backbone and not ns.chain:
        raise SettingsError("--multichain-backbone needs --chain: the chain to design (every chain of the file conditions it)")
    return {"det": bool(ns.det), "seed": int(ns.seed), "batch": int(ns.batch)}


def script_argv(ns: argparse.Namespace, pdbfile: str, outpath: str) -> list:
    """upstream sample_sequences.py's own argv for ONE structure under ``--mode off``: ``<pdbfile> [--chain C] --temperature T --num-samples N
    --outpath <path> [--multichain-backbone] [--nogpu]`` — its flags exactly; ``--seed`` / ``--batch_size`` are not the script's and are not passed."""
    argv = [pdbfile]
    if ns.chain is not None:
        argv += ["--chain", ns.chain]
    argv += ["--temperature", str(float(ns.temperature)), "--num-samples", str(int(ns.num_samples)), "--outpath", outpath]
    if ns.multichain_backbone:
        argv += ["--multichain-backbone"]
    if ns.nogpu:
        argv += ["--nogpu"]
    return argv


def driver_argv(ns: argparse.Namespace, files_arg: str, out_dir: str = None, outpath: str = None) -> list:
    """The kit child's driver arguments (``--mode fast``) recomposed from the parsed namespace (``batched.driver_arguments`` names): the input as ``--input``,
    the output as ``--out DIR`` or ``--outpath FILE`` (whichever layout the parent resolved), then the sampling values and switches."""
    argv = ["--input", files_arg] + (["--outpath", outpath] if outpath else ["--out", out_dir])
    argv += ["--temperature", str(float(ns.temperature)), "--num-samples", str(int(ns.num_samples)), "--seed", str(int(ns.seed)), "--batch_size", str(int(ns.batch))]
    if ns.chain is not None:
        argv += ["--chain", ns.chain]
    if ns.multichain_backbone:
        argv += ["--multichain-backbone"]
    if ns.nogpu:
        argv += ["--nogpu"]
    return argv

