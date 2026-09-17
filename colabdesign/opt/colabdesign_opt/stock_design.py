"""The design script both arms execute — BindCraft's design step through the driver (bindcraft.py), and nothing else. `stock_launch.py`
(mode off) and `kit_launch.py` (mode fast = nosub + the Pallas kernel: the mode's levers installed by `levers.install` before this script starts)
hand their process to it through `runpy.run_path(..., run_name="__main__")`. It knows no lever, reads no kit variable, sets no sub-batch.

    prep = bindcraft.prepare(tree, params_dir, out, binder_len, seed, name, advanced, filters)   # the settings files (sha256 recorded), BindCraft's modules,
                                                                                # bindcraft.py's derivation, the named refusals, the observer
    report.log(report.settings_line(prep["settings"], prep["derived"], hotspot))  # the SETTINGS line, right before the design call: its arrival is the call's start
    bindcraft.run_design(prep, target, chain, binder_len, hotspot, seed, out)   # binder_hallucination(...) as bindcraft.py calls it; the outputs, the `[run]` line

A refusal (bindcraft.Refusal, settings.SettingsError) exits 3 with its name before any model is built. Outputs under `--out`: BindCraft's own tree and the design's files (bindcraft.py); the design's counts and timing
leave this process as its `SETTINGS` / `[run]` lines (report.py: the calling process reads them back).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_LAUNCH = sys.modules.get("colabdesign_opt.launch")                      # the arm launcher, when this script runs through it
T_START = getattr(_LAUNCH, "T_START", None) or time.perf_counter()          # the arm's clock origin: the launcher's start in both arms (else this import) — bindcraft.run_design's t_start
PINS_PATH = getattr(_LAUNCH, "PINS_PATH", None)                             # stock/PINS.json: the tree is the directory around it
EXIT_REFUSED = 3


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="stock_design.py", description=__doc__.split("\n")[0])
    ap.add_argument("--starting-pdb", required=True, help="target PDB file (BindCraft's starting_pdb)")
    ap.add_argument("--chains", required=True, help="target chain(s), e.g. A or A,B (BindCraft's chains)")
    ap.add_argument("--binder-len", type=int, required=True, help="binder length (one value; ColabDesign's binder_len)")
    ap.add_argument("--target-hotspot-residues", default=None, help="BindCraft's target_hotspot_residues, verbatim (e.g. 56 | 31,33,57 | A31,A33 | A1-10); absent = none")
    ap.add_argument("--seed", type=int, default=0, help="the trajectory's seed (default 0)")
    ap.add_argument("--params-dir", required=True, help="AlphaFold params root: <root>/params/params_model_*_multimer_v3.npz (BindCraft af_params_dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--binder-name", default="design", help="BindCraft's binder_name: the trajectory is named <binder-name>_l<binder-len>_s<seed>")
    ap.add_argument("--advanced", default=None, help="bindcraft.py's --advanced: the advanced settings json (default its settings_advanced/default_4stage_multimer.json, vendored)")
    ap.add_argument("--filters", default=None, help="bindcraft.py's --filters: the filters json (default its settings_filters/default_filters.json, vendored)")
    ap.add_argument("--tree", default=None, help="the colabdesign/ tree (default: around the launcher's --pins, else the package's tree)")
    return ap.parse_args(argv)


def _tree(a) -> str:
    from colabdesign_opt import bindcraft
    if a.tree:
        return os.path.abspath(a.tree)
    t = bindcraft.tree_from_pins(PINS_PATH)
    if t:
        return t
    from colabdesign_opt import stack
    return stack.tree_home()


def main(argv=None) -> int:
    a = parse_args(argv)
    from colabdesign_opt import bindcraft, names, report, settings
    tree = _tree(a)
    os.makedirs(a.out, exist_ok=True)
    try:
        prep = bindcraft.prepare(tree, params_dir=a.params_dir, out_dir=a.out, binder_len=a.binder_len, seed=a.seed, tag=a.binder_name, advanced=a.advanced, filters=a.filters)
    except (bindcraft.Refusal, settings.SettingsError) as e:
        report.log(f"REFUSED: {getattr(e, 'kind', 'settings')}: {e}")
        return EXIT_REFUSED
    report.log(report.settings_line(prep["settings"], prep["derived"], a.target_hotspot_residues))   # right before the design call: the line's arrival marks the call's start for the caller (report.run_lines ready_s)
    try:
        bindcraft.run_design(prep, target=a.starting_pdb, chain=a.chains, binder_len=a.binder_len, hotspot=a.target_hotspot_residues, seed=a.seed, out_dir=a.out, t_start=T_START, report=report,
                             extra={"script_sha256": names.sha256_file(os.path.abspath(__file__))})
    except (bindcraft.Refusal, settings.SettingsError):
        return EXIT_REFUSED
    return 0


if __name__ == "__main__":
    sys.exit(main())
