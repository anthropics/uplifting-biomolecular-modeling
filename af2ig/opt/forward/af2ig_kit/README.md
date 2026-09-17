# af2ig_kit — the driver, the patch series, the example inputs

This directory is the part of the af2ig kit that lives beside the `af2ig_opt` package (`opt/af2ig_opt/`, which locates it as
`AF2IG_OPT_KIT`, default `<af2ig>/opt/forward/af2ig_kit`). Nothing here is run directly: `run.sh install` unpacks the pinned upstream tree
from `stock/` and applies `patches/`, and `af2ig-opt pred|check|warm` composes the driver's command line (`../../../CHANGES.md`).

| path | what it is |
|---|---|
| `MANIFEST.json` | the kit's statement of itself: name and version, the upstream pin and the weights it expects, and the `levers` block the package's mode table reads (`af2ig_opt/modes.py`) |
| `patches/[0-9][0-9]_*.diff` | the patch series applied in name order to nrbennet/dl_binder_design @ `cafa3853` (`patch -p1` from the repository root): `00` adds the PyRosetta-free PDB front end `af2_initial_guess/predict_pdb.py` and the two import-compatibility edits (the stock exception, `../../../STOCK.md`); the rest add the levers behind opt-in flags of that driver (each patch's header names its lever, its numerics class and the files it touches) |
| `patches/patched_files/` | the five patched files as they read after the series — the reference `stock/check_pins.py --checkout` compares an installed tree against, byte for byte |
| `tests/inputs/pdbs/` | six public binder–target complexes (RCSB PDB entries re-chained binder-first; `tests/inputs/PUBLIC_INPUTS_README.md`, `index.csv`) — the example inputs of `../../../README.md` and the complex `af2ig-opt warm` runs |
| `tools/fetch_public_inputs.py` | rebuilds the full public input set (the six files plus the two larger barnase–barstar tilings) from files.rcsb.org with the standard library, checking each file's sha256 |
| `upstream/` | the licences of the code the patches apply to (dl_binder_design MIT; the vendored AlphaFold Apache-2.0) and the pin restated (`UPSTREAM.md`) |
| `LICENSE` | third-party licences and notices for the files the kit redistributes or patches (full texts under `upstream/`) |
