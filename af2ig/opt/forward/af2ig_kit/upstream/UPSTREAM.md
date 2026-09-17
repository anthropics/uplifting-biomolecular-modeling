# Upstream pin

nrbennet/dl_binder_design @ `cafa3853ac94dceb1b908c8d9e6954d71749871a` (2024-01-10), MIT licence (`LICENSE.dl_binder_design.MIT.txt`); its
`af2_initial_guess/alphafold/` is DeepMind's AlphaFold 2 with the initial-guess modification, Apache-2.0 (`LICENSE.alphafold.Apache-2.0.txt`).
The tree ships as `stock/dl_binder_design-cafa3853.tar.gz` (the commit's files, unmodified); `stock/PINS.json` is the machine-readable pin
(commit, archive, the patched files' digests, the weights, the stack) and `stock/check_pins.py` checks an installation against it. The
AlphaFold-2 parameters (`params_model_1_ptm.npz` of `alphafold_params_2022-12-06.tar`, CC BY 4.0) are fetched by `run.sh install --weights`,
never redistributed.
