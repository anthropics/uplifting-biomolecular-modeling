# CALIBY_FAST_INFERENCE_KIT_v1 — HOWTO

## Install (public route)
1. Python 3.12 + `torch==2.6.0` (cu124) + the three git pins (caliby @ 41d31560, atomworks-caliby @ ea2c998a, protpardelle-1c @ 7962da09) + the pinned stack (`environment/requirements.lock` at the tree root); the release package (`caliby_opt`) serves this kit's files by import hook — nothing is copied into site-packages.
2. Weights: `snapshot_download('ProteinDesignLab/caliby-weights')` (HF, ungated, Apache-2.0; both files' sha256 in `YOU_MUST_SUPPLY.md`); `MODEL_PARAMS_DIR=<dir>`.
3. Licences: `NOTICE.md` + `licenses/` (caliby Apache-2.0, atomworks-caliby BSD-3, protpardelle-1c MIT, PyTorch BSD-3); `patches/upstream_diffs/README.md`; the patched files' notices in `../xattempt_addon/` (`NOTICE`, `LICENSE`, `LICENSE_NOTES.md`). No Rosetta/PyRosetta; no MPS.

Pinned stack:
- Environment (`stock/PINS.json` "pinned_stack" at the tree root names the container it was tested on): debian_slim py3.12, torch 2.6.0+cu124, caliby @ `41d31560c3c73d7980d94f40f3c852b90bfab5c0`, atomworks-caliby v1.0.0 @ ea2c998a, protpardelle-1c @ 7962da09, gemmi 0.7.1, biopython 1.85, one NVIDIA H100 80GB. `pip freeze --all` of it: `environment/requirements.lock` at the tree root.
- Weights under `MODEL_PARAMS_DIR` (`HF_HUB_OFFLINE=1`): `caliby/soluble_caliby.ckpt` sha256 `49a720458cb86625a74e73ad2a3351c9b91b06c0371040b85896aad507820c98`; `protpardelle-1c/weights/cc95_epoch3490.pth` sha256 `965ef846976e28d05adbf6aad7d32f1ddd1ceee22791757b8848b6a7f2348543`.
- Every Caliby default is untouched (500 DLMC sweeps, anneal 1.0 -> 0.01, LCP regularisation, batch_size 4, num_workers 2).

## Patches (against the pinned commit)
| # | file | what | env switch |
|---|---|---|---|
| 0002 | `caliby/model/seq_denoiser/denoisers/seq_design/potts.py` | fast Potts sampler: level 1 = host-sync-free eager sweep loop (same ops, same RNG draws); level 2 = level 1 + CUDA-graph capture of the sweep body | `CALIBY_FAST_SAMPLER=0/1/2` |
| 0003 | `caliby/model/seq_denoiser/denoisers/atom_mpnn_denoiser.py` | Potts parameters aggregated/densified on the GPU, dead device->host copy removed (level 1); level 2 keeps the sparse 48-neighbour J when no symmetry groups exist | `CALIBY_FAST_POTTS_PARAMS=0/1/2` |

Switch values: `CALIBY_FAST_SAMPLER` — 0 stock | 1 host-sync-free eager sweep (Tier 1) | 2 = 1 + CUDA-graph sweep (Tier 1; this kit's default).
`CALIBY_FAST_POTTS_PARAMS` — 0 stock | 1 GPU-side aggregation/densification, no dead host copy (Tier 1; this kit's default) | 2 = sparse
48-neighbour J, no densification (opt-in: reordered accumulation over the same coupling set — the same sequences, U differs from stock in the
last bit unless the add-on's `CALIBY_X_SPARSE_EXACT` >= 1).

```bash
# in any environment with caliby @ 41d31560 installed
SP=$(python -c "import caliby, os; print(os.path.dirname(caliby.__file__))")
cd $SP/.. && patch -p1 < patches/0002-potts-fast-sampler.patch && patch -p1 < patches/0003-potts-params-gpu-path.patch
```
Applying 0002 + 0003 to the pinned upstream files gives the partner kit's two patched files; the add-on's `../xattempt_addon/fast/potts.py` and
`atom_mpnn_denoiser.py` are those files plus the add-on's own patches (`X001 .vs_kit0002`, `X006 .vs_kit0003`), and its `stock/` holds the
unmodified upstream files. The package `caliby_opt` never writes site-packages: its kit modes load the add-on's files by import hook
(`opt/caliby_opt/overlay.py`) and hash the add-on's `stock/` and `fast/` copies to classify an installed tree (`stack.expected_digests`;
a reinstall of the pinned upstream puts the stock files back). No package verb copies anything into site-packages.
