# CALIBY_XATTEMPT_ADDON_v1

Exact (same output as stock) inference levers for **Caliby / SolubleCaliby** protein sequence design
(github.com/ProteinDesignLab/caliby, commit `41d31560`), layered on top of **CALIBY_FAST_INFERENCE_KIT_v1** (`../fast_inference/`: the
host-sync-free CUDA-graph Potts sampler and the GPU-side Potts parameter path, patches 0002/0003, credited to its authors). The levers live
inside eight upstream modules (`fast/`: caliby's `potts.py`, `atom_mpnn_denoiser.py`, `api.py`, `seq_des_utils.py`, `inference_dataloader.py`,
chroma's `complexity.py`, Protpardelle-1c's `core/models.py` and `data/pdb_io.py`) and are switched purely by `CALIBY_X_*` environment
variables read at call time; unset = the partner kit's / stock code path, so any caliby entry point (`caliby.load_model(...).sample(...)`,
upstream's `examples/scripts/*.sh`) picks them up unchanged. What each lever does, its switch and where it does not apply: `HOWTO.md`.

How the files reach a process: the release package `caliby_opt` (`../../caliby_opt/`, installed with `pip install -e <release>/common/opt_core
-e <release>/caliby/opt`) serves `fast/` under the upstream module names by import hook inside the one design process and exports a mode's
switch line; site-packages keeps the pinned upstream files. Entry points: the release's `run.sh design --mode fast|exact`, or `CALIBY_OPT=<mode>`
in the environment of any caliby process. The design writer every mode runs is `tests/xcaliby_design.py` (a thin CLI over caliby's public API:
`clean_pdbs` → `load_model` → `sample` / `generate_ensembles` + `ensemble_sample`); `tests/public_inputs/` holds caliby's seven public example
structures (the release's `warm` designs the smallest).

Requirements beyond the pinned stack (`<release>/caliby/environment/requirements.lock`): an NVIDIA GPU with a CUDA 12 driver and, for the fused
LCP kernel (`CALIBY_X_LCP=1`, Triton), a C compiler on `PATH` — Triton JIT-compiles a small launcher at first use; without one the first served
call raises the kernel's named error and the release's kit modes refuse by name (a mode is all of its levers; `CALIBY_X_LCP=0` by hand is pure PyTorch).

## Licences (as stated by the upstream licence texts — a summary, not legal advice; per-file table: `LICENSE_NOTES.md`)
| component | files concerned | licence as stated upstream (where) | commercial use per that licence text |
|---|---|---|---|
| caliby code @41d31560 | the files this add-on modifies: `caliby/model/seq_denoiser/denoisers/seq_design/potts.py`, `caliby/model/seq_denoiser/denoisers/atom_mpnn_denoiser.py`, `caliby/api.py`, `caliby/eval/eval_utils/seq_des_utils.py`, `caliby/eval/eval_utils/inference_dataloader.py` (originals in `stock/`) | Apache License 2.0 — repo file `LICENSE`; `pyproject.toml` `license = "Apache-2.0"`; README "Caliby is licensed under the Apache License 2.0" | permitted (Apache-2.0 grants use/reproduction/distribution incl. commercial, subject to its notice/attribution conditions) |
| chroma-derived files inside caliby | `potts.py` (header: "Copyright Generate Biomedicines, Inc. — Licensed under the Apache License, Version 2.0 … Adapted from Chroma by Richard Shuai"), `chroma/layers/complexity.py` (same Generate Biomedicines Apache-2.0 header; vendored under `chroma/` in the caliby repo) | Apache License 2.0 (per-file headers) | permitted (Apache-2.0). Note: this covers the *code*; Chroma's own model weights are not used by caliby or this add-on |
| caliby model weights | loaded by the design writer (`tests/xcaliby_design.py`): the checkpoint `--model_name` names (`caliby/caliby.ckpt` by default — `load_model`'s default — or `soluble_caliby.ckpt`, sha256 49a72045…, …) | HuggingFace repository `ProteinDesignLab/caliby-weights`: model card front-matter `license: apache-2.0` (tag `license:apache-2.0`); caliby README: "Model weights are hosted on HuggingFace" | permitted per the stated Apache-2.0 tag (no separate weight-specific terms file is present in that repository besides `af2/params/LICENSE` for the AlphaFold2 parameters, which this add-on does not load) |
| Protpardelle-1c code @7962da09 | the two files this add-on modifies: `src/protpardelle/core/models.py`, `src/protpardelle/data/pdb_io.py` (originals in `stock/protpardelle_*.py`) | MIT License — repo file `LICENSE` ("Copyright (c) 2025 Protein Design Lab"; carried verbatim in the kit's `third_party_licenses/protpardelle-1c.MIT.txt`); `pyproject.toml` `license = { file = "LICENSE" }` | permitted (MIT) |
| Protpardelle-1c model weights (ensemble mode only) | `protpardelle-1c/weights/cc95_epoch3490.pth` (sha256 965ef846…) + `protpardelle-1c/configs/cc95.yaml`, as redistributed in `ProteinDesignLab/caliby-weights` | two statements exist: the caliby-weights HuggingFace card tags the whole repository `apache-2.0`; the original Protpardelle-1c weight release ("Protpardelle-1c Models", Zenodo record 16817230, linked from the protpardelle-1c README) is licensed **CC-BY-4.0** | permitted under either statement (Apache-2.0; CC-BY-4.0 permits commercial use with attribution). Which of the two governs the redistributed file is not stated explicitly upstream — attribute per CC-BY-4.0 to be safe |
| ProteinMPNN / AlphaFold2 parameters in caliby-weights | `proteinmpnn/*.pt`, `af2/params/*` | not loaded by anything in this add-on | n/a here (AF2 params carry their own `af2/params/LICENSE` in that repository) |
| CALIBY_FAST_INFERENCE_KIT_v1 patches this add-on builds on | `../fast_inference/patches/0002`, `0003` (the base of `fast/potts.py`, `fast/atom_mpnn_denoiser.py`) | derivative of Apache-2.0 caliby files (`../fast_inference/NOTICE.md`) | per the caliby licence (Apache-2.0) |
| this add-on (xattempt) | everything in `fast/`, `patches/`, `tests/`, docs | the modified upstream files keep their upstream licences (rows above; attributions in `NOTICE`, per-file table in `LICENSE_NOTES.md`); the rest is kit-written | — |

Runtime dependencies that are not redistributed here (PyTorch BSD-3-Clause, Triton MIT, biotite BSD-3-Clause, atomworks-caliby BSD-3-Clause per its `LICENSE.md`) keep their own licences; see `LICENSE_NOTES.md`.

Files: `HOWTO.md` (what each lever does, its switch, where it does not apply), `patches/` (unified diffs vs upstream and vs the partner kit's patched file),
`fast/` (full patched files), `stock/` (upstream originals), `tests/` (the design writer, the public example inputs), `LICENSE`, `NOTICE`, `LICENSE_NOTES.md`.
