# LICENSE_NOTES.md — licences of every file shipped or patched
Licences of the third-party files shipped or patched here, as stated upstream (attributions: NOTICE; Apache-2.0 text: LICENSE; MIT text of the two Protpardelle-1c files: `../../../third_party_licenses/protpardelle-1c.MIT.txt`; kit-level summary: `../../../THIRD_PARTY_NOTICES.md`).

| shipped file | origin | licence |
|---|---|---|
| fast/potts.py, stock/potts.py, patches/X001-* | caliby `caliby/model/seq_denoiser/denoisers/seq_design/potts.py` (header: "Copyright Generate Biomedicines, Inc.", Apache-2.0; repo LICENSE: Apache License 2.0) + kit patch 0002 + this add-on | Apache-2.0 (modifications are provided under the same licence; modified-file notice is the `XATTEMPT add-on` block comment) |
| fast/atom_mpnn_denoiser.py, stock/atom_mpnn_denoiser.py | caliby (Apache-2.0) `atom_mpnn_denoiser.py`; kit patch 0003; + X006 (L-E) in fast/ | Apache-2.0 |
| fast/protpardelle_core_models.py, stock/protpardelle_core_models.py, fast/protpardelle_data_pdb_io.py, stock/protpardelle_data_pdb_io.py | Protpardelle-1c (github.com/ProteinDesignLab/protpardelle-1c @ 7962da09) `src/protpardelle/core/models.py`, `src/protpardelle/data/pdb_io.py` + X008 (L-H) / X009 (L-I) | MIT (Protpardelle-1c LICENSE: 'Copyright (c) 2025 Protein Design Lab'); our modifications to these two files are offered under the same MIT terms so they can be upstreamed as-is |
| fast/inference_dataloader.py, stock/inference_dataloader.py | caliby `caliby/eval/eval_utils/inference_dataloader.py` + X007 (L-F) | Apache-2.0 (caliby) |
| fast/api.py, stock/api.py, patches/X002-* | caliby `caliby/api.py` (repo LICENSE Apache-2.0) + this add-on | Apache-2.0 |
| fast/seq_des_utils.py, stock/seq_des_utils.py, patches/X003-* | caliby `caliby/eval/eval_utils/seq_des_utils.py` (Apache-2.0) + this add-on | Apache-2.0 |
| fast/complexity.py, stock/complexity.py, patches/X004-* | `chroma/layers/complexity.py` vendored inside the caliby repo (header: "Copyright Generate Biomedicines, Inc. — Licensed under the Apache License, Version 2.0") + this add-on | Apache-2.0 (modified-file notice is the `XATTEMPT add-on` block comment) |
| tests/public_inputs/*.cif | copied from the caliby repository `examples/example_data/` (RCSB PDB entries 7URP 7XHZ 7XZ3 8HUZ 8SOT, assemblies 1TNF 5JE6): five unchanged; `7urp.cif`, `8sot.cif` with the `_entry.author` value replaced by `?`, otherwise unchanged | repository Apache-2.0; coordinates are public PDB data (wwPDB, CC0) |
| tests/xcaliby_design.py | written for this add-on (thin CLI over caliby's public API); contains no third-party code | — |

Third-party components exercised but not shipped: PyTorch (BSD-3-Clause), Triton (MIT), Protpardelle-1c (MIT; two files patched, see table), atomworks-caliby (BSD 3-Clause, Institute for Protein Design),
biotite (BSD 3-Clause). The L-D kernel re-implements, in Triton, the arithmetic of `estimate_entropy` from chroma (Apache-2.0) and mirrors the
reduction order of ATen's `Reduce.cuh` (behavioural compatibility only; no PyTorch source is copied).

## Model weights (not redistributed by this add-on; loaded at run time from $MODEL_PARAMS_DIR)
- `caliby/caliby.ckpt`, `caliby/soluble_caliby.ckpt` (the checkpoint `--model_name` names; `caliby` is `load_model`'s default) — from HuggingFace `ProteinDesignLab/caliby-weights`, model card `license: apache-2.0`.
- `protpardelle-1c/weights/cc95_epoch3490.pth` + `configs/cc95.yaml` (ensemble mode only) — redistributed in the same HuggingFace repository (card: apache-2.0); the original
  Protpardelle-1c weight release (Zenodo record 16817230, "Protpardelle-1c Models") is licensed CC-BY-4.0. Attribute accordingly.
- Not used here: `proteinmpnn/*`, `af2/params/*` (the latter carry their own LICENSE file in that repository), other caliby checkpoints.
