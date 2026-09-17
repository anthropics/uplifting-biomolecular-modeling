# NOTICE — CALIBY_FAST_INFERENCE_KIT_v1 (Caliby / SolubleCaliby sequence design: host-sync-free + CUDA-graph Potts sweep, GPU-side Potts params)

## 1. What this kit contains and who licenses it
(a) Program code: the additions the two patches make.
(b) Derivative works of the Apache-2.0 upstream: `patches/0002` (against upstream `caliby/model/seq_denoiser/denoisers/seq_design/potts.py`) and `patches/0003` (against upstream
`caliby/model/seq_denoiser/denoisers/atom_mpnn_denoiser.py`); the patched files ship in the add-on (`../xattempt_addon/fast/`, with the unmodified upstream files in its `stock/`); `patches/0002`, `0003` (Apache-2.0 for the upstream portions).
NO weights (see `YOU_MUST_SUPPLY.md`).

## 2. Upstream projects (licence texts verbatim under licenses/; provenance in licenses/LICENSE_SOURCES.json)
| upstream (pin) | how used | licence | verbatim text |
|---|---|---|---|
| ProteinDesignLab/caliby @ 41d31560c3c73d7980d94f40f3c852b90bfab5c0 | installed from git (pip pin); 2 files PATCHED (the release package serves the patched files by import hook; nothing is written into site-packages) | Apache-2.0 | `licenses/ProteinDesignLab__caliby__LICENSE` (blob `d645695673349e3947e8e5ae42332d0ac3164cd7`) |
| Caliby weights `ProteinDesignLab/caliby-weights` (HF, ungated, `license:apache-2.0`): `caliby/soluble_caliby.ckpt`, `protpardelle-1c/weights/cc95_epoch3490.pth` | loaded from `$MODEL_PARAMS_DIR`; not redistributed | Apache-2.0 (HF model card) | — (URLs + sha256 in YOU_MUST_SUPPLY.md; both == HF LFS oids) |
| richardshuai/atomworks-caliby @ ea2c998a (atomworks-caliby 1.0.0) | dependency, unmodified | BSD-3-Clause | `licenses/richardshuai__atomworks-caliby__ea2c998a__LICENSE.md` |
| ProteinDesignLab/protpardelle-1c @ 7962da09 | dependency (e32 conformers), unmodified | MIT | `licenses/ProteinDesignLab__protpardelle-1c__LICENSE` |
| pytorch/pytorch v2.6.0 (+cu124), triton 3.2.0, numpy 2.1.3, gemmi 0.7.1, biopython 1.85 | unmodified | BSD-3 / MIT / BSD-3 / MPL-2.0 / Biopython | `licenses/pytorch__pytorch__v2.6.0__LICENSE`; others not copied (unmodified PyPI wheels) |

## 3. Statements
- No Rosetta / PyRosetta anywhere (0 hits). No MPS dependency (one process per GPU).
- The weights are not redistributed here: `YOU_MUST_SUPPLY.md` gives the public route with the sha256 of both weight files, and the pinned environment (`stock/PINS.json` "pinned_stack" at the tree root).
- Kit-level summary of all third-party components and verbatim licence texts: `../../../THIRD_PARTY_NOTICES.md`, `../../../third_party_licenses/`.
