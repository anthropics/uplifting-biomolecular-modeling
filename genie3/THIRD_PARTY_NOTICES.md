# Third-party notices — `genie3/`

This kit adapts and restates code of the third-party component below; the files concerned are listed, followed by the data files the
carried upstream archive contains.
The kit's own code is licensed under the Apache License 2.0 (LICENSE in this directory and at the top of the tree); this file concerns the third-party portions only.
The licence text that must travel with the adaptations is reproduced verbatim under `third_party_licenses/`; the vendored upstream sources
under `stock/` are unmodified and keep their own licence file.

## Genie 3 (aqlaboratory `genie3`)

- Upstream: https://github.com/aqlaboratory/genie3 at commit `d77ae5ac04212ff1e8b29b585859a3244c614804` (version 0.0.1 in `setup.py`, no
  tag; the pin in `stock/PINS.json`; carried unmodified as `stock/genie3-d77ae5ac.tar.gz` — 188 members, each byte-identical to the upstream
  file at that commit — and the reading copy `stock/src/`, an excerpt of the same files)
- Licence: Apache License 2.0
- Copyright line: upstream prints none of its own — its `LICENSE` keeps the appendix template line `Copyright [yyyy] [name of copyright owner]`
  and the upstream repository has no `NOTICE` file at that commit. Eight upstream modules carry a per-file header that begins `# Adapted from OpenFold` /
  `# Copyright 2021 AlQuraishi Laboratory`, followed in seven of them by `# Copyright 2021 DeepMind Technologies Limited`, and then the
  Apache License 2.0 header text: `src/genie3/generation/utils/tensor_utils.py`, `src/genie3/generation/utils/affine_utils.py`,
  `src/genie3/generation/model/module/triangular_multiplicative_update.py`, `…/module/invariant_point_attention.py`, `…/module/primitive.py`,
  `…/module/transition.py`, `…/module/backbone_update.py` (both copyright lines) and `…/module/dropout.py` (the first copyright line only).
  The headers travel inside the archive unchanged; the one that concerns the kit's own files is quoted in full below.
- Licence text in this tree: `third_party_licenses/Genie.Apache-2.0.txt`, a byte-identical copy of the upstream `LICENSE` at that commit (the
  same text is `stock/src/LICENSE`). It is also the licence the OpenFold-derived headers name (http://www.apache.org/licenses/LICENSE-2.0).

| Kit file | Relation to the upstream code |
|---|---|
| `opt/forward/fast_inference/driver/g3fast_patches.py` | modified copies of four upstream helper functions with the changes needed by the fast route: `batched_gather` (`src/genie3/generation/utils/tensor_utils.py`, OpenFold-derived header), `quat_to_rot` (`…/utils/affine_utils.py`, OpenFold-derived header), `sinusoidal_encoding` (`…/utils/encode_utils.py`) and `_compute_frenet_frames` (`…/utils/geo_utils.py`); at run time it also re-executes the source of `V1PairFeatureNet.forward` (`…/model/embedder/pair/v1.py`) with one expression replaced, in memory |
| `opt/forward/fast_inference/driver/g3fast.py` | restates `DDIMSampler._step` and its step schedule (`…/diffusion/sampler/ddim.py`, `…/diffusion/sampler/sampler.py`) around the changed statements, the model / sampler / dataset set-up order of `…/generation/workflow.py` and `…/generation/runner/runner.py`, and the call order of the denoiser forward (`…/model/implementation/v1.py`) |
| `opt/forward/g3cap/g3lean.py`, `opt/forward/g3cap/g3cap.py` | restate `LatentTransformer` / `LatentTransformerBlock.forward` (`…/model/latent/transformer.py`), `TriangleMultiplicativeUpdate.forward` (`…/model/module/triangular_multiplicative_update.py`, OpenFold-derived header) and `V1PairFeatureNet.forward` (`…/model/embedder/pair/v1.py`) around the kit's kernel calls and captured graphs |
| `opt/genie3_opt/g3batch.py`, `opt/genie3_opt/sampling.py` | restate the per-batch reverse-diffusion loop (`…/diffusion/sampler/sampler.py`; the per-step noise draw of `ddim.py` `_step`) and the order of upstream's sequence and PDB-output stages (`…/runner/runner.py`, `…/runner/postprocess.py`), calling upstream's own functions for each |
| other `opt/genie3_opt/` modules | bind the above over the installed upstream package at run time; no upstream file is modified on disk |

These files carry no upstream header of their own; this notice and `opt/forward/NOTICE` carry the attribution for them (Apache License 2.0
§4): the rows above name each kit file that contains modified upstream material and the upstream functions modified in it (§4(b)), and the
changes are described in `CHANGES.md`. Three of the upstream modules drawn on — `tensor_utils.py`, `affine_utils.py` and
`triangular_multiplicative_update.py` — carry the OpenFold-derived header, which reads, exactly as printed upstream:

```
# Adapted from OpenFold
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## Data files inside the carried archive

`stock/genie3-d77ae5ac.tar.gz` (and, for three of the problem files, the reading copy `stock/src/`) carries upstream's pre-processed
binder-design example set (the "binder design problem set" of the archive's `README.md`) and upstream's example request files, byte for byte
as committed upstream. No model checkpoint, trained parameter file, sequence-database excerpt, alignment or expected-output fixture is inside
the archive or anywhere else in this directory.

| Archive member(s) | What the data is | Source | Licence / terms |
|---|---|---|---|
| `data/design/binder_design/binderbench/problems/01_bhrf1.json` … `10_tnfa.json` (10 files; `03_il7ra.json`, `04_pdl1.json` and `07_h1.json` also under `stock/src/data/design/binder_design/binderbench/problems/`) | one binder-design problem definition per target: target chain ranges, interface hotspot residue lists, binder length range, a `tag` list naming the published binder-design target sets the problem is taken from, and, in `pdb_id`, the wwPDB identifier of the target structure: 2WH6, 6M0J, 3DI3, 5O45, 1WWW, 4ZXB, 5VLI, 1BJ1, 4HSA, 1TNF | written by upstream's problem-set preparation script from the wwPDB entries named in `pdb_id` (`scripts/problem/binder_design/prepare.py`, described in the archive's `README.md` under "Construction of binder design problem set"; the script itself is not carried) | upstream repository content, Apache License 2.0; the identifiers and residue lists refer to wwPDB archive data (CC0 1.0) |
| `data/design/binder_design/binderbench/targets/pdb/*.pdb` (25 files: one `<key>.pdb` per problem and 15 per-chain files `<key>-chain_<X>.pdb`) | target coordinates (ATOM records) taken from those ten wwPDB entries: in `<key>.pdb` the target chains are labelled B, C, D and renumbered from 1 behind upstream's `REMARK 999 KEY / NAME / TARGET` header lines; the per-chain files hold one target chain each without those header lines, and 10 of the 15 are empty files (0 bytes) exactly as committed upstream. Every target is a wwPDB entry named by the matching problem file; no target is a computed, predicted or designed model | wwPDB archive entries 2WH6, 6M0J, 3DI3, 5O45, 1WWW, 4ZXB, 5VLI, 1BJ1, 4HSA, 1TNF, reformatted by the same upstream script (chain selection and relabelling, renumbering, header lines; per the archive's `README.md` the script accepts only entries without alternative positions or insertion codes) | wwPDB archive data: CC0 1.0 Universal; the reformatting adds no terms beyond the repository's Apache License 2.0 |
| `examples/binder_design/*.yaml`, `examples/motif_scaffolding/*.yaml`, `examples/unconditional/*.yaml` (7 files; two also under `stock/src/examples/`) | upstream's example request files: configuration values only, no structure or sequence data | upstream repository | Apache License 2.0 |

Not carried (the `archive_recipe` `excluded` list in `stock/PINS.json`): the target FASTA and MSA files the problem definitions point at
(`data/design/binder_design/binderbench/targets/fasta`, `…/targets/msa`; per the archive's `README.md` the MSAs are built through a public MSA
web server), upstream's motif-scaffolding problem sets (`data/design/motif_scaffolding`, whose motif structure files name their wwPDB entry in
a `REMARK 999 PDB` header line per that README), `scripts/problem` and `assets`. A request that names one of those paths needs the files from
the upstream repository at the pin. The archive's `scripts/setup/setup.sh` — upstream's installer for its evaluation and sequence-prediction
stages — fetches further third-party programs and Python packages from their own sources at install time, under their own licences; none of
them is carried in this tree, and `environment/Dockerfile` does not run that script.

## Shared core and weights

- The kit's triangle-multiplication route calls a fused kernel of the shared core (`../common/opt_core`); nothing of that kernel is in this
  directory (`opt/genie3_opt/fpf_cells.json` is this kit's own tile table for it), and the third-party notices of the core's kernels stay
  beside them, listed in `../common/opt_core/THIRD_PARTY_NOTICES.md`.
- Model weights are not part of this tree or of the images built from it: `./run.sh install --weights DIR` runs upstream's own downloader
  (`scripts/setup/download.sh`: `hf download yeqinglin/genie3 --include 'pretrained/**'`) against https://huggingface.co/yeqinglin/genie3 and
  then checks the two files the kit reads (`pretrained/v1/checkpoints/step=600000.ckpt`, `pretrained/v1/config.yaml`) against the SHA-256
  digests pinned in `stock/PINS.json`; the model card of that repository states `license: apache-2.0`.
