# Third-party notices — `rfdiffusion1/`

This kit carries the upstream project verbatim — `stock/rfdiffusion-86507b65.tar.gz`, with 17 of its files unpacked unmodified under
`stock/src/` for reading; `stock/` is never edited — and, under `opt/`, files written for this kit, some of which transcribe, restate or
re-implement code from the two third-party components listed below with the files concerned. No model weights are included in this
tree or in the container images built from it: `run.sh install --weights DIR` downloads the checkpoint files from the upstream URLs
into `DIR`. Per-directory notices remain beside the code they describe (`opt/forward/fast_inference/NOTICE`, `LICENSE`, `LICENSES.md`;
`opt/forward/se3fast_addon/NOTICE`, `LICENSES.md`). The kit's own code is licensed under the Apache License 2.0 (LICENSE in this
directory and at the top of the tree); this file concerns the third-party portions only.

## RFdiffusion

- Upstream: https://github.com/RosettaCommons/RFdiffusion at commit `86507b6538f51fce57b5a72477165f03999ed7ae` (version 1.1.0; the pin
  in `stock/PINS.json`)
- Licence: BSD 3-Clause (the upstream file is titled `BSD License`)
- Copyright line, exactly as printed in the upstream `LICENSE`: `Copyright (c) 2023 University of Washington. Developed at the Institute for Protein Design by Joseph Watson, David Juergens, Nathaniel Bennett, Brian Trippe and Jason Yim.` The same file states that this copyright and licence cover both
  the source code and the model weights referenced for download in the upstream README.
- Licence text in this tree: `third_party_licenses/RFdiffusion.BSD-3-Clause.txt`, a byte-identical copy of the upstream `LICENSE` at that
  commit (the same text is `stock/src/LICENSE` and the `LICENSE` member of the archive, and is reproduced inside
  `opt/forward/fast_inference/LICENSE` and `opt/forward/fast_inference/LICENSES.md`)

| Kit file | Relation to the upstream code |
|---|---|
| `stock/rfdiffusion-86507b65.tar.gz`, `stock/src/**` | the upstream release at the pinned commit, verbatim: the archive, and 17 of its members unpacked unmodified for reading |
| `opt/forward/fast_inference/drivers/rfd_fastpath.py`, `opt/forward/fast_inference/drivers/rfd_fullgraph.py`, `opt/forward/fast_inference/drivers/rfd_prep.py` | transcribe control flow from upstream `rfdiffusion/util_module.py`, `rfdiffusion/Embeddings.py`, `rfdiffusion/RoseTTAFoldModel.py`, `rfdiffusion/Track_module.py` and `rfdiffusion/inference/model_runners.py` (the network's forward pass re-issued for CUDA-graph replay, per-design constant tensors memoised, input featurisation cached); derivative works of RFdiffusion that remain under its licence, as stated in `opt/forward/fast_inference/NOTICE` and `opt/forward/fast_inference/LICENSE` |
| `opt/forward/se3fast_addon/rfd_se3fast/dense_torch.py` | `str2str_forward_dense` transcribes the control flow of `Str2Str.forward` from upstream `rfdiffusion/Track_module.py` (the file's `VENDORED_FROM` header); described in `opt/forward/se3fast_addon/NOTICE` |
| `opt/rfdiffusion1_opt/pdbio.py` | re-expresses the PDB writers `writepdb` and `writepdb_multi` of upstream `rfdiffusion/util.py` over numpy arrays so that they produce the same bytes; the ATOM-record format string and the protonated-histidine atom-name table are carried verbatim from that file |
| `opt/forward/fast_inference/inputs_public/insulin_target.pdb`, `opt/forward/fast_inference/inputs_public/5TPN.pdb` | unmodified copies of upstream `examples/input_pdbs/insulin_target.pdb` and `examples/input_pdbs/5TPN.pdb`, distributed with RFdiffusion under its licence. `insulin_target.pdb` is a theoretical model (its header reads `EXPDTA THEORETICAL MODEL`; it is not a wwPDB entry and carries no PDB identifier); the `5TPN.pdb` coordinates originate from wwPDB entry 5TPN (wwPDB archive data, CC0 1.0). `opt/forward/fast_inference/inputs_public/SOURCES.md` beside them says the same per file |

The copyright notice above, the licence conditions and the disclaimer (full text in `third_party_licenses/RFdiffusion.BSD-3-Clause.txt`)
accompany these copies and adaptations.

## SE(3)-Transformer (NVIDIA), as vendored inside RFdiffusion

- Upstream: the directory `env/SE3Transformer/` of the RFdiffusion repository at the same commit (carried inside
  `stock/rfdiffusion-86507b65.tar.gz`; its `LICENSE` and the `se3_transformer` package are also unpacked under `stock/src/env/SE3Transformer/`)
- Licence: MIT License
- Copyright line, exactly as printed in the upstream `LICENSE`: `Copyright 2021 NVIDIA CORPORATION & AFFILIATES`
- Upstream `NOTICE`: carried inside the archive as `env/SE3Transformer/NOTICE` and not among the unpacked `stock/src/` files; a
  byte-identical copy is `third_party_licenses/SE3Transformer.NOTICE.txt`. It records that the package includes software from
  https://github.com/FabianFuchsML/se3-transformer-public and https://github.com/lucidrains/se3-transformer-pytorch, each licensed under
  the MIT License.
- Licence text in this tree: `third_party_licenses/SE3Transformer.MIT.txt`, a byte-identical copy of `env/SE3Transformer/LICENSE` at that
  commit (the same text is `stock/src/env/SE3Transformer/LICENSE`)

| Kit file | Relation to the upstream code |
|---|---|
| `opt/forward/se3fast_addon/rfd_se3fast/` (`dense_torch.py`, `kernels.py`) | re-implements, for inference only, the evaluation of the SE(3)-Transformer layer used by RFdiffusion's `Str2Str` block (ConvSE3, AttentionBlockSE3 and basis layouts) against this package and calls its sub-modules at run time; no source file of the package is copied, as stated in `opt/forward/se3fast_addon/NOTICE` |
| `stock/src/env/SE3Transformer/**` | unmodified upstream files unpacked for reading |

The copyright notice above and the MIT permission notice (full text in `third_party_licenses/SE3Transformer.MIT.txt`) accompany this
re-implementation and the unpacked copies.

## Data files
No alignment, sequence-database or weight file is part of this kit. Besides the two example structures above: `opt/rfdiffusion1_opt/tests/fixtures/pdbio_fixture.npz` and
`pdbio_fixture.json` are kit-generated golden outputs (seeded synthetic coordinates and the text upstream's PDB writers produce for them; the generator is
`fixtures/make_pdbio_fixture.py`); `opt/forward/fast_inference/tests/cases_public.json` and `opt/rfdiffusion1_opt/tests/design_route_resolutions.json` are
kit-written test records. Inside the carried archive, `rfdiffusion/inference/sym_rots.npz` is upstream's own table of symmetry rotations (part of the
RFdiffusion release, BSD 3-Clause). The checkpoint files are fetched by `run.sh install --weights DIR` from the URLs upstream publishes and are covered by
the RFdiffusion licence quoted above; they are not in this tree or in the images.

## Runtime dependencies (not carried in this tree)

PyTorch, DGL, Triton, e3nn, hydra-core, opt_einsum and the other packages pinned in `environment/requirements.lock` are installed from
their package indexes when the environment or image is built; their licences (BSD-3-Clause, Apache-2.0, MIT, MIT, MIT, MIT respectively
for the six named) are those published by each project. None of their source files is carried here.
