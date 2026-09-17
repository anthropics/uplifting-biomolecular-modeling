# Third-party notices — AF3-torch optimization kit (`af3_torch/`)

This kit runs xfold (a PyTorch re-implementation of AlphaFold 3) on the converted OpenFold3-preview2 weights. The sections below list every
third-party component that files of this kit carry, modify, restate or call, with the licence name, the copyright line exactly as printed
upstream, the kit files concerned, and where the verbatim licence text sits (`third_party_licenses/`). Nothing under `stock/` is edited; no
header inside any third-party source file is added or changed by this notice; this file adds no licence terms of its own. The kit's own code is
licensed under the Apache License 2.0 (LICENSE in this directory and at the top of the tree); this file concerns the third-party portions only.

## xfold — commit `22bdeedfa309ef4ff6f9199910d8403915de69d6`

- Upstream: https://github.com/Shenggan/xfold (pin and archive recipe: `stock/PINS.json` `upstream`; `STOCK.md` §Pin).
- Licence: Apache License, Version 2.0 — the `LICENSE` member of the source archive (`xfold-22bdeedfa309ef4ff6f9199910d8403915de69d6/LICENSE`
  inside `stock/xfold-22bdeed.tar.gz`); `stock/PINS.json` `upstream.license` reads "Apache-2.0 (LICENSE in the archive)". Copyright line as printed
  in that `LICENSE`: its appendix keeps the template line `Copyright [yyyy] [name of copyright owner]` unfilled; the per-file notices of the package
  print `# Copyright 2024 xfold authors` (for example `xfold/nn/pairformer.py`, line 1). The archive's `README.md` states: "The xfold source code is
  licensed under the Apache License, Version 2.0."
- Kit files concerned:
  - `stock/xfold-22bdeed.tar.gz` — the upstream source archive of that commit, unmodified (read by `./run.sh stock` and by the kit's byte tests).
  - `opt/forward/af3t/af3_torch/xfold/` — the archive's `xfold` package carried for this kit (35 files): 22 files are byte-identical to the archive
    members; 10 are modified by this kit — `alphafold3.py`, `params.py`, `fastnn/attention.py`, `fastnn/gated_linear_unit.py`, `fastnn/layer_norm.py`,
    `nn/atom_cross_attention.py`, `nn/attention.py`, `nn/diffusion_head.py`, `nn/diffusion_transformer.py` (the port to the OpenFold3-preview2 weights
    and the corrections described in `STOCK.md` §Stock exceptions; list: `stock/PINS.json` `upstream.port.changed_files`) and `nn/template.py` (the
    archive member with `stock/patches/04_equivalent_template_gap.diff` applied); 3 are added by this kit — `of3.py`, `nn/fourier_constants.py`,
    `nn/paircond_rows.py` (`upstream.port.added_files`). The per-file copyright and licence headers of the carried files are exactly as found at the
    pin (24 of the 35 files carry the header naming `Copyright 2024 DeepMind Technologies Limited`; `STOCK.md` §Pin: "the per-file notices of its
    AlphaFold 3-derived files as found"); this notice does not restate them.
  - Files of this kit that restate xfold statements with this kit's changes (no upstream file is edited for them): `opt/forward/af3t/af3_torch/xfold/nn/paircond_rows.py`
    (written for this kit: it restates `DiffusionHead._pair_conditioning` of `xfold/nn/diffusion_head.py` in row chunks; its first line names this
    kit's copyright holder and the restated upstream function with that file's two copyright lines, `Copyright 2024 xfold authors` and
    `Copyright 2024 DeepMind Technologies Limited`, above the Apache-2.0 file header; `nn/diffusion_head.py` itself keeps its header as found),
    `opt/af3_torch_opt/canonical_noise.py` (restates the sampler draws of `xfold/alphafold3.py` `AlphaFold3._sample_diffusion` /
    `_apply_denoising_step`), `opt/forward/af3t/kernels/af3t_glu_proj.py` (restates `xfold/fastnn/gated_linear_unit.py` and the transition of
    `xfold/nn/primitives.py` as one kernel), `opt/forward/af3t/kernels/af3_kernels.py` and `af3t_msa.py` (kernel adapters whose reference statements are
    the patched xfold modules'), and the package levers `opt/af3_torch_opt/tri_layout.py`, `ln_rows.py`, `dev_scalars.py` (restate statements of
    `xfold/nn/triangle_multiplication.py`, `xfold/fastnn/layer_norm.py`, `xfold/alphafold3.py`).
- Per-file header text: 24 source files ported from xfold (the files under `opt/forward/af3t/af3_torch/xfold/` whose header names
  `Copyright 2024 DeepMind Technologies Limited`, counted above) carry CC BY-NC-SA 4.0 header text inherited from AlphaFold 3's initial release;
  xfold's own `LICENSE` and `README.md` license the project under the Apache License, Version 2.0. These headers are carried exactly as found at the
  pin, in the unmodified files and in the files this kit modified alike; upstream has been asked to refresh them
  (https://github.com/Shenggan/xfold/issues/5). Of those 24 files, 7 are modified by this kit — `alphafold3.py`, `params.py`, `nn/atom_cross_attention.py`,
  `nn/attention.py`, `nn/diffusion_head.py`, `nn/diffusion_transformer.py` (the port, `upstream.port.changed_files`) and `nn/template.py` (patch `04`) —
  and 17 are byte-identical to the archive members: `constants/atom_types.py`, `constants/mmcif_names.py`, `constants/periodic_table.py`,
  `constants/residue_names.py`, `constants/side_chains.py`, `feat_batch.py`, `features.py`, `geometry.py`, `nn/atom_layout.py`, `nn/featurization.py`,
  `nn/head.py`, `nn/pairformer.py`, `nn/primitives.py`, `nn/triangle_multiplication.py`, `nn/utils.py`, `protein_data_processing.py`, `scoring.py`. The
  other 3 modified files (`fastnn/attention.py`, `fastnn/gated_linear_unit.py`, `fastnn/layer_norm.py`) carry no per-file header upstream and none here.
  Statement of changes (Apache-2.0 §4(b)): the 10 files named as modified in this section were changed by this kit; the change of each is the difference
  to the archive member of the same path in `stock/xfold-22bdeed.tar.gz`, which the kit's byte tests read (`STOCK.md` §Stock exceptions describes them).
- Verbatim licence text: `third_party_licenses/xfold.Apache-2.0.txt` — byte-identical to the archive's `LICENSE` member
  (sha256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).

## AlphaFold 3 (source repository) and the alphafold3 fork used as the reference stack

- Upstream: https://github.com/google-deepmind/alphafold3 (per-file notices print `# Copyright 2024 DeepMind Technologies Limited`); fork
  https://github.com/sokrypton/alphafold3 at commit `bc32b22ff5902e3daffd5d1f7203d7f2ab6cb997` (`stock/PINS.json` `reference`). Both repositories'
  `LICENSE` files are the Apache License, Version 2.0 text, byte-identical to `third_party_licenses/xfold.Apache-2.0.txt` (same sha256); their
  appendix line is the unfilled template `Copyright [yyyy] [name of copyright owner]`.
- Kit files concerned:
  - `opt/forward/af3t/af3_torch/xfold/nn/fourier_constants.py` — the two constant vectors are copied from `alphafold3/model/network/noise_level_embeddings.py`
    of AlphaFold 3 v3.1; the file's first line reads, exactly: "# Fixed Fourier embedding constants copied from alphafold3/model/network/noise_level_embeddings.py (AF3 v3.1, Apache-2.0)".
  - `opt/forward/af3t/af3_torch/xfold/of3.py` and the modified xfold files listed above mirror seven code paths of the fork (`model.py`, `modules.py`,
    `evoformer.py`, `atom_cross_attention.py`, `diffusion_head.py` ×2, `diffusion_transformer.py`), as `of3.py`'s docstring states; the reference
    statements quoted in `opt/forward/af3t/kernels/af3_kernels.py` mirror AlphaFold 3's JAX modules as well as xfold's.
  - `stock/patches/04_of3_empty_template_restype_gap.diff` — a patch for the fork's `alphafold3/model/network/template_modules.py`, applied to the
    fork's source tree when the reference environment is built (`environment/Dockerfile`); its header reads "licence: Apache-2.0 (same as upstream)".
  - No file of either repository is redistributed in this tree: the fork is cloned and built at install into its own environment (featurisation,
    output writers, weight converter — `STOCK.md` §Pin, §Stack) and is not vendored.
- The AlphaFold 3 model parameters distributed by Google DeepMind are not used, included or fetched by this kit.

## OpenFold3-preview2 parameters

- Upstream: https://github.com/aqlaboratory/openfold-3 — checkpoint `of3-p2-155k.pt`; source URL, byte count, sha256 and the licence field as
  recorded by this kit: `stock/PINS.json` `variants.p2.checkpoint` (licence field: "Apache-2.0 (the OpenFold3 parameters, github.com/aqlaboratory/openfold-3)").
- Kit files concerned: none carries the parameters. `./run.sh install --weights DIR --fetch` (`opt/af3_torch_opt/weights.py`) downloads the
  checkpoint into `DIR/checkpoint/`, checks the digest and converts it on the user's machine with the fork's `convert_of3_weights.py`
  (`STOCK.md` §Pin). No weights are included anywhere in this tree or in the images built from it.

## Fused kernels of the shared core

- The levers `trimul`, `triattn`, `transition`, `apb`, `trimul_exact` and their companions route to the providers of the shared core under
  `../common/opt_core/opt_core/kernels/` (`stock/PINS.json` `hardware`; `opt/af3_torch_opt/registry.py` `KERNEL_ROUTES`). Their third-party
  notices — FlashPairformer / Protenix and others — are the `NOTICE`, `NOTICE.md` and `*.PROTENIX_LICENSE` files beside each provider there; this
  kit carries no copy of those kernels.
- `opt/forward/af3t/kernels/third_party/` holds two files written for this kit and no third-party source: `af3t_opm.py` (the MSA module's
  outer-product-mean kernels) and `lnl_fused.py` (LayerNorm fused into the consuming linear; it re-exports the shared core's
  `opt_core.kernels.lnl_fused.gate_transpose`).

## Data files

- Outside `stock/`: `inputs/1BRS.json` — an AlphaFold 3 input file holding the protein sequences of wwPDB entry 1BRS (barnase–barstar; wwPDB archive
  data, CC0 1.0) as single sequences (its `unpairedMsa` fields hold the query sequence alone, `pairedMsa` and `templates` are empty); no alignment,
  template or structure file is shipped with it and it contains no database-derived sequence. `upstream_issues/*.diff` and `stock/patches/*.diff`
  are unified diffs written for this kit against the files they name.
- Inside `stock/xfold-22bdeed.tar.gz`, unmodified: source code, `README.md`, `LICENSE` and one image (`assets/comparison.gif`, part of the Apache-2.0
  repository). `stock/wheels/` is empty in the source tree (`.keep`); wheels built from the fork at image-build time may be placed there by the user
  (`environment/Dockerfile` `WHEELS_FROM`). No weights, sequence databases or alignments are in this tree.

## Run-time dependencies (not redistributed)

PyTorch (BSD-3-Clause), Triton (MIT), NumPy (BSD-3-Clause), einops (MIT), zstandard bindings (BSD-3-Clause), JAX and Haiku (Apache-2.0) are
installed from their own distributions under their own licences (`environment/requirements-torch.lock`, `environment/requirements-jax.lock`).
