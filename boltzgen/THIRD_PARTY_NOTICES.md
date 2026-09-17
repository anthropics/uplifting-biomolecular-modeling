# Third-party notices — `boltzgen` kit

This kit redistributes the unmodified BoltzGen 0.3.2 release wheel and twelve of its source modules (for reference) under `stock/`, and its own
files under `opt/` replace, at run time and in memory only, methods of BoltzGen modules with re-implementations that follow — and in places
restate — the upstream code. No file of the installed package or of `stock/` is edited. For each third-party component: name, upstream URL,
licence, the upstream copyright line exactly as printed upstream, the kit files concerned, and where the verbatim licence text sits.
The kit's own code is licensed under the Apache License 2.0 (LICENSE in this directory and at the top of the tree); this file concerns the third-party portions only.
The only third-party file outside `stock/` and `third_party_licenses/` is the data file of section 3; every other file outside `stock/` was written for this
kit, and those that restate upstream statements are listed under section 1.

## 1. BoltzGen

- Upstream: https://github.com/HannesStark/boltzgen, release 0.3.2 (tag `v0.3.2`). Licence: MIT License. Copyright line: "Copyright (c) 2025 Hannes Stärk".
- Redistributed verbatim: `stock/boltzgen-0.3.2-py3-none-any.whl` (the package-index wheel; its licence file is the wheel member
  `boltzgen-0.3.2.dist-info/licenses/LICENSE`) and the twelve reference modules under `stock/src/` (byte-identical to the tag; listed in
  `stock/PINS.json`, "touched_modules").
- Kit files that restate or adapt BoltzGen code — each replaces methods of the named upstream modules with a re-implementation of the same
  computation, or restates upstream statements:
  - `opt/forward/fast_inference/src/bg_graph_patch.py` — `boltzgen.model.modules.diffusion` (`AtomDiffusion.sample`; upstream lines 601–615
    restated verbatim, the remainder of the sampling loop restated statement by statement), `boltzgen.model.loss.diffusion`, `boltzgen.model.modules.utils`;
  - `opt/forward/fast_inference/src/bg_hook.py`, `opt/forward/fast_inference/src/bg_inproc.py`, `opt/forward/xattempt_addon/src/xa_run.py` —
    `boltzgen.model.models.boltz`, `boltzgen.task.predict.predict` (the pipeline's predict steps run in one seeded process);
  - `opt/forward/xattempt_addon/src/xa_fastinit.py` — `boltzgen.model.layers.initialize`, `boltzgen.model.layers.triangular_attention.primitives`,
    `boltzgen.model.models.boltz`;
  - `opt/forward/xattempt_addon/src/xa_hoist.py` — `boltzgen.model.layers.attention`, `boltzgen.model.modules.diffusion`, `boltzgen.model.modules.transformers`;
  - `opt/forward/fast_levers/src/fl_levers.py` — `boltzgen.model.modules.diffusion`, `boltzgen.model.modules.encoders`, `boltzgen.model.modules.transformers`;
  - `opt/forward/size_levers/src/sz_levers.py` — `boltzgen.model.models.boltz`, `boltzgen.model.modules.trunk`;
  - `opt/host/writer_levers/src/hl_levers.py` — `boltzgen.task.predict.writer`;
  - `opt/boltzgen_opt/census.py`, `opt/boltzgen_opt/stack.py`, `opt/boltzgen_opt/stock_design.py`, `opt/boltzgen_opt/design.py`,
    `opt/boltzgen_opt/weights.py` — wrap or call `boltzgen.model.models.boltz`, `boltzgen.task.predict.predict` and the `boltzgen` command-line
    entry points.
- The MIT License text with its copyright line is reproduced in `third_party_licenses/BoltzGen.MIT.txt` (byte-identical to the upstream
  `LICENSE` at the tag and to the wheel's licence member).
- Citation requested by upstream (tag `v0.3.2`, `README.md`, section "Cite"): Stark et al., "BoltzGen: Toward Universal Binder Design", 2025.
- Origin lines carried by upstream modules that kit files restate or wrap, exactly as printed at line 1 of each (wheel members
  `boltzgen/model/modules/diffusion.py`, `boltzgen/model/modules/encoders.py`, `boltzgen/model/modules/transformers.py`,
  `boltzgen/model/modules/utils.py`, `boltzgen/model/loss/diffusion.py`): "# started from code from https://github.com/lucidrains/alphafold3-pytorch,
  MIT License, Copyright (c) 2024 Phil Wang"; `boltzgen/model/modules/utils.py` line 135 additionally reads "# the following is copied from
  Torch3D, BSD License, Copyright (c) Meta Platforms, Inc. and affiliates." These notices apply to the upstream portions concerned and travel
  with the redistributed wheel and `stock/src/` copies; the kit files listed above that restate statements of the first three modules
  (`bg_graph_patch.py`, `xa_hoist.py`, `fl_levers.py`) carry this pointer to them.

## 2. Apache-2.0-licensed files inside BoltzGen

- The wheel members `boltzgen/model/layers/triangular_attention/attention.py`, `primitives.py` and `utils.py` (the first two also carried under
  `stock/src/`) bear the headers "Copyright 2021 AlQuraishi Laboratory" and "Copyright 2021 DeepMind Technologies Limited" and the notice
  "Licensed under the Apache License, Version 2.0". The upstream release carries no copy of that licence.
- The Apache License, Version 2.0 text (as published at http://www.apache.org/licenses/LICENSE-2.0) is reproduced in
  `third_party_licenses/BoltzGen.Apache-2.0.txt`.
- Kit files concerned: `opt/forward/xattempt_addon/src/xa_fastinit.py` defers the weight initialisers that `primitives.py` defines; the kit
  modes call these modules unchanged. No statement of these three files is restated in the kit's own files.

## 3. AlphaFold Protein Structure Database entry (example target structure)

- `opt/forward/fast_inference/tests/specs/PDL1_IgV_Q9NZQ7_18-132.pdb` — the target structure of the example design spec `pdl1_ref.yaml` used by
  the `README.md` commands and by `run.sh warm` — is trimmed from https://alphafold.ebi.ac.uk/files/AF-Q9NZQ7-F1-model_v6.pdb (AlphaFold Protein
  Structure Database; DeepMind and EMBL-EBI): chain A, residues 18–132, UniProt numbering kept; the source file's header records are not
  carried (line 1 of the kit file states the trim).
  It is a computed structure model (theoretical modelling) from that database, not an experimentally determined wwPDB entry; no coordinates were
  altered by the trim. The attribution, the change statement and the source notice also sit beside the file, in `opt/forward/fast_inference/tests/specs/NOTICE`.
- The source file's notice, exactly as printed there: "ALPHAFOLD DATA, COPYRIGHT (2021) DEEPMIND TECHNOLOGIES LIMITED. THE INFORMATION PROVIDED IS THEORETICAL MODELLING ONLY AND CAUTION SHOULD BE EXERCISED IN ITS USE. IT IS PROVIDED "AS-IS" WITHOUT ANY WARRANTY OF ANY KIND, WHETHER EXPRESSED OR IMPLIED. NO WARRANTY IS GIVEN THAT USE OF THE INFORMATION SHALL NOT INFRINGE THE RIGHTS OF ANY THIRD PARTY. THE INFORMATION IS NOT INTENDED TO BE A SUBSTITUTE FOR PROFESSIONAL MEDICAL ADVICE, DIAGNOSIS, OR TREATMENT, AND DOES NOT CONSTITUTE MEDICAL OR OTHER PROFESSIONAL ADVICE. IT IS AVAILABLE FOR ACADEMIC AND COMMERCIAL PURPOSES, UNDER CC-BY 4.0 LICENCE."
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/. The licence's legal code is reproduced in `third_party_licenses/AlphaFold-DB.CC-BY-4.0.txt`.
- Reference printed in the source file: Jumper et al., "Highly accurate protein structure prediction with AlphaFold", Nature 596, 583 (2021),
  doi:10.1038/s41586-021-03819-2.

## 4. Not contained in this tree

- Model weights: fetched by `./run.sh install --weights DIR` through upstream's own downloader at pinned snapshots (see `STOCK.md`); their model
  cards state `license: mit`. They are not part of this tree or of the container images built from it.
- Runtime dependencies (PyTorch, Triton, NumPy, PyTorch Lightning, cuEquivariance and the rest of `environment/requirements.lock`) are installed
  by the environment recipes from their package indexes; their licences travel with their own distributions.
