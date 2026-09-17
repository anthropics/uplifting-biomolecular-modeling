# Third-party notices — Boltz-2 optimization kit

This kit wraps stock Boltz-2 at run time and ships no third-party source tree of its own besides the unmodified upstream
release under `stock/` (which carries upstream's own licence and notices). Several kit files under `opt/` restate or adapt
statements of third-party code so that a patched forward pass keeps the original arithmetic; those components are listed
here with their licences. The verbatim licence texts are in `third_party_licenses/`. The per-directory notices
`opt/forward/dit_hoist/NOTICE.md` and `opt/forward/trunk_levers/NOTICE` remain in place and are consistent with this file.
The kit's own code is licensed under the Apache License 2.0 (LICENSE in this directory and at the top of the tree);
this file concerns the third-party portions only.

| Component | Upstream | Licence | Copyright line (as printed upstream) | Kit files concerned | Licence text |
|---|---|---|---|---|---|
| Boltz (version 2.2.1) | https://github.com/jwohlwend/boltz | MIT | Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro | Files that restate or adapt Boltz statements: `opt/forward/dit_hoist/src/boltz_dit_hoist.py`, `opt/forward/dit_hoist/src/boltz_graph_patch.py`, `opt/forward/dit_hoist/src/make_worker_variant.py`, `opt/forward/trunk_levers/boltz_trunk_levers.py`, `opt/forward/trunk_levers/boltz_flash_triattn_patch.py`, `opt/forward/trunk_levers/src/bz_worker_lev.py`, `opt/forward/trunk_levers/src/bz_worker_levf2.py`, `opt/forward/atom/src/boltz_atom.py`, `opt/forward/msa2/exact_hoist.py`, `opt/forward/pairfuse/bz_pairfuse.py`, `opt/forward/waste/boltz_waste_levers.py`, `opt/forward/waste/aligncap.py`, `opt/forward/graph/boltz_graph_trunk.py`, `opt/forward/sampler/src/bz_sampler.py`, `opt/forward/conf/template_levers.py`, `opt/boltz2_opt/rowpair_heads.py`, `opt/boltz2_opt/rowpair.py`, `opt/boltz2_opt/rowpair_msa.py`, `opt/boltz2_opt/msa2_probe.py`, `opt/boltz2_opt/affinity_leg.py`, `opt/boltz2_opt/waste.py`, `opt/boltz2_opt/kernels.py`, `opt/boltz2_opt/registry.py`, `opt/boltz2_opt/modes.py`, `opt/boltz2_opt/settings.py`, `opt/boltz2_opt/skipped.py`, `opt/boltz2_opt/stack.py`, and the tests under `opt/boltz2_opt/tests/` that transcribe stock statements to compare against them. Each such file names the upstream source location it restates. | `third_party_licenses/Boltz.MIT.txt` (identical to `stock/src/LICENSE`) |
| alphafold3-pytorch | https://github.com/lucidrains/alphafold3-pytorch | MIT | Copyright (c) 2024 Phil Wang | The upstream Boltz modules `boltz/model/modules/diffusionv2.py`, `encodersv2.py`, `transformersv2.py` and `utils.py` carry the header "started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang"; the kit files that restate statements of those modules (`opt/forward/dit_hoist/src/*.py`, `opt/forward/atom/src/boltz_atom.py`, `opt/forward/sampler/src/bz_sampler.py`, `opt/boltz2_opt/rowpair_heads.py`) carry that attribution with them. | `third_party_licenses/alphafold3-pytorch.MIT.txt` |
| OpenFold-derived triangle attention, as carried in Boltz (`boltz/model/layers/triangular_attention/`) | https://github.com/aqlaboratory/openfold (via https://github.com/jwohlwend/boltz) | Apache-2.0 | Copyright 2021 AlQuraishi Laboratory; Copyright 2021 DeepMind Technologies Limited (file headers) | `opt/forward/trunk_levers/boltz_flash_triattn_patch.py` wraps `boltz.model.layers.triangular_attention` and restates the opening statements of `TriangleAttention.forward` from `attention.py`; no Apache-licensed source file is redistributed by the kit outside the unmodified upstream release under `stock/`. Changes relative to the original: the attention computation after those statements is replaced by the kit's fused kernel call. | `third_party_licenses/Boltz.Apache-2.0.txt` (the Apache License, Version 2.0) |
| PyTorch3D | https://github.com/facebookresearch/pytorch3d | BSD-3-Clause | Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved. | `opt/forward/sampler/src/bz_sampler.py` restates two statements of `random_quaternions`, which upstream Boltz's `boltz/model/modules/utils.py` marks "copied from Torch3D, BSD License, Copyright (c) Meta Platforms, Inc. and affiliates." The copyright notice, conditions and disclaimer are reproduced in the licence text named at right. | `third_party_licenses/PyTorch3D.BSD-3-Clause.txt` |

Third-party code outside `stock/`: none is carried as a file of its own — every file outside `stock/` and `third_party_licenses/` was written
for this kit; the files in the table restate upstream statements inside kit code and say where. Data files: `inputs/*.yaml` (five Boltz input
files) hold the protein sequences of wwPDB entry 1BRS (barnase, barstar; wwPDB archive data, CC0 1.0) and of the designed Trp-cage
miniprotein (wwPDB entry 1L2Y, CC0 1.0) in the copy counts their names state, each chain with `msa: empty` (single-sequence input): no
alignment, template or structure file is shipped and the tree contains no database-derived sequence. The wheel and the `git archive` under
`stock/` hold Boltz's Python sources, `LICENSE`, `README.md` and `pyproject.toml` only — no weights, no CCD or molecule library (those are the
cache files `run.sh install --weights` fetches, STOCK.md §Pin) and no data.

Runtime dependencies (PyTorch, Triton, NumPy, pandas, PyTorch Lightning, cuEquivariance, gemmi and the rest of the pinned stack in
`environment/requirements.lock`) are installed from the package index by the recipes under `environment/` under their own licences;
none of them is carried in this tree. Model weights are fetched by upstream's own downloader at install time and are not part of the
tree or of an image built from it.

Citation: upstream asks that uses of the Boltz code or models in research cite the Boltz-1 and Boltz-2 papers (see the upstream
README, carried in `stock/boltz-cb04aecc.tar.gz`).
