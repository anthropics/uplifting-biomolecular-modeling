# Third-party notices — `chai1` kit

This kit runs the Chai-1 release `chai_lab` 0.6.1, which it carries unmodified under `stock/`, and re-binds parts of it at run
time. The sections below list every third-party component that the kit's own files under `opt/` adapt, restate or wrap, and
the third-party components inside the carried upstream release whose licence texts that release does not itself include:
component name, upstream URL, licence name, the copyright line exactly as printed in the upstream licence text, the files
concerned, and where the verbatim licence text sits (`third_party_licenses/`). No file under `stock/` is edited; texts the
upstream release lacks are supplied here, beside it. The kit's own code is licensed under the Apache License 2.0 (LICENSE
in this directory and at the top of the tree); this file concerns the third-party portions only.

## Chai-1 (`chai_lab` 0.6.1)

- Upstream: https://github.com/chaidiscovery/chai-lab, tag `v0.6.1` (commit `8d5ac0f93e9b6ea4c3a6545c253a6381c0f3694b`);
  carried verbatim as `stock/chai_lab-0.6.1-py3-none-any.whl` (the PyPI release file) and `stock/src/` (the tag's source tree).
- Licence: Apache License, Version 2.0 — the upstream `LICENSE` file (`stock/src/LICENSE`; the same bytes inside the wheel at
  `chai_lab-0.6.1.dist-info/licenses/LICENSE`). No `NOTICE` file is published upstream at this tag. The upstream README states:
  "Chai-1 is released under an Apache 2.0 License (both code and model weights), which means it can be used for both academic
  and commerical purposes, including for drug discovery." (`stock/src/README.md`, §Licence, quoted as printed).
- Copyright line: `Copyright 2024 Chai Discovery` (as printed in the upstream `LICENSE`); the upstream source-file headers read
  `Copyright (c) 2024 Chai Discovery, Inc.`.
- Kit files concerned:
  - `opt/forward/eager_trunk/chai1_eager/` (`trunk.py`, `ts2eager.py`, `hoist.py`, `stack.py`, `kernels.py`, `msa_kernels.py`,
    `transition_core.py`, `__init__.py`) — re-express Chai-1's exported TorchScript model components as eager PyTorch. Stated
    plainly: these files mirror the module structure, submodule names, operation order and numerics of the exported TorchScript
    archives (`models_v2/*.pt`) that Chai Discovery distributes as the Chai-1 weights — that structure was read from those archives,
    not from chai-lab source — and `ts2eager.py` reads TorchScript code out of the same archives at run time and executes it as
    eager PyTorch. They are to that extent derivative of the Chai-1 model release. Licence terms of those archives (the weights):
    Chai Discovery releases Chai-1 "under an Apache 2.0 License (both code and model weights)" (`stock/src/README.md` §Licence, quoted
    in full above); no separate weights licence file is published at the tag, and the archives themselves are not in this tree
    (`STOCK.md` §Pin: fetched by `run.sh install --weights` from Chai Discovery's download location). Per-directory notice:
    `opt/forward/eager_trunk/NOTICE`.
  - `opt/forward/dstep_megakernel/chai1_fastln/` (`stackx.py`, `aoti.py`, `dit_attn.py`, `cpu_isa.py`, `__init__.py`) — compile
    and re-bind the denoiser-step functions that `chai1_eager/ts2eager.py` transpiles at run time from the same exported
    archives, and so rest on the same archive-derived structure and the same Apache-2.0 weights terms; the directory contains no
    chai-lab source and no model weights.
  - `opt/forward/errata_02/chai_worker.py` — restates the per-seed trunk-sample loop of `chai_lab/chai1.py` (`run_inference`);
    `opt/forward/fast_inference/kit/chai_proto.py` drives and wraps that inference loop. Per-directory notice:
    `opt/forward/fast_inference/NOTICE` (stays as it is).
  - `opt/chai1_opt/postproc.py` — restates statements of `chai_lab/ranking/clashes.py` (the clash census used in ranking) and
    wraps `chai_lab.chai1.rank`, `save_to_cif` and `plot_msa` at run time.
  - `opt/chai1_opt/settings.py`, `opt/chai1_opt/stock_fold.py` — reproduce the option names and defaults of
    `chai_lab.chai1.run_inference` / `chai_lab/main.py` (read from the carried source; no function body is copied).
  - `opt/forward/chai1_exactln/serve.py` — re-binds the layer-normalisation statements of the eager trunk above.
  - The other modules under `opt/chai1_opt/` re-bind `chai_lab` functions at run time (`load_exported`,
    `make_all_atom_feature_context`, `load_chains_from_raw`, the ESM embedding context) and contain no upstream code.
- Licence text: `third_party_licenses/chai-lab.Apache-2.0.txt`, byte-identical to `stock/src/LICENSE` (and to
  `opt/forward/eager_trunk/LICENSE`, `opt/forward/fast_inference/LICENSE`).
- Upstream citation request, as printed in the upstream README (`stock/src/README.md`, §Citations): "If you find Chai-1 useful in
  your research or use any structures produced by the model, we ask that you cite our technical report:"

  ```
  @article{Chai-1-Technical-Report,
  	title        = {Chai-1: Decoding the molecular interactions of life},
  	author       = {{Chai Discovery}},
  	year         = 2024,
  	journal      = {bioRxiv},
  	publisher    = {Cold Spring Harbor Laboratory},
  	doi          = {10.1101/2024.10.10.615955},
  	url          = {https://www.biorxiv.org/content/early/2024/10/11/2024.10.10.615955},
  	elocation-id = {2024.10.10.615955},
  	eprint       = {https://www.biorxiv.org/content/early/2024/10/11/2024.10.10.615955.full.pdf}
  }
  ```

  The same README adds: "Additionally, if you use the automatic MMseqs2 MSA generation described above, please also cite:" the
  ColabFold reference printed there (`stock/src/README.md`).
- Model weights and the conformer library are not part of this tree; `STOCK.md` says where `./run.sh install --weights DIR`
  fetches them from (Chai Discovery's download location, the URLs of the carried source) and how they are verified.

## ESM-2 (`fair-esm`)

- Upstream: https://github.com/facebookresearch/esm (ESM-2 model `esm2_t36_3B_UR50D`).
- Licence: MIT.
- Copyright line: `Copyright (c) Meta Platforms, Inc. and affiliates.`
- How it enters: `chai_lab` 0.6.1 computes protein language-model embeddings with a traced export of ESM-2 3B
  (`stock/src/chai_lab/data/dataset/embeddings/esm.py`: `esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt`), which
  `./run.sh install --weights DIR` fetches from Chai Discovery's download location into `CHAI_DOWNLOADS_DIR/esm/`. Neither the
  carried `chai_lab` release nor this tree contains ESM source code or that file.
- Kit files concerned: `opt/forward/errata_02/chai_worker.py` keeps the loaded ESM-2 module resident across inputs and memoises
  its per-sequence embeddings (levers `W1`, `W5`); `opt/chai1_opt/weights.py` fetches and verifies the file through upstream's
  own download routine. No ESM code is restated.
- Licence text: `third_party_licenses/ESM-fair-esm.MIT.txt` (the upstream `LICENSE`, verbatim). The MIT licence asks that its
  copyright and permission notice accompany copies or substantial portions of the software; it is supplied here because the
  carried upstream release does not include it.

## ColabFold

- Upstream: https://github.com/sokrypton/ColabFold.
- Licence: MIT.
- Copyright line: `Copyright (c) 2021 Sergey Ovchinnikov`
- Files concerned: the carried upstream module `stock/src/chai_lab/data/dataset/msas/colabfold.py` (the same module inside
  `stock/chai_lab-0.6.1-py3-none-any.whl`) states, as printed: "N.B. this function (and this function only) is copied from
  https://github.com/sokrypton/ColabFold and follows the license in that repository / We have made modifications to how
  templates are returned from this function." That function belongs to chai-lab's MSA-server client
  (`--use-msa-server`). No file of this kit restates it; the carried module is not edited.
- Licence text: `third_party_licenses/ColabFold.MIT.txt` (the upstream `LICENSE`, verbatim), supplied beside the carried release
  because that release does not include it.

## Also inside the carried upstream release (headers retained there)

- `stock/src/chai_lab/data/residue_constants.py` and `stock/src/chai_lab/tools/rigid.py` carry, as printed, the header
  `Copyright 2021 AlQuraishi Laboratory` / `Copyright 2021 DeepMind Technologies Limited` — "Licensed under the Apache License,
  Version 2.0". The Apache License 2.0 text they name is the text reproduced in `third_party_licenses/chai-lab.Apache-2.0.txt`
  (whose appendix line is Chai Discovery's).

## Data files

- Outside `stock/`: `opt/forward/fast_inference/tests/public_inputs/1BRS_{1to1,2to2,3to3,4to4}.fasta` — the protein sequences of wwPDB
  entry 1BRS (barnase and barstar; wwPDB archive data, CC0 1.0) written 1 to 4 times each as chai-lab FASTA inputs; `PACK.json` beside
  them states the source per file. They are single sequences: no alignment, template or embedding file is shipped, and the tree
  contains no sequence database or data derived from one.
- Inside `stock/`: the wheel and `stock/src/` hold chai-lab's Python sources, `py.typed`, `LICENSE`, `README.md`, `pyproject.toml`,
  `requirements.in` and `Dockerfile.chailab` — no structures, weights, alignments or other data.

Run-time dependencies (PyTorch, Triton, NVIDIA cuEquivariance, RDKit, gemmi and the rest of `environment/requirements.lock`) are
installed from their own distributions under their own licences and are not redistributed in this tree. The shared kernel library
this kit binds at run time (`../common/opt_core`) carries its own notices.
