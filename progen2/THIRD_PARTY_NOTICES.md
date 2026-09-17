# Third-party notices — `progen2/`

This kit patches and restates code of the third-party component below at run time; the files concerned are listed. The kit's own code is licensed under the Apache License 2.0 (LICENSE in this directory and at the top of the tree); this file
concerns the third-party portions only. Licence texts that must travel with the restated code are
reproduced verbatim under `third_party_licenses/`; the vendored upstream sources under `stock/` are unmodified and keep their own licence file.

## ProGen2 (`salesforce/progen`, subdirectory `progen2/`)

- Upstream: https://github.com/salesforce/progen at commit `c27a419c234a0997923761e1fe7daffcebf0eaf5` (no tag, no package metadata; the pin
  in `stock/PINS.json`; carried unmodified as `stock/progen-c27a419c.tar.gz` and the reading copy `stock/src/`)
- Repository licence: BSD 3-Clause License. Copyright line, exactly as printed in the upstream `LICENSE.txt`: `Copyright (c) 2022, Salesforce.com, Inc.`. Licence text in this
  tree: `third_party_licenses/ProGen2.BSD-3-Clause.txt`, a byte-identical copy of the upstream `LICENSE.txt` at that commit (the same text is
  `stock/src/LICENSE.txt`). Upstream's `README.md` states "Our code and models are BSD-3 licensed. See LICENSE.txt for details."
- Two upstream files carry an Apache License 2.0 header instead: `progen2/models/progen/modeling_progen.py` and `configuration_progen.py`
  (header, exactly as printed: `Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.` / `Licensed under the Apache License,
  Version 2.0`). The upstream repository ships no copy of that licence; `third_party_licenses/ProGen2.Apache-2.0.txt` supplies the Apache
  License 2.0 text as published in the `LICENSE` file of the `transformers` project (https://github.com/huggingface/transformers, first line
  `Copyright 2018- The Hugging Face team. All rights reserved.`), from whose GPT-J modeling code those two files descend; the same bytes are `esmfold2/third_party_licenses/transformers_fork.Apache-2.0.txt` in this release.
- Upstream ships no `NOTICE` file.

| Kit file | Relation to the upstream code |
|---|---|
| `opt/forward/engines/progen2/kits/v0_ew/patches.py` | replacement `forward` functions bound over the installed `modeling_progen` classes at run time; they reproduce statement runs of upstream `ProGenAttention.forward` and `ProGenAttention._attn` around the kit's fused element-wise kernel calls |
| `opt/serving/pipeline_v0_4/components/plm_transfer_t1_v0/kit_t1.py` | the same kind of replacement bodies for the serving pipeline's transfer step |
| `opt/serving/pipeline_v0_4/sampler_exact.py` (the exact sampler) and the one-read checkpoint loader `opt/serving/pipeline_v0_4/oneread_loader.py`, `opt/serving/pipeline_v0_4/progen2_decode.py`; `opt/progen2_opt/` (`generate.py`, `score.py`, `stack.py` and the other modules) | the sampler and loader re-implement, by behaviour, the sampling-loop bookkeeping and checkpoint loading that upstream obtains from `transformers` 4.16.2 (no statement of that package is copied); `progen2_opt` drives the installed upstream package otherwise unchanged |

No upstream file is modified on disk.

## Binaries and weights

- `opt/forward/engines/progen2/kits/v0_ew/libew_progen2.so` is compiled from the kit's own `kernels.cu` beside it (`BUILD.json` records the
  source and compiler; `SHA256SUMS` the digest checked before it is loaded); it links the CUDA runtime dynamically (`libcudart.so.12`, not
  redistributed) and embeds no third-party source or library beyond the device support routines the CUDA compiler itself emits into every
  binary (NVIDIA CUDA Toolkit EULA, https://docs.nvidia.com/cuda/eula/).
- Data files: the vendored archive and `stock/src/` carry upstream's `progen2/tokenizer.json` (the tokenizer definition, part of the BSD-3-licensed
  repository) and no sequence database, alignment or weight file; `opt/serving/pipeline_v0_4/components.json` and the JSON records under the
  tests are written for this kit.
- Model checkpoints (`progen2-small` … `progen2-xlarge`) are not part of this tree or of the images built from it: `./run.sh install --weights DIR`
  fetches them from the upstream addresses pinned in `stock/PINS.json` (sha256 checked). Upstream states they are BSD-3 licensed (sentence
  quoted above) and asks users to cite Nijkamp et al., "ProGen2: Exploring the Boundaries of Protein Language Models", as printed in the
  upstream `README.md`.
