# Third-party notices — `e1/`

Built with Profluent-E1. Profluent-E1 is licensed under the Profluent-E1 Clickthrough License Agreement (copy: `e1/stock/src/LICENSE`);
recipients of this kit receive Profluent-E1 under that Agreement.

This kit installs and drives Profluent-E1 and adapts code of the third-party components below; the files concerned are listed per
component. The kit's own code is licensed under the Apache License 2.0 (LICENSE in this directory and at the top of the tree); this file concerns the third-party portions only. Licence texts that
must travel with the adaptations are reproduced verbatim under `third_party_licenses/`; the vendored upstream sources under `stock/`
are unmodified and keep their own licence and notice files.

## Profluent-E1 (the model code and, when fetched, the model weights)

- Upstream: https://github.com/Profluent-AI/E1, version 1.0.0, at commit `bfd2620a602248499f3d2583d85a7ecddf0b6e02` (the pin in
  `stock/PINS.json`; carried unmodified as `stock/E1-bfd2620a.tar.gz` and the reading copy `stock/src/`), installed unmodified by every
  route of this kit.
- Licence: Profluent-E1 Clickthrough License Agreement, with the Profluent-E1 Attribution Guidelines. Copies in this tree, exactly as
  released upstream: the Agreement `stock/src/LICENSE`, the notice file `stock/src/NOTICE`, the Attribution Guidelines `stock/src/ATTRIBUTION`
  (the same three files are members of `stock/E1-bfd2620a.tar.gz`). Agreement §4 states that use of the Profluent-E1 Model Code separate
  and apart from Profluent-E1 and the Profluent-E1 Model Weights is subject to the Apache License, Version 2.0
  (https://www.apache.org/licenses/LICENSE-2.0).
- Copyright line, exactly as printed in the upstream `NOTICE`: `Copyright 2025 Profluent Bio Inc.`
- Model weights: not part of this tree or of the images built from it. `./run.sh install --weights DIR` downloads the `E1-150m`, `E1-300m`
  and `E1-600m` weights from https://huggingface.co/Profluent-Bio at the revisions pinned in `stock/PINS.json`; their model cards name the
  same Agreement (`license_name: profluent-e1-clickthrough-license`). Downloading or using Profluent-E1 is acceptance of the Agreement
  (Agreement, preamble).
- Kit files concerned: the kit's modes run the installed upstream package (`E1.tools.score`, `E1Predictor`, `E1Scorer`) with functions of
  `opt/e1_opt/` and `opt/forward/engines/e1/` bound over it at run time; `opt/forward/engines/e1/kits/attn/adapter.py` imports upstream's
  `E1.model.flash_attention_utils` at run time and carries no copy of it. No upstream file is modified on disk.

## FlashAttention (BSD 3-Clause License) — origin of two adapted pieces

- Upstream: https://github.com/Dao-AILab/flash-attention · Licence: BSD 3-Clause License · Copyright line, exactly as printed in the upstream
  `LICENSE`: `Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.`
- Licence text in this tree: `third_party_licenses/Dao-AILab-flash-attention.LICENSE.txt` (the upstream `LICENSE`, verbatim).
- Upstream Profluent-E1's `E1/model/flash_attention_utils.py` (archive member and `stock/src/E1/model/flash_attention_utils.py`) states in its
  header that it is adapted from `flash_attn/bert_padding.py` of that project and is licensed under the BSD-3-Clause licence; the archive
  carries no copy of that licence text, which is therefore supplied here. The kit imports that module at run time (above).
- The Triton layer-norm kernel published as `kernels-community/triton-layer-norm` (revision pinned in `stock/PINS.json`, card licence
  `bsd-3-clause`), carried unmodified under `stock/hub_kernel/triton_layer_norm/` with the licence text `stock/hub_kernel/LICENSE`, is that
  project's `layer_norm.py` (file header: `# Copyright (c) 2024, Tri Dao.`). The kit's `add_rmsnorm` Triton kernels in
  `opt/forward/engines/e1/kits/ew1/kernels.py` and `opt/forward/engines/e1/kits/ew1_i64/kernels.py` adapt that kernel's reduction code
  (`_layer_norm_fwd_1pass_kernel` with `IS_RMS_NORM`, `HAS_RESIDUAL`); this notice and the licence text above carry the BSD conditions for them.
