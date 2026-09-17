# Third-party notices — `chrombpnet` kit

This file lists the third-party components that the kit's own files adapt, restate, re-implement or ship in binary form, with the
licence and the copyright line of each exactly as printed in the upstream licence file, the kit files concerned, and where the verbatim
licence text is carried. Upstream ChromBPNet itself is vendored unmodified as `stock/chrombpnet-eaa0fe58.tar.gz` (tag v1.0.1) with its
own `LICENSE` inside the archive; `stock/` is never edited. No model weights are shipped with the kit.

| component | upstream | licence | copyright line (as printed upstream) | kit files concerned | verbatim licence text |
|---|---|---|---|---|---|
| ChromBPNet 1.0.1 | https://github.com/kundajelab/chrombpnet (tag `v1.0.1`, commit `eaa0fe58b6a43da62ea23b75cfc2bef4ecd3550c`) | MIT | Copyright (c) 2019 Kundaje Lab | `opt/kit_ho/tf/chrombpnet_fastkit/__init__.py` (`softmax`, verbatim from `chrombpnet/evaluation/make_bigwigs/predict_to_bigwig.py`); `opt/kit_ho/tf/chrombpnet_fastkit/h5_fast.py` (carries the source text of `predict_to_bigwig.write_predictions_h5py` and executes it unchanged); `opt/kit_ho/tf/chrombpnet_fastkit/jobwall_post.py` (restates `predict_to_bigwig.write_predictions_h5py`, `training/metrics.profile_metrics`, `training/utils/data_utils.get_cts` and the statistics block of `bigwig_helper.write_bigwig` with identical results); `opt/chrombpnet_opt/weights.py` (the one-line log-counts combination from `training/models/chrombpnet_with_bias_model.py`) | `third_party_licenses/ChromBPNet.MIT.txt` (byte-identical to `LICENSE` at the pinned commit and to the copy inside the vendored archive) |
| libBigWig, as bundled in pyBigWig 0.3.22 | https://github.com/deeptools/pyBigWig (PyPI `pyBigWig` 0.3.22; bundles https://github.com/dpryan79/libBigWig) | MIT | Copyright (c) 2015 Devon Ryan | `opt/kit_ho/tf/chrombpnet_fastkit/bigwig_numpy.py` (numpy re-implementation of libBigWig's bigWig writer, `bwWrite.c`, producing the same file bytes for the kit's call pattern); `opt/kit_ho/tf/chrombpnet_fastkit/bigwig_reader.py` (numpy reader with the semantics of libBigWig's `bwGetValues`). No libBigWig or pyBigWig source file is copied; pyBigWig 0.3.22 itself is an installed dependency of the stock stack, not redistributed here | `third_party_licenses/pyBigWig.MIT.txt` (the `LICENSE.txt` of the pyBigWig 0.3.22 source distribution; libBigWig's own `LICENSE` in that distribution names the same licence and holder) |
| Triton 3.0.0 | https://github.com/triton-lang/triton (PyPI `triton` 3.0.0) | MIT | Copyright 2018-2020 Philippe Tillet; Copyright 2020-2022 OpenAI | host objects inside the pre-compiled kernel caches `opt/kit/torch/triton_cache_sm90/` and `opt/kit/torch/triton_cache_sm89_l40s/`: `cuda_utils.so` (one per cache; compiled from `triton/backends/nvidia/driver.c`) and `__triton_launcher.so` (five and four; compiled from the launcher source that `triton/backends/nvidia/driver.py` generates). The `.cubin` / `.ptx` / IR files in the same caches are compiled from the kit's own `opt/kit/torch/chrombpnet_k1/kernels.py` by Triton 3.0.0 and ptxas 12.4. Triton itself is an installed dependency (`environment/requirements-opt-torch.lock`), not redistributed here | `third_party_licenses/Triton.MIT.txt` (the `LICENSE` of the Triton 3.0.x release branch; the PyPI wheel carries no licence file) |
| bpnet-lite | https://github.com/jmschrei/bpnet-lite (commit `b37e766bd7a2bef1614cf18d8bac38167e6f6ff5`; package metadata says 1.0.0) | MIT | Copyright (c) 2020 Jacob Schreiber | vendored unmodified under `opt/kit/torch/vendor/bpnetlite/` (the weight loader used by the Triton forward) | `third_party_licenses/bpnet-lite.MIT.txt` (byte-identical to the installed copy `opt/kit/torch/vendor/bpnet_lite-1.0.0.dist-info/licenses/LICENSE` and to upstream's `LICENSE` at that commit) |
| tangermeme 1.4.1 | https://github.com/jmschrei/tangermeme (PyPI wheel `tangermeme-1.4.1-py3-none-any.whl`) | MIT | Copyright (c) 2024-2026 Jacob Schreiber | vendored unmodified under `opt/kit/torch/vendor/tangermeme/` | `third_party_licenses/tangermeme.MIT.txt` (byte-identical to the installed copy `opt/kit/torch/vendor/tangermeme-1.4.1.dist-info/licenses/LICENSE`, the wheel's licence file) |
| typing_extensions 4.13.2 | https://github.com/python/typing_extensions (PyPI wheel `typing_extensions-4.13.2-py3-none-any.whl`) | PSF-2.0 | (the Python Software Foundation License Version 2 text carries no single copyright line; reproduced whole) | vendored unmodified as `opt/kit/torch/chrombpnet_k1/_vendor/typing_extensions.py` | `third_party_licenses/typing_extensions.PSF-2.0.txt` (byte-identical to `opt/kit/torch/chrombpnet_k1/_vendor/LICENSE.typing_extensions` beside the vendored module and to the wheel's licence file) |

The three vendored packages are carried file for file as released: the `tangermeme/` and `bpnetlite/` package trees are byte-identical
to the members of the named wheel and of the repository at the named commit, and `typing_extensions.py` to the wheel's module; nothing in
them is modified. The `*.dist-info/` directories beside them are pip's install records of that vendoring (`INSTALLER`, `REQUESTED`,
`RECORD`, `direct_url.json` are written by pip; `METADATA`, `WHEEL`, `entry_points.txt`, `top_level.txt` and `licenses/LICENSE` are the
distributions' own), and the two `VENDOR.json` files are the kit's inventory notes, not upstream files. The tangermeme wheel's bundled
documentation (`tangermeme/_skills/`, Markdown reference pages) travels with the package under the same MIT licence.

## Data files

Outside `stock/` the kit carries one data file, `opt/kit/torch/chrombpnet_k1/fixtures/l40s_counts_head_64rows.npz`: synthetic arrays drawn
from a seeded NumPy generator plus the outputs of the independent NumPy reference beside it, as `opt/kit/torch/chrombpnet_k1/fixtures/README.md`
states array by array; it contains no model weights, no model outputs and no genomic data. The pre-compiled Triton caches
(`opt/kit/torch/triton_cache_sm90/`, `opt/kit/torch/triton_cache_sm89_l40s/`, with their `JIT_IDENTITY.json` descriptions) hold compiler
output for the kit's own kernels only (row *Triton 3.0.0* above). No genome, signal track, region file or model is in this tree; STOCK.md
names the ENCODE and upstream tutorial files the kit expects and their terms.

Inside the carried upstream archive `stock/chrombpnet-eaa0fe58.tar.gz` (upstream's repository at the tag, unmodified, MIT) the data-carrying
members are upstream's own: `chrombpnet/data/ATAC.ref.motifs.txt`, `DNASE.ref.motifs.txt` and `motif_to_pwm.{ATAC,DNASE,TF}.tsv` (reference Tn5 /
DNase-I cut-site bias motifs, derived by upstream from the public datasets their headers name — GEO GSE101074, ENCODE ENCSR025EYJ — and the
marker motifs of upstream's quality checks), `chrombpnet/data/motifs.meme.txt` (a MEME-format collection of
2,193 transcription-factor motifs used by upstream's TF-MoDISco report step, whose identifiers name their origin: JASPAR matrix accessions
`MA....` — JASPAR is CC BY 4.0; HOCOMOCO v11 models `..._HUMAN.H11MO...` / `..._MOUSE.H11MO...` — HOCOMOCO is distributed under the WTFPL, which
its publishers state may be treated as CC BY; HT-SELEX-derived models named by factor and family; and upstream's bias motifs),
`chrombpnet/evaluation/custom_sequences/example.fa` (example DNA sequences for upstream's custom-sequence notebooks), upstream's
notebooks under `chrombpnet/evaluation/` and `chrombpnet/helpers/` with their saved cell outputs, and the figures under `images/`,
`chrombpnet/helpers/preprocessing/images/` and `chrombpnet/evaluation/invivo_footprints/figures/`. The kit's `pred_bw` routes read none of
them.

Runtime dependencies installed by the Setup routes (TensorFlow, PyTorch, NumPy, SciPy, h5py, pyBigWig, Triton and the rest of the pinned
stack in `environment/*.lock` and `stock/PINS.json`) are not redistributed in this tree and keep their own licences as installed.
