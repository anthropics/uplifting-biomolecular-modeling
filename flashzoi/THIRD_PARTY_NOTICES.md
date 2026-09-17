# Third-party notices — `flashzoi`

The kit's own code is licensed under the Apache License 2.0 (LICENSE in this directory and at the top of the tree); this file
concerns the third-party portions only.

The kit ships one third-party component, unmodified, under `stock/`, and uses it at run time (`stock/check_pins.py` and
`stock/PINS.json` beside it are the kit's own files). No file outside `stock/` is a copy or a modified copy of an upstream file;
the kit files that restate the call structure of upstream functions so as to reproduce their outputs are named under "Kit files
that follow upstream code" below. Two data files are carried: the track table inside the upstream wheel and the kit's warm-up
input window; both have a section below.

## borzoi-pytorch 0.5.1

- Upstream: https://github.com/johahi/borzoi-pytorch (tag `v0.5.1`, commit `8a05eb15dd56871806771b2870c98a0c1dd9e0c8`);
  the same release is published on PyPI as `borzoi-pytorch` 0.5.1.
- Licence: Apache License, Version 2.0. The upstream `LICENSE` file is the Apache License 2.0 text and carries no
  project-specific copyright line; no `NOTICE` file is published upstream at this commit. The per-file notices inside the package
  read, exactly as printed upstream: `# Copyright 2023 Calico LLC` (`pytorch_borzoi_model.py`) and `# Copyright 2022 Calico LLC`
  (`gene_utils.py`, which also states `# adapted from calico/baskerville`), each followed by the Apache-2.0 header;
  `#Copyright (c) 2021 Phil Wang, 2024 Johannes Hingerl` (`pytorch_borzoi_transformer.py`, which states
  `#Adapted from https://github.com/lucidrains/enformer-pytorch/tree/main` and carries the MIT License text in full) and
  `#Copyright (c) 2021 Phil Wang` (`pytorch_borzoi_utils.py`, which carries the MIT License text in full). `__init__.py`,
  `config_borzoi.py` and `pytorch_borzoi_helpers.py` print no notice of their own and fall under the repository `LICENSE`.
- Kit files concerned: `stock/borzoi_pytorch-0.5.1-py3-none-any.whl` — the upstream wheel as published, which carries the
  licence at `borzoi_pytorch-0.5.1.dist-info/licenses/LICENSE` — and `stock/src/borzoi_pytorch/`, the wheel's package files
  unpacked for reading, byte-identical to the wheel members and to the repository files at the commit above.
  `environment/Dockerfile` (and the Apptainer definition built from it) installs that wheel into the image; the kit's modes call
  the installed package and leave its files unchanged. The MIT-licensed portions travel with their licence text, which upstream
  places at the head of the two files named above; no separate MIT licence file exists upstream for them.
- Verbatim licence text: `third_party_licenses/borzoi-pytorch.Apache-2.0.txt`, byte-identical to the upstream `LICENSE` at the
  commit above and to the copy inside the wheel.

### Kit files that follow upstream code

These files are written for the kit and contain no upstream function body; each names in its docstring the upstream function
whose signature, argument names, operation order or single expressions it restates (all from borzoi-pytorch 0.5.1 above,
Apache-2.0; `pytorch_borzoi_model.py` carries `# Copyright 2023 Calico LLC`, `pytorch_borzoi_helpers.py` carries no header):

- `opt/forward/kits_v1_25/engines/flashzoi/kits/v1_25/fz_exact.py` — `FastNCHW.__call__` applies the loaded model's own modules
  in the operation order of upstream `Borzoi.forward` and `Borzoi.get_embs_after_crop` (`pytorch_borzoi_model.py`), with the
  kit's Triton kernels at the fused sites, and restates the `+ 0 * <other head>(x).sum()` expression of the
  `data_parallel_training` form; `transformer_fused_forward` walks the model's transformer blocks in their upstream layer order.
- `opt/forward/kits_v1_25/engines/flashzoi/kits/v1_25/_wrap.py` — `forward_any` keeps the signature and head selection of
  upstream `Borzoi.forward(x, is_human=True, data_parallel_training=False, return_embeddings=False)`; `embs_after_crop` keeps that
  of `Borzoi.get_embs_after_crop(x)`; `predict_tracks` keeps the signature of
  `borzoi_pytorch.pytorch_borzoi_helpers.predict_tracks(models, sequence_one_hot, slices)` and delegates to the file below.
- `opt/forward/kits_v1_25/engines/flashzoi/predict_tracks_fast.py` — same signature and returned array as upstream
  `predict_tracks` (`pytorch_borzoi_helpers.py`, lines 4-13); its per-fold call
  `models[k](sequence_one_hot[None, ...])[:, None, ...]` and the `np.concatenate(...).swapaxes(3, 2)` result layout are restated,
  the host-side copy is the kit's own; the module docstring restates upstream's line 8 (with shortened names) for reference.
- `opt/flashzoi_opt/stock_pred.py` — the stock route: loads the replicates with the upstream `README.md` call
  `Borzoi.from_pretrained('johahi/flashzoi-replicate-<k>')`, runs them under `torch.autocast("cuda")` as that `README.md` requires,
  and calls the package's `predict_tracks` function; these lines are quoted in its docstring and in `STOCK.md`.

## Data file `stock/src/borzoi_pytorch/precomputed/targets.txt` (also the wheel member `borzoi_pytorch/precomputed/targets.txt`)

- What it is: the tab-separated track table the package reads at import (`pytorch_borzoi_model.py`, `TRACKS_DF`) and describes
  as "the original targets.txt from Borzoi": one row per human output track of the model (7,611 rows) with the columns
  `identifier`, `file`, `clip`, `clip_soft`, `scale`, `sum_stat`, `strand_pair`, `description` — experiment accessions (ENCODE
  `ENCFF…`, FANTOM5 CAGE `CNhs…`, GEO `GSM…`, GTEx `GTEX-…` and others), per-track scaling and clipping constants, free-text
  descriptions, and the upstream authors' local file paths in the `file` column. It holds no sequence, coverage or other assay
  data.
- Source: shipped by borzoi-pytorch 0.5.1 as package data, byte-identical to the repository file at the commit above; the
  package's `README.md` names the Borzoi repository (https://github.com/calico/borzoi) as the original implementation, and the
  file is byte-identical to `examples/targets_human.txt` of that repository.
- Licence: distributed under the Apache License 2.0 of borzoi-pytorch (the wheel's `LICENSE`); the Borzoi repository it
  originates from is also under the Apache License 2.0 (its `LICENSE` file, the licence text without a project-specific copyright
  line). The file itself prints no copyright line. Carried unmodified, inside `stock/` only.

## Data file `opt/forward/kits_v1_25/canary_window_0.npz`

- What it is: the kit's warm-up input — one model input window, no model outputs and no expected results: a NumPy archive with
  one array under the key `x`, shape `(4, 524288)`, dtype `uint8`, the one-hot encoding (rows A, C, G, T; every column has exactly
  one 1) of a genomic sequence. `opt/flashzoi_opt/warm.py` runs one forward on it to fill the kernel cache;
  `opt/forward/kits_v1_25/SOURCES.md` sits beside it.
- Origin: the human reference genome assembly GRCh38 (UCSC `hg38`), chr1:143,315,785-143,840,073 as a 0-based, end-exclusive
  interval (524,288 bp), whose central 196,608 bp are the chr1:143,479,625-143,676,233 named in `opt/forward/kits_v1_25/README.md`;
  the encoded bases equal the assembly sequence over the whole window (soft-masking is not represented: upper- and lower-case
  bases encode alike). No other data set contributed to the file.
- Terms: GRCh38 is public reference data of the Genome Reference Consortium, released through GenBank; no licence text
  accompanies it. The UCSC Genome Browser distributes the same assembly sequence as `hg38`, and the `README.txt` beside its `hg38`
  sequence files states, under "GenBank Data Usage", that NCBI places no restrictions on the use or distribution of the GenBank
  data, while some submitters may claim rights in portions of the data they submitted. Terms for UCSC-hosted annotation,
  conservation and alignment tracks are set separately by UCSC and the track providers; the kit reads, fetches and points at no
  such track — the window above is assembly sequence only.

## Not carried here

The model weights (`johahi/flashzoi-replicate-{0,1,2,3}` on Hugging Face, pinned by revision and digest in `stock/PINS.json`)
are not part of this tree; `STOCK.md` describes how they are fetched or supplied and names their terms (the model cards state
`license: mit` at the pinned revisions). The run-time stack listed in `environment/requirements.lock` (PyTorch, Triton,
flash-attn, transformers, huggingface_hub, the `nvidia-*-cu12` CUDA libraries and the rest) is installed at image-build or
install time from PyPI and from the flash-attention release page under those projects' own licences; none of it is carried in
this directory.
