"""Pins for the E1 kits (every value read from bytes, cited; asserted at apply).

The stock is the authors' package at the pinned commit; its forward is fp32 weights under
``torch.autocast('cuda', bfloat16)`` (``tools/score.py`` + ``predictor.py``). The hub RMSNorm kernel the package
downloads at import is part of the stock and is pinned here with its Triton autotune choice (the DET recipe's
``triton_autotune_pin``): the kernel's ``@triton.autotune`` over ``num_warps`` picks by a timing run at the
first call, so two cold processes differ unless the choice is pinned; the pin selects one of the stock's own outcomes.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as _md
import os

STOCK = {"repo": "https://github.com/Profluent-AI/E1", "commit": "bfd2620a602248499f3d2583d85a7ecddf0b6e02",
         "package": "E1", "version": "1.0.0",
         "cite": "git log (commit 2026-03-11 'update python version', the tip of main; no tags)"}

#: the pinned dependency stack (pixi.lock of the stock repo; flash-attn from the Dao-AILab release wheel)
STACK = {"torch": "2.8.0", "torch_version_str": "2.8.0+cu128", "cuda": "12.8", "triton": "3.4.0", "transformers": "4.56.2",
         "tokenizers": "0.22.1", "kernels": "0.11.0", "flash_attn": "2.8.3.post1", "python": "3.12",
         "cite": "pixi.lock @ bfd2620a (linux-64 pypi pins; the PyPI torch 2.8.0 wheel: pip metadata '2.8.0', torch.__version__ "
                 "'2.8.0+cu128', torch.version.cuda '12.8'); "
                 "flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl "
                 "(github.com/Dao-AILab/flash-attention release v2.8.3.post1, 256,040,449 B)"}

WEIGHTS = {
    "150m": {"repo": "Profluent-Bio/E1-150m", "rev": "c4dbfe827e4aa6ed7f95eaef50dc1e084f4d77dc",
             "sha256": "ba2656339005e6598642836acfdafde480fecc7e145ce0058eb54adf572c3484", "bytes": 308869268, "hidden_size": 768},
    "300m": {"repo": "Profluent-Bio/E1-300m", "rev": "5a2871c587eadbcc9237bc686ea45e5b4d28dfb3",
             "sha256": "31e09a2542f45b04e6ce4adafb3b657f21e2d56d12bf68fd2266b1576a80bc9b", "bytes": 548657620, "hidden_size": 1024},
    "600m": {"repo": "Profluent-Bio/E1-600m", "rev": "52d959fb87a609d15cf223a485127b29ed5c382a",
             "sha256": "cfc108d4b98baaa62932331b40be265eae39dc382595bc3cde4a5ab55db1bf7a", "bytes": 1282911716, "hidden_size": 1280},
}
WEIGHTS_CITE = ("huggingface.co/api/models/<repo> 2026-08-25 (sha = the repo revision; model.safetensors LFS oid = sha256 of the file, "
                "re-hashed on disk after download)")

KERNEL = {"repo": "kernels-community/triton-layer-norm", "rev": "5ebc83aa387c282ff3f233bc3022c3a8be33a013",
          "layer_norm_py_sha256": "e7a4a0000fc6254431adfbc99455ed838c2829d181962f7fa0efb4fbd371d529", "license": "bsd-3-clause",
          "cite": "E1/modeling.py L23 get_kernel('kernels-community/triton-layer-norm') (revision main, resolved 2026-08-25 07:50Z); "
                  "build/torch-universal/triton_layer_norm/layer_norm.py (42,646 B)"}

#: the DET recipe's pin per GPU class and size = the MODE of the stock kernel's own autotune picks at the (1, 202) first-call shape,
#: censused in cold processes per class; any censused W is one of the stock's own outcomes; two processes match bitwise only at equal W
TRITON_AUTOTUNE_PIN = {
    "NVIDIA H100 80GB HBM3": {"150m": 1, "300m": 8, "600m": 8, "first_call_shape": (1, 202),
                              "cite": "the pooled mode of two independent 10-process censuses at the (1, 202) first call: 150m 1 = 14/20, "
                                      "300m 8 = 12/20 (a 4-4 tie with 16 in one census, 8/10 in the other), 600m 8 = 12/20; "
                                      "any censused W is exact by selection; comparisons need equal W"},
    'NVIDIA H200': {"150m": 1, "300m": 8, "600m": 8, "first_call_shape": (1, 202),
                    "cite": "the MODE of this class's censuses at (1, 202), pooled add-only (W_CELLS); ties by the mean warm forward wall"},
    'NVIDIA B200': {"150m": 4, "300m": 16, "600m": 32, "first_call_shape": (1, 202),
                    "cite": "the MODE of this class's censuses at (1, 202), pooled add-only (W_CELLS); ties by the mean warm forward wall EXCEPT 300m: a 3/3 tie 8/16 where 16 is kept (the class's test records ran at 16)"},
    'NVIDIA A100-SXM4-80GB': {"150m": 4, "300m": 4, "600m": 4, "first_call_shape": (1, 202),
                    "cite": "the MODE of this class's censuses at (1, 202), pooled add-only (W_CELLS); ties by the mean warm forward wall"},
    'NVIDIA L40S': {"150m": 1, "300m": 1, "600m": 16, "first_call_shape": (1, 202),
                    "cite": "the MODE of this class's censuses at (1, 202), pooled add-only (W_CELLS); ties by the mean warm forward wall"},
}


#: THE ONE W TABLE: one cell per (gpu, size, FIRST-CALL SHAPE) ->
#: {pin = the MODE of the stock's own draws (ties broken by the mean wall time), the outcome set with counts, n draws, the cite}.
#: The first-call shape is the TRUE shape of the kernel's first call: for the documented command (E1.tools.score, masked marginal)
#: = (min(#unique mutated positions, max_batch_tokens // T) masked-parent rows, T = parent length + 4) — scorer.py L83-96/L183-199:
#: one masked copy of the parent per mutated position, batched under max_batch_tokens; NOT the number of mutants (a cell that
#: carries ``label_shape_was`` keeps the label B_first = min(n_mutants, 65536 // T) it first had; each census cell is one assay = one
#: true shape, so its outcome set stands under either key). The (1, 202) cell and the probe cells come from cold-process
#: censuses per class; the CLI cells from censuses of the documented command (H100 by device name; a100/b200/l40s by slug).
#: Add-only: a later census appends cells, never edits counts. A cell exists ONLY where censused: w_cell raises PinUncensused BY NAME
#: for any other (gpu, size, shape); apply_autotune_pin serves it the default row of its compute-capability class (UNPINNED_LINE).
W_CELLS = [
 {
  "assay": "probe",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 12 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 1,
   "4": 11
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "cold processes, one forward each at this first-call shape (DET or production numerics)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "mean_wall_s_by_W": None,
  "n": 12,
  "pin": 4,
  "pin_rule": "mode",
  "shape": [
   16,
   256
  ],
  "size": "150m"
 },
 {
  "assay": "probe",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 12 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 12
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "cold processes, one forward each at this first-call shape (DET or production numerics)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "mean_wall_s_by_W": None,
  "n": 12,
  "pin": 1,
  "pin_rule": "mode",
  "shape": [
   16,
   256
  ],
  "size": "300m"
 },
 {
  "assay": "probe",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 12 cold processes (picks by num_warps in counts)",
  "counts": {
   "4": 8,
   "8": 4
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "cold processes, one forward each at this first-call shape (DET or production numerics)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "mean_wall_s_by_W": None,
  "n": 12,
  "pin": 4,
  "pin_rule": "mode",
  "shape": [
   16,
   256
  ],
  "size": "600m"
 },
 {
  "assay": "probe",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 12 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 3,
   "4": 6,
   "8": 3
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "cold processes, one forward each at this first-call shape (DET or production numerics)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "mean_wall_s_by_W": None,
  "n": 12,
  "pin": 4,
  "pin_rule": "mode",
  "shape": [
   4,
   256
  ],
  "size": "150m"
 },
 {
  "assay": "probe",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 12 cold processes (picks by num_warps in counts)",
  "counts": {
   "4": 3,
   "8": 9
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "cold processes, one forward each at this first-call shape (DET or production numerics)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "mean_wall_s_by_W": None,
  "n": 12,
  "pin": 8,
  "pin_rule": "mode",
  "shape": [
   4,
   256
  ],
  "size": "300m"
 },
 {
  "assay": "probe",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 12 cold processes (picks by num_warps in counts)",
  "counts": {
   "8": 12
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "cold processes, one forward each at this first-call shape (DET or production numerics)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "mean_wall_s_by_W": None,
  "n": 12,
  "pin": 8,
  "pin_rule": "mode",
  "shape": [
   4,
   256
  ],
  "size": "600m"
 },
 {
  "assay": "probe",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 12 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 6,
   "16": 1,
   "8": 5
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "cold processes, one forward each at this first-call shape (DET or production numerics)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "mean_wall_s_by_W": None,
  "n": 12,
  "pin": 1,
  "pin_rule": "mode",
  "shape": [
   1,
   256
  ],
  "size": "150m"
 },
 {
  "assay": "probe",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 12 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 1,
   "16": 2,
   "4": 4,
   "8": 5
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "cold processes, one forward each at this first-call shape (DET or production numerics)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "mean_wall_s_by_W": None,
  "n": 12,
  "pin": 8,
  "pin_rule": "mode",
  "shape": [
   1,
   256
  ],
  "size": "300m"
 },
 {
  "assay": "probe",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 12 cold processes (picks by num_warps in counts)",
  "counts": {
   "16": 5,
   "8": 7
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "cold processes, one forward each at this first-call shape (DET or production numerics)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "mean_wall_s_by_W": None,
  "n": 12,
  "pin": 8,
  "pin_rule": "mode",
  "shape": [
   1,
   256
  ],
  "size": "600m"
 },
 {
  "assay": "lane",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 30 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 18,
   "16": 3,
   "4": 4,
   "8": 5
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "cold processes, one forward each at this first-call shape (DET or production numerics)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "mean_wall_s_by_W": None,
  "n": 30,
  "pin": 1,
  "pin_prior": 1,
  "pin_rule": "mode",
  "shape": [
   1,
   202
  ],
  "size": "150m"
 },
 {
  "assay": "lane",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 30 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 2,
   "16": 5,
   "4": 4,
   "8": 19
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "cold processes, one forward each at this first-call shape (DET or production numerics)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "mean_wall_s_by_W": None,
  "n": 30,
  "pin": 8,
  "pin_prior": 8,
  "pin_rule": "mode",
  "shape": [
   1,
   202
  ],
  "size": "300m"
 },
 {
  "assay": "lane",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 30 cold processes (picks by num_warps in counts)",
  "counts": {
   "16": 13,
   "32": 2,
   "4": 1,
   "8": 14
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "cold processes, one forward each at this first-call shape (DET or production numerics)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "mean_wall_s_by_W": None,
  "n": 30,
  "pin": 8,
  "pin_prior": 8,
  "pin_rule": "mode",
  "shape": [
   1,
   202
  ],
  "size": "600m"
 },
 {
  "assay": "GLPA_HUMAN_Elazar_2016",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   245,
   154
  ],
  "mean_wall_s_by_W": {
   "1": 17.44
  },
  "n": 6,
  "n_mutants_censused": 245,
  "pin": 1,
  "pin_rule": "mode",
  "positions": 15,
  "shape": [
   15,
   154
  ],
  "size": "150m"
 },
 {
  "assay": "IF1_ECOLI_Kelsic_2016",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 1,
   "4": 5
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   862,
   76
  ],
  "mean_wall_s_by_W": {
   "1": 17.4,
   "4": 16.61
  },
  "n": 6,
  "n_mutants_censused": 1367,
  "pin": 4,
  "pin_rule": "mode",
  "positions": 72,
  "shape": [
   72,
   76
  ],
  "size": "150m"
 },
 {
  "assay": "KCNE1_HUMAN_Muhammad_2023_expression",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "4": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   492,
   133
  ],
  "mean_wall_s_by_W": {
   "4": 16.99
  },
  "n": 6,
  "n_mutants_censused": 2339,
  "pin": 4,
  "pin_rule": "mode",
  "positions": 127,
  "shape": [
   127,
   133
  ],
  "size": "150m"
 },
 {
  "assay": "RL40A_YEAST_Roscoe_2013",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 5,
   "4": 1
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   496,
   132
  ],
  "mean_wall_s_by_W": {
   "1": 17.4,
   "4": 17.1
  },
  "n": 6,
  "n_mutants_censused": 1195,
  "pin": 1,
  "pin_rule": "mode",
  "positions": 75,
  "shape": [
   75,
   132
  ],
  "size": "150m"
 },
 {
  "assay": "VG08_BPP22_Tsuboyama_2023_2GP8",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 4,
   "4": 2
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   723,
   44
  ],
  "mean_wall_s_by_W": {
   "1": 18.22,
   "4": 18.23
  },
  "n": 6,
  "n_mutants_censused": 723,
  "pin": 1,
  "pin_rule": "mode",
  "positions": 40,
  "shape": [
   40,
   44
  ],
  "size": "150m"
 },
 {
  "assay": "GLPA_HUMAN_Elazar_2016",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   245,
   154
  ],
  "mean_wall_s_by_W": {
   "1": 19.72
  },
  "n": 6,
  "n_mutants_censused": 245,
  "pin": 1,
  "pin_rule": "mode",
  "positions": 15,
  "shape": [
   15,
   154
  ],
  "size": "300m"
 },
 {
  "assay": "IF1_ECOLI_Kelsic_2016",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "4": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   862,
   76
  ],
  "mean_wall_s_by_W": {
   "4": 19.69
  },
  "n": 6,
  "n_mutants_censused": 1367,
  "pin": 4,
  "pin_rule": "mode",
  "positions": 72,
  "shape": [
   72,
   76
  ],
  "size": "300m"
 },
 {
  "assay": "KCNE1_HUMAN_Muhammad_2023_expression",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "4": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   492,
   133
  ],
  "mean_wall_s_by_W": {
   "4": 19.38
  },
  "n": 6,
  "n_mutants_censused": 2339,
  "pin": 4,
  "pin_rule": "mode",
  "positions": 127,
  "shape": [
   127,
   133
  ],
  "size": "300m"
 },
 {
  "assay": "RL40A_YEAST_Roscoe_2013",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 1,
   "4": 5
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   496,
   132
  ],
  "mean_wall_s_by_W": {
   "1": 19.91,
   "4": 19.49
  },
  "n": 6,
  "n_mutants_censused": 1195,
  "pin": 4,
  "pin_rule": "mode",
  "positions": 75,
  "shape": [
   75,
   132
  ],
  "size": "300m"
 },
 {
  "assay": "GLPA_HUMAN_Elazar_2016",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "2": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   245,
   154
  ],
  "mean_wall_s_by_W": {
   "2": 18.37
  },
  "n": 6,
  "n_mutants_censused": 245,
  "pin": 2,
  "pin_rule": "mode",
  "positions": 15,
  "shape": [
   15,
   154
  ],
  "size": "600m"
 },
 {
  "assay": "HMDH_HUMAN_Jiang_2019",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "2": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   73,
   892
  ],
  "mean_wall_s_by_W": {
   "2": 23.87
  },
  "n": 6,
  "n_mutants_censused": 16853,
  "pin": 2,
  "pin_rule": "mode",
  "positions": 887,
  "shape": [
   73,
   892
  ],
  "size": "600m"
 },
 {
  "assay": "IF1_ECOLI_Kelsic_2016",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "2": 1,
   "4": 5
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   862,
   76
  ],
  "mean_wall_s_by_W": {
   "2": 17.97,
   "4": 17.43
  },
  "n": 6,
  "n_mutants_censused": 1367,
  "pin": 4,
  "pin_rule": "mode",
  "positions": 72,
  "shape": [
   72,
   76
  ],
  "size": "600m"
 },
 {
  "assay": "KCNE1_HUMAN_Muhammad_2023_expression",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "2": 4,
   "4": 2
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   492,
   133
  ],
  "mean_wall_s_by_W": {
   "2": 17.97,
   "4": 18.41
  },
  "n": 6,
  "n_mutants_censused": 2339,
  "pin": 2,
  "pin_rule": "mode",
  "positions": 127,
  "shape": [
   127,
   133
  ],
  "size": "600m"
 },
 {
  "assay": "RL40A_YEAST_Roscoe_2013",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "2": 2,
   "4": 4
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   496,
   132
  ],
  "mean_wall_s_by_W": {
   "2": 18.57,
   "4": 18.41
  },
  "n": 6,
  "n_mutants_censused": 1195,
  "pin": 4,
  "pin_rule": "mode",
  "positions": 75,
  "shape": [
   75,
   132
  ],
  "size": "600m"
 },
 {
  "assay": "VG08_BPP22_Tsuboyama_2023_2GP8",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "2": 5,
   "4": 1
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   723,
   44
  ],
  "mean_wall_s_by_W": {
   "2": 18.3,
   "4": 17.3
  },
  "n": 6,
  "n_mutants_censused": 723,
  "pin": 2,
  "pin_rule": "mode",
  "positions": 40,
  "shape": [
   40,
   44
  ],
  "size": "600m"
 },
 {
  "assay": "b1",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "16": 1,
   "32": 2,
   "4": 2,
   "8": 1
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   1,
   76
  ],
  "mean_wall_s_by_W": {
   "16": 18.01,
   "32": 18.37,
   "4": 18.68,
   "8": 18.61
  },
  "n": 6,
  "n_mutants_censused": 1,
  "pin": 32,
  "pin_rule": "mode; tie by the mean wall",
  "positions": 1,
  "shape": [
   1,
   76
  ],
  "size": "600m"
 },
 {
  "assay": "VG08_BPP22_Tsuboyama_2023_2GP8",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 2,
   "4": 4
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes, observe-only; the first-of-box draw excluded",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   723,
   44
  ],
  "mean_wall_s_by_W": {
   "1": 20.14,
   "4": 18.83
  },
  "n": 6,
  "pin": 4,
  "pin_rule": "mode",
  "positions": 40,
  "shape": [
   40,
   44
  ],
  "size": "300m"
 },
 {
  "assay": "B2L11_HUMAN_Dutta_2010_binding-Mcl-1",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 16 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 2,
   "2": 3,
   "4": 2,
   "8": 9
  },
  "device_name": None,
  "form": "CLI-shape census: 10 stock draws per cell, observe-only",
  "gpu_key": "a100",
  "label_shape_was": [
   170,
   202
  ],
  "mean_wall_s_by_W": None,
  "n": 16,
  "pin": 8,
  "pin_rule": "mode",
  "positions": 10,
  "shape": [
   10,
   202
  ],
  "size": "600m"
 },
 {
  "assay": "B2L11_HUMAN_Dutta_2010_binding-Mcl-1",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 16 cold processes (picks by num_warps in counts)",
  "counts": {
   "2": 4,
   "4": 9,
   "8": 3
  },
  "device_name": None,
  "form": "CLI-shape census: 10 stock draws per cell, observe-only",
  "gpu_key": "a100",
  "label_shape_was": [
   170,
   202
  ],
  "mean_wall_s_by_W": None,
  "n": 16,
  "pin": 4,
  "pin_rule": "mode",
  "positions": 10,
  "shape": [
   10,
   202
  ],
  "size": "300m"
 },
 {
  "assay": "B2L11_HUMAN_Dutta_2010_binding-Mcl-1",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 16 cold processes (picks by num_warps in counts)",
  "counts": {
   "2": 4,
   "4": 11,
   "8": 1
  },
  "device_name": None,
  "form": "CLI-shape census: 10 stock draws per cell, observe-only",
  "gpu_key": "a100",
  "label_shape_was": [
   170,
   202
  ],
  "mean_wall_s_by_W": None,
  "n": 16,
  "pin": 4,
  "pin_rule": "mode",
  "positions": 10,
  "shape": [
   10,
   202
  ],
  "size": "150m"
 },
 {
  "assay": "B2L11_HUMAN_Dutta_2010_binding-Mcl-1",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 16 cold processes (picks by num_warps in counts)",
  "counts": {
   "2": 3,
   "4": 13
  },
  "device_name": None,
  "form": "CLI-shape census: 10 stock draws per cell, observe-only",
  "gpu_key": "b200",
  "label_shape_was": [
   170,
   202
  ],
  "mean_wall_s_by_W": None,
  "n": 16,
  "pin": 4,
  "pin_rule": "mode",
  "positions": 10,
  "shape": [
   10,
   202
  ],
  "size": "600m"
 },
 {
  "assay": "B2L11_HUMAN_Dutta_2010_binding-Mcl-1",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 16 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 3,
   "2": 2,
   "4": 9,
   "8": 2
  },
  "device_name": None,
  "form": "CLI-shape census: 10 stock draws per cell, observe-only",
  "gpu_key": "b200",
  "label_shape_was": [
   170,
   202
  ],
  "mean_wall_s_by_W": None,
  "n": 16,
  "pin": 4,
  "pin_rule": "mode",
  "positions": 10,
  "shape": [
   10,
   202
  ],
  "size": "300m"
 },
 {
  "assay": "B2L11_HUMAN_Dutta_2010_binding-Mcl-1",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 16 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 9,
   "2": 3,
   "4": 3,
   "8": 1
  },
  "device_name": None,
  "form": "CLI-shape census: 10 stock draws per cell, observe-only",
  "gpu_key": "b200",
  "label_shape_was": [
   170,
   202
  ],
  "mean_wall_s_by_W": None,
  "n": 16,
  "pin": 1,
  "pin_rule": "mode",
  "positions": 10,
  "shape": [
   10,
   202
  ],
  "size": "150m"
 },
 {
  "assay": "B2L11_HUMAN_Dutta_2010_binding-Mcl-1",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 16 cold processes (picks by num_warps in counts)",
  "counts": {
   "2": 16
  },
  "device_name": None,
  "form": "CLI-shape census: 10 stock draws per cell, observe-only",
  "gpu_key": "l40s",
  "label_shape_was": [
   170,
   202
  ],
  "mean_wall_s_by_W": None,
  "n": 16,
  "pin": 2,
  "pin_rule": "mode",
  "positions": 10,
  "shape": [
   10,
   202
  ],
  "size": "600m"
 },
 {
  "assay": "B2L11_HUMAN_Dutta_2010_binding-Mcl-1",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 16 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 10,
   "2": 6
  },
  "device_name": None,
  "form": "CLI-shape census: 10 stock draws per cell, observe-only",
  "gpu_key": "l40s",
  "label_shape_was": [
   170,
   202
  ],
  "mean_wall_s_by_W": None,
  "n": 16,
  "pin": 1,
  "pin_rule": "mode",
  "positions": 10,
  "shape": [
   10,
   202
  ],
  "size": "300m"
 },
 {
  "assay": "B2L11_HUMAN_Dutta_2010_binding-Mcl-1",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 16 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 14,
   "2": 2
  },
  "device_name": None,
  "form": "CLI-shape census: 10 stock draws per cell, observe-only",
  "gpu_key": "l40s",
  "label_shape_was": [
   170,
   202
  ],
  "mean_wall_s_by_W": None,
  "n": 16,
  "pin": 1,
  "pin_rule": "mode",
  "positions": 10,
  "shape": [
   10,
   202
  ],
  "size": "150m"
 },
 {
  "assay": "IF1_ECOLI_Kelsic_2016 (one mutant)",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 1,
   "4": 2,
   "8": 3
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command, observe-only read-back (the first-of-host draw NOT excluded here — stated)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   1,
   76
  ],
  "mean_wall_s_by_W": {
   "1": 17.1,
   "4": 16.07,
   "8": 16.09
  },
  "n": 6,
  "n_mutants_censused": 1,
  "pin": 8,
  "pin_rule": "mode",
  "positions": 1,
  "shape": [
   1,
   76
  ],
  "size": "150m"
 },
 {
  "assay": "ENVZ_ECOLI_Ghose_2023",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 5,
   "4": 1
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command, observe-only read-back (the first-of-host draw NOT excluded here — stated)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   1024,
   64
  ],
  "mean_wall_s_by_W": {
   "1": 14.92,
   "4": 16.06
  },
  "n": 6,
  "n_mutants_censused": 1024,
  "pin": 1,
  "pin_rule": "mode",
  "positions": 60,
  "shape": [
   60,
   64
  ],
  "size": "150m"
 },
 {
  "assay": "HMDH_HUMAN_Jiang_2019",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command, observe-only read-back (the first-of-host draw NOT excluded here — stated)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   73,
   892
  ],
  "mean_wall_s_by_W": {
   "1": 19.17
  },
  "n": 6,
  "n_mutants_censused": 73,
  "pin": 1,
  "pin_rule": "mode",
  "positions": 887,
  "shape": [
   73,
   892
  ],
  "size": "150m"
 },
 {
  "assay": "B2L11_HUMAN_Dutta_2010_binding-Mcl-1",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 3,
   "4": 3
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command, observe-only read-back (the first-of-host draw NOT excluded here — stated)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   170,
   202
  ],
  "mean_wall_s_by_W": {
   "1": 16.96,
   "4": 16.08
  },
  "n": 6,
  "n_mutants_censused": 170,
  "pin": 4,
  "pin_rule": "mode; tie by the mean wall",
  "positions": 10,
  "shape": [
   10,
   202
  ],
  "size": "150m"
 },
 {
  "assay": "DN7A_SACS2_Tsuboyama_2023_1JIC",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 4,
   "4": 2
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command, observe-only read-back (the first-of-host draw NOT excluded here — stated)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   1008,
   59
  ],
  "mean_wall_s_by_W": {
   "1": 15.04,
   "4": 15.55
  },
  "n": 6,
  "n_mutants_censused": 1008,
  "pin": 1,
  "pin_rule": "mode",
  "positions": 55,
  "shape": [
   55,
   59
  ],
  "size": "150m"
 },
 {
  "assay": "IF1_ECOLI_Kelsic_2016 (one mutant)",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "4": 3,
   "8": 3
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command, observe-only read-back (the first-of-host draw NOT excluded here — stated)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   1,
   76
  ],
  "mean_wall_s_by_W": {
   "4": 16.34,
   "8": 16.44
  },
  "n": 6,
  "n_mutants_censused": 1,
  "pin": 4,
  "pin_rule": "mode; tie by the mean wall",
  "positions": 1,
  "shape": [
   1,
   76
  ],
  "size": "300m"
 },
 {
  "assay": "ENVZ_ECOLI_Ghose_2023",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command, observe-only read-back (the first-of-host draw NOT excluded here — stated)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   1024,
   64
  ],
  "mean_wall_s_by_W": {
   "1": 15.0
  },
  "n": 6,
  "n_mutants_censused": 1024,
  "pin": 1,
  "pin_rule": "mode",
  "positions": 60,
  "shape": [
   60,
   64
  ],
  "size": "300m"
 },
 {
  "assay": "HMDH_HUMAN_Jiang_2019",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "4": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command, observe-only read-back (the first-of-host draw NOT excluded here — stated)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   73,
   892
  ],
  "mean_wall_s_by_W": {
   "4": 19.94
  },
  "n": 6,
  "n_mutants_censused": 73,
  "pin": 4,
  "pin_rule": "mode",
  "positions": 887,
  "shape": [
   73,
   892
  ],
  "size": "300m"
 },
 {
  "assay": "B2L11_HUMAN_Dutta_2010_binding-Mcl-1",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "4": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command, observe-only read-back (the first-of-host draw NOT excluded here — stated)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   170,
   202
  ],
  "mean_wall_s_by_W": {
   "4": 15.59
  },
  "n": 6,
  "n_mutants_censused": 170,
  "pin": 4,
  "pin_rule": "mode",
  "positions": 10,
  "shape": [
   10,
   202
  ],
  "size": "300m"
 },
 {
  "assay": "DN7A_SACS2_Tsuboyama_2023_1JIC",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command, observe-only read-back (the first-of-host draw NOT excluded here — stated)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   1008,
   59
  ],
  "mean_wall_s_by_W": {
   "1": 15.93
  },
  "n": 6,
  "n_mutants_censused": 1008,
  "pin": 1,
  "pin_rule": "mode",
  "positions": 55,
  "shape": [
   55,
   59
  ],
  "size": "300m"
 },
 {
  "assay": "ENVZ_ECOLI_Ghose_2023",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "4": 1,
   "8": 5
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command, observe-only read-back (the first-of-host draw NOT excluded here — stated)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   1024,
   64
  ],
  "mean_wall_s_by_W": {
   "4": 17.15,
   "8": 16.76
  },
  "n": 6,
  "n_mutants_censused": 1024,
  "pin": 8,
  "pin_rule": "mode",
  "positions": 60,
  "shape": [
   60,
   64
  ],
  "size": "600m"
 },
 {
  "assay": "B2L11_HUMAN_Dutta_2010_binding-Mcl-1",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "2": 5,
   "4": 1
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command, observe-only read-back (the first-of-host draw NOT excluded here — stated)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   170,
   202
  ],
  "mean_wall_s_by_W": {
   "2": 17.78,
   "4": 17.06
  },
  "n": 6,
  "n_mutants_censused": 170,
  "pin": 2,
  "pin_rule": "mode",
  "positions": 10,
  "shape": [
   10,
   202
  ],
  "size": "600m"
 },
 {
  "assay": "DN7A_SACS2_Tsuboyama_2023_1JIC",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "8": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command, observe-only read-back (the first-of-host draw NOT excluded here — stated)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   1008,
   59
  ],
  "mean_wall_s_by_W": {
   "8": 17.45
  },
  "n": 6,
  "n_mutants_censused": 1008,
  "pin": 8,
  "pin_rule": "mode",
  "positions": 55,
  "shape": [
   55,
   59
  ],
  "size": "600m"
 },
 {
  "assay": "lane",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 10 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 5,
   "4": 1,
   "8": 4
  },
  "device_name": "NVIDIA H200",
  "form": "ONE cold process per size at the (1, 202) first call (1 x 202 rows, the 198-aa B2L11 parent); DET env; hub kernel snapshot 5ebc83aa",
  "gpu_key": "h200",
  "mean_wall_s_by_W": None,
  "mean_warm_forward_s_by_W": {
   "1": 0.0515,
   "4": 0.0468,
   "8": 0.0475
  },
  "n": 10,
  "pin": 1,
  "pin_rule": "mode",
  "shape": [
   1,
   202
  ],
  "size": "150m"
 },
 {
  "assay": "lane",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 10 cold processes (picks by num_warps in counts)",
  "counts": {
   "16": 1,
   "4": 4,
   "8": 5
  },
  "device_name": "NVIDIA H200",
  "form": "ONE cold process per size at the (1, 202) first call (1 x 202 rows, the 198-aa B2L11 parent); DET env; hub kernel snapshot 5ebc83aa",
  "gpu_key": "h200",
  "mean_wall_s_by_W": None,
  "mean_warm_forward_s_by_W": {
   "16": 0.0473,
   "4": 0.0503,
   "8": 0.0522
  },
  "n": 10,
  "pin": 8,
  "pin_rule": "mode",
  "shape": [
   1,
   202
  ],
  "size": "300m"
 },
 {
  "assay": "lane",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 10 cold processes (picks by num_warps in counts)",
  "counts": {
   "16": 4,
   "8": 6
  },
  "device_name": "NVIDIA H200",
  "form": "ONE cold process per size at the (1, 202) first call (1 x 202 rows, the 198-aa B2L11 parent); DET env; hub kernel snapshot 5ebc83aa",
  "gpu_key": "h200",
  "mean_wall_s_by_W": None,
  "mean_warm_forward_s_by_W": {
   "16": 0.0659,
   "8": 0.0687
  },
  "n": 10,
  "pin": 8,
  "pin_rule": "mode",
  "shape": [
   1,
   202
  ],
  "size": "600m"
 },
 {
  "assay": "lane",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 10 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 1,
   "2": 1,
   "32": 2,
   "4": 4,
   "8": 2
  },
  "device_name": "NVIDIA B200",
  "form": "ONE cold process per size at the (1, 202) first call (1 x 202 rows, the 198-aa B2L11 parent); DET env; hub kernel snapshot 5ebc83aa",
  "gpu_key": "b200",
  "mean_wall_s_by_W": None,
  "mean_warm_forward_s_by_W": {
   "1": 0.0484,
   "2": 0.0483,
   "32": 0.0471,
   "4": 0.0491,
   "8": 0.0475
  },
  "n": 10,
  "pin": 4,
  "pin_rule": "mode",
  "shape": [
   1,
   202
  ],
  "size": "150m"
 },
 {
  "assay": "lane",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 10 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 1,
   "16": 3,
   "2": 1,
   "32": 2,
   "8": 3
  },
  "device_name": "NVIDIA B200",
  "form": "ONE cold process per size at the (1, 202) first call (1 x 202 rows, the 198-aa B2L11 parent); DET env; hub kernel snapshot 5ebc83aa",
  "gpu_key": "b200",
  "mean_wall_s_by_W": None,
  "mean_warm_forward_s_by_W": {
   "1": 0.0459,
   "16": 0.0479,
   "2": 0.0466,
   "32": 0.0466,
   "8": 0.0468
  },
  "n": 10,
  "pin": 16,
  "pin_rule": "mode TIE 8/16 (3/3): 16 KEPT (this class's test records ran at 16); the warm-wall rule would pick 8 (0.0468 vs 0.0479 s, n = 3 each, within noise)",
  "shape": [
   1,
   202
  ],
  "size": "300m"
 },
 {
  "assay": "lane",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 10 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 1,
   "16": 2,
   "2": 1,
   "32": 4,
   "4": 1,
   "8": 1
  },
  "device_name": "NVIDIA B200",
  "form": "ONE cold process per size at the (1, 202) first call (1 x 202 rows, the 198-aa B2L11 parent); DET env; hub kernel snapshot 5ebc83aa",
  "gpu_key": "b200",
  "mean_wall_s_by_W": None,
  "mean_warm_forward_s_by_W": {
   "1": 0.0679,
   "16": 0.067,
   "2": 0.0653,
   "32": 0.0668,
   "4": 0.0666,
   "8": 0.0658
  },
  "n": 10,
  "pin": 32,
  "pin_rule": "mode",
  "shape": [
   1,
   202
  ],
  "size": "600m"
 },
 {
  "assay": "lane",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 30 cold processes (picks by num_warps in counts)",
  "counts": {
   "16": 5,
   "2": 4,
   "4": 14,
   "8": 7
  },
  "device_name": "NVIDIA A100-SXM4-80GB",
  "form": "ONE cold process per size at the (1, 202) first call (1 x 202 rows, the 198-aa B2L11 parent); DET env; hub kernel snapshot 5ebc83aa",
  "gpu_key": "a100",
  "mean_wall_s_by_W": None,
  "mean_warm_forward_s_by_W": {
   "16": 0.0815,
   "2": 0.0862,
   "4": 0.0813,
   "8": 0.0753
  },
  "n": 30,
  "pin": 4,
  "pin_rule": "mode",
  "shape": [
   1,
   202
  ],
  "size": "150m"
 },
 {
  "assay": "lane",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 30 cold processes (picks by num_warps in counts)",
  "counts": {
   "16": 6,
   "2": 4,
   "32": 4,
   "4": 12,
   "8": 4
  },
  "device_name": "NVIDIA A100-SXM4-80GB",
  "form": "ONE cold process per size at the (1, 202) first call (1 x 202 rows, the 198-aa B2L11 parent); DET env; hub kernel snapshot 5ebc83aa",
  "gpu_key": "a100",
  "mean_wall_s_by_W": None,
  "mean_warm_forward_s_by_W": {
   "16": 0.0806,
   "2": 0.0707,
   "32": 0.0774,
   "4": 0.0856,
   "8": 0.0776
  },
  "n": 30,
  "pin": 4,
  "pin_rule": "mode",
  "shape": [
   1,
   202
  ],
  "size": "300m"
 },
 {
  "assay": "lane",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 30 cold processes (picks by num_warps in counts)",
  "counts": {
   "16": 4,
   "2": 1,
   "32": 1,
   "4": 20,
   "8": 4
  },
  "device_name": "NVIDIA A100-SXM4-80GB",
  "form": "ONE cold process per size at the (1, 202) first call (1 x 202 rows, the 198-aa B2L11 parent); DET env; hub kernel snapshot 5ebc83aa",
  "gpu_key": "a100",
  "mean_wall_s_by_W": None,
  "mean_warm_forward_s_by_W": {
   "16": 0.1173,
   "2": 0.1296,
   "32": 0.131,
   "4": 0.1104,
   "8": 0.1081
  },
  "n": 30,
  "pin": 4,
  "pin_rule": "mode",
  "shape": [
   1,
   202
  ],
  "size": "600m"
 },
 {
  "assay": "lane",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 30 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 25,
   "16": 5
  },
  "device_name": "NVIDIA L40S",
  "form": "ONE cold process per size at the (1, 202) first call (1 x 202 rows, the 198-aa B2L11 parent); DET env; hub kernel snapshot 5ebc83aa",
  "gpu_key": "l40s",
  "mean_wall_s_by_W": None,
  "mean_warm_forward_s_by_W": {
   "1": 0.061,
   "16": 0.0607
  },
  "n": 30,
  "pin": 1,
  "pin_rule": "mode",
  "shape": [
   1,
   202
  ],
  "size": "150m"
 },
 {
  "assay": "lane",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 30 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 25,
   "4": 5
  },
  "device_name": "NVIDIA L40S",
  "form": "ONE cold process per size at the (1, 202) first call (1 x 202 rows, the 198-aa B2L11 parent); DET env; hub kernel snapshot 5ebc83aa",
  "gpu_key": "l40s",
  "mean_wall_s_by_W": None,
  "mean_warm_forward_s_by_W": {
   "1": 0.0607,
   "4": 0.0626
  },
  "n": 30,
  "pin": 1,
  "pin_rule": "mode",
  "shape": [
   1,
   202
  ],
  "size": "300m"
 },
 {
  "assay": "lane",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 30 cold processes (picks by num_warps in counts)",
  "counts": {
   "16": 16,
   "4": 12,
   "8": 2
  },
  "device_name": "NVIDIA L40S",
  "form": "ONE cold process per size at the (1, 202) first call (1 x 202 rows, the 198-aa B2L11 parent); DET env; hub kernel snapshot 5ebc83aa",
  "gpu_key": "l40s",
  "mean_wall_s_by_W": None,
  "mean_warm_forward_s_by_W": {
   "16": 0.0845,
   "4": 0.0843,
   "8": 0.0852
  },
  "n": 30,
  "pin": 16,
  "pin_rule": "mode",
  "shape": [
   1,
   202
  ],
  "size": "600m"
 },
 {
  "assay": "BRCA1_HUMAN_Findlay_2018",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "4": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command on the BRCA1 canary (326 positions; the same (35, 1867) first call as the fully-mutated unit), observe-only read-back; one warm-up (first-of-box) excluded",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   35,
   1867
  ],
  "mean_wall_s_by_W": {
   "4": 19.6
  },
  "n": 6,
  "n_mutants_censused": 1837,
  "pin": 4,
  "pin_rule": "mode",
  "positions": 1863,
  "shape": [
   35,
   1867
  ],
  "size": "300m"
 },
 {
  "assay": "BRCA1_HUMAN_Findlay_2018",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "1": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   35,
   1867
  ],
  "mean_wall_s_by_W": {
   "1": 22.16
  },
  "n": 6,
  "n_mutants_censused": 1837,
  "pin": 1,
  "pin_rule": "mode",
  "positions": 326,
  "shape": [
   35,
   1867
  ],
  "size": "150m"
 },
 {
  "assay": "BRCA1_HUMAN_Findlay_2018",
  "cite": "autotune census of the stock kernel's first-call pick at this cell: 6 cold processes (picks by num_warps in counts)",
  "counts": {
   "2": 6
  },
  "device_name": "NVIDIA H100 80GB HBM3",
  "form": "6 fresh cold stock processes of the documented command (python -m E1.tools.score), observe-only Autotuner read-back; the first-of-box draw excluded (supplement)",
  "gpu_key": "NVIDIA H100 80GB HBM3",
  "label_shape_was": [
   35,
   1867
  ],
  "mean_wall_s_by_W": {
   "2": 31.01
  },
  "n": 6,
  "n_mutants_censused": 1837,
  "pin": 2,
  "pin_rule": "mode",
  "positions": 326,
  "shape": [
   35,
   1867
  ],
  "size": "600m"
 }
]

W_CELL_TABLE = {(c["gpu_key"], c["size"], tuple(c["shape"])): c for c in W_CELLS}
assert len(W_CELL_TABLE) == len(W_CELLS), "duplicate W cells"
LANE_FIRST_CALL_SHAPE = (1, 202)                  # the first-call shape the DET pins are censused at (E1Predictor.predict of one 198-aa parent: (rows, T) = (1, 202))
_GPU_SLUGS = ("h100", "a100", "b200", "l40s", "h200")


DEFAULT_W_ROW = "NVIDIA H100 80GB HBM3"   # the row whose per-size pins serve a (GPU class, size, first-call shape) with NO censused cell on a compute capability with no default row of its own (DEFAULT_W_ROW_BY_CC): the same lever at a default setting (one named line), a censused cell being a measured override
DEFAULT_W_ROW_BY_CC = {"9.0": "NVIDIA H100 80GB HBM3", "8.0": "NVIDIA A100-SXM4-80GB"}   # the default row per COMPUTE-CAPABILITY CLASS for an uncensused card: a cc 9.0 card is served the H100 row's pins, a cc 8.0 card the A100 row's (its censused sibling on the same silicon class); any other cc → DEFAULT_W_ROW — broad by class, never keyed on a device name (the name keys only the MEASURED cells)
assert all(row in TRITON_AUTOTUNE_PIN for row in DEFAULT_W_ROW_BY_CC.values()) and DEFAULT_W_ROW in TRITON_AUTOTUNE_PIN
UNPINNED_LINE = "[e1/pins] autotune pin: no censused cell for {gpu!r} / {size} at first-call shape {shape} — pinned to W={W} ({how}, row {row!r}); the kit stays bit-consistent with this pinned stock, the card's own stock autotune mode is unmeasured (identity to an unpinned stock on this card is recorded, not bit-exact by construction)"   # THE one line of the uncensused path (apply_autotune_pin), exit 0, the lever counts as engaged


def _device_name_now() -> str:
    """torch's name of device 0 (imported here so a host without torch can still ask by an explicit name)."""
    import torch
    return torch.cuda.get_device_name(0)


def _cc_now() -> str | None:
    """The compute capability of device 0 as 'major.minor' ('9.0', '8.0'), or None on a host without torch / CUDA (the caller asked by name)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return f"{int(major)}.{int(minor)}"
    except Exception:
        return None


def default_w_row(cc: str | None = None) -> str:
    """The TRITON_AUTOTUNE_PIN row whose per-size pins serve an UNCENSUSED (GPU class, size, first-call shape): by compute-capability
    class (DEFAULT_W_ROW_BY_CC — cc 9.0 → the H100 row, cc 8.0 → the A100 row), else DEFAULT_W_ROW (an unknown cc, or a host that cannot
    say). `cc` None = read device 0's capability now (None when there is no device)."""
    key = cc if cc is not None else _cc_now()
    return DEFAULT_W_ROW_BY_CC.get(str(key), DEFAULT_W_ROW) if key is not None else DEFAULT_W_ROW


def gpu_key_of(device_name: str) -> str | None:
    """The table key for a device name: the exact name when a cell carries it, else the slug ('a100', 'b200', 'l40s'; those classes'
    cells are keyed by slug) when a cell carries that slug; None = no cell for this GPU class."""
    if any(c["gpu_key"] == device_name for c in W_CELLS):
        return device_name
    low = device_name.lower()
    for slug in _GPU_SLUGS:
        if slug in low and any(c["gpu_key"] == slug for c in W_CELLS):
            return slug
    return None


def w_cell(size: str, first_call_shape, gpu_name: str | None = None) -> dict:
    """The censused cell for (gpu, size, first_call_shape) or PinUncensused (a PinDrift) BY NAME."""
    gpu = gpu_name or _device_name_now()
    key_gpu = gpu_key_of(gpu)
    shape = tuple(int(x) for x in first_call_shape)
    cell = W_CELL_TABLE.get((key_gpu, size, shape)) if key_gpu else None
    if cell is None:
        have = sorted({c["shape"][0] for c in W_CELLS if c["gpu_key"] == key_gpu and c["size"] == size and c["shape"][1] == shape[1]}) if key_gpu else []
        raise PinUncensused(f"no W census at shape {shape} (rows, T) for {size} on {gpu!r} (table key {key_gpu!r}): run the stock or census the cell"
                       + (f" (censused row counts at T={shape[1]}: {have})" if have else ""))
    return dict(cell, gpu_key_matched=key_gpu, device_name_seen=gpu)

#: GPU CLASS KEYS ('A100-80GB' / 'B200' / 'H200' / 'L40S'; torch's probe name for the
#: H100 class 'NVIDIA H100 80GB HBM3'): the key every class-keyed table uses — lower-cased, the memory suffix dropped, aliases applied.
#: Every A100 part is class 'a100' whatever its memory size or form factor (SXM4 / PCIe, 80 GB / 40 GB: one compute silicon, cc 8.0) — the
#: class-keyed A3 table (kits/v1_2) must see 'a100' on all of them, A3 being not exact on that class.
GPU_CLASS_ALIASES = {"nvidia h100 80gb hbm3": "h100", "nvidia h100 nvl": "h100", "nvidia h200": "h200", "nvidia a100-sxm4-80gb": "a100",
                     "nvidia a100 80gb pcie": "a100", "nvidia a100-sxm4-40gb": "a100", "nvidia a100-pcie-40gb": "a100", "nvidia l40s": "l40s",
                     "nvidia b200": "b200", "a100-80gb": "a100", "a100-40gb": "a100", "h100-80gb": "h100"}


def gpu_class_key(name: str | None) -> str | None:
    """'A100-80GB' -> 'a100'; 'NVIDIA A100-SXM4-40GB' / 'NVIDIA A100-PCIE-40GB' / 'NVIDIA A100 80GB PCIe' -> 'a100'; 'L40S' -> 'l40s';
    'NVIDIA H100 80GB HBM3' -> 'h100'; None -> None."""
    if not name:
        return None
    k = str(name).strip().lower()
    k = GPU_CLASS_ALIASES.get(k, k)
    if "a100" in k.replace("-", " ").replace("_", " ").split():   # any A100 spelling ('… a100-sxm4-40gb', '… a100 80gb pcie', 'a100_40gb'): the class, never the marketing name
        return "a100"
    if "-" in k and k.split("-")[0] in ("a100", "h100"):
        k = k.split("-")[0]
    return k


#: the forward shapes the kits declare as TESTED_SHAPES (masked-marginal batches of one parent: B rows of L residues, no padding; tokens = B x (L + 4))
PROBE_SHAPES = ((1, 256), (16, 256), (252, 256), (127, 512), (63, 1024))

#: the DET recipe (engines.core.apply_det_recipe fields + the E1-specific autotune pin applied by ``apply_autotune_pin``)
DET_RECIPE = {"seed": 0, "deterministic_algorithms": True, "tf32": False, "cudnn_benchmark": False, "dtype": "bfloat16", "batch_size": 1,
              "env": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                      "NPY_DISABLE_CPU_FEATURES": "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR AVX2 FMA3 X86_V3 X86_V4",
                      "OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "1"},
              "framework": "torch", "triton_autotune_pin": TRITON_AUTOTUNE_PIN}


class PinDrift(RuntimeError):
    """A pin does not match the bytes in the process."""


class PinUncensused(PinDrift):
    """No censused autotune cell for this (GPU class, size, first-call shape): not a drift of bytes — the table simply has no row.
    apply_autotune_pin answers it with the size's DEFAULT pin of the card's compute-capability class (default_w_row, one named line, exit 0); w_cell still raises it by name
    for callers that need the cell itself."""


def sha256_file(path: str, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_snapshot(size: str, hf_home: str | None = None) -> str:
    """The HF-cache snapshot dir of the pinned revision (``<HF_HOME>/hub/models--<org>--<name>/snapshots/<rev>``)."""
    w = WEIGHTS[size]
    hf_home = hf_home or os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(hf_home, "hub", "models--" + w["repo"].replace("/", "--"), "snapshots", w["rev"])


def assert_weights(size: str, snapshot: str | None = None) -> dict:
    """Digest ``model.safetensors`` in the snapshot (sha256 + size) and word it against the pin (WEIGHTS_WORDS; the line: ``weights_words``) — either word loads; PinDrift only when the file is not there. Returns the record."""
    w = WEIGHTS[size]
    snap = snapshot or weights_snapshot(size)
    st = os.path.join(snap, "model.safetensors")
    if not os.path.exists(st):
        raise PinDrift(f"{size}: no model.safetensors at {snap}")
    sha, n = sha256_file(st), os.path.getsize(st)
    word = WEIGHTS_WORDS[0] if (sha == w["sha256"] and n == w["bytes"]) else WEIGHTS_WORDS[1]
    return {"size": size, "repo": w["repo"], "snapshot": snap, "sha256": sha, "bytes": n, "rev": w["rev"], "pinned_sha256": w["sha256"], "pinned_bytes": w["bytes"],
            "word": word}


def stack_report() -> dict:
    """Installed versions of the dependency stack beside the pins (STACK, the pinned stack) — never a refusal: a version that
    differs from its pin, or a package that is absent, is NAMED under ``drift`` (the callers word it on their line) and the kit runs;
    what cannot run on the drifted stack says so where it fails. Plain strs (the stamp is a weights_only pickle)."""
    got, drift = {}, []
    for pkg in ("torch", "triton", "transformers", "tokenizers", "kernels", "flash_attn"):
        dist = "flash-attn" if pkg == "flash_attn" else pkg
        try:
            got[pkg] = _md.version(dist)
        except _md.PackageNotFoundError:
            got[pkg] = None
            drift.append(f"{pkg} absent (pin {STACK[pkg]})")
            continue
        if got[pkg] != STACK[pkg]:
            drift.append(f"{pkg} {got[pkg]} (pin {STACK[pkg]})")
    try:
        import torch
        got["torch_version_str"], got["cuda"] = str(torch.__version__), str(torch.version.cuda)
    except ImportError:
        got["torch_version_str"], got["cuda"] = None, None
    got["drift"] = drift
    return got


def assert_stock_version() -> str:
    """The installed E1 package's version == the stock pin, else PinDrift: a different E1 is a different stock (the one refusal by pin)."""
    try:
        v = _md.version("E1")
    except _md.PackageNotFoundError:
        raise PinDrift("E1 package not installed")
    if v != STOCK["version"]:
        raise PinDrift(f"E1 package version {v} != {STOCK['version']}: not the pinned stock")
    return v


def rmsnorm_autotuner():
    """The hub kernel's Triton Autotuner object for ``_layer_norm_fwd_1pass_kernel`` (the package __init__ binds the name
    ``layer_norm`` to a function, so the kernel is reached through the globals of the function the model calls,
    ``modeling.py`` L89 ``layer_norm.rms_norm_fn``); PinDrift when the package fell back to torch's rms_norm."""
    import E1.modeling as M
    if M.layer_norm is None:
        raise PinDrift("E1.modeling.layer_norm is None: the hub Triton RMSNorm kernel did not load (torch rms_norm fallback = a different model)")
    k = M.layer_norm.rms_norm_fn.__globals__["_layer_norm_fwd_1pass_kernel"]
    if type(k).__name__ != "Autotuner":
        raise PinDrift(f"_layer_norm_fwd_1pass_kernel is {type(k).__name__}, expected a triton Autotuner (hub kernel revision drift?)")
    return k


def kernel_revision_report() -> dict:
    """The loaded hub RMSNorm kernel beside its pin: the snapshot file it was loaded from, its layer_norm.py sha256, and ``drift`` — a list
    naming a revision or digest that differs from the pin, or the kernel's absence (upstream then runs torch's rms_norm: the KERNELS line
    words it). Never a refusal."""
    try:
        import E1.modeling as M
    except Exception as e:                                                      # noqa: BLE001 — no E1 in this interpreter: named
        return {"file": None, "layer_norm_py_sha256": None, "rev": KERNEL["rev"], "drift": [f"E1.modeling not importable ({type(e).__name__})"]}
    if M.layer_norm is None:
        return {"file": None, "layer_norm_py_sha256": None, "rev": KERNEL["rev"], "drift": ["hub RMSNorm kernel not loaded (upstream's torch rms_norm fallback)"]}
    f = getattr(M.layer_norm, "__file__", None) or ""
    drift = []
    if KERNEL["rev"] not in f:
        drift.append(f"hub kernel loaded from {f!r}, not the pinned revision {KERNEL['rev'][:12]}")
    ln = os.path.join(os.path.dirname(f), "layer_norm.py")
    try:
        sha = sha256_file(ln)
    except OSError:
        sha = None
    if sha != KERNEL["layer_norm_py_sha256"]:
        drift.append(f"layer_norm.py sha256 {str(sha)[:12]} != pin {KERNEL['layer_norm_py_sha256'][:12]}")
    return {"file": f, "layer_norm_py_sha256": sha, "rev": KERNEL["rev"], "drift": drift}


def apply_autotune_pin(size: str, gpu_name: str | None = None, num_warps: int | None = None, first_call_shape=None, cc: str | None = None) -> dict:
    """Pin the hub RMSNorm kernel's autotune to ONE num_warps BEFORE the first forward (one config => the Autotuner skips its
    timing run) AT A CENSUSED CELL — THE ONE W TABLE (the DET pins 1/8/8 are the (1, 202) cells 'assay: lane' of this table):
    ``first_call_shape`` = (rows, T) of the kernel's first call (None = the (1, 202) cell); ``num_warps`` explicit must be inside THAT cell's outcome set (never the union over the
    GPU's censuses), default = the cell's pin (its mode). A (GPU class, size, shape) with NO censused cell — an unknown class included —
    is served the size's DEFAULT pin of its compute-capability class (default_w_row: cc 9.0 → the H100 row, cc 8.0 → the A100 row, any
    other cc → DEFAULT_W_ROW; ``cc`` None = device 0's capability) with ONE named line (UNPINNED_LINE), never refused: the censused cell
    is a measured override. Refuses BY NAME a W outside a censused cell's set, a host without triton, or a kernel whose cache already holds a timing-run
    pick. The stamp carries the cell (None on the default path)."""
    gpu = gpu_name or _device_name_now()
    shape = tuple(first_call_shape) if first_call_shape is not None else LANE_FIRST_CALL_SHAPE
    try:
        cell = w_cell(size, shape, gpu)
    except PinUncensused:                                                       # no row for this (class, size, shape): the SAME lever at its DEFAULT setting — the size's default W (DEFAULT_W_ROW), one named line, never a refusal (a censused cell is a measured override, not the only way to be served)
        cell = None
    outcomes = {int(w) for w in cell["counts"]} if cell is not None else set()
    row = default_w_row(cc) if cell is None else None                              # the uncensused card's default row: its compute-capability class's (H100 row for cc 9.0, A100 row for cc 8.0), else DEFAULT_W_ROW
    W = (int(cell["pin"]) if cell is not None else int(TRITON_AUTOTUNE_PIN[row][size])) if num_warps is None else int(num_warps)
    if cell is None:
        print(UNPINNED_LINE.format(gpu=gpu, size=size, shape=list(shape), W=W, row=row, how="the caller's num_warps=" if num_warps is not None else "the default row's pin"), flush=True)
    elif W not in outcomes:
        raise PinDrift(f"num_warps {W} is not in the censused outcome set {sorted(outcomes)} at cell {(cell['gpu_key_matched'], size, shape)} (not a stock outcome at this first-call shape)")
    try:
        import triton
    except ImportError as e:                                                     # a CPU host without triton: the pin cannot be applied
        raise PinDrift(f"triton is not installed ({e}): the hub RMSNorm Triton kernel cannot be pinned on this host") from e
    k = rmsnorm_autotuner()
    N = WEIGHTS[size]["hidden_size"]
    for key in list(getattr(k, "cache", {}).keys()):
        if str(key).startswith(f"({N},"):
            raise PinDrift(f"the Autotuner already timed N={N} in this process (its pick {k.cache[key].num_warps}); pin before the first forward")
    before = [c.num_warps for c in k.configs]
    k.configs = [triton.Config({}, num_warps=int(W))]
    return {"kernel": "_layer_norm_fwd_1pass_kernel", "gpu": gpu, "size": size, "num_warps": int(W), "pinned": True, "uncensused": cell is None, "configs_before": before,
            "size_default": num_warps is None, "outcome_set": sorted(outcomes), "cell": None if cell is None else {"gpu_key": cell["gpu_key_matched"], "size": size, "first_call_shape": list(shape), "pin": int(cell["pin"]), "pin_rule": cell["pin_rule"], "counts": cell["counts"], "n": cell["n"], "assay": cell.get("assay"), "cite": (cell["cite"][:200] if cell is not None else f"no censused cell: the default row {DEFAULT_W_ROW!r} pin")},
            "lane_cell_default": first_call_shape is None, "default_row": row, "cite": (cell["cite"][:200] if cell is not None else f"no censused cell: the default row {row!r} pin")}


def autotune_readback() -> dict:
    """What the Autotuner holds now. Under the pin the timing cache stays EMPTY by construction (triton 3.4.0
    runtime/autotuner.py Autotuner.run: a single config is used without timing), so the read-back after a
    forward is ``best_config`` (the config the last launch used) beside ``configs``; ``cache`` carries the timing-run picks of an
    unpinned process."""
    k = rmsnorm_autotuner()
    best = getattr(k, "best_config", None)
    return {"configs": [c.num_warps for c in k.configs], "best_config_num_warps": getattr(best, "num_warps", None),
            "cache": {str(key): cfg.num_warps for key, cfg in getattr(k, "cache", {}).items()}}


def inductor_readback() -> dict:
    """The inductor autotune state and the JIT cache dirs of THIS process, read back (a hygiene stamp;
    no inductor knob is pinned)."""
    import os
    try:
        from torch._inductor import config as ic
        tri = ic.triton
        state = {"autotune_pointwise": bool(getattr(tri, "autotune_pointwise", None)), "coordinate_descent_tuning": bool(getattr(ic, "coordinate_descent_tuning", None)),
                 "max_autotune": bool(getattr(ic, "max_autotune", None)), "max_autotune_gemm": bool(getattr(ic, "max_autotune_gemm", None)),
                 "fx_graph_cache": bool(getattr(ic, "fx_graph_cache", None))}
    except Exception as e:                                                      # noqa: BLE001 — a CPU host without inductor stamps the reason
        state = {"error": repr(e)[:200]}
    cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    n = sum(len(f) for _, _, f in os.walk(cache)) if cache and os.path.isdir(cache) else None
    return {**state, "TORCHINDUCTOR_CACHE_DIR": cache, "inductor_cache_files_now": n, "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), "pid": os.getpid()}


def cpu_model() -> str | None:
    """The host CPU model line (small batches are host-bound, so the stamp names the CPU beside the GPU)."""
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def assert_stock_commit() -> dict:
    """The installed E1 package is the stock commit: the E1 package's version string is '1.0.0' at every commit, so the
    stack assert cannot tell commits apart — the pip install record (PEP 610 direct_url.json, written by `pip install git+…@<commit>`)
    carries the resolved commit id. A DIFFERENT commit is refused (PinDrift: a different stock); an install that carries no VCS record (a
    local path, a wheel) cannot be told apart and is NAMED — ``commit_recorded`` False with the reason — never refused: the version pin
    (assert_stock_version) and the kit's own source digests of the methods it replaces still hold it."""
    import importlib.metadata as md
    import json
    version = assert_stock_version()
    dist = md.distribution("E1")
    raw = dist.read_text("direct_url.json")
    du = json.loads(raw) if raw else {}
    commit = (du.get("vcs_info") or {}).get("commit_id")
    if commit is None:
        why = "no VCS install record (direct_url.json)" if not raw else f"installed from {du.get('url')!r} without a commit id"
        return {"url": du.get("url"), "commit_id": None, "version": version, "commit_recorded": False, "why": f"stock commit not recorded by the installer: {why}; version {version} matches the pin",
                "cite": "PEP 610 direct_url.json of the installed E1 distribution"}
    if commit != STOCK["commit"]:
        raise PinDrift(f"E1 installed from commit {commit!r}, the stock is {STOCK['commit']} ({du.get('url')})")
    return {"url": du.get("url"), "commit_id": commit, "version": version, "commit_recorded": True, "why": None, "cite": "PEP 610 direct_url.json of the installed E1 distribution"}


def triton_cache_keys(cache_dir: str | None = None, kernel_substr: str | None = None) -> dict:
    """The Triton cache key list of THIS process's cache dir: {<hash>: [<kernel>.json names]} + triton.__version__ (same
    source + same triton + same arch => same cache keys; the device code itself is compiled per GPU class). ``kernel_substr``
    filters the kernel names (e.g. a kit's kernel prefix)."""
    import os
    cache = cache_dir or os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")
    keys: dict[str, list[str]] = {}
    if os.path.isdir(cache):
        for h in sorted(os.listdir(cache)):
            d = os.path.join(cache, h)
            if not os.path.isdir(d):
                continue
            names = sorted(f for f in os.listdir(d) if f.endswith(".json") and (kernel_substr is None or kernel_substr in f))
            if names:
                keys[h] = names
    try:
        import triton
        tv = triton.__version__
    except Exception:                                                          # noqa: BLE001
        tv = None
    return {"triton": tv, "cache_dir": cache, "n_keys": len(keys), "keys": keys}


# ------------------------------------------------------------------------------------------------------- the weights line's words
WEIGHTS_WORDS = ("pinned", "not-pinned")                          # the word recorded with the weights digest (assert_weights()["word"]; the activation report's weights check)
WEIGHTS_NOTICE = "NOT PINNED — the kit's numbers apply to the pinned weights only"   # the one notice for a digest that is not the pin's


def weights_words(rec: dict) -> str:
    """The weights clause, said once per activation: ``weights=<repo>@<rev12> sha256=<12> (pinned)`` for the pinned digest, else
    ``weights sha256=<12> NOT PINNED — the kit's numbers apply to the pinned weights only (<size>: <sha> (<n> B) != pin <sha> (<n> B))``."""
    s = str(rec.get("sha256"))[:12]
    if rec.get("word") == WEIGHTS_WORDS[0]:
        return f"weights={rec.get('repo')}@{str(rec.get('rev'))[:12]} sha256={s} (pinned)"
    return (f"weights sha256={s} {WEIGHTS_NOTICE} ({rec.get('size')}: {rec.get('sha256')} ({rec.get('bytes')} B) != pin "
            f"{rec.get('pinned_sha256')} ({rec.get('pinned_bytes')} B))")
