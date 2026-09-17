# opt/kit — the Triton forward's tree

What the kit's entry script `opt/kit_ho/tf/pred_bw_fast.py` reads from here (its `_BASE_KIT`): the torch-side package
`torch/chrombpnet_k1` (the K1 Triton forward) with its vendored dependencies (`torch/vendor`, `torch/chrombpnet_k1/_vendor`), the shipped
Triton kernel caches (`torch/triton_cache_*`, listed in `torch/triton_cache_of_record.json`) and the per-architecture route and tile table
(`torch/chrombpnet_k1/arch_tiles.json`). The entry script, the host-side package `chrombpnet_fastkit`, the seed hook and their CPU tests live
in `opt/kit_ho/tf` (opt/kit_ho/README.md). CPU tests of the torch side sit beside the code (`torch/test_*_cpu.py`, `torch/test_predict_no_io.py`;
they import torch).

Prerequisites per route:
- TensorFlow routes: the pinned stock stack (TensorFlow 2.8.0 + chrombpnet 1.0.1, `stock/PINS.json`); nothing else.
- K1 route (the classes `chrombpnet_fastkit/fastdefault.py` / `arch_tiles.json` give it to): torch + triton at the versions in
  `stock/PINS.json` `pins_s1_only`, importable or installed under that entry's `root` (`/opt/torch`, inserted on `sys.path` by the kit and said
  on its line); bpnet-lite + tangermeme from `torch/vendor` (first on `sys.path`). On a K1 class a stack that does not import or kernels that do
  not start refuse the mode by name; they never fall back silently.
- The driver-cache tarball (`CHROMBPNET_OPT_CACHE_TAR`, TensorFlow routes) is unpacked add-only into the process's CUDA ComputeCache once; a
  marker file in that directory makes later runs skip the install, and existing files are never overwritten.

No flag or environment variable chooses a lever: the class tables in `chrombpnet_fastkit/fastdefault.py` and `chrombpnet_k1/arch_tiles.json` decide.
