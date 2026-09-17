# chrombpnet_k1 — the ChromBPNet forward as Triton kernels under torch

* `chrombpnet_k1/` — the kernels (`kernels.py`: im2col dot, profile head, global-average-pool and dense counts head as fp32 FFMA chains in
  TensorFlow 2.8 / cuDNN 8.1's summation order; `kernels_tc.py`: the same im2col dot at TF32 input precision on the tensor cores), the forward,
  weight loading and Triton-cache handling (`forward.py`: `apply(bias_h5, nobias_h5, precision=, cache_dir=None)`, `predict(k1, X, batch_size=b)`),
  the per-architecture tile / route / counts-order table (`arch_tiles.json`), the finish-stage pre-import thread (`_preimport.py`), the
  typing_extensions guard (`_te_guard.py` with `_vendor/typing_extensions.py`), the out-of-memory classifier (`_oom.py`) and the exit hook
  (`_exit.py`). At `precision="ieee"` the heads are bitwise equal to stock's under TensorFlow's determinism settings on sm_90 / sm_89; at
  `precision="tf32"` the convolutions match stock's shipped TF32 numerics to that precision class.
* `vendor/` — tangermeme 1.4.1 as released on PyPI and bpnet-lite from its upstream repository at commit `b37e766` (package metadata
  1.0.0), both unmodified (versions and licences in `vendor/VENDOR.json`; notices in `../../../THIRD_PARTY_NOTICES.md`), first on `sys.path`: the weight
  loader `bpnetlite.chrombpnet.ChromBPNet.from_chrombpnet(bias_scaled.h5, nobias.h5)` and its import `tangermeme.predict`.
* `triton_cache_sm90/`, `triton_cache_sm89_l40s/` — pre-compiled fp32 kernels, one directory per entry of `triton_cache_of_record.json`; each
  carries `JIT_IDENTITY.json` (the kernel source hash and stack it was built for) and `SHA256SUMS`. `apply()` copies the matching one into a
  writable directory; a cache that does not match this install is not used, the kernels compile in-process and the line says so.
* Runtime: the torch stack of `stock/PINS.json` `pins_s1_only`; numpy / pandas / h5py from the stock stack. cuDNN is not in the convolution path.
* Driven by `opt/kit_ho/tf/pred_bw_fast.py` (the K1 route inside the kit's pipelined job). Tests: `test_*_cpu.py`, `test_predict_no_io.py` (CPU;
  they import torch).
