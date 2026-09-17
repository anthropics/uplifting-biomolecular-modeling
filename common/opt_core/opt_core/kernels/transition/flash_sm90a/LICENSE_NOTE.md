# Third-party notices — fpf_flash_transition

* **NVIDIA CUTLASS 4.2.0 (BSD-3-Clause)** — `csrc/flash_transition_sm90.cu` is written against the CuTe/CUTLASS headers (TMA descriptors,
  wgmma atoms, cluster barriers, pipeline conventions) and follows the structure of the CuTe sm90 tutorial `wgmma_tma_sm90.cu` and the sm90
  warp-specialized collective mainloop.  The headers are a BUILD-TIME dependency (`pip download nvidia-cutlass==4.2.0.0`; not vendored here); the
  prebuilt `.so` contains code compiled from those templates, so the CUTLASS licence text ships next to it: `LICENSE.cutlass`.
* **FlashAttention-3, Dao AI Lab (BSD-3-Clause)** — the 6-line accumulator-to-A-operand register relayout `convert_layout_acc_Aregs` in
  `csrc/flash_transition_sm90.cu` re-implements `hopper/utils.h::convert_layout_acc_Aregs`; the intra-warpgroup GEMM/SwiGLU overlap and the
  two-consumer-warpgroup ping-pong follow the scheduling ideas of that work.  Licence text: `LICENSE.flash-attention`.

Everything else in this directory is this kit's own source.
