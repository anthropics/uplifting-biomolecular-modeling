# l3a — batched shared-bias flash attention (one launch for S samples sharing one pair bias)

`fab_batched.py`, `bias_layout.py` and `__init__.py` are this project's original source (source revision
a275f728eecac653ac9f9a14bc938576ceeda90c, "L3a"; carried with comment-line edits only); no third-party code.  `fab_batched.flash_bias_attn_batched(..., stage_c=False)` is
op-for-op the per-sample kernel `opt_core.kernels.dtk_kernels.flash_bias_attn` with a sample grid axis (measured bitwise to the
per-sample loop in every cell of the provider table, both cards); `stage_c=True` adds aligned-pitch / peeled-loop variants whose bits
differ when a tail tile runs.  The reference path (no triton) imports `opt_core.kernels.dtk_kernels`.
