# `cueq_cache_shipped/` — optional user cache for cuEquivariance's triangle-multiplication tiles

The kit ships **no** tile table here. Every kit mode exports `CUEQ_TRITON_CACHE_DIR=<this directory>` (lever
`cueq_tuned_cache`); cuEquivariance (`cuequivariance-ops-cu12` 0.10.0) looks for
`fused_sigmoid_gated_dual_gemm_forward_kernel_wrapper.<major>.<minor>.json` for the running GPU and, when no such file is
present, uses the per-card table packaged inside the installed library — the same tiles the stock route reads. Nothing is
tuned or written at run time.

You may place your own tuned entries here: run once with `CUEQ_TRITON_TUNING=ONDEMAND` (minutes) or `AOT` (hours) and a
writable tree, and the library writes the file itself. Note that the file it writes merges the library's packaged entries
with the newly tuned ones unless `CUEQ_TRITON_IGNORE_EXISTING_CACHE=1` is also set; treat such a file as local data and do
not redistribute it. A user table changes tile parameters for the kit modes only (the stock route does not read this
directory), so `exact`'s bit-identity to stock is certified without one. The LEVER line reports what serves:
`tiles=packaged_defaults(no_table_shipped)` or `tiles=user_table_sm<cc>`.
