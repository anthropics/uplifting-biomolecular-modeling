# cueq_cache/ — the kit's cuEquivariance TriMul autotune cache (lever `cueq_tuned_tiles`)

`env.sh` points `CUEQ_TRITON_CACHE_DIR` at this directory in every kit mode. cuEquivariance's Triton triangle-multiplication kernel
(`fused_sigmoid_gated_dual_gemm`, cuequivariance_ops 0.11) reads its tile table for the running card from
`fused_sigmoid_gated_dual_gemm_forward_kernel_wrapper.<cc>.json` here, `<cc>` being the card's compute capability (a cc 10.3 card reads the
10.0 file); only when no such file exists does it open the table packaged inside its own wheel — a file here replaces the packaged table for
that card entirely. Either way the table is a pure lookup of tile shapes per GEMM key: the same kernel with the same per-element arithmetic
(`CUEQ_TRITON_TUNING` stays unset in every mode, so nothing is tuned or written at run time, and a key absent from the table runs the kernel's
built-in default tiles); the lookup was tested bitwise against the default tiles. With no file for the card, the packaged table applies — the
same tiles the stock route (`run.sh pred --mode off`) reads.

Shipped here — **kit-generated data** only, in the library's cache-file format: `*.9.0.json` (compute capability 9.0: H100 / H200) and
`*.10.0.json` (compute capability 10.0, also read on 10.3: B200 / B300). Both tables were produced by running this kit's own protenix-v2
workloads through the library's autotuner (`CUEQ_TRITON_TUNING`, with `CUEQ_TRITON_IGNORE_EXISTING_CACHE=1` so that no existing table is read
or merged) on cards of that compute capability, for the large-M TriMul shapes where the tile choice matters: the 9.0 table holds four entries
(M = 262,144 and 40,448 rows × N 512 · K 256 and N 256 · K 256), the 10.0 table six (M = 262,144, 126,976 and 34,304 rows × the same two
N · K pairs); every other shape runs the kernel's built-in default tiles. No entry of the library's packaged tables and no other NVIDIA SDK
data is included: `opt/protenix_opt/tests/test_cueq_tiles.py` checks EVERY `*.json` here against the installed wheel's packaged table of the
same name whenever `cuequivariance_ops` is importable (no entry may carry a packaged entry's measured time). Every kit-mode run states whether
a table for its card is present on its `[protenix-opt] LEVER name=F2.cueq_tuned_tiles … tiles=table_sm<NN> | tiles=packaged_defaults(no_table_sm<NN>) …`
line; cards without a file here (compute capability 8.0 among them) read the packaged table.

Regenerating or extending a table: run the model on the target card with `CUEQ_TRITON_TUNING=ONDEMAND` (or `AOT`) and
`CUEQ_TRITON_IGNORE_EXISTING_CACHE=1`; the library writes `<kernel>.<cc>.json` here. Cover every shape you run (the file replaces the packaged
table for that card), and never commit a file written without `CUEQ_TRITON_IGNORE_EXISTING_CACHE=1` — it would carry the packaged entries.
