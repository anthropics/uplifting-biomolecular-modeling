# offload — the single-GPU memory statements of `big`

The add-on directory behind `big` on one GPU at or above the item gate (`OF3O_MIN_TOKENS` polymer tokens): the hook module `of3o/` and
the row-blocked re-statements it installs over OpenFold3 0.5.0. The package (`opt/openfold3_ob0_opt/`) owns the CLI, the mode table
(`modes.LINES`), the gates and the lever registry, and exports every switch below; nothing here is called directly when you use `run.sh`.
Upstream's own memory options (the `low_mem` preset, `offload_inference` of the template module / MSA module / confidence heads, the
per-sample confidence cutoffs, the chunk-size tuner) stay upstream's, as the runner YAML sets them; the units here act on what remains
resident after them: GEMM operands stay stock's, tensor residency and the lifetime of the O(N²) transients change. Numerics: `big`'s class.

## Units (`of3_offload.apply_core` under `OF3O_LAYER=1`; each a switch, default on; lever names as printed on the kit's LEVER lines)

- `OF3O_TRIMUL` (`trimul_hostsnap`) — triangle multiplication streams its update in row blocks (`OF3O_ROWS`) from a pinned host snapshot
  of z instead of a second device copy, with upstream's PyTorch `_inference_forward` statements at their own block boundaries.
- `OF3O_TRIATT` (`triatt_lean`) — the ending triangle attention builds LN(zᵀ) column block by column block into one lean transposed
  buffer and adds its output in place (bias in upstream's `transpose_bias=True` orientation).
- `OF3O_TRANS` (`trans_inplace`) — the pair transition applied in place per row block.
- `OF3O_INPUT` (`input_rows`) — the input embedder's relative-position pair features (cyclic offsets included) built per row block
  (`OF3O_INPUT_ROWS`) and the pairformer embedding per row block (`OF3O_EMBED_ROWS`).
- `OF3O_RUN_TRUNK` (`recycle_rows`) — the recycling embedder's pair add per row block from the host snapshot (`OF3O_RECYCLE_ROWS`).
- `OF3O_COND` (`cond_once`) — the diffusion conditioning's pair path computed once per item in row blocks (`OF3O_COND_ROWS`).
- `OF3O_TEMPL`, `OF3O_TEMPL_HOST` (`templ_host`) — template pair features stay on the host and are embedded per row block; the per-template
  pair stacks are staged on the host and averaged per row block (`OF3O_TEMPL_ROWS`, `OF3O_TEMPL_EMBED_ROWS`, `OF3O_TEMPL_DEDUP`).
- `OF3O_FORWARD` — the confidence heads embed the trunk pair representation in place per row block; `OF3O_CONF_MODE=chunked
  OF3O_CONF_TM_BACKEND=blockreduce` (`conf_chunked`) takes the confidence scores from the row-chunked scorer (`of3o_confidence.py` over
  `of3o_blockreduce.py`, `OF3O_CONF_ROWS`) from `modes.CONF_MIN_TOKENS` polymer tokens; below it `OF3O_CONF_MODE=stock` runs upstream's path.
- `OF3O_LNSAFE=1` (`bigln_guard`) — LayerNorm over ≥ 2³¹ elements runs per leading-dim chunk, scoped to the model's own LayerNorm instances.
- Host pool (`host_pool`) — one pinned buffer per tag; a buffer past `OF3O_PIN_BUDGET_GB` is pageable with a NOTE line and a count under
  `OF3O_PIN_POLICY=census` and refused by name under `strict`; a pinned allocation the host refuses raises by name.

A unit that meets a call it does not restate (e.g. `PairBlock.forward` with `chunk_size: null`) runs the stock body and records a named
`fallback` event (`[of3o] {"event": "fallback", …}` on stderr); `census()` is the record the package turns into the exit line's `offload_*`
counters, and a `big` process with any fallback exits 3 by name. The units restate stock at stock's chunk boundaries, so the line runs
under a fixed chunk plan — `opt/openfold3_ob0_opt/big_bf16_c16_predict.yml` (chunk 16, tuner off, upstream's third-party pair kernels off,
`bf16-mixed`); a caller's runner YAML is composed under it.

Diagnostics (logging only; usable on the directory's own `PYTHONPATH` route, refused by name under a kit mode): `OF3O_LOG_EVERY_BLOCKS=k`
(a `pairformer_block` timing event on the first 3 blocks and every k-th), `OF3O_ABORT_AFTER_BLOCKS=n` (stop by name after n blocks; nothing is
written), `OF3O_ZSTATS=1` (per-block statistics of the pair / single representations), `OF3O_BIGCHECK=1` (logs and nan/inf-checks every torch
op whose output holds ≥ 2³¹ elements), `OF3O_LAYER=0` (no unit: stock under phase timers; `2` is refused by name).
Files: `of3o/sitecustomize.py` (the hook; `OF3O_KIT_LEVERS=<dir>` chains the next hook directory), `of3_offload.py` (the units, host pool, LayerNorm guard, runner wrap, diagnostics),
`of3o_confidence.py` (the row-chunked confidence scorer), `of3o_blockreduce.py` (the row-block TM / gPDE reducer it uses).
