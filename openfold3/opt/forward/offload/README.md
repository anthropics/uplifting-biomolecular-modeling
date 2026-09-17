# offload — `big`'s row-block units (`of3o`)

The hook directory the one-GPU `big` line puts on the import-hook chain (`modes.LINES[("big", "resident")]`, hook key `offload`).
The pair representation z (N×N×128 fp32 plus its transients) is what bounds OpenFold3's input size on one GPU; these units keep the
model's statements and apply them at stock's chunk boundaries with less resident memory. Nothing here is called directly: the mode
table sets the switches and `registry.py` names the levers.

## Line `resident`: the units (`OF3O_LAYER=1`)
* Each install is a switched unit of `apply_core`, default on (`OF3O_TRIMUL OF3O_TRIATT OF3O_TRANS OF3O_COND OF3O_INPUT OF3O_TEMPL
  OF3O_RUN_TRUNK OF3O_FORWARD`; `modes.OFFLOAD_UNIT_SWITCHES`). z stays on the GPU; the triangle multiplication reads its residual from a
  pinned host snapshot instead of a second GPU copy and streams the update in row blocks (`OF3O_ROWS`); the ending triangle attention
  runs on a lean transposed copy; the pair transition is applied in place per row block; the input embedder's relative-position features
  (`OF3O_INPUT_ROWS`, `OF3O_EMBED_ROWS`) and the recycling add (`OF3O_RECYCLE_ROWS`) are built per row block; the diffusion conditioning's
  pair path is computed once per item (`OF3O_COND_ROWS`); template pair features stay on the host and are embedded per row block
  (`OF3O_TEMPL_ROWS`, `OF3O_TEMPL_EMBED_ROWS`).
* The LayerNorm ≥ 2^31-element guard scoped to the model's own LayerNorm instances (`OF3O_LNSAFE=1`); the pinned host pool under
  `OF3O_PIN_POLICY=census` (a buffer over `OF3O_PIN_BUDGET_GB` is pageable, with a NOTE line and a count; `strict` refuses it by name; a
  pinned allocation the host refuses raises by name); the expandable-segments allocator; the fast-inference add-on's fast init chained
  behind this hook (`OF3O_KIT_LEVERS`).
* The confidence phase: from `modes.CONF_MIN_TOKENS` polymer tokens up, the chunked scorer with the BLOCKREDUCE TM backend
  (`OF3O_CONF_MODE=chunked OF3O_CONF_TM_BACKEND=blockreduce`, `modes.conf_gate`) beneath the package's row-block confidence head
  (`OPENFOLD3_OPT_CONFHEAD=1`, `opt/openfold3_opt/confhead.py`, hook `opt/openfold3_opt/hooks/confhead` ahead of `of3o` in the chain);
  below it upstream's confidence path (`OF3O_CONF_MODE=stock`, no confhead hook: the resolver drops the switches and the hook). The
  template data-pipeline change is off (`OF3O_TEMPL_FIX=0`; upstream fixes are the package's `--upstream-fix`). No CUDA graphs on this line.

Numerics: the same statements at the same chunk boundaries; the line's class is `modes.LINE_TIER` (tolerance: `fast`'s cells and bf16
run beneath these units).

## The chunk plan is the line's precondition
The units restate stock at stock's chunk boundaries, so the line needs a FIXED chunk plan: its runner YAML pins chunk size 16 with the
tuner off (`modes.BIG_BF16_C16_YAML`; `config/stock_predict_c16.yml` and `config/shipped_predict_c16.yml` carry the same plan on the
kernels-off and on the shipped attention configuration). Under upstream's chunk tuner an input that fits runs unchunked (`chunk_size:
None`), the PairBlock unit meets no chunked path, names it (`fallback site=PairBlock.forward chunk_size=null`) and the run exits PARTIAL
(rc 3) — fail-closed, not a defect. Under kernels with several diffusion samples stock runs the confidence head's 4 pair blocks
unchunked by its own rule (`prediction_heads.py`, `use_kernels and si.shape[0] > 1`); the units run them on stock's path and count them
(`census()` `conf_head_stock_blocks`, event `conf_head_unchunked`) — not a fallback: the pair-block unit covers the trunk.

## Composition seams
* ONE hook chain, by path: `OF3O_KIT_LEVERS=<dir>` executes that directory's `sitecustomize.py` after this hook installs its finder (the
  `big` line: confhead > of3o > cells > of3t_hook > of3_levers). The LAST installed finder fires FIRST, so `apply_core` runs after the
  chained hooks' installs: its `model_forward` is the outermost `OpenFold3.forward` binding and a prior hook's forward wrapper is
  superseded by name (`census()` `forward_prior`, event `forward_superseded`); no hook rebinds the sampler after import on this line
  (the graph switches are unset), so the pair cache's import-time boundary wraps stand. A line on another hook chain turns the
  conflicting units off (`OF3O_TRIMUL=0 OF3O_TRIATT=0` where fused cells own those methods).
* `tri_att_end_lean` bypasses `TriangleAttention.forward` and reads the installed dispatcher's backend flags itself
  (`of3t_levers._backend_kwargs(OF3T_TRIATT)`, recorded as `triatt_end_dispatch`).

## Fail-closed behaviour
The six stock-body fallbacks (`PairBlock.tri_att_start_end`, `PairBlock.forward`, `DiffusionConditioning.forward`, `OpenFold3.run_trunk`,
`OpenFold3.forward`, `TemplateEmbedderAllAtom.forward`) are NAMED EVENTS — `fallback(site)` counts them in `STATE["fallbacks"]` and logs
`[of3o] {"event": "fallback", ...}`; a failed LNSAFE or template patch raises instead of logging; the confidence path as run is recorded
(`STATE["conf"]`); the chunked scorer's TM backend is explicit (an absent BLOCKREDUCE raises, never the native backend silently);
`census()` is what the package reads (`stack.levers_record` → the exit tally's `offload_*` counters) — a `big` process with any fallback,
pageable buffer or native TM element exits 3, by name: a mode is all of its levers. `embed_zij_rows` materialises x_pred's sample
dimension on zij before the in-place distance add (stock's broadcast with several samples). `census()` requires the chunked scorer's
census once that path ran (an unreadable one raises); an incomplete LNSAFE tally is a named event (`ln_tally`).

## Files
    of3o/of3_offload.py      the hook module: the units, the host pool, the LayerNorm guard, the runner wrap, apply_core / apply_runner
    of3o/of3o_confidence.py  the chunked confidence scorer (PAE/PDE/pLDDT/TM in row blocks; conf_chunked)
    of3o/of3o_blockreduce.py the row-block TM reducer the chunked scorer uses
    of3o/sitecustomize.py    the hook file: post-import finder, always strict, chain-by-path (`OF3O_KIT_LEVERS`)
    config/shipped_predict_c16.yml                          the pinned chunk plan on the shipped attention kernel (DS4Sci on)
    config/stock_predict_c16.yml                            the pinned chunk plan on the kernels-off configuration
