# ProGen2 kit — what changes vs stock

Stock = ProGen2 (`salesforce/progen` @ `c27a419c`) at the pin (STOCK.md). Each lever is a module the kit installs over one stock
function, class or load step at start-up under the kit mode `exact`; `off` loads none and runs the stock scripts in a clean subprocess.
A mode is all of its levers. Lever names are the ones printed in the `on=` field of the run's `ACTIVE` line (`opt/progen2_opt/registry.py`;
mode → composition: `opt/progen2_opt/modes.py`). The stock code is never edited: the generation route still calls `sample.py`'s own
`sample()` and `truncate()` after its `set_seed`, and the scoring route calls `ProGenForCausalLM.forward` verbatim and reduces with the
stock `cross_entropy`. The kit's one native library is its own prebuilt `libew_progen2.so`; nothing is compiled at install or run time.

## exact — outputs identical to stock

Route `sample` (`opt/serving/pipeline_v0_4`; load order as in `sample.py`'s `main`, the `--sanity` pass once before the first item):

- `oneread_mmap` — `pytorch_model.bin` read once (`torch.load(…, mmap=True)`) into a meta-constructed model with the tensors and dtypes
  `create_model(ckpt, fp16)` installs (as stored under `--fp16 true`, cast to float32 under `--fp16 false`; `oneread_loader.py`).
  Numerics: bitwise (loading only). Steps aside: never.
- `sampler_exact` — `model.generate` shadowed for the one stock `sample()` call: the stock warpers, softmax and `torch.multinomial` on
  the stock random stream, with the per-step integer bookkeeping (positions, mask, padding, stop test) fused into static buffers
  (`sampler_exact.py`). Numerics: bitwise (same arithmetic in the same order). Steps aside: never.
- `resident_rotary` — the stock `fixed_pos_embedding` sin/cos tables computed once on the device for the slot length and sliced per
  decode step instead of recomputed on the CPU per layer per step (`components/plm_transfer_t1_v0/kit_t1.py`). Numerics: bitwise
  (caching only). Steps aside: never.
- `static_kv` — the per-step `torch.cat` K/V cache replaced by static slot buffers per (layer, batch size) in the stock key/value dtypes
  (K float32, V the model's), written at position t and viewed over [0, t]; attention sees the same shapes and strides (`kit_t1.py`,
  `progen2_decode.py`). Numerics: bitwise (placement only). Steps aside: never — a batch size whose slots cannot fit (fit check: slot
  bytes × batch size against free device memory less `STATIC_KV_FIT_MARGIN` = 4 % of the card, idle batch sizes released first) fails
  that item by name (`OUT OF MEMORY`), exit 1, no stock fallback. Slot length: the smallest of `KIT_BUCKETS` (256, 512, 1024, 2048)
  covering `--max-length`, capped at `n_positions`; a longer item re-installs the levers at the next bucket.

Route `score` (`opt/forward/engines/progen2/kits/v0_score_r3_1` composed over `v0_ew`; load order as in `likelihood.py`'s `main`,
batch 1 per direction; under `--sanity true` likelihood.py's own sanity section runs before each item):

- `rotary_tables` — `fixed_pos_embedding` (a CPU arange/einsum, a host-to-device copy and device sin/cos once per layer per forward)
  evaluated once per (device, dim) for `n_positions` positions; calls receive views of those bytes. Numerics: bitwise (caching only).
  Steps aside: never.
- `ew:gelu` · `ew:rotary` · `ew:residual` · `ew:glue` · `ew:ln` — the MLP activation (`gelu_new`), the qkv split + rotary application,
  the block's two residual adds, the attention scale / causal mask / mask add, and the LayerNorms, each as one fused CUDA kernel
  reproducing the stock rounding chain op for op. Kernel: the kit's own `v0_ew/kernels.cu` (CUDA C++; SASS sm_80 / sm_90 / sm_100 +
  PTX compute_90, CUDA 12.8) prebuilt as `libew_progen2.so`, loaded through ctypes against `BUILD.json`. Numerics: bitwise (same
  arithmetic in the same order, explicit round-to-nearest intrinsics). Steps aside: all five together, by name, when the library is
  missing, does not load, binds another CUDA runtime major, or has no image for the card (below compute capability 8.0) —
  `ew:gelu, ew:rotary, … cannot run: <reason>`; the mode then refuses (exit 3).
- `one_forward_per_direction` — sum and mean log-likelihood of a direction from one forward's logits (the stock CLI runs one forward
  per reduction per direction), reduced by the stock `cross_entropy` over token ids 5..29. Numerics: bitwise. Steps aside: never.
- `host_pipeline` — tokenisation, pinned host↔device copies behind a CUDA event and row assembly on a host thread, overlapping the
  GPU. Numerics: bitwise (scheduling only). Steps aside: never.

After the items the scoring route checks its own composition (the two kits' rotary tables equal byte for byte; the per-forward kernel
counters equal the lever set × forwards); a contradiction is `NOT ACTIVE: partial activation — …`, exit 3, outputs kept. The scoring
kit's exact-length batching (`Scorer(max_rows=…)`) and resident-fp16 weights (`apply(resident_fp16=True)`) exist in the code and are
not used by `progen2-opt score`.

## Every mode

- The stock exception STACK (STOCK.md §Stock exceptions) applies identically on every mode, `off` included. No upstream fixes ship.
- Refused by name in every mode, `off` included: the stock files at `$PROGEN2_STOCK_DIR` off the digests in `stock/PINS.json`
  (`NOT ACTIVE: stock files differ from the pinned commit (…) …`, exit 3). Refused under `exact` only: no CUDA device or `--device cpu`,
  the kit tree incomplete, a `sample` lever the load did not install (`<lever> cannot run: not installed by the load`).
- Named and engaged on, never refused: interpreter / torch / transformers / tokenizers off the pins (`stack … notes=stack differs from
  the pinned one …`), a card outside H100 · H200 · B200 · A100, `--fp16 false` / `--rng-deterministic false` / another `--device`
  (`notes=settings outside the tested defaults: …`), `ew` kernels JIT-compiled by the driver from the compute_90 PTX on a card whose
  compute-capability major has no SASS image in the library (majors 8, 9 and 10 have one), transformers'
  `activations.py` off the pinned bytes (`ACTIVE … notes=…`).
- Outputs: a single call prints the stock block on stdout and writes nothing; `--input FILE --out_dir DIR` writes
  `DIR/items/<item_id>/block.txt` per item and nothing else (`score`: likelihood.py's stdout without its timer lines and the closing
  `done.`). A sequence longer than `n_positions` fails by name for that item; the job goes on and exits 1. Stderr lines: `stack …`,
  `ACTIVE …`, `load variant=<size> t=<s>s slots=<batch sizes held> max_length=<L>` (`sample`), `APPLIED …` (`score`), `ready …`,
  `item <id> <s>s`, `PEAK item=<id> alloc_gib=… reserved_gib=… pid=…` (`sample`), `score path ok …`, `EXIT … kit_modules=…`.

## Switches

- `--mode exact|off` and `--input FILE --out_dir DIR` are the kit's only switches (FILE: JSON lines, one object per item, keyed by the
  per-item flag names — `item_id`, `context`, and for `sample` `max_length`, `num_samples`, `t`, `p`, `rng_seed`; a missing key takes the stock
  default; `progen2-opt <verb> …` / `python -m progen2_opt <verb> …` are the same entry points as `bash run.sh <verb> …`); no environment variable changes lever behaviour
  (deployment-path variables, names and defaults: STOCK.md §Variables).
