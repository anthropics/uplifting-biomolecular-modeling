# RoseTTAFold3 kit — what changes vs stock

The stock interpreter is never modified. Kit modes run on `opt/venv`, whose `rf3` package carries the add-on's five files — four of
upstream's, modified, plus the new `graph_flags.py` (`opt/forward/rf3_xattempt_addon/patched/rf3/`: the sampler CUDA graph, capture-safe
rewrites of a few masked ops, the step-invariant conditioning cache; on under `RF3_CUDAGRAPH=1 RF3_HOIST=1`, which every kit mode exports).
Trunk kernels are bound at run time by `opt/forward/rf3_fpf_trimul_addon/` (class-level rebinding, no file edits); the diffusion megakernel
is `opt/forward/rf3_mk_dit_addon/`; `opt/rosettafold3_opt/` resolves modes to levers (`modes.py`, `registry.py`), composes the memory levers
over `opt_core.mem` (`big.py`) and runs the multi-GPU pair stack (`rowpair.py`). Every lever prints `LEVER name=<lever> state=…` at exit.
Numerics: **bitwise** = the bits of `--mode off` on the same card (given a deterministic `scatter_mean` in both interpreters; stock's uses
CUDA atomics); **tolerance** = bf16 operands, fp32 accumulation, reproducible run to run, inside stock's seed-to-seed variation;
**placement** = same arithmetic, only where tensors live or when work runs changes.

## exact — outputs identical to stock

This mode binds FlashPairformer kernels via the shared core (opt_core).

- `graph`, `graph_safe_ops`, `warm`, `hoist` — each diffusion roll-out captured as one CUDA graph per sampler call (RNG pre-drawn, eager
  warm-up steps before a process's first capture, the atom encoder's masked ops in dense / slice form so they capture) and replayed; the
  denoiser's step-invariant sub-graphs (pair and atom-encoder conditioning, every block's pair bias) computed once per roll-out. Bitwise.
- `fpf_tg`, `fpf_sapb` — the 48-block pairformer stack captured once per token count and replayed as one CUDA graph per recycle, within a
  per-card token budget (`tgbudget.py`; above it uncaptured, `levers_budget_skipped=fpf_tg(…)`); attention pair bias keeps stock's ops with
  a per-call host-to-device constant cached so the block captures. Bitwise.
- `fpf_xatt`, `fpf_xmul`, `fpf_xln` — triangle attention, the triangle multiplication's cuEquivariance call and every `nn.LayerNorm` are served through the
  exact tiers of `opt_core.kernels.triattn` / `trimul` / `ln` where they have a row for the card, stack and shape — triangle attention's row is a fused kernel
  bit-identical to the stock op, on H100 stacks the core vouches for; per-head stock calls above the core's split size where its table says so — the stock op by name otherwise. Bitwise.
- `fpf_smsa` — the MSA pair-weighted averaging keeps RF3's pair-side statements and takes the sequence-side reduction from `opt_core`;
  each (token, depth) class is checked `torch.equal` against stock's statements at first call and runs them by name if that fails. Bitwise.
- `xtr` — the SwiGLU transitions through `opt_core.kernels.transition`'s exact tier where the provider lists it bitwise for the card and
  width, stock's statements by name elsewhere (`XTR … route:<key>:<word>`). Kernel: FlashPairformer transition via the shared core —
  Triton; exact variant. Bitwise.
- `confhoist` — the confidence head's sample-invariant prologue (fp32 casts, whole-tensor layer norms of the trunk outputs) run once per
  prediction instead of once per sample; engages only while `ConfidenceHead.forward` has the pinned source. Bitwise.
- `hostlean` — `validation_step`'s two symmetry resolutions without per-element host synchronisations. Output-identical.
- `prefetch`, `awrite` — the next item's featurisation in one helper process forked before CUDA, overlapping the current forward; the
  item's output writers on a background thread. Output-identical; in-process (`state=off reason=…`) for a one-item run, `--n_gpu > 1`,
  or a platform that refuses the side process.
- `mem` — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.5` and the allocator's free blocks released
  at the roll-out seams. Placement.

## fast (default) — within stock's seed-to-seed variation

Keeps `graph`, `graph_safe_ops`, `warm`, `hoist`, `fpf_tg` (smaller budget; none on compute capability 8.0), `fpf_xmul` for the
template track's c=64 triangle multiplications, `fpf_xln`, `confhoist`, `hostlean`, `prefetch`, `awrite`, `mem`; replaces the rest:

- `fpf_trimul` — `TriangleMultiplication.forward` (c_z 128) through `opt_core.kernels.trimul`'s fast tier; calls below the provider's
  token floor stay on the exact row. Kernel: FlashPairformer triangle multiplication via the shared core — Triton (launch cells per
  compute capability); fast variant. Tolerance.
- `fpf_gflash` — triangle attention with a fused LayerNorm + cast + transpose prologue and one concatenated q|k|v|g projection. Kernel:
  FlashPairformer triangle attention via the shared core — CUDA-native (sm_90a) on H100, Triton on sm_80, the stock op by name where
  the provider has no row; fast variant. Tolerance.
- `fpf_ttr` — the pair / MSA transitions, hidden activation tiled inside the kernel; below its size gate stock's transition serves the
  call (`levers_size_gated=`). Kernel: FlashPairformer transition via the shared core — Triton; fast variant. Tolerance.
- `fpf_apb`, `fpf_res` — the pairformer's attention pair bias (LayerNorm + bias projection) as one Triton kernel; the residual adds of
  the triangle-multiplication and transition outputs fused into those kernels' epilogues. Tolerance.
- `fpf_msa` — the MSA module's outer-product mean and pair-weighted averaging on `opt_core.ops.msa_opm` / `msa_pwa` (Triton); on compute
  capability 8.0 the pair-weighted averaging alone, the outer-product mean staying stock's (`msa_aside=`). Tolerance.
- `mkdit` — the diffusion module's 24-block token transformer on the `opt/forward/rf3_mk_dit_addon` megakernel (Triton, sm_90 / sm_80 tile
  rows; one launch per sample per step inside the sampler graph; stock's blocks below its token floor, size-gated). Tolerance. `confln` —
  the confidence prologue's three whole-tensor layer norms by a grid-wide mean / variance reduction. Tolerance.
- `fpf_dattn` — an adapter component (the diffusion transformer's pair-biased attention as one `scaled_dot_product_attention` call) that
  no shipped mode selects; its LEVER line reads `state=off`.

## big — lowest peak GPU memory

`fast`'s row with the CUDA graphs off (`RF3_CUDAGRAPH=0`; `hoist` stays on), `xtr` off, `mkdit` replaced by `dtk`, the fused triangle
attention stepping aside for row-block processing, and the memory levers of `big.LEVERS` (printed as `big_<lever>`; each alone is
placement) over `opt_core.mem`. Tolerance. A memory lever that ran the stock path on part of its units makes the run exit 3 unless
`--allow-partial` (`ROSETTAFOLD3_BIG_ALLOW_PARTIAL=1`), which records `PARTIAL allowed` instead.

- `dtk` — the diffusion transformer's pair-biased token attention on `opt_core`'s flash kernel (online softmax, no [I, I] logits held).
- `triatt_chunk` — the trunk's triangle attention in query-row blocks on the stock cuEquivariance op (full bias once, rows streamed).
- `transition_chunk` — transitions in row blocks of the leading dimension wherever the fused transition kernel does not own the call.
- `opm_chunk` — the MSA outer-product mean over row blocks.
- `cond_chunk` — the diffusion pair conditioning over row blocks.
- `atom_pair_local` — the atom-pair conditioning built directly in the atom transformer's window form instead of [L, L, c].
- `confidence_offload` — pae / pde logits staged per sample on pinned host memory; their consumers run per sample on the device.
- `feature_park` — trunk-only input features (MSA stack, template conditioning, fp32 originals) parked on the host while the sampler
  and confidence head run.

### `--n_gpu P`

- `rowpair` (P = 2, 4, 8 GPUs of one host; another mode, or another P, is refused by name) — the pair stack row-sharded over the P GPUs
  through `opt_core.mem.rowpair`: each rank holds N/P pair rows; triangle multiplication as ring / gathered contractions, triangle
  attention bias-gathered per row block, the outer-product mean, pair conditioning, confidence logits and the diffusion blocks' pair
  bias born as each rank's rows; the fused trunk kernels serve each shard above `rowpair.TP_KERNEL_MIN_TOKENS`. Kernel: FlashPairformer
  triangle multiplication via the shared core — Triton (`fpf_trimul_rows` for the c=64 template stacks). Rank 0 featurises each query and
  broadcasts the features (`FEATS … feats_ranks_equal=yes`; disagreeing ranks refuse the run); templated queries build the noised-template
  distogram per pair row, never as an N×N tensor. `triatt_chunk`, `opm_chunk`, `cond_chunk`, `confidence_offload` report
  `site_owned:rowpair`; `prefetch` / `awrite` run in-process (`conflict:n_gpu`). Sharded reductions reorder sums: tolerance, as `big`.
- Placement words exported to the ranks (`modes.TP_ENV`; values printed in the `LEVER name=rowpair` census): the initial pair shard and the
  raw MSA stack held on pinned host memory between uses, the trunk shard released or parked under the confidence head, in-place pair-block
  transposes, row-block transients sized from a per-rank budget, CPU threads split per rank, one leased pinned-host slab per rank, and the
  diffusion blocks' pair bias written by a fused LayerNorm + projection (`opt_core.kernels.ln_proj`).

## Every mode

- No stock exception: `off` is upstream as shipped; no kit mode changes an input, a setting or an output file's format. A mode is all of
  its levers on the running card, or it refuses by name.

## Switches

- `MODEL_OPT_LEVERS_OFF=<lever>[,…]` — a mode without the named levers (`withheld=<names>` on the ACTIVE / EXIT lines); `compile`
  (`--no-compile`) is a no-op reported as `compile=none`. `--allow-partial` / `ROSETTAFOLD3_BIG_ALLOW_PARTIAL=1` — see `big`.
  `RF3_CUDAGRAPH`, `RF3_HOIST`, `FPF_RF3_TG_MAX` and the `ROWPAIR_*` words are set by the mode, never by the user; variables: STOCK.md.
