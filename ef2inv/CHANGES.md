# ESMFold2 binder design kit — what changes vs stock

Stock = the cookbook `binder_design.py` at the pin (STOCK.md), at `--chunk-size none --kernel-backend cuequivariance`. Each lever is an instance-level,
reversible patch installed at start-up under a kit mode on the loaded models or on the imported cookbook module — `opt/ef2inv_opt/fastkit.py` installs
ONE composition per mode (`COMPOSITION`); the lever modules are `opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k/ef2_*.py` — and `off` loads none.
A mode is all of its levers: one that does not install or confirm without a declared reason makes the run NOT ACTIVE (exit 3); one that steps aside by
a declared word says so (`LEVER name=… state=stepped_aside reason=<word>`) and the run completes. Names below are the `LEVER name=…` names. Enable order:
loop-level levers and the per-step state guard, then every kernel / helper rebinding, then the memory plan, then the CUDA-graph captures.

## exact — outputs identical to stock

Bit-identical to stock (`off --chunk-size none --kernel-backend cuequivariance`) when both run `--det 1` on the same card.
- `ef2_bwd_ckpt` — the memory plan. Stock checkpoints every `PairUpdateBlock` and recomputes it in backward; the plan keeps the activations of as many
  pair blocks as fit (rule `budget`: chosen at enable from the free device memory, the kernels' bytes per pair position and what the graph pools will
  take; policies `none` / `ckpt:m` / `block`), written into the patched trunk forward of `ef2_autograd_kernels`. Bitwise. A design that does not fit
  ends by name (exit 4).
- `cueq_tiles` — the stock cuEquivariance triangle-multiplication kernels launched with a tuned tile per compute capability (entries for 9.0 and 8.0;
  the K loop unchanged), process-wide: `exact`'s design steps and every mode's hero-critic folds (`ef2_trimul` `variant="cueq_tiles"`). Tensor-equal to
  the library's own launch. A card without an entry keeps the library's tile and says so; `reason=user_kernel_backend:<v>` under `--kernel-backend fused|None`.
- `skip_unused_confidence` — `ESMFold2ExperimentalModel.forward` honours `calculate_confidence=False`, which the cookbook passes on every step outside
  the confidence phase while stock runs the head regardless; the head consumes detached tensors, so no loss or gradient changes. Bitwise.
- `ef2_lazy_structure` — the one-step structure sample every design fold draws and the loss never reads sits behind a lazy attribute. Bitwise.
- `ef2_pairbias_attn`, `ef2_sampler_graph` — in the diffusion sampler the pair-bias projection is computed once per sample instead of per denoise
  step, and one denoise step is captured as a CUDA graph per sample and replayed (RNG outside the graph); design models and hero critics. Bitwise.
  A capture that fails is counted and the eager step serves (`fallback=` on the LEVER line).
- `ef2_esmc_rope` — ESM-C's differentiable torch rotary (the branch `--upstream-fix EF2INV-0003` selects) as one Triton kernel per tensor, same
  arithmetic; inert while the flash-attn rotary is bound. Bitwise.
- `ef2_loop_prep` — the fold entry re-plumbed: one device→host transfer of the designed sequence per fold, the ESMC-6B feature pass launched before the
  CPU featurisation, a one-entry hidden-state memo, the target's featurisation reused from the first full one (compared field by field once), features
  uploaded through one page-locked staging buffer. Bitwise (hidden states checked against the model's own path at the first call).
- `ef2_loop_pppl` — the pseudo-perplexity body without host synchronisations (same draws, same gathers). Requires `ef2_loop_prep`. Bitwise.
- `ef2_esmc_overlap` — the pseudo-perplexity term launched on a side stream right after the fold, so its ESMC-6B forward+backward runs concurrently
  with the structure backward. Requires `ef2_loop_pppl`. Bitwise.
- `ef2_esmc_graph` — the 80-layer ESM-C forward that feeds every fold, captured once per (shape, dtype, flags) and replayed. Bitwise.
- `ef2_esmc_hoist` — after the first full pass of a design the ESM-C forward runs for the binder's rows only (under the per-chain attention mask the
  target rows' states cannot change within a design), as a CUDA graph; the first hoisted pass of every shape is compared tensor-equal with the full
  pass. Bitwise. Steps aside `reason=cc_unproven:sm_80` on compute capability 8.0 (TransformerEngine's GEMMs are not row-count invariant there).
- `ef2_pppl_graph` — the grad-enabled ESM-C pass of `compute_esmc_pseudoperplexity_nll` captured with its backward and replayed every step. Bitwise.
- `ef2_stepgraph` — the folding trunk's grad-enabled forward and backward captured per slot (2 slots; the pair mask a live input) and replayed every
  step, for complexes of at most `fastkit.POOL_MAX_TOKENS` (256) tokens; above it the lever is off by rule (`state=off … rule=<=256`) and the trunk
  runs eager. Bitwise.
- `featurisation_cache` — the tensors `prepare_esmfold2_tensors` builds for a (sequence set, max_atoms) pair, built once per process; inputs it cannot
  key pass through to the stock function (`fastkit.py`). Bitwise.
- Model switches: `exact` pins `set_chunk_size(None)` + `set_kernel_backend("cuequivariance")` on every loaded model; a differing `--chunk-size` /
  `--kernel-backend` is applied as given (`chunk` `state=user`, `cueq_tiles` stepping aside as above).

## fast — within stock's seed-to-seed variation

Numerics class: bf16 operands, reordered fp32 accumulation — never bitwise. The default mode. `exact`'s levers, with the design models' pair-trunk kernels replaced under grad and the memory plan (`budget`) filling the card:
- `trimul` — `TriangleMultiplicativeUpdate` forward + backward (+ residual) of every grad-enabled pair block on one fused path: LayerNorm + projections
  + gating in Triton, the triangle contraction as two cuBLAS batched GEMMs over channel-major bf16 operands zero-padded to a multiple of 8 tokens (fp32
  accumulation), a frozen-weight backward that computes the input gradient only and recomputes the LayerNorm outputs; gated-GEMM launches from a
  per-compute-capability table (9.0: persistent weight-resident TMA kernels; 8.0: tuned tiles; other cards: the default launches, named on the LEVER
  line). `ef2_trimul` `variant="fused"`; replaces `cueq_tiles` on the design models (the critics keep it).
- `agk_transition` — the pair `Transition` forward + backward as fused Triton kernels (LayerNorm → W1‖W2 → SwiGLU → W3 + residual; LayerNorm and W12
  activations recomputed in backward, no weight gradients; stock's rounding points, reordered accumulation). `ef2_autograd_kernels` `transition="refround_lean"`.
- `ef2_t16_transition` — that forward as one persistent warp-specialised CUDA C++ / CuTe kernel for sm_90a, loaded from the cubin shipped under
  `k/ef2_t16/sm_90a/` through the CUDA driver API (no compiler at run time; sha256, source key and register / spill record checked against
  `manifest.json`, an install canary once per process). Within one bf16 ulp of `agk_transition`'s forward. Steps aside on any compute capability other
  than 9.0 or on a cubin / manifest mismatch; `agk_transition`'s own forward serves.
- `ef2_kd3_gemmswiglu` — where `agk_transition`'s own forward serves, its W12 projection with the SwiGLU as the GEMM epilogue in one Triton kernel (the
  `[M, 2h]` pre-activation never written; same rounding points), from a per-compute-capability launch table (entry: 8.0). Steps aside
  `reason=cc_untuned:sm_NN` without an entry; not asked (`t16_serves`) where `ef2_t16_transition` runs the forward.
- `ef2_fused_ln` — the LayerNorm input-gradient inside those backwards as one row-major Triton kernel without the dγ/dβ reductions; tensor-equal to
  the kernel it replaces.
- `ef2_trimul_nosave` — the first forward of each checkpointed pair block (the pass whose activations checkpointing discards) computes its two triangle
  multiplications on the shared core's forward-only kernels (`opt_core.kernels.trimul`, tier word `fast`: the prebuilt sm_90a kernel where the stack
  has it, else its Triton kernel) and records no autograd graph; the recompute in backward and every kept block run `trimul`. Compute capability 9.0
  only (steps aside by name elsewhere); inert where the plan checkpoints nothing.
- `ef2_bf16_confidence` — the confidence head of the confidence-phase steps under bf16 autocast, design models only (the hero critics keep stock precision).
- `chunk` — the fork's own `set_chunk_size(64)` on the design models: the grad-mode kernels never read it, so it tiles the no-grad confidence steps only
  (same digits, lower peak there). Yields by name to a user `--chunk-size` (`state=user … yielded=64`).
- `critic_switches` — the four hero critics (stock models the kernels above never touch) on `set_chunk_size(None)` + `set_kernel_backend("cuequivariance")`
  right after load, read back after the loop; a user `--chunk-size` / `--kernel-backend` covers every model instead (`scope=all`).

## big — lowest peak GPU memory

Numerics as `fast`. `fast`'s levers with `ef2_bwd_ckpt` pinned to its floor (rule `memory_floor`, policy `block`: every pair block of every grad pass checkpointed, kept 0;
the estimate and budget still printed), `ef2_stepgraph` off at every size (`rule=never`), and `ef2_pppl_graph`, `ef2_esmc_graph`, `ef2_esmc_hoist`,
`ef2_sampler_graph`, `ef2_pairbias_attn`, `ef2_esmc_overlap` switched off by name (`LEVER name=<lever> state=off … reason=mode:big`) — each holds
activations or a private graph pool resident. Everything else as `fast`; `ef2_trimul_nosave` then serves every block.

## Every mode

- The stock exception (STOCK.md §Stock exception, `opt/ef2inv_opt/patches.py`) applies identically on every mode, `off` included. On request only,
  every mode alike: `--det 1` (`opt/ef2inv_opt/det.py`) and `--upstream-fix EF2INV-0003` (`upstream_issues/`; README §Known upstream issues).
- Kit modes: a process-state guard (`k/ef2_state_guard.py`) snapshots torch's global numeric switches before the levers go on and asserts them
  unchanged at every design step; the launcher sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless the operator set that variable.
- Quiet fallback paths inside the lever modules (a failed capture, an anchor that did not hold, a card or batch a lever is not priced for) are counted
  by the levers and read back after the loop: `LEVER … fallback=<n>`, `EVIDENCE levers=applied fallback=<names|none>` (`opt/ef2inv_opt/evidence.py`).

## Switches

- `MODEL_OPT_LEVERS_OFF=<lever>[,…]` — a kit mode without the named levers (`ablated=<names>` on the ACTIVE line, `state=ablated` on their lines); a
  lever that only serves an ablated one goes with it (`ef2_loop_pppl`, `ef2_esmc_overlap` with `ef2_loop_prep`; `ef2_fused_ln`, `ef2_t16_transition`,
  `ef2_kd3_gemmswiglu` with `agk_transition`); a name the mode does not compose is
  exit 3; `compile` (= `--no-compile`) is accepted on every mode and switches nothing — the kit adds no `torch.compile` lever (`compile=stock:not_kit_added`).
- `--chunk-size none|N`, `--kernel-backend fused|cuequivariance|None` — upstream's setters on every loaded model, read back per model (`MODEL-SWITCH …`
  lines), never refused, recorded under `opt_manifest.json` `overrides`. `--allow-partial` — a lever refusal is recorded and the run returns 0, not 3.
- Variables that change behaviour (defaults: STOCK.md §Variables): `EF2INV_JIT_CACHE`, `MODEL_OPT_JIT_ROOT`, `EF2INV_REQUIRE_FAST_ENV`.
