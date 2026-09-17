# ESMFold2 design kit `k/` — autograd-path kernels, CUDA graphs and exact levers

Autograd-path (design / hallucination loop) fast kernels and exact levers for Biohub **ESMFold2-Experimental(-Fast)** and the ESM-C trunk it
embeds. Everything is an opt-in, instance-level (or, where marked, process-wide), reversible monkeypatch with `enable(model, …)` / `disable(model)`;
no source edits; nothing changes precision flags. The one composition per mode and the enable ORDER live in `opt/ef2inv_opt/fastkit.py`
(`COMPOSITION`, `enable`); this table is the parts list.

| module | lever | what | numerics class |
|---|---|---|---|
| `ef2_autograd_kernels.py` | `enable(model, trimul=None\|"bmm2", transition=None\|"refround_lean", checkpoint=…)` | the patched `FoldingTrunk` / `PairUpdateBlock` forwards: the checkpoint policy (`"block"` = stock, `"none"`, `"ckpt:m"`, `"keep:m"`, `"every:k"`) and, when asked, the **K-A2** triangle multiplication fwd+bwd (`trimul="bmm2"`: fused LN, one projection GEMM, Triton gating, bf16 `torch.bmm` contraction with fp32 accumulation, frozen-weight dX-only backward) and the **K-D3** pair Transition fwd+bwd (one Triton kernel, LN + W12 recomputed in backward, stock's rounding points, no dW); `enable_skip_unused_confidence(model)` honours `calculate_confidence=False`; the K-A / K-D / K-D2 kernels are selectable by argument, no mode uses them | policy: exact (bitwise); K-A2: bf16 contraction; K-D3: reordered accumulation, stock rounding points |
| `ef2_t16_transition.py` (+ `ef2_t16_nvjit.py`, `ef2_t16/`) | `engage()`; `agk.enable(model, transition="refround_lean", transition_fwd="t16")` | **t16 transition forward**: K-D3 lean's forward on ONE persistent warp-specialised sm_90a CUDA C++ / CuTe kernel (LN → W1‖W2 → SwiGLU → W3 → + x; d 256, hidden 1024), loaded from the shipped cubin `ef2_t16/sm_90a/ef2_transition_cute.cubin` through the CUDA driver API (`ef2_t16_nvjit`: manifest source key / sha256 / register + local-memory record checked; install canary once per process; no compiler at run time); descriptors over K-D3's own cached bf16 weights (no second copy); K-D3's backward unchanged. Kernel source, cubin and manifest: `ef2_t16/` (`ef2_t16/PROVENANCE.md`). Compute capability 9.0 only: elsewhere `engage()` records the reason and K-D3's own forward serves (`describe()["state"] == "stepped_aside"`) | tolerance (fp32 LN statistics, bf16 operands, fp32 accumulation; within one bf16 ulp of K-D3's forward) |
| `ef2_kd3_gemmswiglu.py` | `engage()` (after `agk`; before any capture) | **K-D3 W12 + SwiGLU forward**: K-D3 lean's OWN forward (where no out-only kernel serves it) computes hidden = SwiGLU(x̂ W12ᵀ) in ONE Triton kernel — two fp32 accumulators over the W1 / W2 columns, K-D3's rounding chain in registers, the `[M, 2h]` pre-activation never written — through the hook `agk._kd3_w12_swiglu` (**process-wide**, read per call, lean only); the W3 GEMM, the LayerNorm kernels, K-D3's backward and the memory plan unchanged. A per-compute-capability launch table (`_CFG_BY_CC`: sm_80); a card without an entry engages nothing and says so (`describe()["state"] == "stepped_aside"`, reason `cc_untuned:sm_NN`); on sm_90 the kit runs K-D3's forward on t16 and never asks it | K-D3's class (reordered fp32 accumulation inside the MMA tiles vs cuBLAS's; the same rounding points as K-D3's chain) |
| `ef2_bwd_ckpt.py` | `enable(model, n_tokens, num_passes=2, copies=, extra_reserved_bytes=, reserve_frac=0.10)` | the memory plan: picks the checkpoint policy that keeps the most blocks that fit, from the device memory in use at that moment, the bytes per pair position of the kernel set (`stock` / `agk3` / `fused3`), a transient bound and a reserve; writes it into the agk config; `describe(model)` = its `LEVER` line | exact |
| `ef2_trimul.py` | `enable(model, variant="fused")` | **fused triangle multiplication** (fast / big; K-A2 is the wired alternative no mode selects): LN_in + projections + gating (Triton), the contraction as two cuBLAS batched GEMMs over x8-padded channel-major bf16 operands, LN_out + output gating; frozen-weight backward (dX only; LN outputs recomputed); the four gated-GEMM launches through the per-compute-capability table `_GG_BY_CC` (sm_90: the kit's persistent weight-resident TMA kernels `_k_gg_dual_fwd` / `_k_gg_dual_bwd` — the vendored `fused_dual_gemm.py` kernels' arithmetic (the pinned transformers fork) re-laid as weight-chunk-resident row walks — and tuned tiles on the vendored `trimul_with_residual.py` kernels; tensor-equal to the vendored launches, `test_gemm_table_bitwise`; other cards: the vendored launches, `h.gemm` names it); composes with `agk.enable(trimul=None)`; residual included (the block's dropout-residual is made the identity on that path) | bf16 contraction, fp32 accumulation |
| | `enable(model, variant="cueq_tiles")` | tuned (TILE_M, TILE_N, warps, stages) entries for the cuEquivariance `fused_sigmoid_gated_dual_gemm` fwd / bwd-pregemm kernels at ≥ 65 536 rows (≥ 256 tokens), written into the library's tuning cache — **process-wide**, restored by `disable` | exact per library pin (TILE_K and the K loop unchanged; tensor-equal) |
| `ef2_fused_ln.py` | `enable(model)` | the dx-only LayerNorm backward the agk backwards call, as one row-major Triton kernel (rebinding `agk._ln_bwd_dx_only`, **process-wide**); `channel_major()` opt-in form = 1 ulp, not used by any mode | tensor-equal to the kernel it replaces |
| `ef2_stepgraph.py` | `enable(model, n_slots=2)` | trunk fwd+bwd CUDA-graph pool: one `GraphedGradSegment` per trunk pass of a step (fixed shapes; the pair mask a live static input; no retained autograd graph), wrapping whatever trunk forward is installed (install it LAST); `stats(model)` | exact (bitwise vs eager) |
| `ef2_esmc_graph.py` | `enable(model)` | CUDA-graph replay of the shared ESMC-6B forward per (shape, dtype, flags) | exact |
| `ef2_esmc_hoist.py` | `enable(model, anchor=1, graphs=True)`; `stats` / `release` / `disable` | **ESMC target-chain hidden-state hoist**: the trunk's `forward` wrapped OVER the ESMC graph's (install it after `ef2_esmc_graph`); per feature-pass call one host read-back of the ids / chain ids, then either the inner forward (first call of a shape, a changed target: its hidden states remembered) or the reduced forward — the blocks' own modules on the last chain's rows, the rotary at the rows' original positions (`seqlen_offset`), masked SDPA against key/value rows at their original columns, the states written into the live rows of the remembered `[81, B, L, D]` tensor — captured as a CUDA graph per shape; the first hoisted call of a shape is anchored (all rows tensor-equal with the inner forward) or the lever switches itself off by name; installs only on a compute capability in `PROVEN_CC` (sm_90) — elsewhere (`sm_80`: TE's cuBLASLt GEMMs pick their algorithm by row count) `enable` registers a handle with `stepped_aside='cc_unproven:sm_NN'` and leaves the forward untouched (`enable(model, cc=…)` overrides the probe) | exact (row-wise kernels on a row subset; bitwise vs the full pass on sm_90) |
| `ef2_pppl_graph.py` | `enable(esmc_model)` | the pseudo-perplexity forward+backward through ESMC-6B captured once and replayed | exact |
| `ef2_esmc_rope.py` | `enable(model)` | ESM-C `RotaryEmbedding.forward` (the torch branch) as one Triton kernel per tensor and direction with that branch's arithmetic; inert on the flash rotary branch | exact (tensor-equal fwd and bwd) |
| `ef2_loop_prep.py` | `enable(BD, memo=1, anchor=1, splice=True, pinned=True)` | the cookbook's `fold_and_get_distogram` re-plumbed: one host transfer, LM input built on the host, the ESMC-6B feature pass launched before the CPU featurisation, a one-entry hidden-state memo, the target's featurisation spliced from the first full one, the feature uploads staged through one pinned buffer with non-blocking copies (`PinnedUploader`: no per-tensor stream synchronisation); a run-time anchor compares its hidden states with the model's own path once | exact |
| `ef2_loop_pppl.py` | `enable(BD)` | `compute_esmc_pseudoperplexity_nll` / `build_gradient_mask` without host synchronisations (same RNG draws, `torch.where` mask placement, sorted integer gather) | exact |
| `ef2_esmc_overlap.py` | `enable(BD or app, esmc_model=)` | the pseudo-perplexity term launched on a side stream right after the design fold returns; enable it after `ef2_loop_pppl` so the side stream runs the sync-free body | exact (concurrency only) |
| `ef2_lazy_structure.py` | `enable(model)` | the one-step structure sample of design folds deferred behind a lazy attribute | exact |
| `ef2_pairbias_attn.py` | `enable(model, variant="hoist")` | the diffusion transformer's pair-bias projection computed once per `DiffusionStructureHead.sample` instead of per denoise step (`variant="fused"`: a fused pair-bias attention, fast class, not used by any mode) | hoist: exact |
| `ef2_sampler_graph.py` | `enable(model)` | one denoise step (`diffusion_module.forward`) captured per sample and replayed for the remaining steps; RNG outside the graph; a capture out of device memory retried once after `empty_cache` (`capture_retries`), any other failure named (`capture_failed`, `last_failure`) and that sample eager | exact |
| `ef2_bf16_confidence.py` | `enable(model)` | the confidence head under bf16 autocast | fast class (confidence logits only; no loss or gradient path) |
| `ef2_state_guard.py` | `StateGuard(mode=)` | process-global numeric state (TF32 / matmul precision / SDPA toggles / deterministic flags) snapshotted and checked | — (a check) |
| `_ef2_testmodel.py`, `test_ef2_*.py`, `run_tests_nopytest.py` | tests | GPU unit tests per lever (random-initialised modules; `test_ef2_loop.py` / `test_ef2_esmc_overlap.py` need the installed cookbook and weights) and their pytest-free runner | — |

## Running the unit tests

One GPU with the Biohub transformers fork (`transformers.models.esmfold2`, `transformers.models.esmc`) importable:

```bash
python k/run_tests_nopytest.py                       # every k/test_ef2_*.py; last line "RESULT passed=N failed=0"
python k/run_tests_nopytest.py test_ef2_trimul       # one module
pytest -q k/test_ef2_trimul.py                       # the same tests under pytest
```

## Using it in a design loop

The order matters: every rebinding before any capture, the memory plan after every kernel choice and before the pools it budgets for, the trunk
pool last. `opt/ef2inv_opt/fastkit.py` `enable` is the reference composition; by hand, for one model at N tokens (design use, frozen weights,
bf16 autocast):

```python
import sys; sys.path.insert(0, "k")
import ef2_autograd_kernels as agk, ef2_trimul, ef2_fused_ln, ef2_bwd_ckpt, ef2_stepgraph, ef2_esmc_graph
model.set_chunk_size(None)
# exact class: model.set_kernel_backend("cuequivariance"); ef2_trimul.enable(model, "cueq_tiles"); agk.enable(model, trimul=None, transition=None, checkpoint="block")
# fast class:
ef2_fused_ln.enable(model)
agk.enable(model, trimul=None, transition="refround_lean", checkpoint="block")   # K-D3; the policy is re-planned below
ef2_trimul.enable(model, "fused")                                              # the fused TriMul (the alternative: agk.enable(model, trimul="bmm2", …) and no ef2_trimul)
agk.enable_skip_unused_confidence(model)                                       # exact
ef2_bwd_ckpt.enable(model, n_tokens=N, num_passes=2, copies=2 if N <= 256 else 1)   # exact: the checkpoint policy that fits
if N <= 256:
    ef2_stepgraph.enable(model, n_slots=2)                                     # exact; launch-bound sizes only
ef2_esmc_graph.enable(model)                                                   # exact
```
`only_under_grad=True` (default) leaves no-grad folds (confidence trunk, critics) on the stock path.

## Known limitations
* bf16 autocast only (the backward kernels require bf16 operands); weights are used as detached bf16 copies cached per module
  (call `disable(model)` / `enable(...)` again after loading new weights).
* K-A2, the fused TriMul, K-D3 and the dx-only LN return `None` parameter gradients by design (design use, frozen predictor).
* graph pools: shapes fixed at capture; a trunk pool slot permanently holds one trunk pass of activations (the memory plan budgets for it);
  `cueq_tiles` and `ef2_fused_ln` are process-wide and must be enabled before any capture that would bake the previous kernels in.
* K-A2's stacked backward weight is a precomputed tensor per module, not a data_ptr-keyed cache (such a cache can alias freed memory under
  selective checkpointing).
