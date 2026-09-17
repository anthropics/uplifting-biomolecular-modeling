# chai1_eager — Chai-1 (chai_lab 0.6.1) as eager PyTorch: trunk, diffusion module (hoisted + CUDA-graphed step), confidence head and embedders; the `tier1` drop-in stack

`models_v2/trunk.pt` of chai-lab 0.6.1 (48 pairformer blocks + MSA module + template embedder) re-expressed as plain `nn.Module`s
(`chai1_eager/trunk.py`) whose `state_dict` keys are those of the scripted module, plus a generic transpiler (`chai1_eager/ts2eager.py`) that runs
an exported archive's printed TorchScript op-for-op in eager mode (any of the six Chai-1 components). The trunk with the copy-free layouts of
`chai1_eager/kernels.py` (`trimul_bmm`) is bitwise-identical to the scripted trunk on the pinned stack; the transpiled denoiser
step is bitwise to the scripted one under the deterministic recipe and inside its run-to-run floor at default numerics. The six scripted modules that
chai_lab loads stay the weight source (memoised); nothing is downloaded here.

    import chai1_eager.stack as S
    h = S.install(levers="tier1")       # (hoister="hoist2" for the value-taint hoister below; "base" is the default)
                                        # patches chai_lab.chai1.load_exported: eager trunk + copy-free layouts, the denoiser step hoisted
                                        # (step-invariant work once per sample) and replayed from a CUDA graph (un-graphed under
                                        # torch.use_deterministic_algorithms, install's own rule), flat eager embedders / confidence head
    chai_lab.chai1.run_inference(...)   # unchanged call; h.restore() undoes the patch; h.stats() -> hoist / capture records
    # lower level: S.Components / S.build_parts / S.make_loader (what chai1_fastln.stackx composes over); chai1_eager.trunk.load_trunk + CFG;
    # chai1_eager.ts2eager.load_eager_component; chai1_eager.hoist.HoistedForward (precompute once per sample, step / step_graphed per call)

Layout
    chai1_eager/trunk.py      structured modules: Trunk > TemplateEmbedder / MSAModule / Pairformer(48 x PairformerBlock{TriangleMultiplication,
                              TriangleAttention, Transition x2, AttentionPairBias}); numerics policy of the export mirrored exactly (LN fp32, GEMM/SDPA bf16,
                              trace chunking rules); CFG switches: precast_bf16 (cache bf16 weight copies; bitwise-neutral), trimul_impl / triattn_impl
                              plug points, record_ranges (torch.profiler ranges per op class).
    chai1_eager/kernels.py    trimul_bmm (channel-major projections + strided-batched matmuls; removes einsum's permute copies).
    chai1_eager/ts2eager.py   TorchScript code-tree parser + shim namespace (torch.X schema names, int dtype codes, prim ops) + SSA dead-value freeing.
    chai1_eager/hoist.py      HoistedForward / HoistedForward2: split a traced denoiser method into its step-invariant and per-step parts; CUDA-graph capture of the step.
    chai1_eager/stack.py      the drop-in: Components, EagerTrunkWrapper, HoistedDiffusionWrapper, FlatWrapper, build_parts / make_loader / install.

Limits: B = 1 layouts (as chai-lab runs); crop sizes are whatever the caller pads to (the eager trunk is shape-generic, the scripted one has 9 fixed
crops); bitwise equality of the layouts and of the transpiled step holds per torch / CUDA stack — re-check it on another stack under the deterministic recipe.
Derived from Chai-1 (Chai Discovery, Apache-2.0) — see NOTICE; LICENSE reproduces upstream's licence. No weights are distributed here.

## The denoiser step's two hoisters (`chai1_eager/hoist.py`; `HoistedDiffusionWrapper(hoister=...)`)

* `base` — `HoistedForward`: a root statement of the traced `forward_<crop>` is per-step when it reads ANY name derived from
  `atom_noised_coords` / `noise_sigma` (name taint); everything else runs once per sample (`precompute`) and is cached.
* `hoist2` — `HoistedForward2`: taint propagates through VALUE reads only; a statement that reads just the metadata of a step-dependent tensor
  (`torch.size` / `new_empty` / `ops.prim.device` — e.g. the diffusion-sample axis length used to `expand` step-invariant conditioning) stays
  step-invariant. The atom-pair update block, the two N=16 atom-pair LayerNorms, the six blocked pair-bias projections + `masked_fill`s and the atom
  transformer's AdaLN scale / shift / gate projections leave the step. `precompute` runs the full traced forward once on the first call's inputs
  (every metadata read resolves to its real, static value) and caches what the step reads; a cached name a per-step statement mutates in place is
  snapshotted right before its first per-step use and cloned at step entry; a step-invariant in-place statement an earlier per-step statement could
  observe is demoted to per-step (`stats["demoted"]`). Same ops, same operands, same order inside every dependency chain: bitwise to the base
  hoister's step and to the un-hoisted call wherever each op is run-to-run deterministic. Cost: a larger hoist cache and a higher per-step peak
  allocation than `base` (the stand-down hook below hands an item back to `base` when a memory line asks).

### Scalar-cache folding
`HoistedForward2` stores `int(x)` / `float(x)` / `bool(x)` for a cached CPU 0-dim tensor the step reads only through that one cast (`scalar_only_uses`;
`stats["n_scalar_folded"]`): bitwise identical, and the per-step function carries no data-dependent scalar read at those sites when traced
(torch.compile / make_fx / export). `hf.src` keeps the generated `_pre` / `_step` source.

### Per-item stand-down hook
`HoistedDiffusionWrapper.stand_down = "base"` (set by the kit at an item's trunk call; `None` restores the wrapper's own hoister) runs that item's
denoiser on the base hoister: same outputs bit for bit, the base's smaller hoist cache. The wrapper re-hoists when the effective hoister changes
(`hoister_now()`), keeps one live hoist cache at a time, records a `<hoister>_stood_down` event and reports
`diffusion_hoister = {row, used, stand_down}` in `StackHandle.stats()`.
