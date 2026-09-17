"""The package's cell levers on the tree's shared core: the pair track (`pairfused`: PairBlock's triangle multiplication / triangle attention /
transition as opt_core cells), the diffusion transformer's attention (`dit_attn`: opt_core's flash kernel with pair bias), the token diffusion
transformer block as one fused schedule (`dit_glue`), the atom -> token aggregation kernel (`token_agg`), the sampler's sequence-local atom
attention on the tree's fused window kernels (`atom_window`), the atom path's per-rollout memo (`atom_hoist`, exact class), the single track's attention with pair bias (`apb_trunk`: the bias produced head-major + opt_core's pair-bias
attention entry) and the host-side scheduling cells, exact class — memoised bf16 weight casts (`castcache`), the memoised key-mask bias
(`apb_hoist`), the trunk stack captured into a CUDA graph (`trunk_graph`), the item-boundary memory release (`post_release`) — thin adapters
binding this kit's switch names onto the tree's one implementation, `opt_core.of3_sampler` (+ its `rollout_memo`, bound once for this kit by
`bind_rollout_memo()` below) / `opt_core.of3_trunk` — the template embedder's fused feature embedding and mean/relu/linear_t tail
around the untouched template pair stack (`templ_embed`: the same kind of adapter onto `opt_core.of3_trunk`, whose kernels are the core's
carried `templ_embed`), and the bf16 rollout region (`rollout`); and the runner-side cells, exact class — the confidence scoring after the forward per (sample, row block) (`postfwd_mem`, over
`openfold3_opt.of3_postfwd`) and the predict loader capped at the item count (`loader_workers`, over `openfold3_opt.of3_loader`;
both) — installed by the hook `opt/openfold3_opt/hooks/cells` on the `fast` line
(all), the `exact` line (`atom_hoist`, `castcache`, `apb_hoist`, `triatt_exact` — the library's triangle-attention call served by the core's ONE provider on word exact, over `openfold3_opt.of3_triattn` —, `transition_exact` — the pair-stack SwiGLU transitions after their own LayerNorm as one kernel with the engine's arithmetic (the core's transition provider row v1), over `openfold3_opt.of3_transition` —, `trimul_exact` — the pair stacks' triangle multiplication on the core's ONE trimul provider on word exact —, `post_release`, `postfwd_mem`, `loader_workers`) and the `big` resident line (the last two).
The TriMul cell's launch cells are the core's own table (opt_core kernels/fpf_trimul_v4/table.json)."""


def bind_rollout_memo():
    """The ONE binding of the tree's per-rollout sampler store (`opt_core.of3_sampler.rollout_memo`) to this engine: the sampler / model modules
    and classes that are the rollout boundary, the log prefix. Every cell that keeps entries in the store calls this (idempotent; the store
    refuses a second, different binding by name) — never its own `configure`."""
    from opt_core.of3_sampler import rollout_memo
    rollout_memo.configure(M_DIFFUSION="openfold3.core.model.structure.diffusion_module", SAMPLER_CLASS="SampleDiffusion",
                           M_MODEL="openfold3.projects.of3_all_atom.model", MODEL_CLASS="OpenFold3", PREFIX="[openfold3-opt/rollout_memo]")
    return rollout_memo
