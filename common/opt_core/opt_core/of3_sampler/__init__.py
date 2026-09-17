"""opt_core.of3_sampler — the diffusion-sampler levers of the AF3-architecture engines of this tree (the OF3 code family: the 0.4.x release and
its 0.5.x fork share one module layout), one implementation each; the engines' kits bind them through thin adapters (`<kit>/cells/{atom_hoist,dit_glue,token_agg,atom_window,apb_trunk,post_release}.py`: switch names, log
prefix, the engine's module paths, registry row, modes line) with `configure(...)` and re-export `STATE / requested / install / census_line`.

    atom_hoist   exact class: the atom path's step-invariant work (AtomAttentionEncoder.get_atom_reps -> c_l, p_lm; the atom transformers'
                 LN_z(p_lm) and per-block pair-bias linear_z; the block-index utilities) computed once per rollout and answered from
                 address-stable buffers (opt_core.of3_sampler.rollout_memo) at the other steps; publishes `_of3opt_atom_inv` per atom block
    dit_glue     fast class: DiffusionTransformerBlock.forward as ONE fused schedule (dit_rows AdaLN -> one q|k|v|g GEMM -> pair-bias flash
                 attention on the column slices in place -> o GEMM -> gate+residual -> dit_rows conditioned transition); refusals by name
                 run the stock block, counted
    token_agg    fast class: aggregate_atom_feat_to_tokens as one deterministic segment-reduce kernel (dtk_kernels.seg_reduce) over runs
                 derived once per rollout
    atom_window  fast class: the atom-attention encoder / decoder blocks (DiffusionTransformerBlock with use_cross_attention) on the
                 carried atom_window kernels (ln_qkvg: LayerNorm + both AdaLN modulations + q|k|v|g in one pass, k / v once per atom;
                 window_attn: shifted key windows in place + bias + mask + softmax + p v + gates + W_o + residual); step-invariant operands
                 once per rollout in rollout_memo; refusals by name run the stock block, counted
    rollout_memo the per-rollout store the memo levers share (not a lever): shape-keyed, byte-capped entries refreshed IN PLACE by one eager
                 denoiser call per rollout placed inside the engine's graphed-step wrapper — a captured graph reads their addresses, so under
                 graphs an entry is never re-addressed or evicted while a live graph can hold it; the rollout boundary / epoch on the sampler
    dit_rows     the AdaLN / conditioned-transition row schedules on dtk_kernels (periodic conditioning rows; shared by dit_glue; not a lever)
                 and the pair-bias attention core resolution (resolve_apb_core) the attention levers share
    apb_trunk    fast class: the pairformer single track's AttentionPairBias (trunk + confidence head; not the sampler's) — the pair bias
                 produced head-major by one GEMM and one launch of opt_core.attn.apb_core; the same code family and adapter pattern, hence here
    post_release exact class: at large token counts the graph machinery's between-items residents (the graphed step's generation + pool +
                 gather tables, the pair cache's static buffers, trunk_graph's graphs, cuBLAS workspaces, cached-free allocator segments) released
                 right after the sampler returns (before the confidence head's pair batch) and after the item's forward; gated by token count /
                 free memory, MiB freed per source on the census; the next item re-captures

One engine per process: `configure()` rebinds module globals, so a process binds one kit's names. Torch and the engine's modules are
imported inside `install()` / at call time; this module imports nothing (PEP 562).
"""
from __future__ import annotations

_MODULES = ("atom_hoist", "dit_glue", "token_agg", "atom_window", "dit_rows", "apb_trunk", "post_release")


def __getattr__(name):
    if name in _MODULES:
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
