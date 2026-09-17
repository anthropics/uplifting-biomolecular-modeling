"""The `atom_window` lever (fast class): the diffusion sampler's sequence-local atom attention (the atom-attention encoder / decoder's
`DiffusionTransformerBlock`s with `use_cross_attention`, 6 blocks x 200 steps over the samples) on the tree's two fused kernels — `ln_qkvg`
(LayerNorm + both AdaLN modulations from per-atom conditioning rows + the q|k|v|gate projections in one pass, k / v once per atom) and
`window_attn` (the shifted key window read in place, pair bias + validity mask, fp32 softmax, p v, sigmoid gate, W_o, AdaLN-Zero gate, residual in
one kernel), then upstream's conditioned transition; refusals by name run the stock block, counted. The implementation is the tree's
(`opt_core.of3_sampler.atom_window` on `opt_core.kernels.atom_window`); this module binds openfold3_ob0_opt's switches and prefix and re-exports its
record. openfold3 0.5.0 asks `use_high_precision_attention=True` on every rollout call: the atom-attention blocks run in the sampler's fp32
context on both sides of `rollout_bf16`'s token gate and the kernels' logits / softmax are fp32, so the word is served as asked (no override);
the lever serves at every token count.

Switches: OPENFOLD3_OB0_OPT_ATOM_WINDOW=1; OPENFOLD3_OB0_OPT_ATOM_WINDOW_PRECISION=tf32rn|tf32x3|ieee (the kernels' dot precision, default tf32rn);
OPENFOLD3_OB0_OPT_ATOM_WINDOW_INV=own|hoist (the conditioning-side invariants: computed here, default, or read from `atom_hoist`'s published
`_of3opt_atom_inv`). Exit line `[openfold3_ob0-opt/atom_window] LEVER name=atom_window …`.

The `ln_qkvg` launch tile is the core's per compute capability (opt_core.kernels.atom_window's LN_QKVG_TILE_BY_CC / ln_qkvg_tile: 128 rows x 8 warps on cc 9.x,
32 rows x 8 warps on cc 8.0 whose shared-memory limit the 128-row tile exceeds; the 3-pass `tf32x3` word 16 rows everywhere)."""
from opt_core.of3_sampler import atom_window as _core

from . import bind_rollout_memo

_memo = bind_rollout_memo()                                   # the kit's one engine binding of the shared per-rollout store

ENV = "OPENFOLD3_OB0_OPT_ATOM_WINDOW"
ENV_PRECISION = "OPENFOLD3_OB0_OPT_ATOM_WINDOW_PRECISION"
ENV_INV = "OPENFOLD3_OB0_OPT_ATOM_WINDOW_INV"
PREFIX = "[openfold3_ob0-opt/atom_window]"
_core.configure(PREFIX=PREFIX, ENV=ENV, ENV_PRECISION=ENV_PRECISION, ENV_INV=ENV_INV,
                M_DIT="openfold3.core.model.layers.diffusion_transformer", M_DM="openfold3.core.model.structure.diffusion_module")

STATE = _core.STATE
VALUES, PRECISION_VALUES, INV_VALUES, KERNEL = _core.VALUES, _core.PRECISION_VALUES, _core.INV_VALUES, _core.KERNEL
requested, serving, plan = _core.requested, _core.serving, _core.plan

install, census_line = _core.install, _core.census_line
