"""The `atom_hoist` lever (exact class; the `exact` and `fast` lines): the atom path's step-invariant work — AtomAttentionEncoder.get_atom_reps
(c_l, p_lm), the atom transformers' LN_z(p_lm) and per-block pair-bias linear_z, the block-index utilities — computed once per rollout and
answered from address-stable buffers at the other denoiser steps (CUDA-graph cooperative: one eager step per rollout refreshes them). The
implementation is the tree's (`opt_core.of3_sampler.atom_hoist` + `opt_core.of3_sampler.rollout_memo`); this module binds openfold3_opt's switches
and prefix and re-exports its record for the stack's probe and the exit census.

Switches: OPENFOLD3_OPT_ATOM_HOIST=1; OPENFOLD3_OPT_ATOM_HOIST_MAX_GB (buffer cap, default 1.5); OPENFOLD3_OPT_ATOM_HOIST_INV=full|bias (the per-block
`_of3opt_atom_inv` buffers published for atom-transformer cells). Exit line `[openfold3-opt/atom_hoist] LEVER name=atom_hoist …`."""
from opt_core.of3_sampler import atom_hoist as _core

from . import bind_rollout_memo

_memo = bind_rollout_memo()                                   # the kit's one engine binding of the shared per-rollout store

ENV = "OPENFOLD3_OPT_ATOM_HOIST"
ENV_MAX_GB = "OPENFOLD3_OPT_ATOM_HOIST_MAX_GB"
ENV_INV = "OPENFOLD3_OPT_ATOM_HOIST_INV"
_core.configure(PREFIX="[openfold3-opt/atom_hoist]", ENV=ENV, ENV_MAX_GB=ENV_MAX_GB, ENV_INV=ENV_INV,
                M_SLAA="openfold3.core.model.layers.sequence_local_atom_attention", M_BLOCK_UTILS="openfold3.core.utils.atom_attention_block_utils")

STATE = _core.STATE
VALUES, INV_VALUES, INV_ATTR = _core.VALUES, _core.INV_VALUES, _core.INV_ATTR
requested, install, census_line, serving = _core.requested, _core.install, _core.census_line, _core.serving
