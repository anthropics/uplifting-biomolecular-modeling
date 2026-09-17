"""The `token_agg` lever (fast class): the atom -> token mean at the end of the atom-attention encoder (`aggregate_atom_feat_to_tokens`, both
bindings) as ONE deterministic segment-reduce kernel over each token's contiguous atom run (`opt_core.kernels.dtk_kernels.seg_reduce`; the runs
derived once per rollout). The implementation is the tree's (`opt_core.of3_sampler.token_agg`); this module binds openfold3_ob0_opt's switch and prefix
and re-exports its record.

Switch: OPENFOLD3_OB0_OPT_TOKEN_AGG=seg_reduce. Exit line `[openfold3_ob0-opt/token_agg] LEVER name=token_agg …`."""
from opt_core.of3_sampler import token_agg as _core

from . import bind_rollout_memo

_memo = bind_rollout_memo()                                   # the kit's one engine binding of the shared per-rollout store

ENV = "OPENFOLD3_OB0_OPT_TOKEN_AGG"
_core.configure(PREFIX="[openfold3_ob0-opt/token_agg]", ENV=ENV,
                M_ATOMIZE="openfold3.core.utils.atomize_utils", M_SLAA="openfold3.core.model.layers.sequence_local_atom_attention")

STATE = _core.STATE
VALUES, KERNEL = _core.VALUES, _core.KERNEL
requested, install, census_line, serving = _core.requested, _core.install, _core.census_line, _core.serving
