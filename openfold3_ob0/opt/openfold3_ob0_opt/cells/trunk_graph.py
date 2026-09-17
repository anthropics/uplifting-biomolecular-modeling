"""The `trunk_graph` lever (exact class; the `fast` line): `PairFormerStack.forward` — the 48-block trunk stack once per recycle pass and the
confidence head's 4-block stack once per call, both launch-bound at small token counts — captured into a CUDA graph per (instance, input
signature, flags, numerics mode) at its second call and replayed after (first call eager; one process pool; refusals by name run the stack's
own forward: token count above the gate, grad, CPU, nested capture, DS4Sci attention, lma, a failed capture). Same kernels on the same operands:
bitwise the line without it. The implementation is the tree's (`opt_core.of3_trunk.trunk_graph`); this module binds openfold3_ob0_opt's switches
and prefix.

Switches: OPENFOLD3_OB0_OPT_TRUNK_GRAPH=1; OPENFOLD3_OB0_OPT_TRUNK_GRAPH_NMAX (token gate, default 1024). Exit line
`[openfold3_ob0-opt/trunk_graph] LEVER name=trunk_graph state=on nmax=… eager_first=… captures=… replays=… replayed_block_calls=… fallback=…`."""
from opt_core.of3_trunk import trunk_graph as _core

ENV = "OPENFOLD3_OB0_OPT_TRUNK_GRAPH"
ENV_NMAX = "OPENFOLD3_OB0_OPT_TRUNK_GRAPH_NMAX"
_core.configure(PREFIX="[openfold3_ob0-opt/trunk_graph]", ENV=ENV, ENV_NMAX=ENV_NMAX, M_PF="openfold3.core.model.latent.pairformer")

STATE = _core.STATE
VALUES, NMAX_DEFAULT = _core.VALUES, _core.NMAX_DEFAULT
requested, install, census_line, serving, nmax = _core.requested, _core.install, _core.census_line, _core.serving, _core.nmax
