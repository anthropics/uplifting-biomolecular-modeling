"""`postfwd_mem` — this kit's binding of the post-forward confidence-scoring lever (exact class; `openfold3_opt.of3_postfwd`):
the runner scores the S diffusion samples after `OpenFold3.forward` returns with statements that materialise [S, N, N, 64] probabilities
(pde, pae, every `compute_ptm` call) and the [S, N_atom, N_atom, 3] all-atom difference tensor of `get_token_frame_atoms` — the process
peak of every line at >= 800 tokens is set THERE, not in the forward. The
lever restates those groups per (sample, row block) and assembles the engine's full-size result tensors by copy; every reduction over a
long dimension runs on the engine's own tensors: bitwise. Switches `OPENFOLD3_OPT_POSTFWD_MEM=1` (the exact, fast and big/resident lines
export it; on big/resident it serves below the confidence gate, modes.CONF_MIN_TOKENS — above it the port's chunked scorer does) and
`OPENFOLD3_OPT_POSTFWD_MEM_MIB=<MiB>` (the block budget, default 256; 0 = the engine's statements: the lever off by name)."""
from .. import of3_postfwd as _core

ENV = "OPENFOLD3_OPT_POSTFWD_MEM"
ENV_MIB = "OPENFOLD3_OPT_POSTFWD_MEM_MIB"
_core.configure(ENV=ENV, ENV_MIB=ENV_MIB, PREFIX="[openfold3-opt/postfwd_mem]",
                M_ACR="openfold3.core.metrics.aggregate_confidence_ranking", M_SR="openfold3.core.metrics.sample_ranking",
                M_CONF="openfold3.core.metrics.confidence", M_ATOMIZE="openfold3.core.utils.atomize_utils",
                DIGESTS={"_get_confidence_scores": ("ffcbb85bb56be288",), "compute_ptm": ("5786eda0ad0a4723",),      # sha256[:16] of the OpenFold3 0.4.1 function texts the lean
                         "get_token_frame_atoms": ("301fb497720c2bfd",), "probs_to_expected_error": ("70c6c0a1eed14a17",),   # statements restate / call (stock/PINS.json pins the wheel;
                         "compute_global_predicted_distance_error": ("1b53c4ebe5e41891",)})                                # another text refuses the lever by name)

STATE = _core.STATE
VALUES = _core.VALUES
requested, install, census_line, serving = _core.requested, _core.install, _core.census_line, _core.serving
