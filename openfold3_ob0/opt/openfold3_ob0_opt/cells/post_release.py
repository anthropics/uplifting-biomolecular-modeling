"""The `post_release` lever (exact class; every graphed line: `exact`, `fast`): at large token counts (>= OPENFOLD3_OB0_OPT_POST_RELEASE_NTOK, default
2400, or when free device memory falls below OPENFOLD3_OB0_OPT_POST_RELEASE_MIN_FREE_GB, default 16) the graph machinery's between-items residents
(of3_graphs' generation + pool + gather tables, of3t_paircache's static pair buffers, trunk_graph's graphs, cuBLAS workspaces, the allocator's
cached-free segments) are released right after the sampler returns — before the confidence head's per-sample pair batch, the forward's largest
late allocation — and again after the item's forward; the next item re-captures as a first item does. Below the gate nothing happens. Outputs
untouched: bitwise the line without it. The implementation is the tree's (`opt_core.of3_sampler.post_release`); this module binds
openfold3_ob0_opt's switches, prefix and module paths.

Switches: OPENFOLD3_OB0_OPT_POST_RELEASE=1; OPENFOLD3_OB0_OPT_POST_RELEASE_NTOK; OPENFOLD3_OB0_OPT_POST_RELEASE_MIN_FREE_GB. Exit line
`[openfold3_ob0-opt/post_release] LEVER name=post_release state=on … pre_confidence=… post_forward=… kept=… freed_mib=<per source> fallback=…`."""
from opt_core.of3_sampler import post_release as _core

ENV = "OPENFOLD3_OB0_OPT_POST_RELEASE"
ENV_NTOK = "OPENFOLD3_OB0_OPT_POST_RELEASE_NTOK"
ENV_MIN_FREE_GB = "OPENFOLD3_OB0_OPT_POST_RELEASE_MIN_FREE_GB"
_core.configure(PREFIX="[openfold3_ob0-opt/post_release]", ENV=ENV, ENV_NTOK=ENV_NTOK, ENV_MIN_FREE_GB=ENV_MIN_FREE_GB,
                M_DIFFUSION="openfold3.core.model.structure.diffusion_module", SAMPLER_CLASS="SampleDiffusion",
                M_MODEL="openfold3.projects.of3_all_atom.model", MODEL_CLASS="OpenFold3", M_GRAPHS="of3_graphs", M_PAIRCACHE="of3t_paircache")

STATE = _core.STATE
VALUES, NTOK_DEFAULT, MIN_FREE_GB_DEFAULT = _core.VALUES, _core.NTOK_DEFAULT, _core.MIN_FREE_GB_DEFAULT
requested, install, census_line, serving = _core.requested, _core.install, _core.census_line, _core.serving
