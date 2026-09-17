"""The `templ_embed` lever (fast class): the template embedder around the template pair stack — the eight-Linear template feature embedding as
ONE kernel writing the stack's input and the mean over templates / relu / linear_t as ONE kernel (`opt_core.kernels.templ_embed`), the stack
itself untouched (the pair cells, cuEquivariance and templ_distinct serve it as the line says). The implementation is the tree's
(`opt_core.of3_trunk.templ_embed`); this module binds openfold3_opt's switch and prefix and re-exports its record.

Switch: OPENFOLD3_OPT_TEMPL_EMBED=1. Exit line `[openfold3-opt/templ_embed] LEVER name=templ_embed …`."""
from opt_core.of3_trunk import templ_embed as _core

ENV = "OPENFOLD3_OPT_TEMPL_EMBED"
_core.configure(PREFIX="[openfold3-opt/templ_embed]", ENV=ENV, M_TEMPLATE="openfold3.core.model.latent.template_module")

STATE = _core.STATE
VALUES, KERNEL = _core.VALUES, _core.KERNEL
requested, install, census_line, serving = _core.requested, _core.install, _core.census_line, _core.serving
