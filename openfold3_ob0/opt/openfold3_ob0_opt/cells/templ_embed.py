"""The `templ_embed` lever (fast class): the template embedder around the template pair stack — the eight-Linear template feature embedding as
ONE kernel writing the stack's input and the mean over templates / relu / linear_t as ONE kernel (`opt_core.kernels.templ_embed`), the stack
itself untouched (the pair cells, cuEquivariance and templ_distinct serve it as the line says). The implementation is the tree's
(`opt_core.of3_trunk.templ_embed`, whose served forward takes no `offload_inference` argument); this module binds openfold3_ob0_opt's switch, prefix and
template-module path, re-exports its record, and carries OpenFold3 0.5.0's forward signature over it: `TemplateEmbedderAllAtom.forward(...,
offload_inference=False)`. The resident path (upstream's `_forward`: the pair embedder on all templates, then the stack; `_mask_trans` pinned
True by upstream's forward) is the served one; the offloaded path (`offload_inference=True`: upstream's per-template loop through host memory,
`_forward_offload`, which the model selects above its `memory.eval.offload_inference` token cutoff) is upstream's own statement, counted by
name on the census (`fallback=offload_inference:<n>`).

Switch: OPENFOLD3_OB0_OPT_TEMPL_EMBED=1. Exit line `[openfold3_ob0-opt/templ_embed] LEVER name=templ_embed state=… impl=… served=<n> fallback=<word:n,…|none> first=<TxN>`."""
from opt_core.of3_trunk import templ_embed as _core

ENV = "OPENFOLD3_OB0_OPT_TEMPL_EMBED"
M_TEMPLATE = "openfold3.core.model.latent.template_module"
_core.configure(PREFIX="[openfold3_ob0-opt/templ_embed]", ENV=ENV, M_TEMPLATE=M_TEMPLATE)

STATE = _core.STATE
VALUES, KERNEL = _core.VALUES, _core.KERNEL
requested, census_line, serving = _core.requested, _core.census_line, _core.serving
SHIM_ATTR = "_ob0opt_templ_embed_sig"        # marks the 0.5.0-signature forward this module lays over the tree's served forward


def install(environ=None) -> dict:
    """The tree's install (routes the kernels, replaces TemplateEmbedderAllAtom.forward by its served forward), then OpenFold3 0.5.0's signature
    over it. Idempotent; not requested -> nothing installed; raises by name when the kernel module is not importable (the tree's words)."""
    rec = _core.install(environ)
    if not (rec.get("installed") and rec.get("state") == "on"):
        return rec
    import importlib
    cls = importlib.import_module(M_TEMPLATE).TemplateEmbedderAllAtom
    served = cls.forward
    if getattr(served, SHIM_ATTR, False) or not getattr(served, "_of3opt_templ_embed", False):
        return rec                                                   # already laid, or another copy's patch (the tree named it): nothing to add
    stock = served.__wrapped__                                       # upstream's own forward (0.5.0: ..., offload_inference=False)

    def forward(self, batch, z, pair_mask, chunk_size=None, _mask_trans=True, use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False,
                use_triton_triangle_kernels=False, use_lma=False, inplace_safe=False, offload_inference=False):
        if offload_inference:                                        # upstream's per-template offloaded path: its own statement, counted by name
            _core._count_fallback("offload_inference")
            return stock(self, batch, z, pair_mask, chunk_size=chunk_size, _mask_trans=_mask_trans, use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                         use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels, use_lma=use_lma,
                         inplace_safe=inplace_safe, offload_inference=True)
        return served(self, batch, z, pair_mask, chunk_size=chunk_size, _mask_trans=True, use_deepspeed_evo_attention=use_deepspeed_evo_attention,   # 0.5.0 pins _mask_trans=True on this path
                      use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels, use_lma=use_lma,
                      inplace_safe=inplace_safe)
    forward._of3opt_templ_embed = True; forward.__wrapped__ = stock
    setattr(forward, SHIM_ATTR, True)
    cls.forward = forward
    return rec
