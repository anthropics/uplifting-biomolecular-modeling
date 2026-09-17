# Part of OF3_TRUNK_KERNELS_ADDON (inference-speed add-on for OpenFold3); attributions: NOTICE.
"""OF3 TRUNK levers (OF3 trunk-kernels add-on).  All default OFF; selected by env vars; no source modification
(installed by the of3t sitecustomize import hook after `openfold3.projects.of3_all_atom.model` is imported; stacks with the fast-inference add-on's levers).

EXACT-class levers (same math, same kernels; tested bitwise under the deterministic reference config):
  OF3T_TEMPL_DISTINCT=1   TemplateEmbedder: with templates OFF the featuriser emits n_templ=4 IDENTICAL dummy templates; stock embeds all 4 and averages
                          (sum/4). We detect identical template feature rows by VALUE (torch.equal on every template feature) and, only then, run the
                          2-block template pair stack on the distinct set and re-expand before the mean.  Different templates -> stock path (no change).
                          NOTE: sum over 4 identical fp32 values /4 vs 1 value: (x+x+x+x)/4 == x exactly in IEEE fp32 for finite x (x+x=2x exact, 2x+x=3x
                          exact? NO: 3x is not always exact) -> we therefore keep the stock reduction: expand distinct outputs back to n_templ copies and
                          call the SAME torch.sum(...)/n_templ, so the arithmetic is byte-identical to stock.
KERNEL-SWAP levers (numerics class: different kernels, same math -> appendix unless tested inside the stock-vs-stock floor):
  OF3T_TRIATT=ds|cueq|triton    backend for TriangleAttention (starting+ending) only — upstream's own kernel flags for the statement beneath the kit's
                                pair cells (the triangle attention of every pair stack is the core's provider's by the line's tier word, openfold3_opt
                                of3_triattn; this switch only picks the flags of the stock statement the cells name as their fallback)
  OF3T_APB=ds|cueq|triton       backend for AttentionPairBias in the trunk/confidence pairformers (NOT the diffusion transformer: AdaLN instances excluded)
  OF3T_TRIMUL=cueq              cuEquivariance fused triangle multiplicative update (fp32 in OF3's predict preset), via PairBlock.tri_mul_out_in
Diagnostics: one line per lever decision class (first occurrence).
"""
import os, sys
import torch

_SEEN = set()
STATS = {"templ_distinct_hits": 0, "templ_distinct_miss": 0, "templ_calls": 0, "triatt_routed": 0, "apb_routed": 0, "trimul_routed": 0}


def _log_once(key, msg):
    if key not in _SEEN:
        _SEEN.add(key)
        sys.stderr.write(f"[of3t_levers] {msg}\n"); sys.stderr.flush()


def _backend_kwargs(name):
    name = (name or "stock").lower()
    if name == "stock":
        return None
    kw = dict(use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False, use_triton_triangle_kernels=False, use_lma=False)
    if name == "ds":
        kw["use_deepspeed_evo_attention"] = True
        os.environ["CUTLASS_PATH"] = os.environ.get("CUTLASS_PATH", "DS_USE_CUTLASS_PYTHON_BINDINGS")
        if os.environ["CUTLASS_PATH"] == "placeholder":
            os.environ["CUTLASS_PATH"] = "DS_USE_CUTLASS_PYTHON_BINDINGS"
    elif name == "cueq":
        kw["use_cueq_triangle_kernels"] = True
    elif name == "triton":
        kw["use_triton_triangle_kernels"] = True
    else:
        raise ValueError(f"unknown backend {name}")
    return kw


def install():
    from openfold3.core.model.layers.triangular_attention import TriangleAttention
    from openfold3.core.model.layers.attention_pair_bias import AttentionPairBias
    from openfold3.core.model.latent.base_blocks import PairBlock
    from openfold3.core.model.latent import template_module as TM

    triatt = _backend_kwargs(os.environ.get("OF3T_TRIATT"))
    apb = _backend_kwargs(os.environ.get("OF3T_APB"))
    trimul = (os.environ.get("OF3T_TRIMUL", "stock").lower() == "cueq")

    if triatt is not None:
        _orig = TriangleAttention.forward
        def triatt_forward(self, x, mask=None, chunk_size=None, **k):
            k.update(triatt); STATS["triatt_routed"] += 1
            _log_once("triatt", f"TriangleAttention routed to {os.environ.get('OF3T_TRIATT')} (x={tuple(x.shape)} {x.dtype} chunk_size={chunk_size})")
            return _orig(self, x, mask=mask, chunk_size=chunk_size, **k)
        TriangleAttention.forward = triatt_forward

    if apb is not None:
        _orig_apb = AttentionPairBias.forward
        def apb_forward(self, a, z, s=None, mask=None, **k):
            if not self.use_ada_layer_norm and not k.get("use_high_precision_attention"):
                k.update({kk: vv for kk, vv in apb.items() if kk != "use_lma" or "use_lma" in k}); STATS["apb_routed"] += 1
                _log_once("apb", f"AttentionPairBias (non-AdaLN) routed to {os.environ.get('OF3T_APB')} (a={tuple(a.shape)} {a.dtype})")
            return _orig_apb(self, a, z, s=s, mask=mask, **k)
        AttentionPairBias.forward = apb_forward

    if trimul:
        _orig_tm = PairBlock.tri_mul_out_in
        def tri_mul_out_in(self, z, pair_mask, inplace_safe, use_cueq_triangle_kernels=False, use_triton_triangle_kernels=False):
            use_cueq_triangle_kernels = True; STATS["trimul_routed"] += 1
            _log_once("trimul", f"triangle multiplicative update routed to cuEquivariance (z={tuple(z.shape)} {z.dtype})")
            return _orig_tm(self, z, pair_mask, inplace_safe, use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels)
        PairBlock.tri_mul_out_in = tri_mul_out_in

    # template module: distinct-template evaluation (exact class).  With --use-templates false OpenFold3's featuriser emits n_templ (=4) IDENTICAL
    # dummy templates; stock runs the 2-block template pair stack on all of them and averages.  Implementation = "stack" (default, version-robust,
    # 0.4.x and 0.5.x): wrap TemplatePairStack.forward: if every template slice of its input t is identical BY VALUE (torch.equal) and the mask is
    # broadcast over templates, run the stack on t[..., :1, :, :, :] and return it expanded to n_templ copies, so the caller's unchanged
    # `torch.sum(t, dim=-4) / n_templ` sees the same values as stock.  Different templates (real template search) -> stock path, no change.
    distinct = os.environ.get("OF3T_TEMPL_DISTINCT") == "1"
    _orig_tps = TM.TemplatePairStack.forward

    def tps_forward(self, t, mask, *a, **k):
        if distinct and t.dim() >= 5 and t.shape[-4] > 1:
            n_templ = t.shape[-4]
            STATS["templ_calls"] += 1
            t0 = t.narrow(-4, 0, 1)
            ident = torch.equal(t, t0.expand_as(t)) and (mask is None or mask.shape[-3] == 1 or torch.equal(mask, mask.narrow(-3, 0, 1).expand_as(mask)))
            if ident:
                STATS["templ_distinct_hits"] += 1
                m = mask if (mask is None or mask.shape[-3] == 1) else mask.narrow(-3, 0, 1)
                out = _orig_tps(self, t0, m, *a, **k)
                _log_once("templ_hit", f"template distinct-eval: {n_templ} identical template embeddings -> pair stack run once, expanded x{n_templ} (exact class)")
                return out.expand(*out.shape[:-4], n_templ, *out.shape[-3:])
            STATS["templ_distinct_miss"] += 1
            _log_once("templ_miss", f"template embeddings differ across the {n_templ} templates -> stock path")
        return _orig_tps(self, t, mask, *a, **k)
    TM.TemplatePairStack.forward = tps_forward


    sys.stderr.write(f"[of3t_levers] installed: TRIATT={os.environ.get('OF3T_TRIATT','stock')} APB={os.environ.get('OF3T_APB','stock')} TRIMUL={'cueq' if trimul else 'stock'} "
                     f"TEMPL_DISTINCT={distinct}\n")
    import atexit
    atexit.register(lambda: sys.stderr.write(f"[of3t_levers] stats {STATS}\n"))
