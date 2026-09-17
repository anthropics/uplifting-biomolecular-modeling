# Part of OF3_TRUNK_KERNELS_ADDON (inference-speed add-on for OpenFold3); attributions: NOTICE.
"""Template distinct-evaluation lever of the trunk_kernels add-on (OpenFold3 0.5.0). Default OFF; selected by environment variable; no source modification
(installed by the of3t sitecustomize import hook after `openfold3.projects.of3_all_atom.model` is imported; stacks with the fast_inference add-on's levers).

EXACT-class lever (same math, same kernels; bitwise under the deterministic recipe):
  OF3T_TEMPL_DISTINCT=1   TemplateEmbedder: with templates OFF the featuriser emits n_templ=4 IDENTICAL dummy templates; stock embeds all 4 and averages
                          (sum/4). The lever detects identical template feature rows by VALUE (torch.equal on the pair-stack input and its mask) and, only then,
                          runs the 2-block template pair stack on the distinct set and re-expands before the mean.  Different templates -> stock path (no change).
                          Why re-expand instead of returning one slot: (x+x+x+x)/4 is not x for every finite fp32 x (x+x is exact, (x+x)+x can
                          round), so the distinct output is expanded back to n_templ copies and the caller's SAME torch.sum(...)/n_templ runs:
                          the arithmetic is byte-identical to stock.
                          Engages on TemplateEmbedderAllAtom._forward (all templates in one pair-stack call); under
                          settings.memory.eval.offload_inference.template_module (per-template CPU-offload loop, _forward_offload) every pair-stack call
                          carries one template and the lever is inactive for that item (named in the per-item census line).  A census line per predicted
                          item states whether distinct-eval fired (identical dummy templates = untemplated chains) or the stock path ran (templates differ).
The triangle attention of every pair stack belongs to the kit's pair cells (the shared core's provider by the line's tier word, openfold3_ob0_opt
of3_triattn) and, beneath them, to stock's own statement under the line's runner configuration: this add-on carries no triangle-attention switch
(OF3T_TRIATT set is an error, named at install).
Diagnostics: one line per lever decision class (first occurrence); `fallbacks()` = this module's per-call degraded paths ({}: the
template lever has none — templated input and the offload loop are its named stock cases).
"""
import os, sys
import torch

_SEEN = set()
STATS = {"templ_distinct_hits": 0, "templ_distinct_miss": 0, "templ_calls": 0, "templ_offload_items": 0, "templ_items": 0}


def fallbacks():
    """Per-call degraded paths of this module's levers as {"<lever>:<reason>": n}; always {} — the template distinct-eval lever either fires (identical
    templates, bitwise to stock) or leaves the stock path in place by name (templated input / offload loop / no call), never a degraded computation.
    """
    return {}


def _census(d):
    return ",".join(f"{k}={v}" for k, v in sorted(d.items())) if d else "none"


def _log_once(key, msg):
    if key not in _SEEN:
        _SEEN.add(key)
        sys.stderr.write(f"[of3t_levers] {msg}\n"); sys.stderr.flush()


def _install_templ_distinct(TM):
    """Distinct-template evaluation (exact class).  For a chain without templates OpenFold3's featuriser emits n_templ (=4) IDENTICAL dummy templates;
    stock runs the 2-block template pair stack on all of them and averages.  Wrap TemplatePairStack.forward: if every template slice of its input t
    is identical BY VALUE (torch.equal) and the mask is broadcast over templates, run the stack on t[..., :1, :, :, :] and return it expanded to n_templ
    copies, so the caller's unchanged `torch.sum(t, dim=-4) / n_templ` (TemplateEmbedderAllAtom._forward) sees the same values as stock.  Different
    templates (a templated chain) -> stock path, no change.  One census line per predicted item (OpenFold3.forward wrapper)."""
    from openfold3.projects.of3_all_atom.model import OpenFold3
    _orig_tps, _orig_te, _orig_model_fwd = TM.TemplatePairStack.forward, TM.TemplateEmbedderAllAtom.forward, OpenFold3.forward
    _ITEM = {"calls": 0, "hits": 0, "miss": 0, "offload": False, "n_templ": None}          # per-item template census (reset by the OpenFold3.forward wrapper)

    def tps_forward(self, t, mask, *a, **k):
        if t.dim() >= 5 and t.shape[-4] > 1:
            n_templ = t.shape[-4]
            STATS["templ_calls"] += 1; _ITEM["calls"] += 1; _ITEM["n_templ"] = n_templ
            t0 = t.narrow(-4, 0, 1)
            ident = torch.equal(t, t0.expand_as(t)) and (mask is None or mask.shape[-3] == 1 or torch.equal(mask, mask.narrow(-3, 0, 1).expand_as(mask)))
            if ident:
                STATS["templ_distinct_hits"] += 1; _ITEM["hits"] += 1
                m = mask if (mask is None or mask.shape[-3] == 1) else mask.narrow(-3, 0, 1)
                out = _orig_tps(self, t0, m, *a, **k)
                _log_once("templ_hit", f"template distinct-eval: {n_templ} identical template embeddings -> pair stack run once, expanded x{n_templ} (exact class)")
                return out.expand(*out.shape[:-4], n_templ, *out.shape[-3:])
            STATS["templ_distinct_miss"] += 1; _ITEM["miss"] += 1
            _log_once("templ_miss", f"template embeddings differ across the {n_templ} templates -> stock path")
        return _orig_tps(self, t, mask, *a, **k)

    def te_forward(self, batch, z, pair_mask, *a, **k):
        if k.get("offload_inference") or (len(a) >= 8 and bool(a[7])):
            # settings.memory.eval.offload_inference.template_module: TemplateEmbedderAllAtom._forward_offload embeds and stacks ONE template per call
            # (template_module.py: `for i in range(n_templ)`), so no pair-stack call carries the identical set -> stock path for this item, named below.
            _ITEM["offload"] = True
        return _orig_te(self, batch, z, pair_mask, *a, **k)

    def model_forward(self, batch):
        # per-item census (one stderr line per predicted item, never silent): distinct-eval fired (untemplated: identical dummy templates) | stock path
        # (templated chains: embeddings differ) | inactive (per-template offload loop) | no template pair-stack call with n_templ > 1.
        _ITEM.update(calls=0, hits=0, miss=0, offload=False, n_templ=None)
        try:
            return _orig_model_fwd(self, batch)
        finally:
            STATS["templ_items"] += 1
            try:
                q = batch.get("query_id"); q = ",".join(q) if isinstance(q, (list, tuple)) else q
                n = int(batch["token_mask"].shape[-1])
            except Exception:
                q, n = None, -1
            if _ITEM["offload"]:
                STATS["templ_offload_items"] += 1
                verdict = "INACTIVE (offload_inference.template_module: per-template loop, stock computation)"
            elif _ITEM["calls"] == 0:
                verdict = "no template pair-stack call with n_templ > 1 (stock computation)"
            elif _ITEM["hits"] == _ITEM["calls"]:
                verdict = f"FIRED on {_ITEM['hits']}/{_ITEM['calls']} pair-stack calls (n_templ={_ITEM['n_templ']} identical dummy templates: untemplated input)"
            elif _ITEM["hits"] == 0:
                verdict = f"stock path on {_ITEM['miss']}/{_ITEM['calls']} pair-stack calls (templates differ: templated input)"
            else:
                verdict = f"FIRED on {_ITEM['hits']}/{_ITEM['calls']} pair-stack calls, stock path on {_ITEM['miss']} (mixed)"
            sys.stderr.write(f"[of3t_levers] templ distinct census: item query={q} tokens={n}: {verdict}\n"); sys.stderr.flush()

    TM.TemplatePairStack.forward, TM.TemplateEmbedderAllAtom.forward, OpenFold3.forward = tps_forward, te_forward, model_forward


def install():
    from openfold3.core.model.latent import template_module as TM

    triatt_mode = os.environ.get("OF3T_TRIATT", "").strip()
    if triatt_mode:
        raise RuntimeError(f"[of3t_levers] OF3T_TRIATT={triatt_mode!r}: this add-on carries no triangle-attention switch (the pair stacks' triangle attention is the "
                           "kit's pair cells' — the core's provider by the line's tier word; OpenFold3's own kernel switches are runner-YAML settings, settings.memory.eval)")
    distinct = os.environ.get("OF3T_TEMPL_DISTINCT") == "1"
    if distinct:
        _install_templ_distinct(TM)
    sys.stderr.write(f"[of3t_levers] installed: TEMPL_DISTINCT={distinct}\n")
    import atexit
    atexit.register(lambda: sys.stderr.write(f"[of3t_levers] stats {STATS}\n[of3t_levers] fallbacks={_census(fallbacks())}\n"))
