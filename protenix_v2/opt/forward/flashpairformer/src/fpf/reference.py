"""Reference implementations = the STOCK Protenix v2 module objects with checkpoint weights.
build_model() -> the full Protenix model exactly as the stock CLI builds it (protenix-v2, bf16, cuequivariance x2, 10 cycles, msa on).
stock_call(name, module, *args) runs the stock forward under bf16 autocast (the production numerics); ref64_call(...) runs the SAME module in float64 on the 'torch' backend
(no cuEq) = the error reference: report your kernel's max|err| and RMS err vs ref64 DIVIDED BY stock's own (ratio <= 1.5 => same error class)."""
import os, copy, torch, contextlib
_DUMP_DIR = None
def _private_dump_dir():
    """FPF_DUMP_DIR's default: one private directory (tempfile.mkdtemp, mode 0700) per process, made on first use."""
    global _DUMP_DIR
    if _DUMP_DIR is None:
        import tempfile
        _DUMP_DIR = tempfile.mkdtemp(prefix="fpf_pred-")
    return _DUMP_DIR
def build_runner(seeds=(101,), cycles=10):
    import runner.batch_inference as BI
    from runner.batch_inference import get_default_runner
    BI.inference_configs["dump_dir"] = os.environ.get("FPF_DUMP_DIR") or _private_dump_dir()   # a directory of this process's own, never a fixed name every account shares
    r = get_default_runner(seeds=list(seeds), n_cycle=cycles, n_step=200, n_sample=1, dtype="bf16", model_name="protenix-v2", use_msa=True,
                           trimul_kernel="cuequivariance", triatt_kernel="cuequivariance", enable_cache=True, enable_fusion=True, enable_tf32=True,
                           use_template=False, use_rna_msa=False, use_seeds_in_json=False, need_atom_confidence=True)
    return r
def amp(): return torch.autocast("cuda", dtype=torch.bfloat16)
_FP64_ATT_DONE = False
def install_fp64_attention_fallback():
    """protenix primitives._attention hard-casts q,k (and bias) to float32 while v keeps the input dtype -> SDPA dtype error for float64 inputs (AttentionPairBias / whole-block fp64 refs).
    Patch: when q arrives as float64, run the same math entirely in float64 (no fp32 downcast; this IS the reference). bf16/fp32 stock paths untouched. Idempotent."""
    global _FP64_ATT_DONE
    if _FP64_ATT_DONE: return
    import protenix.model.modules.primitives as P
    _orig = P._attention
    def _attention64(q, k, v, attn_bias=None, use_efficient_implementation=True, **kw):
        if q.dtype != torch.float64:
            return _orig(q, k, v, attn_bias=attn_bias, use_efficient_implementation=use_efficient_implementation, **kw)
        k64 = k.to(torch.float64); v64 = v.to(torch.float64)
        logits = torch.matmul(q, k64.transpose(-1, -2))
        if attn_bias is not None: logits = logits + attn_bias.to(torch.float64)
        return torch.matmul(torch.softmax(logits, dim=-1), v64)
    P._attention = _attention64; _FP64_ATT_DONE = True
_FP64_LN_DONE = False
def install_fp64_layernorm_fallback():
    """Protenix's fast_layernorm CUDA extension has no float64 kernel -> module.double() forward raises 'Unsupported data type'.
    Patch FusedLayerNorm.forward to use F.layer_norm for float64 inputs ONLY (bf16/fp32 stock paths untouched). Idempotent."""
    global _FP64_LN_DONE
    if _FP64_LN_DONE: return
    import torch.nn.functional as F
    try:
        from protenix.model.layer_norm.layer_norm import FusedLayerNorm
    except Exception:
        _FP64_LN_DONE = True; return
    _orig = FusedLayerNorm.forward
    def fwd(self, x, *a, **kw):
        if x.dtype == torch.float64:
            w = getattr(self, "weight", None); b = getattr(self, "bias", None)
            return F.layer_norm(x, tuple(self.normalized_shape), None if w is None else w.to(x.dtype), None if b is None else b.to(x.dtype), self.eps)
        return _orig(self, x, *a, **kw)
    FusedLayerNorm.forward = fwd; _FP64_LN_DONE = True
@torch.no_grad()
def stock_call(module, method, *a, **kw):
    with amp(): return getattr(module, method)(*a, **kw)
@torch.no_grad()
def ref64_call(module, method, *a, backend_kw=("triangle_multiplicative", "triangle_attention"), **kw):
    """same module, deep-copied to float64, torch backend, fp64 inputs; returns fp64 output."""
    install_fp64_layernorm_fallback(); install_fp64_attention_fallback()
    m64 = copy.deepcopy(module).double()
    kw = dict(kw)
    for k in backend_kw:
        if k in kw: kw[k] = "torch"
    a64 = [x.double() if torch.is_tensor(x) and x.is_floating_point() else x for x in a]
    kw64 = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kw.items()}
    with torch.autocast("cuda", enabled=False):
        return getattr(m64, method)(*a64, **kw64)
def err_table(candidate, stock, ref64):
    c = candidate.double(); s = stock.double(); r = ref64.double()
    ec = (c - r); es = (s - r)
    out = {"cand_max_abs": float(ec.abs().max()), "cand_rms": float(ec.pow(2).mean().sqrt()), "stock_max_abs": float(es.abs().max()), "stock_rms": float(es.pow(2).mean().sqrt())}
    out["ratio_max"] = out["cand_max_abs"] / max(out["stock_max_abs"], 1e-30); out["ratio_rms"] = out["cand_rms"] / max(out["stock_rms"], 1e-30)
    out["bitwise_vs_stock"] = bool(torch.equal(candidate, stock)); out["same_class(<=1.5x)"] = out["ratio_max"] <= 1.5 and out["ratio_rms"] <= 1.5
    return out
