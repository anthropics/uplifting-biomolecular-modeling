"""install.py — MSA-module levers: instance-level `forward` replacements on the stock OuterProductMean / MSAPairWeightedAveraging modules of the
(single) MSAModule; stock weights used unchanged; no stock file edited.

opm_fused   OuterProductMean.forward (the eval call MSABlock makes: mask=None, chunk_size = the runner's): fused LayerNorm(m)+[linear_1|linear_2]
            (one pass over m; ln_linear) -> stock's own einsum statement on the same transposed views (the same cuBLAS GEMM, per row chunk of
            `chunk_size`) -> opm_out: linear_out computed directly on the einsum's native [b,c,d,e] layout (no permute copy) with + bias and
            / norm in the epilogue (norm = stock's own einsum(mask, mask) + eps statement, i.e. the same bf16 count). Peak memory: one
            [chunk, N, 32, 32] bf16 einsum result at a time (stock additionally materialises its flattened copy).
pwa_fused   MSAPairWeightedAveraging.forward: fused LayerNorm(m)+[linear_mv|linear_mg] writing v HEAD-MAJOR [h, N, S, c] (pwa_ln_vg) so the
            pair-weighted average is one torch.matmul (cuBLAS bmm, contiguous operands, fp32 accumulation — stock's einsum needs two permute
            copies around the same GEMM); z-path = fused LayerNorm(z)+linear_z producer (protenix_fpf_apb.pf_bias, fp32 logits) + softmax over j,
            OR softmax weights found cached on the module for this MSAStack call (`_fpf_zcache_w`, when another lever set them); then pwa_out2:
            sigmoid(g) * wv -> linear_out in one kernel.
Numerics class TOLERANCE (both): bf16 operands exactly where stock's autocast Linears use bf16, fp32 LayerNorm statistics and accumulations;
reduction orders differ from cuBLAS.
Composition: a `_fpf_zcache_w` attribute on the module (softmax weights another lever cached for this MSAStack call) is honoured; any lever
that already put an instance-level forward on one of these modules is left in place and NAMED (count in STATE[...]['left_alone']). In this kit
ptxfpf/trunk2_ptx1.py performs the equivalent instance-level install itself with its own cells; these are the package's own entry points.
Refuse-by-name: unknown topology / geometry / a training-mode or masked call raises RuntimeError naming the lever (no stock fallback).
"""
from __future__ import annotations
import functools
import torch


def _pf_triton():
    """fpf_apb's pf_triton module (pf_bias: the fused LayerNorm(z)+Linear pair-bias producer pwa_fused's z-path uses): the shared core's carry
    (opt_core.kernels.apb.fpf_apb), else a top-level package `protenix_fpf_apb` when one is importable. ImportError when neither is importable (the caller records it as a refusal by name)."""
    try:
        from opt_core.kernels.apb.fpf_apb import pf_triton
    except ImportError:
        from protenix_fpf_apb import pf_triton
    return pf_triton

CELLS = {
    "opm_fused": {"9.0": dict(ln_rows=64, td=64, cg=2, num_warps=4, num_stages=2)},     # H100: ln_linear 64 rows/program; opm_out 64 d-rows x 256 z, 2 c-slices per MMA step, 4 warps, 2 stages
    "pwa_fused": {"9.0": dict(mb=8, nb=16, ln_warps=8, out_mb=8, out_nb=8)},           # H100: LN+vg tile 8 msa rows x 16 tokens / 8 warps; epilogue tile 8 x 8 / 4 warps
}
STATE = {
    "opm_fused": {"installed_on": 0, "calls": 0, "chunks": 0, "cell": None, "cell_key": None, "named": None, "left_alone": [], "error": None},
    "pwa_fused": {"installed_on": 0, "calls": 0, "zcache_hits": 0, "cell": None, "cell_key": None, "named": None, "left_alone": [], "error": None},
}


def _cell(lever: str) -> dict:
    """The lever's cell for this card; a card without a row engages the sm_90 cell and is NAMED."""
    cc = "%d.%d" % torch.cuda.get_device_capability() if torch.cuda.is_available() else "cpu"
    table = CELLS[lever]
    key = cc if cc in table else "9.0"
    STATE[lever]["cell_key"] = key
    if key != cc:
        STATE[lever]["named"] = f"untested arch sm_{cc.replace('.', '')}: engaged with the sm_90 cell"
    STATE[lever]["cell"] = dict(table[key])
    return dict(table[key])


def _msa_module(model):
    import protenix.model.modules.pairformer as PF
    mm = [x for x in model.modules() if isinstance(x, PF.MSAModule)]
    if len(mm) != 1:
        raise RuntimeError(f"expected one MSAModule in the model, found {len(mm)}")
    return mm[0]


# ----------------------------------------------------------------------------------------------------------------- opm_fused
def _opm_forward(self, cell, m, mask=None, chunk_size=None, inplace_safe=False):
    from opt_core.ops.msa_fused.msa_triton import ln_linear, opm_out
    st = STATE["opm_fused"]
    if self.training or mask is not None or m.dim() != 3:
        raise RuntimeError(f"opm_fused: unsupported call (training={self.training}, mask given={mask is not None}, m.dim={m.dim()}); the lever covers MSABlock's eval call")
    from protenix.model.utils import is_fp16_enabled
    if is_fp16_enabled():
        raise RuntimeError("opm_fused: fp16 autocast is not covered (the kit runs bf16 autocast)")
    st["calls"] += 1
    S, N, C = m.shape
    ln = self.layer_norm
    a, b = ln_linear(m if m.is_contiguous() else m.contiguous(), ln.weight, getattr(ln, "bias", None), (self.linear_1.weight, self.linear_2.weight),
                     getattr(ln, "eps", 1e-5), R=cell["ln_rows"])                                   # [S, N, c] bf16 each (stock: linear_k(layer_norm(m)))
    mask1 = m.new_ones(m.shape[:-1]).unsqueeze(-1)                                                  # stock: mask = ones -> a, b unchanged; norm below verbatim
    norm = torch.einsum("...abc,...adc->...bdc", mask1, mask1) + self.eps                          # [N, N, 1]
    at = a.transpose(-2, -3); bt = b.transpose(-2, -3)                                              # [N, S, c] views, as stock
    wt = getattr(self, "_fpf_msa_wout_t", None)
    if wt is None or wt.device != m.device:
        wt = self.linear_out.weight.detach().t().contiguous().to(torch.bfloat16); self._fpf_msa_wout_t = wt
    bias = self.linear_out.bias
    bias_b = None if bias is None else bias.detach().to(torch.bfloat16)
    cz = wt.shape[1]
    out = torch.empty((N, N, cz), device=m.device, dtype=torch.bfloat16)
    step = N if chunk_size is None else max(1, int(chunk_size))
    for r0 in range(0, N, step):
        r1 = min(N, r0 + step)
        outer = torch.einsum("...bac,...dae->...bdce", at[r0:r1], bt)                             # stock's GEMM statement (cuBLAS), result laid out [b, c, d, e]
        opm_out(outer, wt, bias_b, norm[r0:r1], out[r0:r1], TD=cell["td"], CG=cell["cg"], num_warps=cell["num_warps"], num_stages=cell["num_stages"])
        st["chunks"] += 1
        del outer
    return out


def install_opm_fused(model) -> dict:
    st = STATE["opm_fused"]
    try:
        from opt_core.ops.msa_fused import msa_triton  # noqa: F401
        cell = _cell("opm_fused")
        mm = _msa_module(model)
        n = 0
        for i, blk in enumerate(mm.blocks):
            opm = blk.outer_product_mean_msa
            if "forward" in opm.__dict__:
                st["left_alone"].append(f"block{i}:{getattr(opm.forward, '__name__', type(opm.forward).__name__)}"); continue
            if opm.linear_out.weight.shape[1] != opm.c_hidden ** 2 or opm.c_hidden & (opm.c_hidden - 1) or opm.layer_norm.weight.numel() & (opm.layer_norm.weight.numel() - 1):
                raise RuntimeError(f"opm_fused: MSA block {i}: c_hidden={opm.c_hidden} c_m={opm.layer_norm.weight.numel()} linear_out={tuple(opm.linear_out.weight.shape)} (cells measured for power-of-two c_m / c_hidden)")
            fwd = functools.partial(_opm_forward, opm, cell)
            fwd.__wrapped__ = type(opm).forward; fwd._fpf_msa = "opm_fused"
            opm.forward = fwd; n += 1
        st["installed_on"] = n
        if st["left_alone"]:
            st["named"] = ((st["named"] + "; ") if st["named"] else "") + f"{len(st['left_alone'])} module(s) owned by another lever left alone"
        return dict(st)
    except Exception as e:
        st["error"] = repr(e)
        raise


# ----------------------------------------------------------------------------------------------------------------- pwa_fused
def _pwa_forward(self, cell, m, z):
    from opt_core.ops.msa_fused.msa_triton import pwa_ln_vg, pwa_out2
    st = STATE["pwa_fused"]
    if self.training or m.dim() != 3 or z.dim() != 3:
        raise RuntimeError(f"pwa_fused: unsupported call (training={self.training}, m.dim={m.dim()}, z.dim={z.dim()}); the lever covers MSAStack's eval call")
    st["calls"] += 1
    H, CC = self.n_heads, self.c
    ln = self.layernorm_m
    v_hm, g = pwa_ln_vg(m if m.is_contiguous() else m.contiguous(), ln.weight, getattr(ln, "bias", None), self.linear_no_bias_mv.weight, self.linear_no_bias_mg.weight,
                        getattr(ln, "eps", 1e-5), H=H, CC=CC, MB=cell["mb"], NB=cell["nb"], num_warps=cell["ln_warps"])
    M, N = m.shape[0], m.shape[1]
    w = getattr(self, "_fpf_zcache_w", None)                              # softmax weights another lever may have cached across the chunks of one MSAStack call (else None)
    if w is None:
        _pf = _pf_triton()                                                # fpf_apb's pair-bias producer: fused LN(z)+Linear(c_z -> H)
        lz = self.layernorm_z
        logits = _pf.pf_bias(z if z.is_contiguous() else z.contiguous(), lz.weight, getattr(lz, "bias", None), self.linear_no_bias_z.weight,
                             getattr(lz, "eps", 1e-5), out_dtype=torch.float32)                    # [H, i, j] fp32 (row pitch 8)
        w8 = torch.softmax(logits, dim=-1).to(torch.bfloat16).contiguous()                         # softmax over j; bf16 = the einsum's autocast operand dtype
    else:
        st["zcache_hits"] += 1
        w8 = w.permute(2, 0, 1).to(torch.bfloat16).contiguous()                                   # stock's [i, j, H] fp32 weights -> [H, i, j] bf16
    wv = torch.matmul(w8, v_hm.view(H, N, M * CC)).view(H, N, M, CC)                               # the pair-weighted average: one bmm, fp32-accumulated
    wt = getattr(self, "_fpf_msa_wout_t", None)
    if wt is None or wt.device != m.device:
        wt = self.linear_no_bias_out.weight.detach().t().contiguous().to(torch.bfloat16); self._fpf_msa_wout_t = wt
    return pwa_out2(g, wv, wt, MB=cell["out_mb"], NB=cell["out_nb"])


def install_pwa_fused(model) -> dict:
    st = STATE["pwa_fused"]
    try:
        from opt_core.ops.msa_fused import msa_triton  # noqa: F401
        _pf_triton()                       # (ImportError = refusal by name: the z-path producer lives in the fpf_apb package)
        cell = _cell("pwa_fused")
        mm = _msa_module(model)
        n = 0
        for i, blk in enumerate(mm.blocks):
            if not hasattr(blk, "msa_stack"):
                continue
            pwa = blk.msa_stack.msa_pair_weighted_averaging
            if "forward" in pwa.__dict__:
                st["left_alone"].append(f"block{i}:{getattr(pwa.forward, '__name__', type(pwa.forward).__name__)}"); continue
            hc = pwa.n_heads * pwa.c
            c_m = pwa.layernorm_m.weight.numel(); c_z = pwa.linear_no_bias_z.in_features
            if pwa.linear_no_bias_mv.weight.shape != (hc, c_m) or hc & (hc - 1) or c_m & (c_m - 1) or c_z & (c_z - 1) or c_z < 16:
                raise RuntimeError(f"pwa_fused: MSA block {i}: heads={pwa.n_heads} c={pwa.c} c_m={c_m} c_z={c_z} (cells measured for power-of-two sizes)")
            fwd = functools.partial(_pwa_forward, pwa, cell)
            fwd.__wrapped__ = type(pwa).forward; fwd._fpf_msa = "pwa_fused"
            pwa.forward = fwd; n += 1
        st["installed_on"] = n
        if st["left_alone"]:
            st["named"] = ((st["named"] + "; ") if st["named"] else "") + f"{len(st['left_alone'])} module(s) owned by another lever left alone"
        return dict(st)
    except Exception as e:
        st["error"] = repr(e)
        raise


def report() -> dict:
    return {k: dict(v) for k, v in STATE.items()}
