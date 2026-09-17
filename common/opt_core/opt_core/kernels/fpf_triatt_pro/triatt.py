"""fpf_triatt_pro/triatt.py — FPF_SPEC_v0 registry op `triatt` (the stock TriangleAttention.forward contract) built on the prologue producer.

    fn(module, x, mask=None, chunk_size=None, triangle_attention='cuequivariance', inplace_safe=True)  -> update [.., I, J, C]  (caller does z +=)

Variants (all share the code path; only the LayerNorm source differs):
    fn          : fused in-kernel two-pass fp32 LayerNorm  (TIER2 a priori)                       == fn_fused
    fn_exactln  : stock module.layer_norm(x) kernel call, then the fused 5-projection kernel      (EXACT candidate; fallback)
    fn_welford  : in-kernel emulation of fast_layernorm's Welford/butterfly statistics           (bit-exact candidate; label decided by measurement)
Pipeline per call (OP mode; module.starting is True for BOTH tri_att_start and tri_att_end in the pinned stock — the ending role is realised by the
caller's transposes — so there is no internal transpose here unless module.starting is False, which is honoured for generality):
    prologue(z) -> q,k,v [I,H,J,D] bf16, g [I,J,HD] bf16, bias [H,I,J] fp32
    o = cuequivariance triangle_attention(q, k, v, bias[None], mask=None|bool, scale=1/sqrt(D))   (the STOCK attention kernel, same operands as stock)
    out = linear_o( bf16( fp32(o[i,h,j,d]) * fp32(bf16(sigmoid_fp32(g[i,j,hD+d]))) ) )            (the OG gate kernel below + the stock cuBLAS GEMM for linear_o)
Falls back to the STOCK forward (type(module).forward._fpf_orig, i.e. the unpatched class method) for anything outside the checked envelope:
(C, H*D, gpu_class) not in prologue.VERIFIED (e.g. template pair stack c=64/H=2 until tested, c=128 pair stacks, non-H100), chunk_size set,
non-cuEq backend, N <= 16, non-bf16 input.  STATS['fallback'] counts such calls.  Extra leading batch dims are looped (B=1 per call, like stock in-model).
"""
from __future__ import annotations
import math, os
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .prologue import stock_word, triatt_prologue, get_cache, PINNED_CONFIG, VERIFIED, config_table_sha256, eligible

try:
    from triton.language.extra import libdevice as _ld
except Exception:  # pragma: no cover
    try:
        from triton.language.extra.cuda import libdevice as _ld
    except Exception:
        _ld = None
_HAS_LD = tl.constexpr(_ld is not None)

ENGINE_STOCK_CLASS = ("the stock TriangleAttention starting/ending (triangular.TriangleAttention + layers.Attention, fast_layernorm, "
                      "bf16-autocast Linear, cuEq triangle_attention 0.8.0, fp32 bias)")
STATS = {"calls": 0, "fallback": 0}


# ----------------------------------------------------------------------------------------------------------------- OG epilogue (_gate_out_kernel2: the stock gate math, fused)
@triton.jit
def _gate_out_kernel2(O, G, OUT, J, so_i, so_h, so_j, sg_i, sg_j, H: tl.constexpr, D: tl.constexpr, BLOCK_J: tl.constexpr):
    # OUT[i, j, h*D + d] = bf16( fp32(O[i,h,j,d]) * fp32( bf16( sigmoid_fp32(G[i,j,hD+d]) ) ) )  -- stock: g = sigmoid(linear_g(x)) (bf16 under autocast), o = o * g
    i = tl.program_id(0).to(tl.int64); jb = tl.program_id(1)
    js = jb * BLOCK_J + tl.arange(0, BLOCK_J).to(tl.int64)
    d = tl.arange(0, D)
    jm = js < J
    for h in tl.static_range(H):
        o = tl.load(O + i * so_i + h * so_h + js[:, None] * so_j + d[None, :], mask=jm[:, None], other=0.0)
        g = tl.load(G + i * sg_i + js[:, None] * sg_j + h * D + d[None, :], mask=jm[:, None], other=0.0)
        gf = g.to(tl.float32)
        if _HAS_LD:
            e = _ld.exp(-gf)
        else:
            e = tl.exp(-gf)
        sg = (1.0 / (1.0 + e)).to(g.dtype).to(tl.float32)
        r = o.to(tl.float32) * sg
        tl.store(OUT + (i * J + js[:, None]) * (H * D) + h * D + d[None, :], r.to(o.dtype), mask=jm[:, None])


def gate_out(o, g, H, D):
    """o [I,H,J,D] bf16 contiguous (cuEq output), g [I,J,H*D] bf16 pre-sigmoid -> [I,J,H*D] bf16 gated, flattened (== stock o.transpose*sigmoid(g) then flatten)."""
    I, _, J, _ = o.shape
    out = torch.empty((I, J, H * D), dtype=o.dtype, device=o.device)
    BJ2 = 64 if J >= 512 else 32
    _gate_out_kernel2[(I, triton.cdiv(J, BJ2))](o, g, out, J, o.stride(0), o.stride(1), o.stride(2), g.stride(0), g.stride(1), H=H, D=D, BLOCK_J=BJ2, num_warps=4)
    return out


def gate_out_torch(o, g, H, D):
    """plain-torch epilogue with stock op order (for A/B against the OG kernel): g = sigmoid(g) bf16; o[I,J,H,D] * g.view; flatten."""
    gg = torch.sigmoid(g).view(g.shape[:-1] + (H, D))
    oo = o.transpose(-2, -3) * gg
    return oo.reshape(oo.shape[:-2] + (H * D,))


# ----------------------------------------------------------------------------------------------------------------- the op
def _stock_forward(module):
    f = type(module).forward
    return getattr(f, "_fpf_orig", f)


def _core(module, x, mask, ln_mode, epilogue="og", fma_flags=(True, True, True)):
    """x: [I, J, C] bf16 (3-D, x-frame of the module, i.e. AFTER the module's own transpose if starting=False). Returns update [I, J, C] bf16."""
    from protenix.model.triangular.layers import cuequivariance_triangular_attn
    cch = get_cache(module, x.device)
    H, D = cch["H"], cch["D"]
    if ln_mode == "stock":
        x_ln = module.layer_norm(x)                       # stock fast_layernorm kernel (does .contiguous() itself if x is a strided view)
        q, k, v, g, bias = triatt_prologue(module, x_ln, ending=False, ln_mode="stock", x_ln=x_ln)
    else:
        q, k, v, g, bias = triatt_prologue(module, x, ending=False, ln_mode=ln_mode, fma_flags=fma_flags)
    if mask is None:
        mask_bool = None      # cuEq: mask=None == all-True mask (the no-mask lever: op-level bit-exact checked, and re-checked in this package's tests)
    else:
        # stock: mask_bias = inf*(mask-1) [.., I,1,1,J] fp32-ish; cuEq gets (mask_bias == 0) -> bool [I,1,1,J]
        mask_bias = (module.inf * (mask - 1))[..., :, None, None, :]
        mask_bool = (mask_bias == 0)
    scale = 1.0 / math.sqrt(D)
    o = cuequivariance_triangular_attn(q, k, v, bias.unsqueeze(0), mask_bool, scale)
    o = o[0] if isinstance(o, (tuple, list)) else o
    if o.dim() == 5:
        o = o[0]
    # o: [I, H, J, D] bf16 contiguous
    if epilogue == "og" and o.is_contiguous():
        og = gate_out(o, g, H, D)
    else:
        og = gate_out_torch(o, g, H, D)
    lin_o = module.mha.linear_o
    return lin_o(og)                                      # the stock Linear: bf16 input -> F.linear(input, W.to(bf16)) == stock


def _make_fn(ln_mode, epilogue="og", fma_flags=(True, True, True)):
    def fn(module, x, mask=None, chunk_size=None, triangle_attention="torch", inplace_safe=False):
        STATS["calls"] += 1
        ok = (triangle_attention == "cuequivariance" and chunk_size is None and x.is_cuda and x.dtype == torch.bfloat16
              and x.dim() >= 3 and x.shape[-2] > 16 and x.shape[-3] > 16 and x.stride(-1) == 1 and module.mha.linear_g is not None
              and (ln_mode != "welford" or module.c_in == 256))
        word = "precondition" if not ok else stock_word(module.c_in, module.mha.no_heads * module.mha.c_hidden, x.device)   # a VERIFIED key or an UNKNOWN key on SAFE settings runs; a
        if word:                                                                                                             # measured-off key is the stock forward BY NAME
            STATS["fallback"] += 1
            STATS.setdefault("fallback_by", {})[word] = STATS["fallback_by"].get(word, 0) + 1
            return _stock_forward(module)(module, x, mask=mask, chunk_size=chunk_size, triangle_attention=triangle_attention, inplace_safe=inplace_safe)
        lead = x.shape[:-3]
        xs = x.reshape((-1,) + tuple(x.shape[-3:]))
        ms = None if mask is None else mask.reshape((-1,) + tuple(mask.shape[-2:]))
        outs = []
        for b in range(xs.shape[0]):
            xb = xs[b]
            mb = None if ms is None else ms[b]
            if not module.starting:                       # generality only; the pinned stock builds both tri_att modules with starting=True
                xb = xb.transpose(-2, -3)
                mb = None if mb is None else mb.transpose(-1, -2)
                # x-frame is the transposed view; the prologue reads it through strides (no copy) for fused modes; stock LN mode copies like stock does
                if ln_mode == "stock":
                    ob = _core(module, xb, mb, ln_mode, epilogue, fma_flags)
                else:
                    ob = _core_strided(module, xb, mb, ln_mode, epilogue, fma_flags)
                ob = ob.transpose(-2, -3)
            else:
                ob = _core(module, xb, mb, ln_mode, epilogue, fma_flags)
            outs.append(ob)
        out = outs[0].unsqueeze(0) if len(outs) == 1 else torch.stack(outs, 0)
        return out.reshape(tuple(lead) + tuple(out.shape[-3:]))
    fn.__name__ = f"fn_{ln_mode}_{epilogue}"
    fn.LN_MODE = ln_mode
    return fn


def _core_strided(module, xv, mask, ln_mode, epilogue, fma_flags):
    """xv is a transposed VIEW [I, J, C] of a contiguous [J, I, C] tensor: run the prologue with ending=True on the base (no copy)."""
    base = xv.transpose(-2, -3)          # contiguous [J, I, C]
    from protenix.model.triangular.layers import cuequivariance_triangular_attn
    cch = get_cache(module, xv.device)
    H, D = cch["H"], cch["D"]
    q, k, v, g, bias = triatt_prologue(module, base, ending=True, ln_mode=ln_mode, fma_flags=fma_flags)
    mask_bool = None if mask is None else ((module.inf * (mask - 1))[..., :, None, None, :] == 0)
    o = cuequivariance_triangular_attn(q, k, v, bias.unsqueeze(0), mask_bool, 1.0 / math.sqrt(D))
    o = o[0] if isinstance(o, (tuple, list)) else o
    if o.dim() == 5:
        o = o[0]
    og = gate_out(o, g, H, D) if (epilogue == "og" and o.is_contiguous()) else gate_out_torch(o, g, H, D)
    return module.mha.linear_o(og)


fn = _make_fn("fused")                 # Tier-2 candidate (fully fused prologue: z read once)
fn_fused = fn
fn_exactln = _make_fn("stock")         # EXACT candidate / fallback (stock LN kernel + fused projections)
fn_welford = _make_fn("welford")       # fast_layernorm-emulating LN (bit-exact candidate; decided by measurement)
fn_fused_torchepi = _make_fn("fused", epilogue="torch")
fn_exactln_torchepi = _make_fn("stock", epilogue="torch")

__all__ = ["fn", "fn_fused", "fn_exactln", "fn_welford", "fn_fused_torchepi", "fn_exactln_torchepi", "triatt_prologue", "gate_out",
           "PINNED_CONFIG", "config_table_sha256", "ENGINE_STOCK_CLASS", "STATS"]
