"""E63: a Hyena block's ``proj_norm`` (pre_norm RMSNorm -> pad L to a multiple of 16 -> the FP8 input projection) with the RMSNorm tail
writing the projection's FP8 (e4m3) GEMM input directly. Per element: E11's tail arithmetic (bf16(norm * H^-0.5), bf16(+eps), bf16(x / d),
bf16(scale_w * y)) followed by Transformer Engine's cast of that bf16 value with the module's forward input scale (widen to fp32, * scale,
round-to-nearest saturating e4m3); the projection is TE's own ``general_gemm`` on the module's cached FP8 weight workspace (W1) with the
arguments ``_Linear.forward`` gives it (TE workspace, bf16 out, the recipe's accumulation mode, the module's bias rule), so E61's output
buffer serves it. The bf16 normed tensor, its zero-padded copy and TE's cast kernel are never materialised; the module's FP8 scales are not
updated under no-grad by TE either (the amax reduction runs at autocast exit under grad only), which is the one mode this path serves:
under grad, before the module's first FP8 forward (no weight workspace yet), or for any other module / dtype / layout, the stock composition
runs. One (rows, hidden) uint8 buffer per (rows, hidden, device) is shared by every block (blocks run in stream order)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.module import linear as te_linear
from transformer_engine.pytorch.module.base import get_workspace
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from vortex.model.layers import RMSNorm
from vortex.model.model import ParallelGatedConvBlock

from evo2_opt.kit import i32 as _I32
from evo2_opt.kit import rmsnorm as RN
from evo2_opt.kit.base import CTR

LEVER = "E63_pre_norm_fp8_emit"
ORIG = {"proj_norm": ParallelGatedConvBlock.proj_norm}
FWD_INPUT = int(tex.FP8FwdTensors.GEMM1_INPUT)
_bufs: dict = {}                  # (rows, hidden, device) -> Float8Tensor over a zero-initialised uint8 buffer
_scale_invs: dict = {}            # id(projection module) -> (module, 1 / its forward input scale): the scale is fixed under no-grad, so its reciprocal is taken once
NUM_WARPS = 8
CHANNELS_FIRST = {"on": False}   # E64, set by the route whose featurizer reads it: the GEMM issued with operand roles swapped (D^T = W @ X^T, the same
                                 # TN fp8 GEMM: per element the same K products in the same order) into E61's kept (3*hidden, B*L_pad) buffer, handed on
                                 # as its (B, L, 3*hidden) view - the channels contiguous along L for the featurizer FIR (fir3rows)
_named = set()


@triton.jit
def _rms_tail_e4m3_kernel(x_ptr, n_ptr, s_ptr, q_ptr, scale_ptr, H, L, LP, inv_sqrt_h, eps, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    orow = (row // L) * LP + row % L                                   # the row of the L-padded (B, LP, H) GEMM input
    offs = tl.arange(0, BLOCK_H)
    m = offs < H
    n = tl.load(n_ptr + row).to(tl.float32)
    d = (n * inv_sqrt_h).to(tl.bfloat16).to(tl.float32)               # bf16(norm * H^-0.5)
    d = (d + eps).to(tl.bfloat16).to(tl.float32)                      # bf16(. + eps)
    x = tl.load(x_ptr + row * H + offs, mask=m, other=0.0).to(tl.float32)
    y = tl.math.div_rn(x, d).to(tl.bfloat16).to(tl.float32)           # bf16(x / d), IEEE round-to-nearest division as torch's
    s = tl.load(s_ptr + offs, mask=m, other=0.0).to(tl.float32)
    o = (s * y).to(tl.bfloat16).to(tl.float32)                        # bf16(scale_w * y): the RMSNorm output, widened
    scale = tl.load(scale_ptr)
    tl.store(q_ptr + orow * H + offs, (o * scale).to(tl.float8e4nv), mask=m)   # TE cast: rn.satfinite e4m3 of fp32(out) * scale


def _use_split_accumulator(p) -> bool:
    """``_Linear.forward``'s accumulation mode: the recipe's fp8_gemm_fprop when it carries one, else linear._2X_ACC_FPROP."""
    recipe = getattr(p, "fp8_recipe", None) or p.fp8_meta.get("recipe")
    if recipe is not None and hasattr(recipe, "fp8_gemm_fprop"):
        return bool(recipe.fp8_gemm_fprop.use_split_accumulator)
    return bool(te_linear._2X_ACC_FPROP)


def takes(block, x) -> bool:
    """True when this proj_norm call is the one the lever reproduces: no grad, an unpadded-forward Hyena block whose pre_norm is vortex's
    RMSNorm and whose projection is a Transformer Engine Linear that has run its FP8 forward once (fp8 on, e4m3 forward format, the W1
    weight workspace held, one bias-free GEMM, no tensor/sequence parallelism or comm overlap), a bf16 CUDA (B, L, hidden) input."""
    if torch.is_grad_enabled() or getattr(block, "print_activations", False):
        return False
    p, norm = getattr(block, "projections", None), getattr(block, "pre_norm", None)
    if not isinstance(p, te.Linear) or not isinstance(norm, RMSNorm) or not getattr(p, "use_fp8_input_projections", False):
        return False
    if not getattr(p, "fp8", False) or "scaling_fwd" not in p.fp8_meta or "weight" not in getattr(p, "_fp8_workspaces", {}):
        return False
    if p.fp8_meta["recipe"].__class__.__name__ != "DelayedScaling":
        return False
    ws = p._fp8_workspaces["weight"]
    if getattr(ws, "_fp8_dtype", None) != tex.DType.kFloat8E4M3 or getattr(ws, "_data", None) is None:
        return False
    if p.use_bias or p.return_bias or getattr(p, "tp_size", 1) != 1 or getattr(p, "sequence_parallel", False):
        return False
    if getattr(p, "ub_overlap_ag", False) or getattr(p, "ub_overlap_rs", False) or getattr(p, "ub_bulk_dgrad", False) or getattr(p, "ub_bulk_wgrad", False):
        return False
    if x.dim() != 3 or x.dtype != torch.bfloat16 or x.device.type != "cuda" or getattr(p, "activation_dtype", None) != torch.bfloat16:
        return False
    H = int(x.shape[-1])
    return H == int(norm.scale.numel()) and int(ws._data.shape[-1]) == H and norm.scale.dtype == torch.bfloat16


def _input_tensor(rows: int, H: int, device, quantizer) -> Float8Tensor:
    key = (rows, H, str(device))
    q = _bufs.get(key)
    if q is None:
        CTR["e63_fp8_input_alloc"] += 1
        data = torch.zeros(rows, H, dtype=torch.uint8, device=device)             # padded rows stay e4m3 +0.0 = the cast of the zero pad
        q = _bufs[key] = Float8Tensor(shape=(rows, H), dtype=torch.bfloat16, data=data, fp8_scale_inv=torch.empty(1, dtype=torch.float32, device=device),
                                      fp8_dtype=tex.DType.kFloat8E4M3, requires_grad=False, data_transpose=None, quantizer=quantizer)
    q._quantizer = quantizer
    return q


def proj_norm(self, x):
    """ParallelGatedConvBlock.proj_norm on the kit path (gated; the stock composition when ``takes`` says no).  ``x`` may be the previous
    block's closing pair (m, s) not yet added (kit/base.py _stateless_forward_pairs): the result is then (u, projected) with u = bf16(m + s)
    formed inside the norm launch (kit/rmsnorm.py add_rmsnorm_e4m3) when the rows are served, by torch's add otherwise."""
    pair = None
    if isinstance(x, tuple):
        pair, x = x, x[0]
    if not takes(self, x):
        CTR["e63_stock_composition"] += 1
        if pair is None:
            return ORIG["proj_norm"](self, x)
        u = pair[0] + pair[1]; CTR["e66_added_apart"] += 1
        return u, ORIG["proj_norm"](self, u)
    p, norm = self.projections, self.pre_norm
    B, L, H = (int(s) for s in x.shape)
    LP = int(self.pad_to_multiple(torch.empty((1, L, 1), device="meta")).shape[1])           # the stock's pad of L (a multiple of 16 under FP8)
    rows, rows_p = B * L, B * LP
    _I32.check(LEVER, f"the RMSNorm input {tuple(x.shape)} padded to L={LP}", (rows_p * H,), f"batch x padded length <= {_I32.EXTENT // max(1, H):,} rows at width {H}")
    meta = p.fp8_meta["scaling_fwd"]
    quantizer = p.quantizers["scaling_fwd"][FWD_INPUT]
    with torch.cuda.device(x.device):
        q = _input_tensor(rows_p, H, x.device, quantizer)
        si = _scale_invs.get(id(p))
        if si is None or si[0] is not p:
            si = _scale_invs[id(p)] = (p, torch.reciprocal(meta.scale[FWD_INPUT:FWD_INPUT + 1]))   # the cast's scale_inv = 1 / scale, once per module
        q._scale_inv = si[1]
        emitted = False
        if pair is not None:
            m2d, s2d = pair[0].contiguous().view(rows, H), pair[1].contiguous().view(rows, H)
            if (B == 1 or LP == L) and RN.served_add(m2d, s2d, norm.scale):                  # one launch: the residual add + ATen-order norm + tail + cast
                CTR["e66_residual_pre_norm_fused"] += 1
                xc = RN.add_rmsnorm_e4m3(m2d, s2d, norm.scale, norm.eps, q._data.view(torch.float8_e4m3fn)[:rows], meta.scale[FWD_INPUT:FWD_INPUT + 1]).view(B, L, H)
                emitted = True
            else:
                xc = (pair[0] + pair[1]).contiguous(); CTR["e66_added_apart"] += 1
        else:
            xc = x.contiguous()
        x2d = xc.view(rows, H)
        if not emitted and (B == 1 or LP == L) and RN.served(x2d, norm.scale):             # one launch: ATen-order norm + tail + cast (kit/rmsnorm.py);
            CTR["e63_fp8_emit_fused_norm"] += 1                                               # the valid rows lead the padded buffer
            RN.rmsnorm_forward_e4m3(x2d, norm.scale, norm.eps, q._data.view(torch.float8_e4m3fn)[:rows], meta.scale[FWD_INPUT:FWD_INPUT + 1])
        elif not emitted:
            n = xc.norm(2, dim=-1, keepdim=True)                                              # torch's own reduction (bf16 out), as E11
            _rms_tail_e4m3_kernel[(rows,)](x2d, n.view(rows), norm.scale, q._data.view(torch.float8_e4m3fn), meta.scale[FWD_INPUT:FWD_INPUT + 1],
                                             H, L, LP, float(H ** (-1.0 / 2)), float(norm.eps), BLOCK_H=triton.next_power_of_2(H), num_warps=NUM_WARPS)
        CTR["e63_fp8_emit"] += 1
        w8 = p._fp8_workspaces["weight"]
        N = int(w8._data.shape[0])
        if CHANNELS_FIRST["on"] and _channels_first_fits(N, rows_p, B, LP):
            CTR["e64_gemm_transposed"] += 1                                                # D^T (3H, B*L_pad) = W @ X^T: the projection rows contiguous along L
            out_t, *_ = te_linear.general_gemm(q, w8, get_workspace(), quantization_params=None, out_dtype=p.activation_dtype, bias=None,
                                               use_split_accumulator=_use_split_accumulator(p), ub=None, ub_type=None, extra_output=None)
            bufs = getattr(te_linear.general_gemm, "buffers", None)
            if bufs is not None:
                bufs.pop((rows_p, N, p.activation_dtype, str(x.device)), None)                # the row-major output buffer of this shape is not used again
            out = out_t.t()
        else:
            out, *_ = te_linear.general_gemm(w8, q, get_workspace(), quantization_params=None, out_dtype=p.activation_dtype, bias=None,
                                             use_split_accumulator=_use_split_accumulator(p), ub=None, ub_type=None, extra_output=None)
    projected = out.view(B, LP, -1)
    if LP > L:
        projected = projected[:, :L, :]
    return (xc, projected) if pair is not None else projected


def _channels_first_fits(N: int, rows_p: int, B: int, LP: int) -> bool:
    """The (3H, B*L_pad) output and its featurizer are addressed with 32-bit element offsets: beyond that extent the row-major output serves
    the shape (named once)."""
    if N * rows_p < _I32.EXTENT:
        return True
    if "extent" not in _named:
        _named.add("extent")
        import sys
        print(f"[evo2-kit] E64: {N} x {rows_p} elements at (B, L_pad)=({B}, {LP}) exceed 32-bit element addressing; the row-major projection output serves this shape",
              file=sys.stderr, flush=True)
    return False


def release() -> None:
    _bufs.clear()
    _scale_invs.clear()
