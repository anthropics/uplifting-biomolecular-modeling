"""Rows ``esm_k1ptr`` / ``esm_k1ptr:f32in`` of kernels.trimul: the POINTER-load K1 line of ``opt_core.kernels.trimul_esm_shapes`` bound by name.

The sub-package serves each cell's descriptor-load K1 tile where the Triton at hand has ``tl.make_tensor_descriptor`` and its measured
pointer-load tile elsewhere; these rows serve the pointer tile EVERYWHERE, so one word gives the same instruction stream on a triton-3.3
stack and on a triton-3.7 stack (the planes of the two K1 kernels are equal byte for byte: they share every statement after the loads; the
bmm and K3 are the same launches).  Envelope, refused BY NAME outside it: cc >= 8.0; triton >= 3.3; c_z == c_hidden in {128, 256, 384};
B = 1 pairs; ``esm_k1ptr`` for bf16 z (or the bf16 rounding point of an fp32 z under bf16 autocast), ``esm_k1ptr:f32in`` for fp32 | tf32
trunks (fp32 in / out, bf16 tensor-core operands: a TOLERANCE class, numerics = the sub-package's fp32 cells).  Residual fused or absent.
Pure until served (no framework import at module level)."""

ROWS = ("esm_k1ptr", "esm_k1ptr:f32in")
WIDTHS = (128, 256, 384)          # c_z == c_hidden; the sub-package's (64,64) / (64,128) template cells stay behind row esm_shapes
CC_MIN = 8.0
TRITON_MIN = (3, 3)               # pointer loads, tl.dot, tl.range: nothing newer is asked of the compiler
SUBPACKAGE = "opt_core.kernels.trimul_esm_shapes"


def fallback(row, C, H):
    """The row a refusal names: v4 where it serves the width (bf16 row), else the stock op."""
    if row == "esm_k1ptr":
        return "v4" if (int(C) in (128, 256) and int(H) == int(C)) else "cueq"
    return "cueq"


def refusal(row, cc, prec, C, H, N, *, batch=1, triton=None):
    """None inside the envelope, else (reason, fallback) -- words only; the provider raises its Refusal with them."""
    C, H, N = int(C), int(H), int(N)
    fb = fallback(row, C, H)
    if float(cc) < CC_MIN:
        return "cc:%s<8.0" % cc, "cueq"
    if row == "esm_k1ptr" and prec not in ("bf16", "f32z_bf16"):
        return "dtype:%s(fp32 | tf32 callers use esm_k1ptr:f32in)" % prec, ("esm_k1ptr:f32in" if prec in ("fp32", "tf32") else "cueq")
    if row == "esm_k1ptr:f32in" and prec not in ("fp32", "tf32"):
        return "dtype:%s(bf16 callers use esm_k1ptr)" % prec, ("esm_k1ptr" if prec in ("bf16", "f32z_bf16") else "cueq")
    if C != H or C not in WIDTHS:
        return "shape:C%d/H%d(c_z == c_hidden in %s)" % (C, H, "/".join(str(w) for w in WIDTHS)), fb
    if triton is not None:
        try:
            tv = tuple(int(x) for x in str(triton).split("+")[0].split(".")[:2])
        except ValueError:
            tv = None
        if tv is not None and tv < TRITON_MIN:
            return "triton:%s<3.3" % triton, fb
    if int(batch) != 1:
        return "batch:%s(B=1 pairs)" % batch, fb
    try:
        S = __import__(SUBPACKAGE, fromlist=["supported"])
    except ImportError as e:
        return "needs:kernels.trimul_esm_shapes(import:%s)" % getattr(e, "name", "?"), fb
    ok, why = S.supported(str(cc) if not isinstance(cc, str) else cc, "bf16" if row == "esm_k1ptr" else "fp32", C, H, N, batch=1, descriptor_api=False)
    if not ok:
        return why, fb
    return None


def serve(row, z, mask, direction, w, residual, eps, cache, config, compute_input, Refusal):
    """Serve one call through the sub-package with ``pointer_k1=True``.  ``compute_input`` / ``Refusal`` are the provider's (passed in: this
    module imports nothing of the provider at load time)."""
    import torch
    C = int(z.shape[-1])
    fb = fallback(row, C, C)
    try:
        S = __import__(SUBPACKAGE, fromlist=["triangle_multiplication"])
    except ImportError as e:
        raise Refusal("needs:kernels.trimul_esm_shapes(import:%s)" % getattr(e, "name", "?"), row, fb)
    if row == "esm_k1ptr":
        zin = compute_input(z, cache)
        if zin.dtype != torch.bfloat16:
            raise Refusal("dtype:%s(bf16 z or bf16 autocast; fp32 trunks use esm_k1ptr:f32in)" % str(zin.dtype).replace("torch.", ""), row,
                          "esm_k1ptr:f32in" if zin.dtype == torch.float32 else fb)
    else:
        if z.dtype == torch.bfloat16:
            raise Refusal("dtype:bf16(bf16 callers use esm_k1ptr)", row, "esm_k1ptr")
        if z.dtype != torch.float32:
            raise Refusal("dtype:%s(not fp32)" % str(z.dtype).replace("torch.", ""), row, "cueq")
        zin = z.contiguous()
    m = None if mask is None else (mask[0] if (mask.dim() == 3 and int(mask.shape[0]) == 1) else mask)
    sub = cache.setdefault("esm_k1ptr", {}) if cache is not None else None
    try:
        out = S.triangle_multiplication(zin, m, direction=direction, weights=w, residual=residual, cache=sub, eps=eps, cell=config, pointer_k1=True)
    except S.Unsupported as e:
        raise Refusal(e.word, row, fb)
    return out if out.shape == z.shape else out.reshape(z.shape)
