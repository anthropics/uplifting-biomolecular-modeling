"""Row ``of3_form``: the OpenFold-family TriangleMultiplicativeUpdate INFERENCE statement, issued whole-tensor (no host chunk loop).

The statement (an OpenFold-family engine's ``TriangleMultiplicativeUpdate._inference_forward``: what its template pair stack -- and every
pair block called without the cuEquivariance flag -- executes; the arithmetic of this provider's ``torch_math`` row under bf16)::

    x    = LN_in(z)                                        OpenFold LayerNorm: bf16 z -> the bf16 kernel with bf16-cast gamma/beta, bf16 out
                                                           (autocast disabled); fp32 z -> fp32 (under bf16 autocast too: layer_norm is fp32 there)
    p_a  = sigmoid(x W_ag^T) ; p_a *= (x W_ap^T) ; p_a *= mask     OpenFold Linear: bf16 x -> bf16 GEMM with bf16-cast weights; fp32 x -> the
    p_b  = sigmoid(x W_bg^T) ; p_b *= (x W_bp^T) ; p_b *= mask     fp32 GEMM (TF32 by the process flags) or, under bf16 autocast, the bf16 GEMM
                                                           of the bf16-cast operands; every product rounded in this order, in place
    X[i,j,h] = sum_k p_a[i,k,h] p_b[j,k,h]   (outgoing)  |  sum_k p_a[k,i,h] p_b[k,j,h]   (incoming)
                                                           cuBLAS batched NN GEMM over h on contiguous [h,i,k] x [h,k,j] operands -- the layouts
                                                           and, column block by column block of 256, the problem shapes upstream's einsum hands it
    u    = LN_out(X) W_o^T ; u *= sigmoid(x W_og^T)        (LN_out / Linear by the same dtype rules)
    returns u, or z + u (residual=True; upstream adds the update into z in z's dtype)

Upstream computes these statements per 256-column block with LN_in and the a-plane recomputed per block and a rotating z-cache (an
in-place algorithm for activation memory).  LayerNorm is row-wise and the projections reduce over c_z only, so issuing them once over the
whole tensor changes no bit; the contraction keeps upstream's 256-column blocks so cuBLAS is handed upstream's problem shapes.  Per direction:
2 LayerNorms (the core's exactln row -- ATen's LayerNorm kernel replicated bit for bit -- where it admits the width on this device, else
ATen's own), 6 GEMMs, ceil(N/256) batched GEMMs, ~10 elementwise kernels, 3 layout copies; no Python loop over token blocks except the
contraction's.

Class: EXACT vs that statement BY MEASUREMENT (cuBLAS chooses its kernel per problem shape; TRIMUL_CELLS.json records the stacks and cells
where the row measured bitwise-identical to the engine module, and a kit's first-call proof holds it per process).  Versus the cuEquivariance
op the row is TOLERANCE class (LayerNorm statistics and the gate products round differently: max abs ~1 bf16 ulp of the pair activations).

Envelope: CUDA; z bf16 | fp32 (fp16 refused by name); any c_z, any c_hidden (unfused projections: c_hidden != c_z served); mask None or
[B,N,N] of any float dtype (multiplied into the bf16 plane in place, as upstream); B >= 1; forward only (in-place elementwise chain); the ten
canonical weights without biases (OpenFold's projections carry none).  torch only inside functions.
"""
CHUNK = 256                      # upstream's _inplace_chunk_size default: the contraction's column block
# Layout configurations of the SAME statements (which tensor is a copy and which a view handed to cuBLAS with a leading dimension); each names
# arithmetic-identical calls, but cuBLAS chooses its kernel per layout, so a configuration is exact only where MEASURED bitwise (TRIMUL_CELLS.json
# rows.of3_form.configs records which):
#   planes  "rows": the four projections as x W^T ([M,c] x [c,H] -> planes [i,k,h], then layout copies to [h,i,k]) -- upstream's calls
#           "hm":   the projections as W x^T ([H,c] x [c,M] -> planes [h,i,k] directly: no layout copy; the transposed problem of the same GEMM)
#   bview   False: the contraction's second operand copied contiguous per 256-column block (what upstream's einsum does with its view)
#           True:  handed as a strided view (leading dimension N / a transposed operand) -- no copy
#   xout    "chunk": each block's product copied into X[i,j,h] (upstream's layout at once)  |  "direct": written by the batched GEMM into an
#           [h,i,j] buffer through a leading-dimension view, ONE transpose to [i,j,h] before LN_out
#   fused   True: the two gate chains (p = sigmoid(G); p *= P; p *= mask  and  u *= sigmoid(Gout)) each in ONE elementwise kernel keeping ATen's rounding
#           points (sigmoid = 1 / (1 + expf(-x)) in fp32 rounded to the GEMM dtype; each product in fp32 rounded to the GEMM dtype) when the planes are
#           16-bit (fp32 planes keep ATen's three kernels: the two expf differ in the last fp32 bit), and the [h,i,j] -> [i,j,h] transpose by a tiled
#           transpose kernel (loads and stores only) -- the kernels of row af3t_form (kernels/trimul/af3t_form.py); needs triton (else = hm_view)
CONFIGS = {"replica":  {"planes": "replica", "bview": False, "xout": "chunk", "fused": False},   # the module's OWN call sequence per 256-column block (its operand layouts
            #             exactly: `a` materialised contiguous as the module stores it, b / LN_out / linear / gate per block on compact blocks, the same einsum call) without
            #             its per-block recomputation of the a-plane and its z-cache: bitwise at EVERY token count where measured (the other configs only at N % 16 == 0)
            "upstream": {"planes": "rows", "bview": False, "xout": "chunk", "fused": False},
           "hm": {"planes": "hm", "bview": False, "xout": "direct", "fused": False},
           "hm_view": {"planes": "hm", "bview": True, "xout": "direct", "fused": False},
           "hm_chunk": {"planes": "hm", "bview": False, "xout": "chunk", "fused": False},
           "hm_fused": {"planes": "hm", "bview": True, "xout": "direct", "fused": True}}
DEFAULT_CONFIG = "replica"        # the every-N-exact layout (measured bitwise at every token count swept); the fast layouts are named per cell and served at the aligned sizes through the vouch classes
LN_WORD = "exactln"              # the core LayerNorm row asked first (bitwise ATen replica); ATen's own kernel when it refuses

_KEYS = ("ln_in_w", "ln_in_b", "w_ag", "w_ap", "w_bg", "w_bp", "ln_out_w", "ln_out_b", "w_o", "w_og")


def supported(z, mask, w, backward=False):
    """None when the row can serve these tensors, else the refusal word."""
    import torch
    if not z.is_cuda:
        return "device:%s" % z.device.type
    if z.dtype not in (torch.bfloat16, torch.float32):
        return "dtype:%s(bf16|fp32)" % str(z.dtype).replace("torch.", "")
    if z.dim() != 4:
        return "rank:%d(z [B,N,N,c])" % z.dim()
    if mask is not None and (mask.dim() != 3 or tuple(mask.shape) != tuple(z.shape[:3])):
        return "mask_shape:%s(z %s)" % (tuple(mask.shape), tuple(z.shape[:3]))
    if backward or (torch.is_grad_enabled() and (z.requires_grad or any(getattr(w.get(k), "requires_grad", False) for k in _KEYS))):
        return "backward(forward-only row: in-place elementwise chain)"
    missing = [k for k in _KEYS if w.get(k) is None]
    if missing:
        return "weights_missing:%s" % "+".join(missing)
    if int(w["w_ap"].shape[1]) != int(z.shape[-1]) or int(w["w_o"].shape[0]) != int(z.shape[-1]):
        return "weight_shape:w_ap%s,w_o%s(c_z %d)" % (tuple(w["w_ap"].shape), tuple(w["w_o"].shape), int(z.shape[-1]))
    return None


def _gemm_dtype(x):
    """The dtype OpenFold's Linear multiplies in for an input x: bf16 x -> bf16; fp32 x -> bf16 under bf16 autocast, else fp32."""
    import torch
    if x.dtype == torch.bfloat16:
        return torch.bfloat16
    if torch.is_autocast_enabled():
        try:
            adt = torch.get_autocast_dtype("cuda")
        except AttributeError:                                   # older torch
            adt = torch.get_autocast_gpu_dtype()
        if adt == torch.bfloat16:
            return torch.bfloat16
    return x.dtype


def _wsig(t):
    """A weight tensor's identity for the casts memoised on it: the tensor OBJECT (held by the cache entry, so neither its id nor its storage
    address can be recycled while the entry lives), its version counter where it has one (in-place updates; inference tensors have none), and
    its storage address / shape / stride / dtype / device.  Shape alone never keys a cache: another module's weight of the same shape is another
    tensor."""
    try:
        ver = t._version
    except RuntimeError:  # inference tensors: 'Inference tensors do not track version counter'
        ver = None
    return (id(t), ver, t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))


def _cast(cache, name, t, dtype):
    """``t`` in ``dtype``: a .to() copy memoised in ``cache`` under ``name`` ON THIS TENSOR (identity + version + storage signature, see _wsig;
    the entry holds the source tensor so the signature cannot be satisfied by a recycled address); the tensor itself when already in dtype.
    One entry per name: a cache shared across modules re-casts when the module changes, never serves another module's weight."""
    if t.dtype == dtype:
        return t
    key = ("of3_form.cast", name, str(dtype))
    sig = _wsig(t)
    hit = cache.get(key) if cache is not None else None
    if hit is not None and hit[0] == sig and hit[1] is t:
        return hit[2]
    v = t.detach().to(dtype=dtype)
    if cache is not None:
        cache[key] = (sig, t, v)
    return v


def _layer_norm(x, w, b, eps, cache, name):
    """OpenFold's LayerNorm forward: bf16 x -> bf16 statistics kernel with bf16-cast affine (autocast off); otherwise fp32 arithmetic on x as
    given (what autocast's layer_norm does with an fp32 input).  Served by the core's exactln row where it admits (bitwise ATen); else ATen."""
    import torch
    import torch.nn.functional as F
    C = int(x.shape[-1])
    if x.dtype == torch.bfloat16:
        w, b = _cast(cache, name + "_w", w, torch.bfloat16), _cast(cache, name + "_b", b, torch.bfloat16)
    else:
        w, b = _cast(cache, name + "_w32", w, x.dtype), _cast(cache, name + "_b32", b, x.dtype)
    x2 = x.reshape(-1, C)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    with torch.autocast(device_type="cuda", enabled=False):
        skey = ("of3_form.ln_sel", str(x.dtype), C)
        served = cache.get(skey) if cache is not None else None            # a Selection (the core row serves) | "aten" | None (first call)
        if served != "aten" and x2.data_ptr() % 16 == 0:
            from opt_core.kernels import ln as KL
            try:
                y, sel = KL.layer_norm(x2, (C,), w, b, eps, word=LN_WORD, selection=(served if served is not None else None))
            except KL.Refusal as e:                                          # the core row cannot serve here (capability, bindings, width): ATen's kernel IS the statement
                if cache is not None:
                    cache[skey] = "aten"
                    cache["of3_form.ln"] = "aten"
                    cache["of3_form.ln_reason"] = ("%s" % e)[:160]
            else:
                if cache is not None and served is None:
                    cache[skey] = sel
                    cache["of3_form.ln"] = "%s%s" % (sel.row, (":" + sel.variant) if getattr(sel, "variant", None) else "")
                return y.reshape(x.shape)
        return F.layer_norm(x2, (C,), w, b, eps).reshape(x.shape)


def _linear(x, wt, cache, name, gdt):
    """OpenFold's Linear forward without bias: the GEMM in ``gdt`` (see _gemm_dtype) with the weight cast to it; autocast off (the cast operands
    are exactly what autocast would hand the same GEMM)."""
    import torch
    import torch.nn.functional as F
    with torch.autocast(device_type="cuda", enabled=False):
        return F.linear(x if x.dtype == gdt else x.to(gdt), _cast(cache, name, wt, gdt))


def _plane(x, wg, wp, m, cache, tag, gdt):
    """p = sigmoid(x Wg^T); p *= x Wp^T; p *= mask -- upstream's in-place order.  [B,N,N,H] in gdt."""
    p = _linear(x, wg, cache, tag + "g", gdt)
    p.sigmoid_()
    p.mul_(_linear(x, wp, cache, tag + "p", gdt))
    if m is not None:
        p.mul_(m)
    return p


def _contract_rows(xg, m, w, out, cache, gdt, B, N, chunk):
    """Planes as x W^T (upstream's projection calls), layout copies to the contraction's operands, the contraction as upstream's einsum per
    256-column block, each block copied into X[b,i,j,h]."""
    import torch
    pa = _plane(xg, w["w_ag"], w["w_ap"], m, cache, "a", gdt)                           # [B,N,N,H]  (row, col, h)
    pb = _plane(xg, w["w_bg"], w["w_bp"], m, cache, "b", gdt)
    H = int(pa.shape[-1])
    if out:
        A = pa.permute(0, 3, 1, 2).contiguous()            # A[b,h,i,k] = pa[b,i,k,h]
    else:
        A = pa.permute(0, 3, 2, 1).contiguous()            # A[b,h,i,k] = pa[b,k,i,h]
    del pa
    X = torch.empty((B, N, N, H), dtype=A.dtype, device=xg.device)                     # X[b,i,j,h]
    for j0 in range(0, N, chunk):                                                       # ambient autocast state, as upstream's einsum runs in
        j1 = min(N, j0 + chunk)
        if out:
            bv = pb[:, j0:j1, :, :].permute(0, 3, 2, 1)                                 # [B,H,k,j] view of pb[b,j,k,h] rows j0:j1
        else:
            bv = pb[:, :, j0:j1, :].permute(0, 3, 1, 2)                                 # [B,H,k,j] view of pb[b,k,j,h] columns j0:j1
        Xc = torch.einsum("...ij,...jk->...ik", A, bv)                                  # [B,H,i,j1-j0]
        X[:, :, j0:j1, :].copy_(Xc.permute(0, 2, 3, 1))
        del Xc, bv
    return X


def _plane_hm(x2t, wg, wp, m1, cache, tag, gdt, H, N, fused=False):
    """p[h, (i,k)] = sigmoid(Wg x^T); p *= Wp x^T; p *= mask -- the projections' transposed problem: planes h-major without a layout copy.
    ``fused``: the three elementwise statements in one kernel (ATen's rounding points), the result in the Wp x^T buffer."""
    import torch
    g = torch.matmul(_cast(cache, tag + "g", wg, gdt), x2t)                             # [H,c] x [c,M] (x^T is a column-major view: no copy)
    if fused and gdt in (torch.bfloat16, torch.float16):                                # the fused kernel's sigmoid is ATen's to the 16-bit rounding (measured); fp32 planes keep ATen's kernels
        from .af3t_form import gate_
        p = torch.matmul(_cast(cache, tag + "p", wp, gdt), x2t)
        mk = None if m1 is None else m1.reshape(-1)
        if mk is not None and not mk.is_contiguous():
            mk = mk.contiguous()
        gate_(p, g, mk, mask_mod=(int(mk.numel()) if mk is not None else None), order=1)   # p = (sigmoid(g) * p) * mask[e % M]
        return p.view(H, N, N)
    g.sigmoid_()
    g.mul_(torch.matmul(_cast(cache, tag + "p", wp, gdt), x2t))
    if m1 is not None:
        g.mul_(m1)                                                                      # [1,M] broadcast over h
    return g.view(H, N, N)


def _contract_hm(xg, m, w, out, cache, gdt, N, chunk, cfg):
    """B == 1.  Planes h-major from the transposed projections; the contraction per 256-column block by the batched GEMM with the operands as
    (leading-dimension) views or one contiguous copy (cfg bview); the product either written straight into an [h,i,j] buffer and transposed once
    (cfg xout direct) or copied per block into X[i,j,h] (chunk)."""
    import torch
    H = int(w["w_ap"].shape[0])
    M = N * N
    x2t = xg.reshape(M, -1).t()                                                         # [c, M] view
    m1 = None if m is None else m.reshape(1, M)
    with torch.autocast(device_type="cuda", enabled=False):
        PaT = _plane_hm(x2t, w["w_ag"], w["w_ap"], m1, cache, "a", gdt, H, N, cfg.get("fused", False))   # [H, row, col]
        PbT = _plane_hm(x2t, w["w_bg"], w["w_bp"], m1, cache, "b", gdt, H, N, cfg.get("fused", False))
    if out:                                             # X[i,j] = sum_k a[i,k] b[j,k]:  A[h,i,k] = PaT as is; Bm[h,k,j] = PbT[h,j,k]^T
        A3 = PaT
        Bm = PbT.transpose(1, 2)                        # [H,k,j] column-major view
        if not cfg["bview"]:
            Bm = Bm.contiguous()                        # one layout copy; blocks are leading-dimension views of it
    else:                                               # X[i,j] = sum_k a[k,i] b[k,j]:  A[h,i,k] = PaT[h,k,i]^T; Bm[h,k,j] = PbT as is
        A3 = PaT.transpose(1, 2)
        if not cfg["bview"]:
            A3 = A3.contiguous()
        Bm = PbT
    if cfg["xout"] == "direct":
        XT = torch.empty((H, N, N), dtype=PaT.dtype, device=xg.device)                 # XT[h,i,j]
        for j0 in range(0, N, chunk):
            j1 = min(N, j0 + chunk)
            torch.bmm(A3, Bm[:, :, j0:j1], out=XT[:, :, j0:j1])
        del A3, Bm, PaT, PbT
        if cfg.get("fused", False):
            from .af3t_form import channels_last
            return channels_last(XT)[None]                                              # [1,N,N,H]: the tiled transpose kernel
        return XT.permute(1, 2, 0).contiguous()[None]                                   # [1,N,N,H]: one transpose
    X = torch.empty((1, N, N, H), dtype=PaT.dtype, device=xg.device)
    for j0 in range(0, N, chunk):
        j1 = min(N, j0 + chunk)
        Xc = torch.einsum("...ij,...jk->...ik", A3, Bm[:, :, j0:j1])                    # [H,i,j1-j0]
        X[0, :, j0:j1, :].copy_(Xc.permute(1, 2, 0))
        del Xc
    del A3, Bm, PaT, PbT
    return X


def _forward_replica(xg, m, w, out, cache, gdt, B, N, chunk, eps):
    """Config replica: the module's inference path CALL BY CALL for everything downstream of LN_in (whose whole-tensor issue is row-wise exact):
    a = the a-plane stored CONTIGUOUS [B,H,N,N] exactly as the module's row-block writes leave it (outgoing a[b,h,i,k] = p_a[b,i,k,h]; incoming
    a[b,h,i,k] = p_a[b,k,i,h]); then per output-column block j0:j1 -- the MODULE's blocks: [0, ceil(N/2)) and [ceil(N/2), N) each in steps of
    256 (its z-cache halves), NOT 256-blocks from 0 -- the b-plane from the module's z block (outgoing: ROWS j0:j1, a
    contiguous block; incoming: COLUMNS j0:j1 made contiguous, as its LayerNorm output is) -> permute_final_dims(2,0,1) view (+ transpose for
    outgoing) -> the SAME `torch.einsum('...ij,...jk->...ik', a, b_blk)` call (same operand shapes AND strides, so the same cuBLAS kernel) ->
    permute (1,2,0) view -> LN_out -> W_o -> * sigmoid(W_og LN_in(z)[:, :, j0:j1] made contiguous) -> written to the update's column block.
    The module recomputes the a-plane's LayerNorm input per block and rotates a z-cache to survive its own in-place writes; the update here is
    out of place, so neither is needed and no statement changes."""
    import torch
    H = int(w["w_ap"].shape[0]); C = int(w["w_o"].shape[0])
    pa = _plane(xg, w["w_ag"], w["w_ap"], m, cache, "a", gdt)                           # [B,N,N,H] (row r, col c, h): the module's a-projection (row blocks give the same bytes)
    if out:
        a = pa.permute(0, 3, 1, 2).contiguous()                                          # a[b,h,i,k] = pa[b,i,k,h]   (need_transpose False: p[..., rows, :] = chunk)
    else:
        a = pa.permute(0, 3, 2, 1).contiguous()                                          # a[b,h,i,k] = pa[b,k,i,h]   (need_transpose True:  p[..., :, rows] = chunk^T)
    del pa
    u = torch.empty((B, N, N, C), dtype=gdt, device=xg.device)
    half = N // 2 + N % 2                                                                # the module's blocks: [0, half) in steps of `chunk`, then [half, N) in steps of `chunk`
    bounds = [(j0, min(j0 + chunk, half)) for j0 in range(0, half, chunk)] + [(j0, min(j0 + chunk, N)) for j0 in range(half, N, chunk)]
    for j0, j1 in bounds:
        if out:                                                                          # b rows j0:j1 (z[:, j0:j1, :, :]: contiguous)
            xb = xg[:, j0:j1, :, :]
            mb = None if m is None else m[:, j0:j1, :, :]
            pb = _plane(xb, w["w_bg"], w["w_bp"], mb, cache, "b", gdt)                   # [B,cols,N,H]
            bv = pb.permute(0, 3, 1, 2).transpose(-1, -2)                                # permute_final_dims (2,0,1) -> [B,H,cols,N]; need_transpose (outgoing) -> [B,H,N(k),cols(j)]
        else:                                                                            # b columns j0:j1 (z[:, :, j0:j1, :]: the LayerNorm's contiguous output of that slice)
            xb = xg[:, :, j0:j1, :].contiguous()
            mb = None if m is None else m[:, :, j0:j1, :].contiguous()
            pb = _plane(xb, w["w_bg"], w["w_bp"], mb, cache, "b", gdt)                   # [B,N,cols,H]
            bv = pb.permute(0, 3, 1, 2)                                                  # [B,H,N(k),cols(j)]
        Xc = torch.einsum("...ij,...jk->...ik", a, bv)                                   # the module's call: [B,H,N(i),cols(j)]
        del pb, bv
        xc = Xc.permute(0, 2, 3, 1)                                                      # permute_final_dims (1,2,0): [B,N,cols,H] view
        y = _layer_norm(xc, w["ln_out_w"], w["ln_out_b"], eps, cache, "ln_out")          # LN over a contiguous copy (ATen's own first step)
        del Xc, xc
        gdt2 = _gemm_dtype(y)
        ub = _linear(y, w["w_o"], cache, "o", gdt2)                                      # [B,N,cols,C]
        del y
        g = _linear(xg[:, :, j0:j1, :].contiguous(), w["w_og"], cache, "og", gdt)        # the gate on the module's column block of LN_in(z) (made contiguous by its LayerNorm)
        g.sigmoid_()
        ub.mul_(g)
        del g
        u[:, :, j0:j1, :].copy_(ub)
        del ub
    return u


def forward(z, mask, direction, w, eps=1e-5, residual=False, cache=None, chunk=CHUNK, config=None):
    """z [B,N,N,c_z] (pre-LayerNorm pair tensor), mask None | [B,N,N], direction 'outgoing' | 'incoming', w = the ten canonical tensors as the
    module holds them (any float dtype: cast per OpenFold's rules, casts memoised in ``cache``).  Returns the update [B,N,N,c_z] in the GEMM
    dtype (bf16 under bf16 residency or autocast, else z's), or z + update when residual."""
    import torch
    why = supported(z, mask, w)
    if why:
        raise ValueError("of3_form: " + why)
    cache = cache if cache is not None else {}
    out = str(direction).lower().startswith("out")
    B, N, _, C = z.shape
    x = _layer_norm(z, w["ln_in_w"], w["ln_in_b"], eps, cache, "ln_in")             # [B,N,N,C]  bf16 (bf16 z) | fp32
    gdt = _gemm_dtype(x)
    xg = x if x.dtype == gdt else x.to(gdt)                                              # one cast shared by the five projections (autocast casts per call: same values)
    m = None if mask is None else mask.unsqueeze(-1)
    cfg = CONFIGS[config or DEFAULT_CONFIG]
    if cfg.get("fused", False):
        try:
            import triton  # noqa: F401
        except ImportError:                                                              # no triton: the same statements by ATen's kernels
            cfg = CONFIGS["hm_view"]
    if cfg["planes"] == "replica":
        u = _forward_replica(xg, m, w, out, cache, gdt, B, N, int(chunk), eps)          # [B,N,N,C]: the module's per-block sequence (contraction .. gate)
        if residual:
            return z + u if z.dtype == u.dtype else (z + u.to(z.dtype))
        return u
    if cfg["planes"] == "hm" and B == 1:
        X = _contract_hm(xg, m, w, out, cache, gdt, N, int(chunk), cfg)                # [1,N,N,H]
    else:
        X = _contract_rows(xg, m, w, out, cache, gdt, B, N, int(chunk))
    y = _layer_norm(X, w["ln_out_w"], w["ln_out_b"], eps, cache, "ln_out")            # [B,N,N,H]
    del X
    gdt2 = _gemm_dtype(y)
    u = _linear(y, w["w_o"], cache, "o", gdt2)                                            # [B,N,N,C]
    del y
    g = _linear(xg, w["w_og"], cache, "og", gdt)
    if cfg.get("fused", False) and u.dtype == g.dtype and u.dtype in (torch.bfloat16, torch.float16) and u.is_contiguous() and g.is_contiguous():
        from .af3t_form import gate_
        gate_(u, g, order=1)                                                             # u = sigmoid(g) * u: one pass, ATen's roundings
    else:
        g.sigmoid_()
        u.mul_(g)
    del g
    if residual:
        return z + u if z.dtype == u.dtype else (z + u.to(z.dtype))
    return u
