"""Row ``af3t_form``: the AlphaFold3-family (xfold-form) TriangleMultiplication MODULE statement, issued whole-tensor in measured layouts.

The statement (an AF3-family torch engine's ``TriangleMultiplication.forward``; c_pair = c; no batch dimension: the module runs per item
``[N, N, c]``; its trunk pair block AND its template pair stack (c 64) call the same class)::

    x   = LN_in(z)                     the engine's fused row LayerNorm (a Triton kernel): fp32 block-lane partial sums reduced once, two-pass
                                       variance, 1/sqrt, affine in fp32, ONE rounding to z's dtype (fp32 gamma/beta as the module holds them)
    P   = x Wp^T   [N, N, 2c]          ONE Linear c -> 2c without bias (the a / b planes INTERLEAVED: channel 2h = a_h, 2h+1 = b_h); under bf16
    G   = x Wg^T   [N, N, 2c]          autocast the bf16 GEMM of the bf16-cast operands
    P  *= mask[i, j]                   all 2c channels at (i, j) (the module multiplies its channel-major view: the same elements)
    P  *= sigmoid(G)                   sigmoid rounded to the GEMM dtype, then the product -- mask FIRST, gate second (the module's order)
    a, b = P[..., 0::2], P[..., 1::2]  the interleaved halves as [c, N, N] channel-major views
    X   = einsum('cik,cjk->cij', a, b)  (outgoing)  |  einsum('ckj,cki->cij', a, b)  (incoming)
                                       torch clones each strided half into a contiguous operand and runs ONE batched GEMM over c (no column
                                       blocks); the incoming result is a permuted view of storage (c, j, i)
    y   = LN_out(X as [N, N, c] contiguous)   the same LayerNorm on the channels-last copy of the product
    u   = y Wo^T ; u *= sigmoid(x Wog^T)
    returns u  (the caller adds it into the pair tensor), or z + u (residual=True)

Canonical weights (this provider's ten names, as the engine kit's adapter hands them): outgoing (w_ag, w_ap) = the even channels of (Wg, Wp),
(w_bg, w_bp) = the odd ones; INCOMING swaps them ((w_ag, w_ap) = odd, (w_bg, w_bp) = even) because the provider's incoming statement
sum_k A[k,i] B[k,j] is the module's sum_k a[k,j] b[k,i] with (A, B) = (b, a).  The row re-interleaves them into the module's [2c, c] weights
(memoised in ``cache``) so the module's ONE projection GEMM is issued, not two.

Layout configurations (``CONFIGS``; the SAME statements, differing in which tensor is a copy and which a view handed to cuBLAS; each exact only
where MEASURED bitwise -- TRIMUL_CELLS.json rows.af3t_form.configs):
    upstream   the module's own calls: planes [N, N, 2c], the strided halves handed to torch.einsum (its clones = generic strided copies),
               the product's channels-last copy by a generic strided copy
    hm_view    the projections' transposed problem (Wp x^T -> planes [2c, N*N]: the halves are leading-dimension VIEWS, no copy), the
               contraction as ONE batched GEMM on those views in the module's orientation per equation, the product made channels-last by a
               tiled transpose kernel (loads and stores only)
    hm_copy    as hm_view with the product's channels-last copy by the generic strided copy (the transpose kernel isolated)
    hm_fused   as hm_view with the two gate chains (P *= mask; P *= sigmoid(G)  and  u *= sigmoid(Gout)) each in ONE elementwise kernel that
               keeps ATen's rounding points (product with the mask rounded to P's dtype; sigmoid = 1 / (1 + expf(-x)) in fp32 rounded to G's
               dtype; the final product rounded to P's dtype) -- three passes over the planes instead of seven

Class: EXACT vs that module BY MEASUREMENT (cuBLAS chooses its kernel per problem shape and layout; the cells record where each config measured
bitwise).  Versus the cuequivariance op and versus the OpenFold-family module (row of3_form: sigmoid-then-product gate before the mask, ATen
LayerNorm, per-256-column contraction) the row is TOLERANCE class.

Envelope: CUDA; z bf16 | fp32; z [N, N, c] or [B, N, N, c] (served per item); any c (c_hidden == c: the module has one width); mask None |
[N, N] / [B, N, N] of any float dtype; forward only; no biases; Triton >= 3.0 for the LayerNorm / transpose kernels (else refused by name).
torch / triton only inside functions.  The LayerNorm kernel restates the arithmetic order of the Triton tutorial's fused layer-norm forward
(one program's 1-D body per row, several rows per program); the transpose kernel is a plain tiled copy.  Both are this tree's own source.
"""
CONFIGS = ("upstream", "hm_view", "hm_copy", "hm_fused")
DEFAULT_CONFIG = "upstream"        # from 0.5.78.0: the every-N-exact layout (measured bitwise at every token count swept); the fast layouts are named per cell and served at the aligned sizes through the vouch classes
LN_ROWS = 8                       # rows per LayerNorm program (each row = the 1-D body: bitwise whatever the count)
TILE = 64                         # transpose tile edge
_KEYS = ("ln_in_w", "ln_in_b", "w_ag", "w_ap", "w_bg", "w_bp", "ln_out_w", "ln_out_b", "w_o", "w_og")
_STATE = {"ln": None, "tr": None, "gate": None}
EQ_OUT, EQ_IN = "cik,cjk->cij", "ckj,cki->cij"


def supported(z, mask, w, backward=False):
    """None when the row can serve these tensors, else the refusal word."""
    import torch
    if not z.is_cuda:
        return "device:%s" % z.device.type
    if z.dtype not in (torch.bfloat16, torch.float32):
        return "dtype:%s(bf16|fp32)" % str(z.dtype).replace("torch.", "")
    if z.dim() not in (3, 4):
        return "rank:%d(z [N,N,c] | [B,N,N,c])" % z.dim()
    lead = tuple(z.shape[:-3])
    if mask is not None and tuple(mask.shape) != lead + tuple(z.shape[-3:-1]):
        return "mask_shape:%s(z %s)" % (tuple(mask.shape), tuple(z.shape))
    if backward or (torch.is_grad_enabled() and (z.requires_grad or any(getattr(w.get(k), "requires_grad", False) for k in _KEYS))):
        return "backward(forward-only row: in-place elementwise chain)"
    missing = [k for k in _KEYS if w.get(k) is None]
    if missing:
        return "weights_missing:%s" % "+".join(missing)
    C = int(z.shape[-1])
    for k in ("w_ap", "w_bp", "w_ag", "w_bg", "w_o", "w_og"):
        if tuple(w[k].shape) != (C, C):
            return "weight_shape:%s%s(c %d: the module has c_hidden == c)" % (k, tuple(w[k].shape), C)
    try:
        import triton  # noqa: F401
    except ImportError:
        return "import:triton"
    return None


# ----------------------------------------------------------------------------------------------------------------------------- kernels
def _ln_kernel():
    """The engine-form row LayerNorm: per row, fp32 block-lane accumulation of x (one block when C <= BLOCK), tl.sum once, the same for the
    centred squares, rstd = 1 / sqrt(var + eps), y = (x - mean) * rstd * w (+ b) stored in Y's dtype.  ROWS consecutive rows per program, each by
    the identical 1-D body (the arithmetic order of the Triton tutorial's fused layer-norm forward)."""
    if _STATE["ln"] is not None:
        return _STATE["ln"]
    import triton
    import triton.language as tl

    @triton.jit
    def _ln_fwd_rows(X, Y, W, B, M, N, eps, ROWS: tl.constexpr, BLOCK_SIZE: tl.constexpr, HAS_W: tl.constexpr, HAS_B: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        for r in range(ROWS):
            row = pid * ROWS + r
            if row < M:
                Xr = X + row * N
                Yr = Y + row * N
                mean = 0
                _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
                for off in range(0, N, BLOCK_SIZE):
                    cols = off + tl.arange(0, BLOCK_SIZE)
                    a = tl.load(Xr + cols, mask=cols < N, other=0.).to(tl.float32)
                    _mean += a
                mean = tl.sum(_mean, axis=0) / N
                _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
                for off in range(0, N, BLOCK_SIZE):
                    cols = off + tl.arange(0, BLOCK_SIZE)
                    x = tl.load(Xr + cols, mask=cols < N, other=0.).to(tl.float32)
                    x = tl.where(cols < N, x - mean, 0.)
                    _var += x * x
                var = tl.sum(_var, axis=0) / N
                rstd = 1 / tl.sqrt(var + eps)
                for off in range(0, N, BLOCK_SIZE):
                    cols = off + tl.arange(0, BLOCK_SIZE)
                    m = cols < N
                    if HAS_W:
                        w = tl.load(W + cols, mask=m)
                        if HAS_B:
                            b = tl.load(B + cols, mask=m)
                    x = tl.load(Xr + cols, mask=m, other=0.).to(tl.float32)
                    x_hat = (x - mean) * rstd
                    if HAS_W:
                        y = x_hat * w
                        if HAS_B:
                            y += b
                    else:
                        y = x_hat
                    tl.store(Yr + cols, y, mask=m)

    _STATE["ln"] = _ln_fwd_rows
    return _ln_fwd_rows


def layer_norm_rows(x2, w, b, eps, rows=LN_ROWS):
    """x2 [M, C] contiguous (any float dtype), w / b fp32 (or None) -> y [M, C] in x2's dtype: the engine-form LayerNorm."""
    import torch
    import triton
    assert x2.dim() == 2 and x2.is_contiguous(), (tuple(x2.shape), x2.stride())
    M, C = int(x2.shape[0]), int(x2.shape[1])
    y = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(C)
    if BLOCK > 65536 // x2.element_size():
        raise ValueError("af3t_form: layer_norm width %d over the fused kernel's 64 KB row" % C)
    nw = min(max(BLOCK // 256, 1), 8)
    grid = (triton.cdiv(M, rows),)
    _ln_kernel()[grid](x2, y, w if w is not None else x2, b if b is not None else x2, M, C, eps,
                       ROWS=rows, BLOCK_SIZE=BLOCK, HAS_W=w is not None, HAS_B=b is not None, num_warps=nw)
    return y


def _tr_kernel():
    """dst[b, x, y] = src[b, x, y] between two layouts of one [B, X, Y] index space: a tile [TY, TX] loaded along the source's unit-stride axis (x)
    and stored along the destination's (y).  Loads and stores only."""
    if _STATE["tr"] is not None:
        return _STATE["tr"]
    import triton
    import triton.language as tl

    @triton.jit
    def _tile_transpose(SRC, DST, X, Y, s_sb, s_sx, s_sy, s_db, s_dx, s_dy, TX: tl.constexpr, TY: tl.constexpr):
        b = tl.program_id(2).to(tl.int64)
        xs = (tl.program_id(0) * TX + tl.arange(0, TX)).to(tl.int64)
        ys = (tl.program_id(1) * TY + tl.arange(0, TY)).to(tl.int64)
        m = (ys[:, None] < Y) & (xs[None, :] < X)
        v = tl.load(SRC + b * s_sb + ys[:, None] * s_sy + xs[None, :] * s_sx, mask=m)
        tl.store(DST + b * s_db + ys[:, None] * s_dy + xs[None, :] * s_dx, v, mask=m)

    _STATE["tr"] = _tile_transpose
    return _tile_transpose


def channels_last(X3):
    """[c, N, N] (contiguous, or the transposed view of a contiguous (c, j, i) product) -> contiguous [N, N, c] by the tiled transpose."""
    import torch
    import triton
    c, N, _ = X3.shape
    out = torch.empty((N, N, c), dtype=X3.dtype, device=X3.device)
    sc, si, sj = X3.stride()
    if sj == 1:                                   # storage (c, i, j): index space (b=i, x=j, y=c): source unit-stride along j, destination along c
        grid = (triton.cdiv(N, TILE), triton.cdiv(c, TILE), N)
        _tr_kernel()[grid](X3, out, N, c, si, sj, sc, N * c, c, 1, TX=TILE, TY=TILE, num_warps=4)
    else:                                         # storage (c, j, i) (the incoming product's permuted view): (b=j, x=i, y=c)
        grid = (triton.cdiv(N, TILE), triton.cdiv(c, TILE), N)
        _tr_kernel()[grid](X3, out, N, c, sj, si, sc, c, N * c, 1, TX=TILE, TY=TILE, num_warps=4)
    return out


def _gate_kernel():
    """P[e] = round(round(P[e] * mask[m(e)]) * round(sigmoid(G[e]))): ATen's three elementwise statements (mul_ by the mask in opmath fp32 rounded
    to P's dtype, sigmoid in fp32 with expf and an IEEE division rounded to G's dtype, mul_ rounded to P's dtype) in one pass.  m(e) = e % M for
    channel-major planes [2c, M] (MASK_MOD) or e // W for row-major planes [M, W]."""
    if _STATE["gate"] is not None:
        return _STATE["gate"]
    import triton
    import triton.language as tl
    try:
        from triton.language.extra import libdevice
    except ImportError:                                       # older trees
        from triton.language.extra.cuda import libdevice

    @triton.jit
    def _gate_fused(P, G, MASK, n, M, W, HAS_MASK: tl.constexpr, MASK_MOD: tl.constexpr, BLOCK: tl.constexpr, ORDER: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        ok = offs < n
        p = tl.load(P + offs, mask=ok, other=0.)
        pf = p.to(tl.float32)
        if HAS_MASK:
            if MASK_MOD:
                midx = offs % M
            else:
                midx = offs // W
            mk = tl.load(MASK + midx, mask=ok, other=0.).to(tl.float32)
        if ORDER == 0:                                        # P *= mask; P *= sigmoid(G)            (this row's module)
            if HAS_MASK:
                pf = (pf * mk).to(p.dtype).to(tl.float32)
        g = tl.load(G + offs, mask=ok, other=0.)
        sg = tl.fdiv(1.0, 1.0 + libdevice.exp(-g.to(tl.float32)), ieee_rounding=True)
        sg = sg.to(g.dtype).to(tl.float32)
        if ORDER == 0:
            tl.store(P + offs, (pf * sg).to(p.dtype), mask=ok)
        else:                                                 # G = sigmoid(G); G *= P; G *= mask    (the OpenFold-family order; result stored in P's buffer)
            v = (sg * pf).to(g.dtype).to(tl.float32)
            if HAS_MASK:
                v = (v * mk).to(g.dtype).to(tl.float32)
            tl.store(P + offs, v.to(p.dtype), mask=ok)

    _STATE["gate"] = _gate_fused
    return _gate_fused


def gate_(P, G, mask=None, mask_mod=None, block=2048, order=0):
    """In place into P.  order 0: P *= mask; P *= sigmoid(G)  (this row's module).  order 1: P = (sigmoid(G) * P) * mask -- the OpenFold-family
    module's `g.sigmoid_(); g.mul_(p); g.mul_(mask)` with every intermediate rounded as ATen rounds it (row of3_form).  Broadcast mask index =
    flat % mask_mod when given, else flat // P.shape[-1].  P, G same shape, same dtype, contiguous; mask a contiguous float tensor or None."""
    import triton
    assert P.is_contiguous() and G.is_contiguous() and P.shape == G.shape and P.dtype == G.dtype, (P.shape, G.shape, P.dtype, G.dtype)
    n = P.numel()
    grid = (triton.cdiv(n, block),)
    _gate_kernel()[grid](P, G, mask if mask is not None else P, n, int(mask_mod or 1), int(P.shape[-1]),
                         HAS_MASK=mask is not None, MASK_MOD=mask_mod is not None, BLOCK=block, ORDER=int(order), num_warps=4)
    return P


# ----------------------------------------------------------------------------------------------------------------------------- statements
def _gemm_dtype(x):
    """The dtype the module's Linear multiplies in for an input x: bf16 under bf16 autocast (or bf16 x), else x's dtype."""
    import torch
    if x.dtype == torch.bfloat16:
        return torch.bfloat16
    if torch.is_autocast_enabled():
        dt = torch.get_autocast_dtype("cuda") if hasattr(torch, "get_autocast_dtype") else torch.get_autocast_gpu_dtype()
        if dt == torch.bfloat16:
            return torch.bfloat16
    return x.dtype


def _wsig(*ts):
    """The identity of the weight tensor(s) a memoised value was built from: per tensor the OBJECT id (the entry holds the tensors, so ids and
    storage addresses cannot be recycled while it lives), the version counter where the tensor has one (inference tensors have none), storage
    address, shape, stride, dtype, device.  Shape alone never keys a cache: another module's weight of the same shape is another tensor."""
    out = []
    for t in ts:
        try:
            ver = t._version
        except RuntimeError:  # 'Inference tensors do not track version counter'
            ver = None
        out.append((id(t), ver, t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device)))
    return tuple(out)


def _cached(cache, key, make, *srcs):
    """``make()`` memoised in ``cache`` under ``key`` FOR THESE SOURCE TENSORS: a hit requires the same tensor objects with the same signature
    (_wsig); the entry holds the sources.  One entry per key: a cache shared across modules rebuilds when the module changes and never serves a
    value built from another module's weights."""
    sig = _wsig(*srcs)
    hit = cache.get(key)
    if hit is not None and isinstance(hit, tuple) and len(hit) == 3 and hit[0] == sig and all(a is b for a, b in zip(hit[1], srcs)):
        return hit[2]
    v = make()
    cache[key] = (sig, tuple(srcs), v)
    return v


def _w2(cache, name, w_even, w_odd, dt):
    """The module's interleaved [2c, c] weight rebuilt from its two halves (rows 0::2 = w_even, 1::2 = w_odd), cast to dt; memoised on the two
    source tensors (see _cached)."""
    import torch

    def make():
        c = int(w_even.shape[0])
        out = torch.empty((2 * c, int(w_even.shape[1])), dtype=dt, device=w_even.device)
        out[0::2].copy_(w_even.detach())
        out[1::2].copy_(w_odd.detach())
        return out
    return _cached(cache, "%s_%s" % (name, str(dt).replace("torch.", "")), make, w_even, w_odd)


def _ln(x, w, b, eps, cache, name):
    """The module's LayerNorm on x [..., C]: x made contiguous (the module's first statement), rows through the engine-form kernel, output in
    x's dtype; gamma / beta as the module holds them (fp32 for fp32 masters; cast up once if held narrower)."""
    import torch
    C = int(x.shape[-1])
    xc = x.contiguous()
    wf = _cached(cache, name + "_w", lambda: w.detach().to(torch.float32).contiguous(), w)
    bf = _cached(cache, name + "_b", lambda: b.detach().to(torch.float32).contiguous(), b)
    return layer_norm_rows(xc.view(-1, C), wf, bf, eps).view(xc.shape)


def _one(z, mask, out, w, eps, residual, cache, config):
    """One item: z [N, N, c], mask None | [N, N]."""
    import torch
    import torch.nn.functional as F
    N, _, C = z.shape
    M = N * N
    x = _ln(z, w["ln_in_w"], w["ln_in_b"], eps, cache, "ln_in")                      # [N, N, C] in z's dtype
    gdt = _gemm_dtype(x)
    xg = x if x.dtype == gdt else x.to(gdt)                                           # autocast's cast of the Linear input (one cast serves the three Linears on x)
    # the module's interleaved weights: even channels = its a plane, odd = its b plane; the provider's incoming (A, B) = the module's (b, a)
    if out:
        wp_e, wp_o, wg_e, wg_o = w["w_ap"], w["w_bp"], w["w_ag"], w["w_bg"]
    else:
        wp_e, wp_o, wg_e, wg_o = w["w_bp"], w["w_ap"], w["w_bg"], w["w_ag"]
    Wp = _w2(cache, "wp_" + ("out" if out else "in"), wp_e, wp_o, gdt)
    Wg = _w2(cache, "wg_" + ("out" if out else "in"), wg_e, wg_o, gdt)
    eq = EQ_OUT if out else EQ_IN
    with torch.autocast(device_type="cuda", enabled=False):                            # the operands are cast exactly as autocast casts them; the calls below are the module's
        if config == "upstream":
            P = F.linear(xg, Wp)                                                      # [N, N, 2c]
            if mask is not None:
                P.mul_(mask.unsqueeze(-1))                                            # every channel at (i, j) by mask[i, j] (the module: its [2c, N, N] view *= mask[None])
            P.mul_(torch.sigmoid(F.linear(xg, Wg)))                                   # sigmoid rounded to the GEMM dtype, then the product
            Pv = P.permute(2, 0, 1).reshape(C, 2, N, N)                               # the module's channel-major view split into interleaved halves (views)
            a, b = Pv[:, 0], Pv[:, 1]
            X = torch.einsum(eq, a, b)                                                # torch clones the strided halves (generic copies) and runs one batched GEMM
            Xc = X.permute(1, 2, 0).contiguous()                                      # center_norm's x.contiguous(): generic strided copy
        else:
            x2t = xg.reshape(M, C).t()                                                # [C, M] column-major view (no copy)
            PT = torch.matmul(Wp, x2t)                                                # [2c, M]: the projection's transposed problem -- planes channel-major as written
            if config == "hm_fused" and gdt in (torch.bfloat16, torch.float16):      # (fp32 planes: ATen's kernels -- the fused sigmoid is ATen's to the 16-bit rounding only)
                mk = None if mask is None else mask.reshape(M).contiguous()
                gate_(PT, torch.matmul(Wg, x2t), mk, mask_mod=M)                         # P *= mask[m]; P *= sigmoid(G): ATen's roundings, one pass
            else:
                if mask is not None:
                    PT.mul_(mask.reshape(1, M))
                PT.mul_(torch.sigmoid(torch.matmul(Wg, x2t)))
            ev = PT[0::2].reshape(C, N, N)                                            # even channels: [c, p, q] = P[p, q, 2c]   (leading dimension N, batch stride 2M: views)
            od = PT[1::2].reshape(C, N, N)                                            # odd channels:  [c, p, q] = P[p, q, 2c+1]
            if out:                                       # X[c, i, j] = sum_k a[c, i, k] b[c, j, k]: bmm(a, b^T) -> storage (c, i, j)
                X = torch.bmm(ev, od.transpose(1, 2))
            else:                                         # the module's orientation: X'[c, j, i] = sum_k a[c, k, j] b[c, k, i] = bmm(a^T, b) -> storage (c, j, i), viewed [c, i, j]
                X = torch.bmm(ev.transpose(1, 2), od).permute(0, 2, 1)
            del PT, ev, od
            Xc = X.permute(1, 2, 0).contiguous() if config == "hm_copy" else channels_last(X)   # hm_view / hm_fused: the tiled transpose
        del X
        y = _ln(Xc, w["ln_out_w"], w["ln_out_b"], eps, cache, "ln_out")               # [N, N, C]
        del Xc
        gdt2 = _gemm_dtype(y)
        u = F.linear(y if y.dtype == gdt2 else y.to(gdt2), _cached(cache, "wo_" + str(gdt2), lambda: w["w_o"].detach().to(gdt2).contiguous(), w["w_o"]))
        del y
        Gout = F.linear(xg, _cached(cache, "wog_" + str(gdt), lambda: w["w_og"].detach().to(gdt).contiguous(), w["w_og"]))
        if config == "hm_fused" and u.dtype == Gout.dtype and u.dtype in (torch.bfloat16, torch.float16):
            gate_(u, Gout)                                                            # u *= sigmoid(Gout): one pass, ATen's roundings
        else:
            u.mul_(torch.sigmoid(Gout))
    if residual:
        return z + u                                                                  # the caller's `pair += u` (z's dtype)
    return u


def forward(z, mask, direction, w, eps=1e-5, residual=False, cache=None, config=None):
    """z [N, N, c] | [B, N, N, c] (pre-LayerNorm pair tensor), mask None | [N, N] | [B, N, N], direction 'outgoing' | 'incoming', w = the ten
    canonical tensors.  Returns the update in the GEMM dtype (z + update when residual), shaped like z."""
    import torch
    why = supported(z, mask, w)
    if why:
        raise ValueError("af3t_form: " + why)
    cache = cache if cache is not None else {}
    config = config or DEFAULT_CONFIG
    if config not in CONFIGS:
        raise ValueError("af3t_form: config:%s(not %s)" % (config, "|".join(CONFIGS)))
    out = str(direction).lower().startswith("out")
    if z.dim() == 3:
        return _one(z, mask, out, w, eps, residual, cache, config)
    outs = [_one(z[i], None if mask is None else mask[i], out, w, eps, residual, cache, config) for i in range(int(z.shape[0]))]
    return torch.stack(outs, 0)
