"""ef2_hoist — exact-class removals of per-recycle / per-fold data-movement work in the ESMFold2 forward.

    import ef2_hoist
    ef2_hoist.install(model, trimul=True, disto=True)      # after ef2_server.configure()/ef2_opt.install(); instance patches only; strict by default
    ef2_hoist.uninstall(model); ef2_hoist.stats(); ef2_hoist.describe()

Each lever is its own flag, inference-only (grad enabled -> the stock statements run), and BITWISE identical to stock under the deterministic
recipe by construction: every floating-point operation is stock's, in stock's order, on identical operand bits; only copies, layouts and the
host<->device schedule change.  Nothing under stock/ is edited: the levers patch the loaded model instance.  A lever engages or install() raises
(strict) / reports (strict=False) by name — no silent fallback at install; the per-call fall-through conditions below are shape/dtype guards that
select the verbatim stock statements for cases the lever does not cover (batch > 1, chunked mode, CPU, grad), counted in stats().

 H3 trimul  MSA-module TriangleMultiplicativeBlock, reference path (Full model; the exact tier runs it — in the fast tier ef2_opt's M1 replaces the
            MSA block's trimul with the fused Triton kernel and H3 refuses by name).  Stock, under the model's bf16 autocast:
                left, right = routed.float().chunk(2, dim=-1)                       # bf16 -> fp32 (4 GB written at 1400 tokens)
                contracted = torch.einsum('bikd,bjkd->bijd' | 'bkid,bkjd->bijd', left, right)
                mixed = self.proj_emit(self.norm_mix(contracted))
            autocast casts left/right straight back to bf16 (bf16 -> fp32 -> bf16 is the identity), einsum's sumproduct_pair makes two transposing
            `.reshape` copies to feed cuBLAS bmm([d,i,k] @ [d,k,j]) and returns the [d,i,j] product as a permuted view; norm_mix's autocast cast
            `.to(float32)` preserves those strides and F.layer_norm's internal `.contiguous()` then transposes it in fp32.  Measured at 1400 tokens
            (H100): 42 ms in the einsum region + 7-10 ms of fp32 transpose inside norm_mix per call; the bmm itself is 3.1 ms (456 TFLOP/s).
            H3 builds the SAME two contiguous bf16 operands with a tiled Triton transpose read directly from `routed` (identical bits, identical
            layout), calls torch.bmm on them (the identical cuBLAS problem einsum issues: same m, n, k, batch, dtypes, strides and transposition flags for both flows), and
            writes the [d,i,j] product to a contiguous fp32 [b,i,j,d] tensor in one transpose+widen pass (bf16 -> fp32 is exact; it is the tensor
            norm_mix's cast + contiguous() would have built), so the LayerNorm runs on it directly.  8 calls per recycle x 21 recycles = 168/fold.
            Per-call fall-through to the verbatim stock statements: batch > 1, chunked mode (_chunk_size set), grad enabled, CPU / non-bf16 routed,
            cuEquivariance backend (_use_kernels).
 H2 disto   distogram_logits [B,L,L,64] fp32 (0.5 GB at 1400 tokens, 2.1 GB at 2900) are copied device -> pinned host on a side stream right after
            distogram_head, overlapped with the diffusion sampler that follows; a model forward hook waits on that copy's event (complete long before)
            and swaps a pageable host clone (14 ms per 0.5 GB) into the output dict before it leaves forward(), so processor.decode()'s `.cpu()` finds
            a host tensor: the 139 ms pageable D2H (1400 tokens) inside the timed window disappears.  Values untouched (a copy is a copy).  One
            grow-only pinned staging buffer per process; the caller never aliases it.  Refuses by name when ef2_xl's X10 hook already owns the move.

 Withdrawn after measurement (kept here as a note so nobody re-derives it): a fused recycle-injection lever ("H1 inject", z = a*z + F.linear(...) in
 one pass carrying z in bf16) is vacuous — forward() casts a and b_mat to z.dtype = bf16 before the loop, so stock already carries z in bf16 between
 recycles and FoldingTrunk performs no entry/exit casts; the remaining injection glue is ~2 ms per recycle at 1400 tokens.
"""
import types, collections
import torch

VERSION = "hoist.0.6"
STATS = collections.Counter()
_STATE = {"levers": {}}
LEVERS = ("trimul", "glue", "disto")


# =====================================================================================================================
# H3: MSA-module reference TriMul — contraction data movement (tiled transposes + the identical bmm), exact
# =====================================================================================================================
_TP = {"fn": None, "err": None}


def _build_tp():
    if _TP["fn"] is not None or _TP["err"] is not None:
        return _TP["fn"]
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _tp_kernel(S, D, R, Cn, s_p, s_r, s_c, d_p, d_r, d_c, TR: tl.constexpr, TC: tl.constexpr):
            # D[p, c, r] = S[p, r, c]   (pure data movement; S read coalesced along c, D written coalesced along r; store casts to D's element
            # type: bf16 -> bf16 or bf16 -> fp32, both exact)
            pid_p = tl.program_id(0); pid_r = tl.program_id(1); pid_c = tl.program_id(2)
            rr = pid_r * TR + tl.arange(0, TR); cc = pid_c * TC + tl.arange(0, TC)
            m = (rr[:, None] < R) & (cc[None, :] < Cn)
            src = S + pid_p.to(tl.int64) * s_p + rr[:, None].to(tl.int64) * s_r + cc[None, :].to(tl.int64) * s_c
            x = tl.load(src, mask=m)
            dst = D + pid_p.to(tl.int64) * d_p + cc[None, :].to(tl.int64) * d_c + rr[:, None].to(tl.int64) * d_r
            tl.store(dst, x.to(D.dtype.element_ty), mask=m)

        def tp(S_base, P, R, Cn, s_strides, D_base, d_strides, TR=64, TC=64):
            grid = (P, triton.cdiv(R, TR), triton.cdiv(Cn, TC))
            _tp_kernel[grid](S_base, D_base, R, Cn, s_strides[0], s_strides[1], s_strides[2], d_strides[0], d_strides[1], d_strides[2], TR=TR, TC=TC, num_warps=4)
        _TP["fn"] = tp
    except Exception as e:  # noqa: BLE001
        _TP["err"] = repr(e)
    return _TP["fn"]


def contract_bmm(routed, flow, out_dtype=torch.float32):
    """Values of torch.einsum(eq[flow], *routed.float().chunk(2, -1)) under autocast(bf16) (a bf16 [d,i,j] product viewed as [b,i,j,d]), returned as
    a CONTIGUOUS [1,L,L,c] tensor of `out_dtype` (fp32 = what norm_mix's autocast cast + layer_norm's contiguous() build from stock's view;
    bf16 -> fp32 widening is exact).  routed: [1, L, L, 2c] bf16 contiguous."""
    tp = _TP["fn"]
    B, L, L2, C2 = routed.shape; c = C2 // 2
    A = torch.empty((c, L, L), dtype=routed.dtype, device=routed.device)      # [d, i, k]  (== einsum's reshape copy of the permuted left operand)
    Bm = torch.empty((c, L, L), dtype=routed.dtype, device=routed.device)     # [d, k, j]  (== einsum's reshape copy of the permuted right operand)
    sL, sK = L * C2, C2
    if flow == "outgoing":      # 'bikd,bjkd->bijd':  A[d,i,k] = routed[i,k,d]      B[d,k,j] = routed[j,k,c+d]
        tp(routed, L, L, c, (sL, sK, 1), A, (L, 1, L * L))
        tp(routed[..., c:], L, L, c, (sK, sL, 1), Bm, (L, 1, L * L))
    else:                       # 'bkid,bkjd->bijd':  A[d,i,k] = routed[k,i,d]      B[d,k,j] = routed[k,j,c+d]
        tp(routed, L, L, c, (sK, sL, 1), A, (L, 1, L * L))
        tp(routed[..., c:], L, L, c, (sL, sK, 1), Bm, (L, 1, L * L))
    Cm = torch.bmm(A, Bm)                                                     # [d, i, j]: the cuBLAS problem einsum's sumproduct_pair issues
    del A, Bm
    out = torch.empty((B, L, L, c), dtype=out_dtype, device=routed.device)
    tp(Cm, L, c, L, (L, L * L, 1), out, (L * c, 1, c))                        # out[i,j,d] = Cm[d,i,j]  (widened to fp32 in the same pass)
    return out


def _trimul_forward_h3(self, pair_grid, visibility=None):
    """TriangleMultiplicativeBlock.forward (modeling_esmfold2_common.py) verbatim, except the contraction segment (H3) when eligible."""
    if self._use_kernels or torch.is_grad_enabled():
        return self._ef2hoist_eager_forward(pair_grid, visibility)
    if visibility is None:
        visibility = pair_grid.new_ones(pair_grid.shape[:-1])
    normalized_grid = self.norm_start(pair_grid)
    bundled = self.proj_bundle(normalized_grid)
    signal, gate_logits = bundled.split(2 * self.latent_channels, dim=-1)
    routed = signal * torch.sigmoid(gate_logits)
    routed = routed * visibility.unsqueeze(-1)
    if (self._chunk_size is None and routed.is_cuda and routed.dtype == torch.bfloat16 and routed.dim() == 4 and routed.shape[0] == 1
            and routed.shape[1] == routed.shape[2] and routed.shape[3] == 2 * self.latent_channels and routed.is_contiguous()
            and self.flow in ("outgoing", "incoming") and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16):
        STATS["trimul_calls"] += 1
        contracted = contract_bmm(routed, self.flow, out_dtype=torch.float32)   # H3 (norm_mix under autocast computes in fp32 on exactly these values)
    else:
        STATS["trimul_fallthrough"] += 1
        left_stream, right_stream = routed.float().chunk(2, dim=-1)
        if self._chunk_size is not None:
            contracted = self._triangular_contract_chunked(left_stream, right_stream, self._chunk_size)
        else:
            contracted = self._triangular_contract(left_stream, right_stream)
    mixed = self.proj_emit(self.norm_mix(contracted))
    output_gate = torch.sigmoid(self.proj_gate(normalized_grid))
    return mixed * output_gate


def _msa_trimul_engines(model):
    enc = getattr(model, "msa_encoder", None)
    if enc is None:
        return []
    out = []
    for blk in enc.blocks:
        for nm in ("tri_mul_out", "tri_mul_in"):
            eng = getattr(getattr(blk, nm, None), "_engine", None)
            if eng is not None:
                out.append((blk, eng))
    return out


def install_trimul(model):
    import transformers.models.esmfold2.modeling_esmfold2_common as C
    engines = _msa_trimul_engines(model)
    if not engines:
        return False, "model has no msa_encoder (Fast variant): nothing to apply"
    if any(getattr(blk, "_ef2opt_fused", False) for blk, _ in engines):
        return False, "ef2_opt M1 (fused MSA trimul) owns the MSA blocks in this mode"
    if _build_tp() is None:
        return False, f"triton transpose kernel unavailable: {_TP['err']}"
    n = 0
    for blk, eng in engines:
        if getattr(eng, "_ef2hoist_trimul", False):
            n += 1
            continue
        if type(eng).forward is not C.TriangleMultiplicativeBlock.forward or "forward" in eng.__dict__:
            return False, f"TriangleMultiplicativeBlock.forward is not upstream's on {type(eng).__name__} (another lever owns it)"
        eng._ef2hoist_eager_forward = eng.forward
        eng.forward = types.MethodType(_trimul_forward_h3, eng)
        eng._ef2hoist_trimul = True
        n += 1
    return True, f"ok ({n} engines)"


def uninstall_trimul(model):
    for blk, eng in _msa_trimul_engines(model):
        if getattr(eng, "_ef2hoist_trimul", False):
            del eng.forward
            eng._ef2hoist_trimul = False


# =====================================================================================================================
# H5: TriMul elementwise glue fused into H3's transposes (exact; the sigmoid is checked exhaustively against torch's at install)
# =====================================================================================================================
_GL = {"fn": None, "sig": None, "err": None, "ok": None, "checked": 0}


def _build_glue():
    if _GL["fn"] is not None or _GL["err"] is not None:
        return _GL["fn"]
    try:
        import triton
        import triton.language as tl
        try:
            from triton.language.extra import libdevice as _ld
        except Exception:  # noqa: BLE001
            from triton.language.extra.cuda import libdevice as _ld  # older layout

        @triton.jit
        def _sigmoid_f32(g):
            # ATen sigmoid for reduced-precision inputs: opmath float, one / (one + std::exp(-x)) -> __nv_expf, add.rn, div.rn
            e = _ld.exp(-g)
            den = e + 1.0
            one = tl.full(den.shape, 1.0, tl.float32)
            return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=f,f,f", [one, den], dtype=tl.float32, is_pure=True, pack=1)

        @triton.jit
        def _sig_all_kernel(X, OUT, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = offs < n
            g = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
            tl.store(OUT + offs, _sigmoid_f32(g).to(tl.bfloat16), mask=m)

        @triton.jit
        def _tp_glue_kernel(SIG, GATE, VIS, D, R, Cn, s_p, s_r, v_p, v_r, d_p, d_r, d_c, TR: tl.constexpr, TC: tl.constexpr):
            # D[p, c, r] = bf16( float(bf16(float(SIG[p,r,c]) * float(bf16(sigmoid(float(GATE[p,r,c])))))) * float(VIS[p,r]) )
            pid_p = tl.program_id(0); pid_r = tl.program_id(1); pid_c = tl.program_id(2)
            rr = pid_r * TR + tl.arange(0, TR); cc = pid_c * TC + tl.arange(0, TC)
            m = (rr[:, None] < R) & (cc[None, :] < Cn)
            off = pid_p.to(tl.int64) * s_p + rr[:, None].to(tl.int64) * s_r + cc[None, :].to(tl.int64)
            sig = tl.load(SIG + off, mask=m, other=0.0).to(tl.float32)
            gate = tl.load(GATE + off, mask=m, other=0.0).to(tl.float32)
            s = _sigmoid_f32(gate).to(tl.bfloat16).to(tl.float32)                 # torch.sigmoid(gate_logits): bf16 result
            r = (sig * s).to(tl.bfloat16).to(tl.float32)                          # signal * sig: exact fp32 product of two bf16 values, RNE once
            vis = (tl.load(VIS + pid_p.to(tl.int64) * v_p + rr.to(tl.int64) * v_r, mask=rr < R, other=0) != 0).to(tl.float32)   # gate, never scale: a malformed bool byte (>1) still reads as 1; identical for real 0/1 masks
            r = (r * vis[:, None]).to(tl.bfloat16)                                  # routed * visibility (0/1): exact
            dst = D + pid_p.to(tl.int64) * d_p + cc[None, :].to(tl.int64) * d_c + rr[:, None].to(tl.int64) * d_r
            tl.store(dst, r, mask=m)

        @triton.jit
        def _gate_out_kernel(MIXED, PG, OUT, n, BLOCK: tl.constexpr):
            # OUT = bf16( float(MIXED) * float(bf16(sigmoid(float(PG)))) )       (== mixed * torch.sigmoid(proj_gate(.)))
            offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            m = offs < n
            mixed = tl.load(MIXED + offs, mask=m, other=0.0).to(tl.float32)
            pg = tl.load(PG + offs, mask=m, other=0.0).to(tl.float32)
            s = _sigmoid_f32(pg).to(tl.bfloat16).to(tl.float32)
            tl.store(OUT + offs, (mixed * s).to(tl.bfloat16), mask=m)

        def sig_all(x):
            out = torch.empty_like(x)
            n = x.numel(); _sig_all_kernel[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024, num_warps=4)
            return out

        def tp_glue(sig_base, gate_base, vis, vis_strides, P, R, Cn, s_strides, D_base, d_strides, TR=64, TC=64):
            grid = (P, triton.cdiv(R, TR), triton.cdiv(Cn, TC))
            _tp_glue_kernel[grid](sig_base, gate_base, vis, D_base, R, Cn, s_strides[0], s_strides[1], vis_strides[0], vis_strides[1],
                                  d_strides[0], d_strides[1], d_strides[2], TR=TR, TC=TC, num_warps=4)

        def gate_out(mixed, pg):
            assert mixed.shape == pg.shape and mixed.is_contiguous() and pg.is_contiguous()
            out = torch.empty_like(mixed)
            n = mixed.numel(); _gate_out_kernel[(triton.cdiv(n, 2048),)](mixed, pg, out, n, BLOCK=2048, num_warps=8)
            return out

        _GL["fn"] = (tp_glue, gate_out); _GL["sig"] = sig_all
    except Exception as e:  # noqa: BLE001
        _GL["err"] = repr(e)
    return _GL["fn"]


def check_sigmoid_exhaustive(device="cuda"):
    """Evaluate the fused sigmoid on every bf16 bit pattern and compare with torch.sigmoid (bf16 in -> bf16 out) on this device.
    Returns (ok, n_mismatch_nonnan, n_checked)."""
    if _build_glue() is None:
        return False, -1, 0
    bits = torch.arange(-32768, 32768, dtype=torch.int32, device=device).to(torch.int16)
    x = bits.view(torch.bfloat16)
    ref = torch.sigmoid(x)
    mine = _GL["sig"](x)
    nonnan = ~torch.isnan(x)
    mism = (ref.view(torch.int16) != mine.view(torch.int16)) & nonnan
    n_bad = int(mism.sum().item())
    nan_ok = bool(torch.isnan(mine[~nonnan]).all().item()) if (~nonnan).any() else True
    ok = (n_bad == 0) and nan_ok
    _GL["ok"] = ok
    _GL["checked"] = int(nonnan.sum().item())
    STATS["glue_sigmoid_exhaustive_mismatches"] = n_bad
    return ok, n_bad, _GL["checked"]


def contract_bmm_glue(bundled, visibility, flow, c):
    """H3's contraction with the routed = signal*sigmoid(gate)*mask glue folded into the operand builds.
    bundled: [1,L,L,4c] bf16 contiguous (proj_bundle output: signal = [..., :2c], gate = [..., 2c:]); visibility: [1,L,L] bool/0-1."""
    tp = _TP["fn"]; tp_glue, _ = _GL["fn"]
    B, L, L2, C4 = bundled.shape
    vis = visibility
    if vis.dtype == torch.bool:
        vis = vis.view(torch.uint8)                                            # reinterpret, no copy
    vis = vis.reshape(L, L)
    if not vis.is_contiguous():
        vis = vis.contiguous()
    A = torch.empty((c, L, L), dtype=bundled.dtype, device=bundled.device)     # [d, i, k]
    Bm = torch.empty((c, L, L), dtype=bundled.dtype, device=bundled.device)    # [d, k, j]
    sX, sY = L * C4, C4                                                        # bundled strides for (x, y); channel stride 1
    sigA, gateA = bundled, bundled[..., 2 * c:]                                # routed channels [0, c):   signal[..., 0:c],  gate[..., 0:c]
    sigB, gateB = bundled[..., c:], bundled[..., 3 * c:]                       # routed channels [c, 2c):  signal[..., c:2c], gate[..., c:2c]
    if flow == "outgoing":      # A[d,i,k] = routed[i,k,d]  (x=p=i, y=r=k);   B[d,k,j] = routed[j,k,c+d]  (x=r=j, y=p=k)
        tp_glue(sigA, gateA, vis, (L, 1), L, L, c, (sX, sY), A, (L, 1, L * L))
        tp_glue(sigB, gateB, vis, (1, L), L, L, c, (sY, sX), Bm, (L, 1, L * L))
    else:                       # A[d,i,k] = routed[k,i,d]  (x=r=k, y=p=i);   B[d,k,j] = routed[k,j,c+d]  (x=p=k, y=r=j)
        tp_glue(sigA, gateA, vis, (1, L), L, L, c, (sY, sX), A, (L, 1, L * L))
        tp_glue(sigB, gateB, vis, (L, 1), L, L, c, (sX, sY), Bm, (L, 1, L * L))
    Cm = torch.bmm(A, Bm)
    del A, Bm
    out = torch.empty((B, L, L, c), dtype=torch.float32, device=bundled.device)
    tp(Cm, L, c, L, (L, L * L, 1), out, (L * c, 1, c))
    return out


def _trimul_forward_h5(self, pair_grid, visibility=None):
    """TriangleMultiplicativeBlock.forward with H3 + H5; verbatim stock statements whenever a guard fails."""
    if self._use_kernels or torch.is_grad_enabled():
        return self._ef2hoist_eager_forward(pair_grid, visibility)
    if visibility is None:
        visibility = pair_grid.new_ones(pair_grid.shape[:-1])
    normalized_grid = self.norm_start(pair_grid)
    c = self.latent_channels
    ok = (self._chunk_size is None and normalized_grid.is_cuda and normalized_grid.dim() == 4 and normalized_grid.shape[0] == 1
          and normalized_grid.shape[1] == normalized_grid.shape[2] and self.flow in ("outgoing", "incoming")
          and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
          and normalized_grid.dtype == torch.float32 and tuple(visibility.shape) == tuple(normalized_grid.shape[:-1])
          and visibility.dtype in (torch.bool, torch.uint8) and self.proj_bundle.out_features == 4 * c and self.input_channels == self.proj_gate.out_features)
    if not ok:
        STATS["glue_fallthrough"] += 1
        return _trimul_forward_h3(self, pair_grid, visibility)
    STATS["glue_calls"] += 1
    ng16 = normalized_grid.to(torch.bfloat16)                                   # (a) == autocast's cast inside each Linear; done once
    bundled = self.proj_bundle(ng16)                                           # bf16 [1,L,L,4c]
    contracted = contract_bmm_glue(bundled, visibility, self.flow, c)          # (b) + H3
    mixed = self.proj_emit(self.norm_mix(contracted))
    pg = self.proj_gate(ng16)
    if mixed.dtype != torch.bfloat16 or pg.dtype != torch.bfloat16 or not mixed.is_contiguous() or not pg.is_contiguous():
        return mixed * torch.sigmoid(pg)
    return _GL["fn"][1](mixed, pg)                                             # (c)


def install_glue(model):
    engines = _msa_trimul_engines(model)
    if not engines:
        return False, "model has no msa_encoder (Fast variant): nothing to apply"
    if not all(getattr(eng, "_ef2hoist_trimul", False) for _, eng in engines):
        return False, "needs lever trimul (H3) installed first"
    if _build_glue() is None:
        return False, f"triton glue kernels unavailable: {_GL['err']}"
    dev = next(model.parameters()).device
    if dev.type != "cuda":
        return False, "model not on CUDA"
    ok, n_bad, n_chk = check_sigmoid_exhaustive(dev)
    if not ok:
        return False, f"exhaustive sigmoid check FAILED on this device/stack: {n_bad} of {n_chk} bf16 inputs differ from torch.sigmoid (would be tolerance-tier)"
    for _, eng in engines:
        eng.forward = types.MethodType(_trimul_forward_h5, eng)
        eng._ef2hoist_glue = True
    return True, f"ok ({len(engines)} engines; sigmoid checked on {n_chk} bf16 inputs, 0 mismatches)"


def uninstall_glue(model):
    for _, eng in _msa_trimul_engines(model):
        if getattr(eng, "_ef2hoist_glue", False):
            eng.forward = types.MethodType(_trimul_forward_h3, eng)
            eng._ef2hoist_glue = False


# =====================================================================================================================
# H2: distogram logits -> pinned host, asynchronously, overlapped with the sampler
# =====================================================================================================================
_D2H = {"stream": None, "staging": None, "pending": []}


def _staging(nbytes, like):
    st = _D2H["staging"]
    if st is None or st.numel() < nbytes:
        _D2H["staging"] = st = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)   # grow-only, once per process and size class
        STATS["disto_staging_allocs"] += 1
    return st[:nbytes].view(like.dtype).view(like.shape)


class _Pending:
    __slots__ = ("host", "event", "dev")

    def __init__(self, host, event, dev):
        self.host, self.event, self.dev = host, event, dev


def _disto_hook(mod, args, out):
    if not (torch.is_tensor(out) and out.is_cuda and out.dim() == 4) or torch.is_grad_enabled() or torch.cuda.is_current_stream_capturing():
        return out
    _D2H["pending"].clear()                                                           # a previous forward that raised before its end hook: dropped
    if _D2H["stream"] is None:
        _D2H["stream"] = torch.cuda.Stream(device=out.device)
    side = _D2H["stream"]
    host = _staging(out.numel() * out.element_size(), out)
    side.wait_stream(torch.cuda.current_stream(out.device))                          # the head's GEMM is ordered before the copy
    with torch.cuda.stream(side):
        host.copy_(out, non_blocking=True)
        done = torch.cuda.Event(); done.record(side)
    out.record_stream(side)                                                           # the device block stays reserved until the side-stream copy has run
    _D2H["pending"].append(_Pending(host, done, out))
    STATS["disto_async_copies"] += 1; STATS["disto_bytes"] += out.numel() * out.element_size()
    return out                                                                        # forward() keeps the device tensor; the end hook swaps the host copy in


def _forward_end_hook(mod, args, output):
    pend, _D2H["pending"] = _D2H["pending"], []
    if not pend or not isinstance(output, dict):
        return output
    for p in pend:
        p.event.synchronize()                                                         # waits for the copy only (issued before the sampler ran)
        STATS["disto_waits"] += 1
        for k, v in list(output.items()):
            if v is p.dev:
                output[k] = p.host.clone()                                             # pageable host tensor; decode()'s .cpu() is then a no-op
                STATS["disto_swapped"] += 1
    return output


def install_disto(model):
    if getattr(model, "_ef2hoist_disto_hooks", None):
        return True, "already"
    if getattr(model, "_ef2xl_disto_hook", None) is not None:                       # X10 (big) moves it synchronously: one owner
        return False, "ef2_xl X10 distocpu hook present"
    if not hasattr(model, "distogram_head"):
        return False, "model has no distogram_head"
    h1 = model.distogram_head.register_forward_hook(_disto_hook)
    h2 = model.register_forward_hook(_forward_end_hook)
    model._ef2hoist_disto_hooks = (h1, h2)
    return True, "ok"


def uninstall_disto(model):
    hs = getattr(model, "_ef2hoist_disto_hooks", None)
    if hs:
        for h in hs:
            h.remove()
        model._ef2hoist_disto_hooks = None



# =====================================================================================================================
_INSTALL = {"trimul": install_trimul, "glue": install_glue, "disto": install_disto}
_UNINSTALL = {"trimul": uninstall_trimul, "glue": uninstall_glue, "disto": uninstall_disto}


def install(model, *, trimul=False, glue=False, disto=False, strict=True):
    """Apply the flagged levers to the model instance; returns {lever: (applied, reason)}.  strict: a flagged lever that cannot engage raises
    RuntimeError naming it (the mode refuses by name); strict=False records (False, reason) instead."""
    want = {"trimul": trimul, "glue": glue, "disto": disto}
    import ef2_srcguard                                                                # trimul / glue re-issue TriangleMultiplicativeBlock.forward: refuse by name on another upstream source
    ef2_srcguard.check_many(want)
    out = {}
    for name in LEVERS:
        if not want[name]:
            continue
        ok, why = _INSTALL[name](model)
        out[name] = (ok, why)
        _STATE["levers"][name] = ok
        if strict and not ok:
            raise RuntimeError(f"ef2_hoist: lever {name} NOT applied: {why}")
    return out


def install_named(model, levers, strict=True):
    """install() by a comma-separated lever list: install_named(model, 'trimul,disto')."""
    names = [x.strip() for x in (levers.split(",") if isinstance(levers, str) else levers) if x.strip()]
    bad = [n for n in names if n not in LEVERS]
    if bad:
        raise ValueError(f"ef2_hoist: unknown lever(s) {bad}; known: {LEVERS}")
    return install(model, strict=strict, **{n: True for n in names})


def uninstall(model, levers=None):
    for n in (levers.split(",") if isinstance(levers, str) else (levers or list(_UNINSTALL))):
        _UNINSTALL[n.strip()](model)
        _STATE["levers"][n.strip()] = False


COUNTERS = ("trimul_calls", "trimul_fallthrough", "glue_calls", "glue_fallthrough", "glue_sigmoid_exhaustive_mismatches",
            "disto_async_copies", "disto_swapped", "disto_waits", "disto_staging_allocs", "disto_bytes")


def stats():
    """Counters for the LEVER / EXIT lines: every name in COUNTERS is present (0 when the path never ran), plus the install-time exhaustive
    sigmoid check of H5 (glue_sigmoid_check: True / False / None = not run; glue_sigmoid_checked: number of non-NaN bf16 inputs)."""
    d = {k: int(STATS.get(k, 0)) for k in COUNTERS}
    d.update({k: v for k, v in STATS.items() if k not in d})
    d["glue_sigmoid_check"] = _GL["ok"]
    d["glue_sigmoid_checked"] = _GL.get("checked", 0)
    d["levers"] = dict(_STATE["levers"])
    return d


def sigmoid_check_word():
    """`ok:<n>` / `mismatch:<k>of<n>` / `not_run` — the H5 install-time sigmoid check as one blank-free token for the glue LEVER line."""
    if _GL["ok"] is None:
        return "not_run"
    n = _GL.get("checked", 0)
    return f"ok:{n}" if _GL["ok"] else f"mismatch:{int(STATS.get('glue_sigmoid_exhaustive_mismatches', 0))}of{n}"


def describe():
    return f"ef2_hoist {VERSION}: " + ", ".join(f"{k}={'on' if v else 'off'}" for k, v in _STATE["levers"].items())
