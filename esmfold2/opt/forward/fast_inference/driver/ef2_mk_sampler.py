"""ef2_mk_sampler — per-fold hoisting + fused per-step execution of the ESMFold2 diffusion token transformer.

    import ef2_mk_sampler as MK
    MK.enable(model, gemm="cublas", ew="triton", attn="sdpa")   # BEFORE ef2_opt.install(...); patches the model INSTANCE only
    MK.disable(model)

Levers (execution only; model math / inputs / RNG draws / step count unchanged; fp32 storage + fp32 accumulation kept everywhere stock is fp32):
  A  per-fold hoist of step-invariant conditioning.  s = DiffusionConditioning-s-path(s_inputs, t_hat) depends only on t_hat, which takes one
     known value per step of a fold -> at the fold's eager step 0 the s rows of ALL steps are computed in one batched pass with the stock modules
     ([S*L, 768], S = number of steps), plus s_to_token(s_step_norm(s)).  Per block and step, the six s-only linears (AdaLN s_gate/s_shift and the
     output gates of the attention and transition blocks) are either (mode "table") precomputed for all steps into static fp32 tables when they
     fit the memory budget, or (mode "grouped") computed per step by ONE grouped cuBLAS GEMM [L,768] x [768, 6*768] per block instead of six.
     Graph-replay safe: the step index is computed ON DEVICE from the live t_hat buffer (argmin |t_hats - t_hat|), no host sync.
  B  per block pair: cuBLAS fp32 GEMMs on concatenated weights (q|k|v|g: N=3072; lin_swish N=3072) + fused Triton elementwise kernels
     (LN+modulate prologue, bias+sigmoid(g) split, sigmoid-gate*out+residual, SiLU(a)*b) + SDPA (fp32, same cutlass kernel family as stock) or a
     Triton fp32 flash kernel.  ~11 launches per block pair instead of ~41.
Tier: reordered fp32 reductions (different GEMM shapes => cuBLAS picks different kernels/split), measured at op level vs fp64.
Pair bias: re-uses the kit's per-fold cached bias (ef2_opt lever 'pb', blk._ef2opt_pb) when present, else computes it once per fold with the
block's stock code into its own static buffer.  B>1 / s_trunk / grad / non-CUDA -> stock forward (counted fallback).
Credit: follows ef2_opt's conventions — instance patching, static per-fold buffers refreshed in place, generation hook.
"""
import os, time, types, collections
import torch
import torch.nn.functional as F

STATS = collections.Counter()
VERSION = "mk1.0"
_CFG = dict(gemm="cublas", ew="triton", attn="sdpa", hoist="auto", cond="stock", max_table_bytes=int(float(os.environ.get("EF2_MK_MAX_TABLE_GB", "2.0")) * 2**30), probe=False)
_ENABLED = []
_EPOCH = {"n": 0}
_RETIRED = []

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

# =====================================================================================================================
# Triton elementwise / attention kernels (fp32, deterministic)
# =====================================================================================================================
if _HAS_TRITON:

    @triton.jit
    def _ln_mod_kernel(X, CS, CH, OUT, L_rows, D: tl.constexpr, BD: tl.constexpr, eps, stride_x, stride_cs, stride_ch, stride_o):
        """OUT[m,:] = LN(X[m,:]) (no affine) * CS[m % L_rows, :] + CH[m % L_rows, :]   (one program per row; two-pass mean/var in fp32 like torch)"""
        m = tl.program_id(0)
        r = m % L_rows
        offs = tl.arange(0, BD)
        msk = offs < D
        x = tl.load(X + m * stride_x + offs, mask=msk, other=0.0)
        mean = tl.sum(x, axis=0) / D
        xc = tl.where(msk, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        cs = tl.load(CS + r * stride_cs + offs, mask=msk, other=0.0)
        ch = tl.load(CH + r * stride_ch + offs, mask=msk, other=0.0)
        tl.store(OUT + m * stride_o + offs, xc * rstd * cs + ch, mask=msk)

    @triton.jit
    def _gate_res_kernel(X, Y, G, OUT, L_rows, D: tl.constexpr, BD: tl.constexpr, stride_x, stride_y, stride_g, stride_o, SIG_G: tl.constexpr):
        """OUT[m,:] = X[m,:] + gate(G[m % L_rows,:]) * Y[m,:]   gate = sigmoid if SIG_G else identity (table already sigmoided)"""
        m = tl.program_id(0)
        r = m % L_rows
        offs = tl.arange(0, BD)
        msk = offs < D
        x = tl.load(X + m * stride_x + offs, mask=msk, other=0.0)
        y = tl.load(Y + m * stride_y + offs, mask=msk, other=0.0)
        g = tl.load(G + r * stride_g + offs, mask=msk, other=0.0)
        if SIG_G:
            g = tl.sigmoid(g)
        tl.store(OUT + m * stride_o + offs, x + g * y, mask=msk)

    @triton.jit
    def _swiglu_kernel(SW, OUT, HID: tl.constexpr, BH: tl.constexpr, stride_sw, stride_o):
        """OUT[m,:] = silu(SW[m, :HID]) * SW[m, HID:2HID]"""
        m = tl.program_id(0)
        offs = tl.arange(0, BH)
        msk = offs < HID
        a = tl.load(SW + m * stride_sw + offs, mask=msk, other=0.0)
        b = tl.load(SW + m * stride_sw + HID + offs, mask=msk, other=0.0)
        tl.store(OUT + m * stride_o + offs, a * tl.sigmoid(a) * b, mask=msk)

    @triton.jit
    def _qbias_sig_kernel(X, BQ, D: tl.constexpr, BD: tl.constexpr, stride_x, SIG_G: tl.constexpr):
        """in place on row m of [M, 4D]: X[m, 0:D] += BQ ; if SIG_G: X[m, 3D:4D] = sigmoid(X[m, 3D:4D])"""
        m = tl.program_id(0)
        offs = tl.arange(0, BD)
        msk = offs < D
        q = tl.load(X + m * stride_x + offs, mask=msk, other=0.0)
        bq = tl.load(BQ + offs, mask=msk, other=0.0)
        tl.store(X + m * stride_x + offs, q + bq, mask=msk)
        if SIG_G:
            g = tl.load(X + m * stride_x + 3 * D + offs, mask=msk, other=0.0)
            tl.store(X + m * stride_x + 3 * D + offs, tl.sigmoid(g), mask=msk)

    @triton.jit
    def _sig_cols_kernel(X, N0, N1, stride_x, BN: tl.constexpr):
        """in place: X[m, N0:N1] = sigmoid(X[m, N0:N1])"""
        m = tl.program_id(0)
        offs = N0 + tl.arange(0, BN)
        msk = offs < N1
        v = tl.load(X + m * stride_x + offs, mask=msk, other=0.0)
        tl.store(X + m * stride_x + offs, tl.sigmoid(v), mask=msk)

    @triton.jit
    def _attn_fp32_kernel(QKVG, BIAS, KM, OUT, L, stride_row, stride_bb, stride_bh, stride_bq, stride_out, scale,
                          H: tl.constexpr, HD: tl.constexpr, HDP: tl.constexpr, D: tl.constexpr,
                          HAS_BIAS: tl.constexpr, HAS_KM: tl.constexpr, BQ: tl.constexpr, BKV: tl.constexpr):
        """flash-style fp32 attention; one program per (q-block, head, batch). QKVG row = [q | k | v | g(sigmoided)]; out cols h*HD.. = (softmax(qk*scale+bias) v) * g"""
        pid_q = tl.program_id(0).to(tl.int64); h = tl.program_id(1).to(tl.int64); b = tl.program_id(2).to(tl.int64)   # 64-bit offsets: h * stride_bh (= h*L*L on the
        offs_q = pid_q * BQ + tl.arange(0, BQ)                                                                          # [B, H, L, L] bias) passes 2^31 from L = 11966 at 16 heads
        offs_d = tl.arange(0, HDP)
        d_mask = offs_d < HD
        q_mask = offs_q < L
        base = QKVG + b * L * stride_row
        q = tl.load(base + offs_q[:, None] * stride_row + (h * HD + offs_d)[None, :], mask=q_mask[:, None] & d_mask[None, :], other=0.0)
        q = q * scale
        m_i = tl.full([BQ], -1e30, dtype=tl.float32)
        l_i = tl.zeros([BQ], dtype=tl.float32)
        acc = tl.zeros([BQ, HDP], dtype=tl.float32)
        for k0 in range(0, L, BKV):
            offs_k = k0 + tl.arange(0, BKV)
            k_valid = offs_k < L
            kt = tl.load(base + offs_k[None, :] * stride_row + (D + h * HD + offs_d)[:, None], mask=k_valid[None, :] & d_mask[:, None], other=0.0)
            s = tl.dot(q, kt, input_precision="ieee")
            if HAS_BIAS:
                bias = tl.load(BIAS + b * stride_bb + h * stride_bh + offs_q[:, None] * stride_bq + offs_k[None, :], mask=q_mask[:, None] & k_valid[None, :], other=0.0)
                s = s + bias.to(tl.float32)
            if HAS_KM:
                km = tl.load(KM + b * L + offs_k, mask=k_valid, other=0)
                k_valid = k_valid & (km != 0)
            s = tl.where(k_valid[None, :], s, -1e30)
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            p = tl.where(k_valid[None, :], p, 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(base + offs_k[:, None] * stride_row + (2 * D + h * HD + offs_d)[None, :], mask=k_valid[:, None] & d_mask[None, :], other=0.0)
            acc = acc * alpha[:, None]
            acc = tl.dot(p, v, acc, input_precision="ieee")
            m_i = m_new
        g = tl.load(base + offs_q[:, None] * stride_row + (3 * D + h * HD + offs_d)[None, :], mask=q_mask[:, None] & d_mask[None, :], other=0.0)
        o = acc / l_i[:, None] * g
        tl.store(OUT + b * L * stride_out + offs_q[:, None] * stride_out + (h * HD + offs_d)[None, :], o, mask=q_mask[:, None] & d_mask[None, :])


def _nw(D):
    return 4 if D <= 1024 else 8


def ln_mod(x, cs, ch, out, L_rows, eps):
    M, D = x.shape
    if _CFG["ew"] == "triton" and _HAS_TRITON:
        _ln_mod_kernel[(M,)](x, cs, ch, out, L_rows, D=D, BD=triton.next_power_of_2(D), eps=eps, stride_x=x.stride(0), stride_cs=cs.stride(0), stride_ch=ch.stride(0), stride_o=out.stride(0), num_warps=_nw(D))
        return out
    an = F.layer_norm(x, (D,), None, None, eps)
    if M != L_rows:
        cs = cs.repeat(M // L_rows, 1); ch = ch.repeat(M // L_rows, 1)
    torch.addcmul(ch, an, cs, out=out)
    return out


def gate_res(x, y, g, out, L_rows, sig_g):
    M, D = x.shape
    if _CFG["ew"] == "triton" and _HAS_TRITON:
        _gate_res_kernel[(M,)](x, y, g, out, L_rows, D=D, BD=triton.next_power_of_2(D), stride_x=x.stride(0), stride_y=y.stride(0), stride_g=g.stride(0), stride_o=out.stride(0), SIG_G=sig_g, num_warps=_nw(D))
        return out
    gg = torch.sigmoid(g) if sig_g else g
    if M != L_rows:
        gg = gg.repeat(M // L_rows, 1)
    torch.addcmul(x, gg, y, out=out)
    return out


def swiglu(sw, out, hid):
    M = sw.shape[0]
    if _CFG["ew"] == "triton" and _HAS_TRITON:
        _swiglu_kernel[(M,)](sw, out, HID=hid, BH=triton.next_power_of_2(hid), stride_sw=sw.stride(0), stride_o=out.stride(0), num_warps=_nw(hid))
        return out
    torch.mul(F.silu(sw[:, :hid]), sw[:, hid:], out=out)
    return out


def qbias_sig_(x, bq, D, sig_g):
    M = x.shape[0]
    if _CFG["ew"] == "triton" and _HAS_TRITON:
        _qbias_sig_kernel[(M,)](x, bq, D=D, BD=triton.next_power_of_2(D), stride_x=x.stride(0), SIG_G=sig_g, num_warps=4)
        return x
    x[:, :D] += bq
    if sig_g:
        x[:, 3 * D:4 * D].sigmoid_()
    return x


def sig_cols_(x, n0, n1):
    M = x.shape[0]
    if _CFG["ew"] == "triton" and _HAS_TRITON:
        _sig_cols_kernel[(M,)](x, n0, n1, x.stride(0), BN=triton.next_power_of_2(n1 - n0), num_warps=4)
        return x
    x[:, n0:n1].sigmoid_()
    return x


def attn_triton(qkvg, bias, key_mask, out, B, L, H, HD, D, scale):
    BQ = 64 if L >= 256 else 32
    grid = (triton.cdiv(L, BQ), H, B)
    _attn_fp32_kernel[grid](qkvg, bias if bias is not None else qkvg, key_mask if key_mask is not None else qkvg, out,
                            L, qkvg.stride(0), (bias.stride(0) if bias is not None else 0), (bias.stride(1) if bias is not None else 0), (bias.stride(2) if bias is not None else 0), out.stride(0),
                            scale, H=H, HD=HD, HDP=64, D=D, HAS_BIAS=bias is not None, HAS_KM=key_mask is not None, BQ=BQ, BKV=64, num_warps=4, num_stages=2)
    return out


# =====================================================================================================================
# per-instance state
# =====================================================================================================================
class _State:
    def __init__(self, tt, sh):
        self.tt = tt; self.sh = sh
        self.nb = len(tt.attn_blocks)
        a0 = tt.attn_blocks[0]
        self.D = a0.d_model; self.H = a0.num_heads; self.HD = a0.head_dim; self.scale = a0.scale
        self.hid = tt.transition_blocks[0].lin_swish.out_features // 2
        self.eps = a0.adaln.eps
        self.refresh_weights()
        self.table = None; self.table_key = None; self.fold_epoch = -1; self.mode = None
        self.pb = {}; self.ws = {}
        self.fold_bad = -1
        self.side = None            # side CUDA stream for the s-only (x-independent) per-step work in mode "overlap"

    def refresh_weights(self):
        tt = self.tt; dev = tt.attn_blocks[0].q_proj.weight.device
        with torch.no_grad():
            self.WqkvgT = [torch.cat([a.q_proj.weight, a.kv_proj.weight, a.g_proj.weight], 0).t().contiguous() for a in tt.attn_blocks]     # [D, 4D] (x @ W layout)
            self.bq = [a.q_proj.bias.detach().contiguous() for a in tt.attn_blocks]
            self.WoutT = [a.out_proj.weight.t().contiguous() for a in tt.attn_blocks]        # [D, D]
            self.WswT = [t.lin_swish.weight.t().contiguous() for t in tt.transition_blocks]   # [D, 2*hid]
            self.WloT = [t.lin_out.weight.t().contiguous() for t in tt.transition_blocks]     # [hid, D]
            # grouped s-only linears per block: columns [s_gate_a | s_shift_a | out_gate_a | s_gate_t | s_shift_t | output_gate_t]; the first two act on LN_a(s), 3rd on s, ...
            # LN(s) uses a different affine (s_scale) per AdaLN => fold s_scale into the weights: Linear(LN(s)*s_scale) = LN(s) @ (W*s_scale)^T
            self.WcondT, self.bcond = [], []
            for a, tr in zip(tt.attn_blocks, tt.transition_blocks):
                Wa_g = a.adaln.s_gate.weight * a.adaln.s_scale[None, :]; Wa_s = a.adaln.s_shift.weight * a.adaln.s_scale[None, :]
                Wt_g = tr.adaln.s_gate.weight * tr.adaln.s_scale[None, :]; Wt_s = tr.adaln.s_shift.weight * tr.adaln.s_scale[None, :]
                Wg = torch.cat([Wa_g, Wt_g], 0).t().contiguous()            # biased, act on LN(s):  [D, 2D] -> cs_a | cs_t (pre-sigmoid)
                Wsh = torch.cat([Wa_s, Wt_s], 0).t().contiguous()          # unbiased, act on LN(s): [D, 2D] -> ch_a | ch_t
                Ws = torch.cat([a.out_gate.weight, tr.output_gate.weight], 0).t().contiguous()   # biased, act on s: [D, 2D] -> og_a | og_t (pre-sigmoid)
                bg = torch.cat([a.adaln.s_gate.bias, tr.adaln.s_gate.bias], 0).contiguous()
                bs = torch.cat([a.out_gate.bias, tr.output_gate.bias], 0).contiguous()
                self.WcondT.append((Wg, Wsh, Ws)); self.bcond.append((bg, bs))
        self._wkey = self.weights_key()

    def weights_key(self):
        ps = []
        for a in self.tt.attn_blocks:
            ps += [a.q_proj.weight, a.kv_proj.weight, a.g_proj.weight, a.out_proj.weight, a.adaln.s_gate.weight, a.out_gate.weight]
        for t in self.tt.transition_blocks:
            ps += [t.lin_swish.weight, t.lin_out.weight]
        return tuple((p.data_ptr(), p._version) for p in ps)


def _find(model):
    sh = model.structure_head
    return sh, sh.diffusion_module, sh.diffusion_module.token_transformer


# ---------------------------------------------------------------------------------------------------------------------
# A: per-fold schedule + conditioning table
# ---------------------------------------------------------------------------------------------------------------------
def _t_hats_for(sh, steps, max_sigma, device):
    sched = sh.inference_noise_schedule(steps, device)
    if max_sigma is not None:
        sched = sched[sched <= float(max_sigma)]
        sched = F.pad(sched, (1, 0), value=float(max_sigma))
    gam = torch.where(sched > sh.gamma_min, torch.full_like(sched, sh.gamma_0), torch.zeros_like(sched))
    sl = sched.tolist(); gl = gam.tolist()
    t_hats = [float(sl[i]) * (1.0 + float(gl[i + 1])) for i in range(len(sl) - 1)]     # same python-double arithmetic as stock / ef2_opt
    return torch.tensor(t_hats, dtype=torch.float32, device=device)


def _fold_schedule(sh, device, t0=None):
    """t_hat values of all steps of the current fold, from the (num_sampling_steps, max_inference_sigma) recorded for THIS model call by the model-level
    forward pre-hook (or by our sample() wrapper when it is outermost).  NOTE the first t_hat is the same for every step count whose schedule is clipped at
    max_sigma, so t0 alone cannot identify the schedule; correctness of rows >= 1 is CHECKED at every eager (non-captured) step (see _dm_forward_mk)."""
    cfg = getattr(sh, "_mk_fold_cfg", None) or {}
    steps = int(cfg.get("steps") or sh.inference_num_steps)
    max_sigma = cfg.get("max_sigma", 256.0)
    STATS["schedule_steps_%d" % steps] += 1
    return _t_hats_for(sh, steps, max_sigma, device), (steps, max_sigma)


def _cond_s_all_steps(dm, s_inputs, t_hats):
    """stock DiffusionConditioning s-path, called PER STEP exactly as the stock forward does (t as a (1,) fp32 tensor, same modules, same shapes =>
    same cuBLAS kernels => the same bits as the stock per-step computation), stacked into s [S, L, D]; s_tok = s_to_token(s_step_norm(s)) per step."""
    cond = dm.conditioning
    S = t_hats.numel(); L = s_inputs.shape[1]
    s_rows, tok_rows = [], []
    with torch.no_grad():
        s_in = cond.s_proj(cond.s_input_norm(s_inputs.to(dtype=torch.float32)))        # [1, L, D]  (identical every step in stock too)
        for i in range(S):
            t = t_hats[i:i + 1]
            t_noise = 0.25 * torch.log((t / cond.sigma_data).clamp(min=1e-20))
            n = cond.fourier(t_noise)
            n = cond.noise_proj(cond.noise_norm(n))
            s = s_in + n.unsqueeze(1)
            for block in cond.s_transitions:
                s = s + block(s)
            s_rows.append(s)
            tok_rows.append(dm.s_to_token(dm.s_step_norm(s)))
    return torch.cat(s_rows, 0).contiguous(), torch.cat(tok_rows, 0).contiguous()


def _cond_tables_all_steps(st, s, which=("cs", "ch", "og")):
    """per block and step, with the STOCK modules on [1, L, D] rows (same kernels as the stock per-step forward): cs = sigmoid(s_gate(LN(s; s_scale))),
    ch = s_shift(LN(s; s_scale)), og = sigmoid(out_gate(s)) for the attention (_a) and transition (_t) blocks -> [nb, S, L, D] fp32 tables."""
    S, L, D = s.shape
    keys = []
    if "cs" in which: keys += ["cs_a", "cs_t"]
    if "ch" in which: keys += ["ch_a", "ch_t"]
    if "og" in which: keys += ["og_a", "og_t"]
    outs = {k: torch.empty(st.nb, S, L, D, device=s.device, dtype=torch.float32) for k in keys}
    tt = st.tt
    with torch.no_grad():
        for b in range(st.nb):
            a = tt.attn_blocks[b]; tr = tt.transition_blocks[b]
            for i in range(S):
                si = s[i:i + 1]                                                   # [1, L, D] like the stock call
                if "cs" in which or "ch" in which:
                    sn_a = F.layer_norm(si, (D,), a.adaln.s_scale, None, a.adaln.eps)
                    sn_t = F.layer_norm(si, (D,), tr.adaln.s_scale, None, tr.adaln.eps)
                    if "cs" in which:
                        outs["cs_a"][b, i] = torch.sigmoid(a.adaln.s_gate(sn_a))[0]; outs["cs_t"][b, i] = torch.sigmoid(tr.adaln.s_gate(sn_t))[0]
                    if "ch" in which:
                        outs["ch_a"][b, i] = a.adaln.s_shift(sn_a)[0]; outs["ch_t"][b, i] = tr.adaln.s_shift(sn_t)[0]
                if "og" in which:
                    outs["og_a"][b, i] = torch.sigmoid(a.out_gate(si))[0]; outs["og_t"][b, i] = torch.sigmoid(tr.output_gate(si))[0]
    return outs


def _cond_block_step_stock(st, b, s_row, need=("cs", "ch", "og")):
    """per step and block with the stock modules (M = L rows): used when the tables do not fit. s_row: [L, D]."""
    a = st.tt.attn_blocks[b]; tr = st.tt.transition_blocks[b]; D = st.D; out = {}
    si = s_row.unsqueeze(0)
    if "cs" in need or "ch" in need:
        sn_a = F.layer_norm(si, (D,), a.adaln.s_scale, None, a.adaln.eps); sn_t = F.layer_norm(si, (D,), tr.adaln.s_scale, None, tr.adaln.eps)
        if "cs" in need:
            out["cs_a"] = torch.sigmoid(a.adaln.s_gate(sn_a))[0]; out["cs_t"] = torch.sigmoid(tr.adaln.s_gate(sn_t))[0]
        if "ch" in need:
            out["ch_a"] = a.adaln.s_shift(sn_a)[0]; out["ch_t"] = tr.adaln.s_shift(sn_t)[0]
    if "og" in need:
        out["og_a"] = torch.sigmoid(a.out_gate(si))[0]; out["og_t"] = torch.sigmoid(tr.output_gate(si))[0]
    return out


def _cond_block_step(st, b, s_row, sn_row, w, need=("cs", "ch", "og")):
    """per step and block: conditioning tensors from grouped GEMMs ([L,D]x[D,2D] each). Returns dict of views into workspaces."""
    Wg, Wsh, Ws = st.WcondT[b]; bg, bs = st.bcond[b]
    D = st.D; out = {}
    if "cs" in need:
        y = torch.addmm(bg, sn_row, Wg, out=w["yg"]); sig_cols_(y, 0, 2 * D); out["cs_a"] = y[:, :D]; out["cs_t"] = y[:, D:]
    if "ch" in need:
        y = torch.mm(sn_row, Wsh, out=w["ysh"]); out["ch_a"] = y[:, :D]; out["ch_t"] = y[:, D:]
    if "og" in need:
        y = torch.addmm(bs, s_row, Ws, out=w["ys"]); sig_cols_(y, 0, 2 * D); out["og_a"] = y[:, :D]; out["og_t"] = y[:, D:]
    return out


def _step_index(t_hats, t):
    return torch.argmin((t_hats - t.reshape(-1)[:1]).abs())


# ---------------------------------------------------------------------------------------------------------------------
# patched DiffusionModule.forward
# ---------------------------------------------------------------------------------------------------------------------
def _dm_forward_mk(self, x_noisy, t_hat, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars, ref_space_uid, tok_idx, s_inputs, s_trunk, z_trunk,
                   relative_position_encoding, asym_id, residue_index, entity_id, token_index, sym_id, sigma_data=None, token_attention_mask=None,
                   num_diffusion_samples=1, return_token_repr=False, return_atom_repr=False, inference_cache=None):
    st = self._mk_state
    eager = self._mk_eager_forward

    def _fallback(reason):
        STATS["fallback_" + reason] += 1
        return eager(x_noisy, t_hat, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars, ref_space_uid, tok_idx, s_inputs, s_trunk, z_trunk, relative_position_encoding,
                     asym_id, residue_index, entity_id, token_index, sym_id, sigma_data=sigma_data, token_attention_mask=token_attention_mask, num_diffusion_samples=num_diffusion_samples,
                     return_token_repr=return_token_repr, return_atom_repr=return_atom_repr, inference_cache=inference_cache)
    if torch.is_grad_enabled() or (not x_noisy.is_cuda):
        return _fallback("grad_or_cpu")
    if s_trunk is not None or return_atom_repr or num_diffusion_samples != 1 or s_inputs.shape[0] != 1 or x_noisy.shape[0] != 1:
        return _fallback("batch_or_args")
    if st._wkey != st.weights_key():
        st.refresh_weights(); STATS["weights_refreshed"] += 1
    sigma = self.sigma_data if sigma_data is None else float(sigma_data)
    if _CFG["hoist"] == "side" or (_CFG["hoist"] == "auto" and _CFG["gemm"] in ("stock", "split")):
        st.mode = "side"
        return _dm_forward_side(self, st, x_noisy, t_hat, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars, ref_space_uid, tok_idx, s_inputs, s_trunk, z_trunk,
                                relative_position_encoding, asym_id, residue_index, entity_id, token_index, sym_id, sigma, token_attention_mask, return_token_repr, inference_cache)
    t = torch.as_tensor(t_hat, dtype=torch.float32, device=x_noisy.device).reshape(-1)[:1]
    capturing = torch.cuda.is_current_stream_capturing()
    if (not capturing) and (inference_cache is None or "z" not in inference_cache):
        _EPOCH["n"] += 1; STATS["folds_seen"] += 1                        # step 0 of a new sample() call (fresh inference cache)
        if inference_cache is None:
            STATS["no_inference_cache"] += 1
    L = s_inputs.shape[1]
    cfg_now = getattr(self._mk_sh, "_mk_fold_cfg", None) or {}
    key = (tuple(s_inputs.shape), s_inputs.dtype, str(s_inputs.device), int(cfg_now.get("steps") or self._mk_sh.inference_num_steps), cfg_now.get("max_sigma", 256.0))
    if getattr(st, "fold_bad", -1) == _EPOCH["n"]:
        return _fallback("schedule_mismatch_fold")
    need_build = (st.table is None) or (st.table_key != key) or (st.fold_epoch != _EPOCH["n"])
    if need_build and (not capturing) and st.table is not None and st.table_key == key and _CFG.get("reuse", True):
        # same shape + schedule as the cached table: reuse it iff this fold's s_inputs is bitwise identical (seeds of one design share s_inputs)
        if st.table.get("s_inputs") is not None and st.table["s_inputs"].shape == s_inputs.shape and bool(torch.equal(st.table["s_inputs"], s_inputs)):
            st.fold_epoch = _EPOCH["n"]; need_build = False; STATS["table_reuse"] += 1
    if need_build:
        if capturing:
            raise RuntimeError("ef2_mk_sampler: per-fold table missing/stale during CUDA-graph capture (the fold's eager step 0 must run first)")
        t0 = time.perf_counter()
        t_hats, sched_key = _fold_schedule(self._mk_sh, x_noisy.device, t0=t)
        S = t_hats.numel()
        s_all, s_tok = _cond_s_all_steps(self, s_inputs, t_hats)
        mode = _CFG["hoist"]
        unit = st.nb * S * L * st.D * 4          # bytes of one [nb, S, L, D] table
        if mode == "auto":
            mode = "table" if 6 * unit <= _CFG["max_table_bytes"] else ("table_cc" if 4 * unit <= _CFG["max_table_bytes"] else "overlap")
        tab = dict(t_hats=t_hats, s=s_all, s_tok=s_tok, sn=F.layer_norm(s_all, (st.D,), None, None, st.eps), s_inputs=s_inputs.detach().clone())
        if mode == "table":
            tab.update(_cond_tables_all_steps(st, s_all, ("cs", "ch", "og")))
        elif mode == "table_cc":
            tab.update(_cond_tables_all_steps(st, s_all, ("cs", "ch")))
        same = (st.table is not None and st.table_key == key and st.mode == mode and set(st.table) == set(tab) and all(st.table[k].shape == tab[k].shape for k in tab))
        if same:
            for k in tab:
                st.table[k].copy_(tab[k])          # in place: live graphs of this shape read these addresses
            STATS["table_refresh_inplace"] += 1
        else:
            if st.table is not None:
                _RETIRED.append(st.table); STATS["table_realloc"] += 1
            st.table = tab
        st.table_key = key; st.fold_epoch = _EPOCH["n"]; st.mode = mode
        torch.cuda.synchronize(); STATS["table_builds"] += 1; STATS["table_build_ms"] += int(1000 * (time.perf_counter() - t0)); STATS["table_MB_" + mode] = int(sum(v.numel() * 4 for v in tab.values()) / 2**20)
    tab = st.table
    idx = _step_index(tab["t_hats"], t)
    i1 = idx.reshape(1)
    if not capturing:
        # exactness guard (eager steps only: step 0 of every fold + ef2_opt's pre-capture warm-up steps): the table row must belong to THIS t_hat exactly
        if float((tab["t_hats"][idx] - t[0]).abs().item()) != 0.0:
            STATS["schedule_mismatch"] += 1; st.fold_bad = _EPOCH["n"]
            return _fallback("schedule_mismatch")
        STATS["verified_eager_steps"] += 1
    # 1. conditioning z (stock cache) ; s from table
    if inference_cache is not None and "z" in inference_cache:
        z = inference_cache["z"]
    else:
        _unused, z = self.conditioning(t_hat=t, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, relative_position_encoding=relative_position_encoding, sigma_data=sigma,
                                       num_diffusion_samples=1, inference_cache=inference_cache)
        STATS["cond_z_calls"] += 1
    # 2-3. atom encoder (stock)
    tt_ = t.expand(x_noisy.shape[0])
    r_noisy = x_noisy / torch.sqrt(tt_ * tt_ + sigma * sigma)[:, None, None]
    a, q_skip, c_skip, p_skip, _ = self.atom_encoder(ref_pos=ref_pos, atom_attention_mask=ref_mask, ref_space_uid=ref_space_uid, ref_charge=ref_charge, ref_element=ref_element,
                                                     ref_atom_name_chars=ref_atom_name_chars, atom_to_token=tok_idx, r_l=r_noisy, s_i=None, num_diffusion_samples=1,
                                                     return_intermediates=False, inference_cache=inference_cache)
    # 4. a += s_to_token(s_step_norm(s))
    a = a + tab["s_tok"].index_select(0, i1)
    # 5. token transformer
    if _CFG["gemm"] == "stock":
        a = tt_forward_exact(st, a, i1, z, token_attention_mask, self.token_transformer)
    elif _CFG["gemm"] == "split":
        a = tt_forward_split(st, a, i1, z, token_attention_mask, self.token_transformer)
    else:
        a = tt_forward_mk(st, a, i1, z, token_attention_mask, self.token_transformer)
    # 6-8 (stock)
    a = self.token_norm(a)
    r_update, _ = self.atom_decoder(a_i=a, q_l=q_skip, c_l=c_skip, p_lm=p_skip, atom_to_token=tok_idx, atom_attention_mask=ref_mask, num_diffusion_samples=1, return_intermediates=False)
    sigma2 = sigma * sigma; t2 = tt_ * tt_
    out = (sigma2 / (sigma2 + t2))[:, None, None] * x_noisy + ((sigma * tt_) / torch.sqrt(sigma2 + t2))[:, None, None] * r_update
    return {"x_denoised": out, "token_repr": a if return_token_repr else None, "atom_intermediates": None}


def _dm_forward_side(self, st, x_noisy, t_hat, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars, ref_space_uid, tok_idx, s_inputs, s_trunk, z_trunk,
                     relative_position_encoding, asym_id, residue_index, entity_id, token_index, sym_id, sigma, token_attention_mask, return_token_repr, inference_cache):
    """mode 'side': the stock step, with every s-only computation (conditioning s-path, s_to_token, and per block the AdaLN gate/shift and output-gate
    Linears) issued on a SIDE stream at step entry and joined per block with events; the x-dependent chain (atom encoder -> 24 blocks -> decoder) runs
    on the current stream. Same modules, same shapes ([1, L, D]) => same kernels => bitwise the stock values. Step 0 of a fold (fresh inference cache:
    z conditioning must be computed first) runs inline on the current stream.
    Stream/event structure per step (inside ef2_opt's captured graph for steps >= 1):
        cur:  ev0 |  r_noisy, atom_encoder ........ | wait ev_s: a += s_tok | wait ev_b0: block0 | wait ev_b1: block1 | ... | block11 | token_norm, decoder, out
        side:      wait ev0 | cond-s(t), s_tok -> ev_s | tables b0 -> ev_b0 | b1 -> ev_b1 | ... | b11 -> ev_b11
    """
    bsz = x_noisy.shape[0]
    t = torch.as_tensor(t_hat, dtype=torch.float32, device=x_noisy.device).reshape(-1)
    if t.numel() == 1:
        t = t.expand(bsz)
    cur = torch.cuda.current_stream()
    fresh = inference_cache is None or "z" not in inference_cache
    capturing = torch.cuda.is_current_stream_capturing()
    if fresh and capturing:
        raise RuntimeError("ef2_mk_sampler: step 0 (fresh inference cache) must run eagerly before capture")
    evs = None; ev_s = None
    if fresh:
        if not capturing:
            _EPOCH["n"] += 1; STATS["folds_seen"] += 1
        s, z = self.conditioning(t_hat=t, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, relative_position_encoding=relative_position_encoding, sigma_data=sigma,
                                 num_diffusion_samples=1, inference_cache=inference_cache)
        s_tok = self.s_to_token(self.s_step_norm(s))
        cd_list = [{k: v.unsqueeze(0) for k, v in _cond_block_step_stock(st, b, s[0]).items()} for b in range(st.nb)]
        STATS["side_step0_inline"] += 1
    else:
        if st.side is None:
            st.side = torch.cuda.Stream(device=x_noisy.device)
        ev0 = torch.cuda.Event(); ev0.record(cur)
        with torch.cuda.stream(st.side):
            st.side.wait_event(ev0)
            s, z = self.conditioning(t_hat=t, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, relative_position_encoding=relative_position_encoding, sigma_data=sigma,
                                     num_diffusion_samples=1, inference_cache=inference_cache)          # z read from the cache; s-path computed (stock code)
            s_tok = self.s_to_token(self.s_step_norm(s))
            ev_s = torch.cuda.Event(); ev_s.record(st.side)
            cd_list, evs = [], []
            for b in range(st.nb):
                cd_list.append({k: v.unsqueeze(0) for k, v in _cond_block_step_stock(st, b, s[0]).items()})
                e = torch.cuda.Event(); e.record(st.side); evs.append(e)
        STATS["side_steps"] += 1
    denom = torch.sqrt(t * t + sigma * sigma)
    r_noisy = x_noisy / denom[:, None, None]
    a, q_skip, c_skip, p_skip, _ = self.atom_encoder(ref_pos=ref_pos, atom_attention_mask=ref_mask, ref_space_uid=ref_space_uid, ref_charge=ref_charge, ref_element=ref_element,
                                                     ref_atom_name_chars=ref_atom_name_chars, atom_to_token=tok_idx, r_l=r_noisy, s_i=s_trunk, num_diffusion_samples=1,
                                                     return_intermediates=False, inference_cache=inference_cache)
    if ev_s is not None:
        cur.wait_event(ev_s)
    a = a + s_tok
    if _CFG["gemm"] == "split":
        a = tt_forward_split(st, a, None, z, token_attention_mask, self.token_transformer, cd_list=cd_list, evs=evs)
    else:
        a = tt_forward_exact(st, a, None, z, token_attention_mask, self.token_transformer, cd_list=cd_list, evs=evs)
    a = self.token_norm(a)
    r_update, _ = self.atom_decoder(a_i=a, q_l=q_skip, c_l=c_skip, p_lm=p_skip, atom_to_token=tok_idx, atom_attention_mask=ref_mask, num_diffusion_samples=1, return_intermediates=False)
    sigma2 = sigma * sigma
    t2 = t * t
    out = (sigma2 / (sigma2 + t2))[:, None, None] * x_noisy
    out = out + ((sigma * t) / torch.sqrt(sigma2 + t2))[:, None, None] * r_update
    if evs is not None:
        cur.wait_event(evs[-1])
    return {"x_denoised": out, "token_repr": a if return_token_repr else None, "atom_intermediates": None}


def _pair_bias(st, blk, z, attention_mask, bsz, n):
    pbd = getattr(blk, "_ef2opt_pb", None)
    if pbd:
        for k, v in pbd.items():
            if len(k) > 1 and k[1] == tuple(z.shape):
                STATS["pb_kit"] += 1
                return v[0]
    key = (id(blk), tuple(z.shape), z.dtype, None if attention_mask is None else tuple(attention_mask.shape))
    e = st.pb.get(key)
    if e is None or e[1] != _EPOCH["n"]:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("ef2_mk_sampler: pair bias missing during capture")
        import transformers.models.esmfold2.modeling_esmfold2_common as C
        with torch.no_grad():
            if blk._can_use_fused_pair_bias(z, n, 0.0):
                kernel_mask = attention_mask if attention_mask is not None else torch.ones(bsz, n, device=z.device, dtype=torch.bool)
                pw = blk.pair_norm.weight; pb = blk.pair_norm.bias if blk.pair_norm.bias is not None else torch.zeros_like(pw)
                z_bf = z if z.dtype == torch.bfloat16 else z.to(torch.bfloat16)
                bias = C._fused_pair_bias(z_bf, kernel_mask, blk.pair_bias_proj.weight, num_heads=blk.num_heads, pair_norm_w=pw, pair_norm_b=pb)
            else:
                bias = blk.pair_bias_proj(blk.pair_norm(z))                     # reference layout [B, Q, K, H]; mask added at use like stock
        if e is None or e[0].shape != bias.shape or e[0].dtype != bias.dtype:
            st.pb[key] = (bias.clone(), _EPOCH["n"]); STATS["pb_new"] += 1
        else:
            e[0].copy_(bias); st.pb[key] = (e[0], _EPOCH["n"])
        STATS["pb_compute"] += 1
    return st.pb[key][0]


def _workspace(st, M, L, dev):
    key = (M, str(dev))
    w = st.ws.get(key)
    if w is None:
        f32 = dict(device=dev, dtype=torch.float32)
        w = dict(xin=torch.empty(M, st.D, **f32), qkvg=torch.empty(M, 4 * st.D, **f32), ctx=torch.empty(M, st.D, **f32), o=torch.empty(M, st.D, **f32),
                 sw=torch.empty(M, 2 * st.hid, **f32), h=torch.empty(M, st.hid, **f32), o2=torch.empty(M, st.D, **f32), xa=torch.empty(M, st.D, **f32), xb=torch.empty(M, st.D, **f32),
                 yg=torch.empty(L, 2 * st.D, **f32), ysh=torch.empty(L, 2 * st.D, **f32), ys=torch.empty(L, 2 * st.D, **f32))
        st.ws[key] = w
    return w


def _cond_all_blocks_side(st, s_row, need, n_events):
    """mode 'overlap': launch the s-only per-step work of ALL blocks (stock modules, M=L rows => stock bits) on a side stream forked from the current
    stream; returns (list of per-block dicts, list of per-block events). Graph-capture safe (fork/join via events inside the capture)."""
    cur = torch.cuda.current_stream()
    if st.side is None:
        st.side = torch.cuda.Stream(device=s_row.device)
    ev0 = torch.cuda.Event(); ev0.record(cur)
    outs, evs = [], []
    with torch.cuda.stream(st.side):
        st.side.wait_event(ev0)
        for b in range(st.nb):
            r = _cond_block_step_stock(st, b, s_row, need)
            outs.append({k: v.unsqueeze(0) for k, v in r.items()})
            e = torch.cuda.Event(); e.record(st.side); evs.append(e)
    return outs, evs


def tt_forward_exact(st, a, i1, z, attention_mask, tt, cd_list=None, evs=None):
    """EXACT-class path: every GEMM / SDPA / LayerNorm / activation is the SAME torch call on the SAME shapes as the stock forward (=> same cuBLAS
    kernels, same bits); the s-only sub-graphs are replaced by their per-fold hoisted values (bitwise identical: computed by the same calls at the
    fold's step 0) or, when the tables do not fit the memory budget (mode 'overlap'), computed per step by the same calls on a side stream that
    overlaps the x-dependent chain.  torch.equal with the stock token transformer at op level (measured 185-705 tok)."""
    bsz, n, D = a.shape
    tab = st.table
    x = a
    side_outs = side_evs = None
    if cd_list is not None:
        side_outs, side_evs = cd_list, evs
    elif st.mode != "table":
        s_row = tab["s"].index_select(0, i1)                     # [1, L, D]
        need = tuple(x_ for x_ in ("cs", "ch", "og") if (x_ + "_a") not in tab)
        if st.mode == "overlap" and need:
            side_outs, side_evs = _cond_all_blocks_side(st, s_row[0], need, st.nb)
    for b in range(st.nb):
        blk = tt.attn_blocks[b]; trb = tt.transition_blocks[b]
        cd = {}
        if cd_list is None:
            for k in ("cs_a", "ch_a", "og_a", "cs_t", "ch_t", "og_t"):
                if k in tab:
                    cd[k] = tab[k][b].index_select(0, i1)            # [1, L, D]
        need_b = tuple(x_ for x_ in ("cs", "ch", "og") if (x_ + "_a") not in cd)
        if need_b:
            if side_outs is not None:
                if side_evs is not None:
                    torch.cuda.current_stream().wait_event(side_evs[b])
                cd.update(side_outs[b])
            else:
                r = _cond_block_step_stock(st, b, s_row[0], need_b)
                cd.update({k: v.unsqueeze(0) for k, v in r.items()})
        # ---- AttentionPairBias.forward (stock op order) with adaln(a,s) = cs * LN(a) + ch
        a_norm = F.layer_norm(x, (D,), None, None, blk.adaln.eps)
        xx = cd["cs_a"] * a_norm + cd["ch_a"]
        q = blk.q_proj(xx).view(bsz, n, blk.num_heads, blk.head_dim)
        kv = blk.kv_proj(xx)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(bsz, n, blk.num_heads, blk.head_dim); v = v.view(bsz, n, blk.num_heads, blk.head_dim)
        bias = _pair_bias(st, blk, z, attention_mask, bsz, n)
        if bias.dim() == 4 and bias.shape[1] == blk.num_heads:          # fused-layout bias [B,H,Q,K] (kit cache or ours)
            attn_out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias.to(q.dtype))
            g = torch.sigmoid(blk.g_proj(xx)).view(bsz, n, blk.num_heads, blk.head_dim)
            ctx = g * attn_out.transpose(1, 2)
            out = blk.out_proj(ctx.reshape(bsz, n, D))
        else:                                                             # reference layout [B,Q,K,H] (backend None)
            g = torch.sigmoid(blk.g_proj(xx)).view(bsz, n, blk.num_heads, blk.head_dim)
            logits = torch.einsum("... i h d, ... j h d -> ... i j h", q, k) * blk.scale
            logits = logits + bias.to(dtype=logits.dtype)
            if attention_mask is not None:
                min_val = torch.finfo(logits.dtype).min
                logits = logits + torch.where(attention_mask.bool()[:, None, :, None], 0.0, min_val).to(dtype=logits.dtype)
            attn = torch.softmax(logits, dim=-2).to(dtype=v.dtype)
            ctx = g * torch.einsum("... i j h, ... j h d -> ... i h d", attn, v)
            out = blk.out_proj(ctx.reshape(bsz, n, D))
        out = cd["og_a"] * out
        x = x + out
        # ---- ConditionedTransitionBlock.forward (stock op order)
        a_norm = F.layer_norm(x, (D,), None, None, trb.adaln.eps)
        xx = cd["cs_t"] * a_norm + cd["ch_t"]
        swish_a, swish_b = trb.lin_swish(xx).chunk(2, dim=-1)
        hb = F.silu(swish_a) * swish_b
        out = trb.lin_out(hb)
        out = cd["og_t"] * out
        x = x + out
    if side_evs is not None:
        torch.cuda.current_stream().wait_event(side_evs[-1])            # join (all side work consumed)
    return x


def tt_forward_split(st, a, i1, z, attention_mask, tt, cd_list=None, evs=None):
    """Tier-2 'split' arm: stock-shaped GEMMs (F.linear per stock Linear => the same cuBLAS kernels/accuracy as stock) + fused Triton elementwise
    (LN+modulate, gate*out+residual, SwiGLU) + SDPA. Conditioning from cd_list (side stream) or tables."""
    bsz, n, D = a.shape
    M = bsz * n
    tab = st.table
    w = _workspace(st, M, n, a.device)
    cur = a.reshape(M, D).contiguous()
    for b in range(st.nb):
        blk = tt.attn_blocks[b]; trb = tt.transition_blocks[b]
        if cd_list is not None:
            if evs is not None:
                torch.cuda.current_stream().wait_event(evs[b])
            cd = {k: v.reshape(n, D) for k, v in cd_list[b].items()}
        else:
            cd = {k: tab[k][b].index_select(0, i1).reshape(n, D) for k in ("cs_a", "ch_a", "og_a", "cs_t", "ch_t", "og_t")}
        bias = _pair_bias(st, blk, z, attention_mask, bsz, n)
        # attention block
        ln_mod(cur, cd["cs_a"], cd["ch_a"], w["xin"], n, st.eps)
        xx = w["xin"].view(bsz, n, D)
        q = blk.q_proj(xx).view(bsz, n, blk.num_heads, blk.head_dim)
        kv = blk.kv_proj(xx); k, v = kv.chunk(2, dim=-1)
        k = k.view(bsz, n, blk.num_heads, blk.head_dim); v = v.view(bsz, n, blk.num_heads, blk.head_dim)
        attn_out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias.to(q.dtype) if bias.dtype != q.dtype else bias)
        g = torch.sigmoid(blk.g_proj(xx)).view(bsz, n, blk.num_heads, blk.head_dim)
        ctx = (g * attn_out.transpose(1, 2)).reshape(bsz, n, D)
        out = blk.out_proj(ctx).reshape(M, D)
        x1 = w["xb"] if cur is not w["xb"] else w["xa"]
        gate_res(cur, out, cd["og_a"], x1, n, sig_g=False)
        # transition block
        ln_mod(x1, cd["cs_t"], cd["ch_t"], w["xin"], n, st.eps)
        sw = trb.lin_swish(w["xin"].view(bsz, n, D)).reshape(M, 2 * st.hid)
        swiglu(sw, w["h"], st.hid)
        out = trb.lin_out(w["h"].view(bsz, n, st.hid)).reshape(M, D)
        x2 = w["xa"] if x1 is w["xb"] else w["xb"]
        gate_res(x1, out, cd["og_t"], x2, n, sig_g=False)
        cur = x2
    if evs is not None:
        torch.cuda.current_stream().wait_event(evs[-1])
    return cur.view(bsz, n, D).clone()


def tt_forward_mk(st, a, i1, z, attention_mask, tt):
    """token transformer, 12 x (attention pair-bias block + conditioned transition), fused execution. a: [1, L, D] fp32; i1: step index tensor [1]."""
    bsz, n, D = a.shape
    M = bsz * n
    tab = st.table; mode = st.mode
    w = _workspace(st, M, n, a.device)
    x = a.reshape(M, D)
    cur = x.contiguous()
    side_outs = side_evs = None
    if mode != "table":
        s_row = tab["s"].index_select(0, i1).reshape(n, D); sn_row = tab["sn"].index_select(0, i1).reshape(n, D)
        need0 = tuple(x_ for x_ in ("cs", "ch", "og") if (x_ + "_a") not in tab)
        if mode == "overlap" and need0 and _CFG.get("cond", "stock") != "grouped":
            side_outs, side_evs = _cond_all_blocks_side(st, s_row, need0, st.nb)
    key_mask = None
    for b in range(st.nb):
        blk = tt.attn_blocks[b]
        bias = _pair_bias(st, blk, z, attention_mask, bsz, n)
        cd = {}
        for k in ("cs_a", "ch_a", "og_a", "cs_t", "ch_t", "og_t"):
            if k in tab:
                cd[k] = tab[k][b].index_select(0, i1).reshape(n, D)
        need = tuple(x for x in ("cs", "ch", "og") if (x + "_a") not in cd)
        if need:
            if side_outs is not None:
                torch.cuda.current_stream().wait_event(side_evs[b])
                cd.update({k: v[0] for k, v in side_outs[b].items()})
            elif _CFG.get("cond", "stock") == "grouped":
                cd.update(_cond_block_step(st, b, s_row, sn_row, w, need))
            else:
                cd.update(_cond_block_step_stock(st, b, s_row, need))
        cs_a, ch_a, og_a, cs_t, ch_t, og_t = cd["cs_a"], cd["ch_a"], cd["og_a"], cd["cs_t"], cd["ch_t"], cd["og_t"]
        # --- attention block
        ln_mod(cur, cs_a, ch_a, w["xin"], n, st.eps)                          # AdaLN(a, s)
        torch.mm(w["xin"], st.WqkvgT[b], out=w["qkvg"])                       # [M, 4D] = q|k|v|g  (cuBLAS fp32, no bias: bias-free like stock kv/g Linears)
        qbias_sig_(w["qkvg"], st.bq[b], D, sig_g=(_CFG["attn"] == "triton" and _HAS_TRITON))   # q += q_bias (exact fp32 add like cuBLAS epilogue... rounding: one add) ; g = sigmoid(g) for the triton attn
        if _CFG["attn"] == "triton" and _HAS_TRITON:
            in_bias_mask = (bias.dtype == torch.bfloat16) or attention_mask is None
            if not in_bias_mask and key_mask is None:
                key_mask = attention_mask.to(torch.int32).contiguous()
            attn_triton(w["qkvg"], bias, None if in_bias_mask else key_mask, w["ctx"], bsz, n, st.H, st.HD, D, st.scale)
            ctx = w["ctx"]
        else:
            qkvg = w["qkvg"].view(bsz, n, 4, st.H, st.HD)
            q = qkvg[:, :, 0].transpose(1, 2); k = qkvg[:, :, 1].transpose(1, 2); v = qkvg[:, :, 2].transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype) if bias.dtype != q.dtype else bias)     # same call as stock (bf16 bias -> fp32 copy like stock)
            g = torch.sigmoid(qkvg[:, :, 3])
            ctx = (g * o.transpose(1, 2)).reshape(M, D)
        torch.mm(ctx, st.WoutT[b], out=w["o"])                                # out_proj
        x1 = w["xb"] if cur is not w["xb"] else w["xa"]
        gate_res(cur, w["o"], og_a, x1, n, sig_g=False)                       # x1 = cur + sigmoid(out_gate(s)) * out
        # --- transition block
        ln_mod(x1, cs_t, ch_t, w["xin"], n, st.eps)
        torch.mm(w["xin"], st.WswT[b], out=w["sw"])                           # [M, 2*hid]
        swiglu(w["sw"], w["h"], st.hid)
        torch.mm(w["h"], st.WloT[b], out=w["o2"])
        x2 = w["xa"] if x1 is w["xb"] else w["xb"]
        gate_res(x1, w["o2"], og_t, x2, n, sig_g=False)
        cur = x2
    if side_evs is not None:
        torch.cuda.current_stream().wait_event(side_evs[-1])
    return cur.view(bsz, n, D).clone()


# ---------------------------------------------------------------------------------------------------------------------
def _wrap_sample(sh):
    if getattr(sh, "_mk_sample_wrapped", False):
        return
    inner = sh.sample

    def sample_mk(*a, **k):
        sh._mk_fold_cfg = dict(steps=(int(k["num_sampling_steps"]) if k.get("num_sampling_steps") is not None else sh.inference_num_steps), max_sigma=k.get("max_inference_sigma", 256.0), src="sample_wrapper")
        return inner(*a, **k)
    sh._mk_inner_sample = inner
    sh.sample = sample_mk
    sh._mk_sample_wrapped = True


def clear():
    _RETIRED.clear()
    for model in _ENABLED:
        st = model.structure_head.diffusion_module._mk_state
        if st is not None:
            st.pb.clear(); st.ws.clear(); st.table = None; st.table_key = None
    STATS["clears"] += 1


def _model_pre_hook(sh):
    def hook(module, args, kwargs):
        ns = kwargs.get("num_sampling_steps", None)
        prev = getattr(sh, "_mk_fold_cfg", None) or {}
        sh._mk_fold_cfg = dict(steps=(int(ns) if ns is not None else sh.inference_num_steps), max_sigma=prev.get("max_sigma_override", 256.0), src="model_pre_hook")
        return None
    return hook


def enable(model, gemm="stock", ew="torch", attn="sdpa", hoist="auto", cond="stock", probe=False, max_table_gb=None, reuse=True):
    _CFG.update(gemm=gemm, ew=ew, attn=attn, hoist=hoist, cond=cond, probe=probe, reuse=reuse)
    if max_table_gb is not None:
        _CFG["max_table_bytes"] = int(float(max_table_gb) * 2**30)
    sh, dm, tt = _find(model)
    if getattr(dm, "_mk_enabled", False):
        return describe()
    dm._mk_state = _State(tt, sh)
    dm._mk_eager_forward = dm.forward
    dm._mk_sh = sh
    dm.forward = types.MethodType(_dm_forward_mk, dm)
    dm._mk_enabled = True
    dm._mk_model_hook = model.register_forward_pre_hook(_model_pre_hook(sh), with_kwargs=True)
    _wrap_sample(sh)
    _ENABLED.append(model)
    try:
        import ef2_opt
        if not getattr(ef2_opt, "_mk_hooked", False):
            _orig = ef2_opt.clear_graphs

            def clear_graphs_mk(*a, **k):
                r = _orig(*a, **k); clear(); return r
            ef2_opt.clear_graphs = clear_graphs_mk; ef2_opt._mk_hooked = True
    except Exception:
        pass
    return describe()


def disable(model):
    sh, dm, tt = _find(model)
    if getattr(dm, "_mk_enabled", False):
        dm.forward = dm._mk_eager_forward
        del dm._mk_eager_forward
        h = getattr(dm, "_mk_model_hook", None)
        if h is not None:
            h.remove(); dm._mk_model_hook = None
        dm._mk_enabled = False; dm._mk_state = None
    if getattr(sh, "_mk_sample_wrapped", False) and getattr(sh.sample, "__name__", "") == "sample_mk":
        sh.sample = sh._mk_inner_sample; sh._mk_sample_wrapped = False
    _ENABLED[:] = [m for m in _ENABLED if m is not model]
    return True


def describe():
    return dict(version=VERSION, cfg=dict(_CFG), triton=_HAS_TRITON, enabled_models=len(_ENABLED))


def stats():
    return {k: int(v) for k, v in STATS.items()}
