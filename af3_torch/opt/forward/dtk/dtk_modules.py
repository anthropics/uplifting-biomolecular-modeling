# dtk_modules.py -- stock-formulation replicas of the AF3-family diffusion transformer blocks (per model family) and the DTK fused path.
# Families are numerics recipes keyed by name: 'xfold' (this engine's eager statements), 'af3t_apb' (SDPA + additive mask), 'of3' (bf16 matmul + softmax);
#           the other keys of LOWP_FAMILIES / FP32_FAMILIES select the precision recipe their branches below state (TF32 or 'highest' fp32, einsum layout, softmax dim).
# Weights are random with trained-like scales; the SAME weights feed stock replica, fused path and the fp64 reference.
import math
import torch
import torch.nn.functional as F
import dtk_kernels as K

LOWP_FAMILIES = {"xfold": torch.bfloat16, "af3t_apb": torch.bfloat16, "of3": torch.bfloat16, "rf3": torch.bfloat16}
FP32_FAMILIES = {"protenix": "tf32", "boltz2": "ieee"}


def family_cfg(fam):
    lowp = fam in LOWP_FAMILIES
    return dict(lowp=lowp, cdtype=(LOWP_FAMILIES[fam] if lowp else torch.float32),
                tf32=(fam == "protenix"))


def set_matmul_policy(fam):
    if fam == "protenix":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    else:  # 'highest' fp32 matmuls; bf16 families do their GEMMs in bf16 anyway (TF32 off)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False


class DiTWeights:
    """Random weights for n_blocks AF3-style diffusion transformer blocks (attention + conditioned transition)."""
    def __init__(self, n_blocks, c, c_s, n_head, dev, gen, hidden_mult=2):
        self.n_blocks, self.c, self.c_s, self.h = n_blocks, c, c_s, n_head
        self.d = c // n_head
        self.hid = hidden_mult * c
        def W(o, i, gain=1.0):
            return (torch.randn(o, i, generator=gen, device="cpu") * (gain / math.sqrt(i))).to(dev)
        def vec(n, val=0.0, noise=0.0):
            return (torch.full((n,), val) + noise * torch.randn(n, generator=gen)).to(dev)
        self.blocks = []
        for _ in range(n_blocks):
            b = dict(
                # AdaLN (attention): LN(a) no affine; LN(s) weight only; scale Linear(c_s->c, bias); shift Linear(c_s->c, no bias)
                ln_s_w1=vec(c_s, 1.0, 0.1), W_scale1=W(c, c_s), b_scale1=vec(c, 0.0, 0.2), W_shift1=W(c, c_s),
                Wq=W(c, c), bq=vec(c, 0.0, 0.2), Wk=W(c, c), Wv=W(c, c), Wg=W(c, c), Wo=W(c, c),
                W_zero1=W(c, c_s), b_zero1=vec(c, -2.0, 0.2),
                ln_s_w2=vec(c_s, 1.0, 0.1), W_scale2=W(c, c_s), b_scale2=vec(c, 0.0, 0.2), W_shift2=W(c, c_s),
                W1=W(self.hid, c), W2=W(self.hid, c), W3=W(c, self.hid),
                W_zero2=W(c, c_s), b_zero2=vec(c, -2.0, 0.2),
            )
            self.blocks.append(b)

    def to(self, dtype):
        nw = DiTWeights.__new__(DiTWeights)
        nw.__dict__.update({k: v for k, v in self.__dict__.items() if k != "blocks"})
        nw.blocks = [{k: (v.to(dtype) if torch.is_tensor(v) else v) for k, v in b.items()} for b in self.blocks]
        return nw


# ----------------------------------------------------------------------------- stock replicas
def _hp(cd):
    return torch.float64 if cd == torch.float64 else torch.float32


def _adaln_ref(a, s, ln_s_w, W_scale, b_scale, W_shift, cd):
    # a: fp32 residual stream [N,c]; s: [N,c_s] fp32. LN in fp32 (autocast keeps LN fp32), linears in compute dtype cd.
    hp = _hp(cd)
    a_n = F.layer_norm(a.to(hp), (a.shape[-1],))
    s_n = F.layer_norm(s.to(hp), (s.shape[-1],)) * ln_s_w.to(hp)
    s_c = s_n.to(cd)
    scale = F.linear(s_c, W_scale.to(cd), b_scale.to(cd))
    shift = F.linear(s_c, W_shift.to(cd))
    return torch.sigmoid(scale) * a_n.to(cd) + shift          # -> cd


def _attn_core_ref(fam, q, k, v, bias, n_head):
    """q,k,v [N,c] in compute dtype; bias [H,N,N]; returns [N,c]."""
    N, c = q.shape
    d = c // n_head
    if fam == "rf3":
        # heads-last einsum: softmax over j at dim=-2 of [i,j,h]
        Q = q.view(N, n_head, d) / math.sqrt(d); Kh = k.view(N, n_head, d); V = v.view(N, n_head, d)
        B = bias.permute(1, 2, 0).to(q.dtype)                     # [i,j,h]  (bias add happens in bf16 under autocast)
        A = torch.softmax((torch.einsum("ihd,jhd->ijh", Q, Kh) + B).float(), dim=-2).to(V.dtype)   # autocast softmax = fp32, strided (dim=-2)
        o = torch.einsum("ijh,jhc->ihc", A, V)
        return o.reshape(N, c)
    Q = q.view(N, n_head, d).permute(1, 0, 2); Kh = k.view(N, n_head, d).permute(1, 0, 2); V = v.view(N, n_head, d).permute(1, 0, 2)
    if fam in ("af3t_apb", "protenix"):
        o = F.scaled_dot_product_attention(Q.unsqueeze(0), Kh.unsqueeze(0), V.unsqueeze(0),
                                           attn_mask=bias.unsqueeze(0).to(Q.dtype), scale=1.0 / math.sqrt(d))[0]
    elif fam == "of3":
        a = torch.matmul(Q, Kh.transpose(-1, -2)) * (1.0 / math.sqrt(d)) + bias.to(Q.dtype)
        a = torch.softmax(a.float(), dim=-1).to(V.dtype)            # conservative: fp32 softmax (stricter stock reference)
        o = torch.matmul(a, V)
    else:  # eager fp32 softmax; ref64 (fp64 everything)
        hp = _hp(q.dtype)
        logits = torch.einsum("hqc,hkc->hqk", Q * (1.0 / math.sqrt(d)), Kh).to(hp) + bias.to(hp)
        w = torch.softmax(logits, dim=-1).to(V.dtype)
        o = torch.einsum("hqk,hkc->hqc", w, V)
    return o.permute(1, 0, 2).reshape(N, c)


def dit_block_ref(fam, a, s, bias, w, cd):
    """One AF3 diffusion-transformer block, stock formulation of family `fam`. a fp32 [N,c] (residual stream), s fp32 [N,c_s]."""
    b = w
    a_n = _adaln_ref(a, s, b["ln_s_w1"], b["W_scale1"], b["b_scale1"], b["W_shift1"], cd)
    q = F.linear(a_n, b["Wq"].to(cd), b["bq"].to(cd)); k = F.linear(a_n, b["Wk"].to(cd)); v = F.linear(a_n, b["Wv"].to(cd))
    o = _attn_core_ref(fam, q, k, v, bias, w_nhead(b))
    g = torch.sigmoid(F.linear(a_n, b["Wg"].to(cd)))
    o = F.linear(g * o, b["Wo"].to(cd))
    zg = torch.sigmoid(F.linear(s.to(cd), b["W_zero1"].to(cd), b["b_zero1"].to(cd)))
    a = a + (zg * o).to(a.dtype)
    a_n = _adaln_ref(a, s, b["ln_s_w2"], b["W_scale2"], b["b_scale2"], b["W_shift2"], cd)
    hdn = F.silu(F.linear(a_n, b["W1"].to(cd))) * F.linear(a_n, b["W2"].to(cd))
    o = F.linear(hdn, b["W3"].to(cd))
    zg = torch.sigmoid(F.linear(s.to(cd), b["W_zero2"].to(cd), b["b_zero2"].to(cd)))
    a = a + (zg * o).to(a.dtype)
    return a


def w_nhead(b):
    return b["_nhead"]


def transformer_ref(fam, a, s, bias_all, W: DiTWeights, cd):
    for i, b in enumerate(W.blocks):
        b["_nhead"] = W.h
        a = dit_block_ref(fam, a, s, bias_all[i], b, cd)
    return a


# ----------------------------------------------------------------------------- fused path
class FusedDiT:
    """DTK fused implementation with the same weights. Precomputes: concatenated per-block weights; folds LN(s) per-block weight into the
    s-side GEMM; s-side conditioning of ALL blocks in one GEMM per step; static bias must be given pre-laid ([nb,H,N,N], bf16 or fp32)."""
    def __init__(self, W: DiTWeights, cd, fam, attn_impl="flash", structure="af3", norm="ln", eps_a=1e-5, eps_s=1e-5, eps_kq=1e-5):
        """structure='af3' (the AlphaFold 3 block: LayerNorm AdaLN, no q/k norm, sequential residual
        a1 = a + g1*Attn(AdaLN1(a)); out = a1 + g2*Trans(AdaLN2(a1)))  |  structure='rf3' (RMSNorm AdaLN (norm='rms'), LayerNorm(q), LayerNorm(k)
        over the full width after the projections, no residual between attention and transition -- the transition applied to the BLOCK INPUT:
        out = a + g1*Attn(AdaLN1(a)) + g2*Trans(AdaLN2(a))).
        For 'rf3' each block dict must also carry ln_q_w, ln_q_b, ln_k_w, ln_k_b."""
        self.W, self.cd, self.fam, self.attn_impl = W, cd, fam, attn_impl
        self.structure, self.rms = structure, (norm == "rms")
        self.eps_a, self.eps_s, self.eps_kq = eps_a, eps_s, eps_kq
        self.h, self.c, self.d = W.h, W.c, W.d
        c = W.c
        Ws, bs = [], []
        self.blk = []
        for b in W.blocks:
            # s-side: [scale1 | shift1 | zero1 | scale2 | shift2 | zero2] ; scale/shift act on LN(s)*w_b  -> fold w_b into columns
            w1 = b["ln_s_w1"].float(); w2 = b["ln_s_w2"].float()
            Ws.append(torch.cat([b["W_scale1"].float() * w1[None, :], b["W_shift1"].float() * w1[None, :], b["W_zero1"].float(),
                                 b["W_scale2"].float() * w2[None, :], b["W_shift2"].float() * w2[None, :], b["W_zero2"].float()], 0))
            zc = torch.zeros(c, device=w1.device)
            bs.append(torch.cat([b["b_scale1"].float(), zc, b["b_zero1"].float(), b["b_scale2"].float(), zc, b["b_zero2"].float()]))
            self.blk.append(dict(
                W_qkvg=torch.cat([b["Wq"], b["Wk"], b["Wv"], b["Wg"]], 0).to(cd).contiguous(),
                b_qkvg=torch.cat([b["bq"], torch.zeros(3 * c, device=w1.device)]).to(cd),
                Wo=b["Wo"].to(cd).contiguous(), W12=torch.cat([b["W1"], b["W2"]], 0).to(cd).contiguous(), W3=b["W3"].to(cd).contiguous(),
                ln_q_w=(b["ln_q_w"].float().contiguous() if "ln_q_w" in b else None), ln_q_b=(b["ln_q_b"].float().contiguous() if "ln_q_b" in b else None),
                ln_k_w=(b["ln_k_w"].float().contiguous() if "ln_k_w" in b else None), ln_k_b=(b["ln_k_b"].float().contiguous() if "ln_k_b" in b else None)))
        self.W_s_all = torch.cat(Ws, 0).to(cd).contiguous()        # [nb*6c, c_s]
        self.b_s_all = torch.cat(bs, 0).to(cd)
        # NOTE zero-gates act on raw s (not LN(s)) in AF3: handled by splitting the GEMM input: cols 2,5 use s, others use LN(s).
        # To keep ONE GEMM we run two GEMMs per step instead (LN(s) -> scale/shift cols, s -> zero cols). Still once per step.
        nb = len(W.blocks)
        idx = torch.arange(nb * 6 * c, device=w1.device).view(nb, 6, c)
        self.cols_ln = idx[:, [0, 1, 3, 4], :].reshape(-1)
        self.cols_raw = idx[:, [2, 5], :].reshape(-1)
        self.W_s_ln = self.W_s_all[self.cols_ln].contiguous(); self.b_s_ln = self.b_s_all[self.cols_ln].contiguous()
        self.W_s_raw = self.W_s_all[self.cols_raw].contiguous(); self.b_s_raw = self.b_s_all[self.cols_raw].contiguous()

    def cond(self, s):
        """Once per step: LN(s) (no affine) and the two concatenated s-side GEMMs -> per block views."""
        cd = self.cd
        s_hat = K.ln_modulate(s.float(), out_dtype=cd, rms=self.rms, eps=self.eps_s)    # [N, c_s]  (LN or RMSNorm, no affine: weight folded into W_s)
        S_ln = F.linear(s_hat, self.W_s_ln, self.b_s_ln)                                 # [N, nb*4c]
        S_raw = F.linear(s.to(cd), self.W_s_raw, self.b_s_raw)                           # [N, nb*2c]
        return S_ln, S_raw

    def block(self, i, a, S_ln, S_raw, bias_i, key_mask=None):
        c, cd = self.c, self.cd
        N = a.shape[0]
        B = self.blk[i]
        o4 = i * 4 * c; o2 = i * 2 * c
        a_hat = K.ln_modulate(a, scale=S_ln[:, o4:o4 + c], shift=S_ln[:, o4 + c:o4 + 2 * c], out_dtype=cd, rms=self.rms, eps=self.eps_a)
        qkvg = F.linear(a_hat, B["W_qkvg"], B["b_qkvg"])                                  # [N, 4c]
        if B.get("ln_q_w") is not None:      # kq_norm (structure 'rf3'): LayerNorm over the full H*D width of q and k (fp32 statistics, written in compute dtype)
            qn = K.ln_modulate(qkvg[:, 0:c], weight=B["ln_q_w"], bias=B["ln_q_b"], eps=self.eps_kq, out_dtype=cd)
            kn = K.ln_modulate(qkvg[:, c:2 * c], weight=B["ln_k_w"], bias=B["ln_k_b"], eps=self.eps_kq, out_dtype=cd)
            q = qn.view(N, self.h, self.d).permute(1, 0, 2)
            k = kn.view(N, self.h, self.d).permute(1, 0, 2)
        else:
            q = qkvg[:, 0:c].view(N, self.h, self.d).permute(1, 0, 2)
            k = qkvg[:, c:2 * c].view(N, self.h, self.d).permute(1, 0, 2)
        v = qkvg[:, 2 * c:3 * c].view(N, self.h, self.d).permute(1, 0, 2)
        if self.attn_impl == "sdpa":
            am = bias_i.unsqueeze(0).to(q.dtype)
            if key_mask is not None:
                am = am + ((1.0 - key_mask.to(q.dtype)) * -1e9)[None, None, None, :]
            o = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn_mask=am, scale=1.0 / math.sqrt(self.d))[0]
            o = o.permute(1, 0, 2).reshape(N, c)
            o = K.gate_residual(o, gate=qkvg[:, 3 * c:4 * c], res=None, out_dtype=cd)                     # sigmoid(g) * o
        else:
            o = K.flash_bias_attn(q, k, v, bias=bias_i, key_mask=key_mask, gate=qkvg[:, 3 * c:4 * c], out_dtype=cd)   # [N, c]
        o = F.linear(o, B["Wo"])
        a1 = K.gate_residual(o, gate=S_raw[:, o2:o2 + c], res=a, out_dtype=torch.float32)          # a + g1 * attn
        t_in = a if self.structure == "rf3" else a1                                                   # structure 'rf3': transition of the BLOCK INPUT (parallel residual)
        a_hat = K.ln_modulate(t_in, scale=S_ln[:, o4 + 2 * c:o4 + 3 * c], shift=S_ln[:, o4 + 3 * c:o4 + 4 * c], out_dtype=cd, rms=self.rms, eps=self.eps_a)
        ab = F.linear(a_hat, B["W12"])                                                    # [N, 2*hid]
        hb = K.swiglu(ab)
        o = F.linear(hb, B["W3"])
        a = K.gate_residual(o, gate=S_raw[:, o2 + c:o2 + 2 * c], res=a1, out_dtype=torch.float32)
        return a

    def forward(self, a, s, bias_laid, key_mask=None):
        if a.dim() == 3:                                         # [S, N, c]: S samples of ONE conditioning (s, bias, key mask shared) -> the sample-batched schedule
            return self.forward_batched(a, s, bias_laid, key_mask)
        S_ln, S_raw = self.cond(s)
        for i in range(len(self.blk)):
            a = self.block(i, a, S_ln, S_raw, bias_laid[i], key_mask)
        return a

    # ------------------------------------------------------------------ sample-batched schedule (AF3-torch lever 'sbatch')
    # a [S, N, c] fp32 = S diffusion samples at ONE noise level: s [N, c_s], the pre-laid bias [nb, H, N, N] and the key mask [N] are the same
    # for every sample. cond() (LN(s) + the two concatenated s-side GEMMs) runs ONCE per step at [N, .] and the row kernels read it PERIODICALLY
    # (dtk_kernels >= 0.2 `mod_period` / `gate_period` = N: row s*N+i takes conditioning row i -- never materialised x S); the a-side GEMMs run
    # on the [S*N, c] row block; the attention core is ONE launch for all samples through the shared core's pair-bias attention PROVIDER by the
    # caller's TIER WORD (`apb_word` = fast | big, set by the caller that also folds the key mask into the hoisted bias once per trajectory:
    # opt_core.kernels.apb.pair_bias_attention, q/k/v/gate = [S, N, h, d] views of the QKVG GEMM's column slices, the block's bias [1, h, N, N]
    # shared by the samples, cell dit_h16d48) -- the provider's measured cell for (card, dtype, S, N bucket, eager | graph replay) names the row
    # (its own apb_attn kernel: the bias plane read once per tile and shared by the samples through L2; a carried package such as fpf_apb or
    # l3a; or its stock SDPA row where SDPA measured fastest) and `batched_attn` names the arm that served. No provider in the process (or
    # `apb_word` None) -> the shared core's apb_attn entry directly (opt_core.attn.apb_core, key mask by name, sigmoid gate fused, out [S*N, c]):
    # `batched_attn = "apb_attn"`; a call class the word's row refuses by name is served that way too (named once in `batched_attn_event`); when
    # that entry is absent or refuses, the DTK flash kernel runs once per sample on the sample's row slice (`"flash_bias_attn:per_sample"`: the
    # serial path's own kernel and bits per sample). dtk_kernels without periodic operands (the v0.1 module) -> `batched_route = "serial"`: the
    # per-sample schedule above, sample by sample (same bits as S serial calls), named.
    batched_attn = "apb_attn"
    batched_attn_event = None
    batched_route = None            # decided on the first batched call: "periodic_rows" | "serial:<why>"
    _apb_entry = None
    apb_word = None                 # the provider word this instance asks (the caller's tier word fast | big; a row word serves exactly that row); None = apb_attn directly
    apb_graph = None                # the batched calls are replayed from a whole-step CUDA graph (True: the provider's graph-replay cells, also for the eager warm-up
                                    # steps, so no row meets its first launch under capture; False: eager cells; None: the stream's state at the call decides)
    _apb_provider = None

    def _decide_batched_route(self):
        import inspect
        try:
            ok = "mod_period" in inspect.signature(K.ln_modulate).parameters and "gate_period" in inspect.signature(K.gate_residual).parameters
        except (TypeError, ValueError):
            ok = False
        self.batched_route = "periodic_rows" if ok else "serial:dtk_kernels_without_periodic_operands(%s)" % getattr(K, "__file__", "?")
        if self.batched_attn == "apb_attn" and FusedDiT._apb_entry is None:
            try:
                from opt_core.attn.apb_core import pair_bias_attention
                FusedDiT._apb_entry = staticmethod(pair_bias_attention)
            except Exception as e:                               # noqa: BLE001 -- the shared core absent from this interpreter: the per-sample flash route, named
                self.batched_attn = "flash_bias_attn:per_sample"; self.batched_attn_event = "apb_core_import:%s" % type(e).__name__

    def _capturing(self):
        if self.apb_graph is not None:
            return bool(self.apb_graph)
        try:
            return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
        except Exception:                                        # noqa: BLE001
            return False

    def _attn_provider(self, q, k, v, g, bias_i, S, N):
        """All samples in one launch through the provider's row for `apb_word`: q/k/v/gate as [S, N, h, d] views of the QKVG rows, the block's
        (mask-folded) bias [1, h, N, N]; the row's Selection is decided once per call class (S, N, eager | graph) and reused (no table walk per
        call). Returns [S*N, c], or None when the word's row refused this call class by name (recorded once; apb_attn serves it directly)."""
        KA = FusedDiT._apb_provider
        if KA is None:
            from opt_core.kernels import apb as KA                   # noqa: N811 -- the shared core's provider face
            FusedDiT._apb_provider = KA
        c, h, d = self.c, self.h, self.d
        if not hasattr(self, "_apb_sel"):
            self._apb_sel = {}
        cap = self._capturing()
        key = (S, N, str(self.apb_word), "graph" if cap else "eager")   # the call class (the word rides in it: an instance whose word changes re-resolves)
        sel = self._apb_sel.get(key)
        if isinstance(sel, str):                                 # refused by name for this class before
            return None
        q4, k4, v4, g4 = (t.view(S, N, h, d) for t in (q, k, v, g))
        cell = KA.cell_word("dit", h, d) or KA.cell_word("pf", h, d)
        try:
            o, sel2 = KA.pair_bias_attention(q4, k4, v4, bias_i.reshape(1, h, N, N), None, gate=g4, word=self.apb_word, layout="snhd", cell=cell, capture=cap, selection=sel)
        except KA.Refusal as r:
            self._apb_sel[key] = "%s:%s" % (getattr(r, "row", None) or self.apb_word, str(getattr(r, "kind", r)).split(" ")[0][:60])
            self.batched_attn_event = "apb_word_refused:%s:S%d:N%d:%s:%s" % (self.apb_word, S, N, key[3], self._apb_sel[key])
            return None
        if sel is None:
            self._apb_sel[key] = sel2
        self.batched_attn = KA.arm_word(sel2.row, sel2.variant)   # the arm that served (apb_attn | fpf_apb | l3a[:c] | sdpa:auto ...): the LEVER line's dtk_attn
        return o.reshape(S * N, c)

    def apb_rows(self):
        """{call class 'S<samples>:N<tokens>:<word>:<eager|graph>': the arm the provider served | '<row>:<kind>' refused by name} of this instance."""
        KA = FusedDiT._apb_provider
        return {"S%d:N%d:%s:%s" % k: (v if isinstance(v, str) else (KA.arm_word(v.row, v.variant) if KA is not None else "?")) for k, v in getattr(self, "_apb_sel", {}).items()}

    def _attn_batched(self, qkvg, q, k, bias_i, key_mask, S, N):
        """[S*N, c] gated attention output in the compute dtype. q / k: [S*N, c] row views (the QKVG slices, or the kq-normed copies)."""
        c, cd, h, d = self.c, self.cd, self.h, self.d
        v = qkvg[:, 2 * c:3 * c]; g = qkvg[:, 3 * c:4 * c]
        if self.apb_word is not None:                            # the provider by the caller's tier word (the caller folded the key mask into bias_i)
            try:
                o = self._attn_provider(q, k, v, g, bias_i, S, N)
                if o is not None:
                    return o
                direct = "apb_attn" if FusedDiT._apb_entry is not None else "flash_bias_attn:per_sample"   # this call class refused by name: the direct entry serves it
            except Exception as e:                               # noqa: BLE001 -- the provider absent or its row failed: apb_attn directly for the rest of the process, named once
                if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                    raise
                low = str(e).lower()
                if isinstance(e, getattr(torch, "OutOfMemoryError", MemoryError)) or "out of memory" in low:
                    raise
                self.batched_attn_event = "apb_word_failed:%s:%s:%s" % (self.apb_word, type(e).__name__, (str(e).splitlines() or [""])[0][:100])
                self.apb_word = None; direct = "apb_attn" if FusedDiT._apb_entry is not None else "flash_bias_attn:per_sample"
            if not str(self.batched_attn).startswith("flash_bias_attn"):
                self.batched_attn = direct
        if self.batched_attn == "apb_attn":
            try:
                return FusedDiT._apb_entry(q, k, v, bias_i, key_mask, gate=g, num_samples=S, num_heads=h, layout="rows")
            except Exception as e:                               # noqa: BLE001 -- refused (alignment / dtype / device) or failed: the per-sample flash route for the rest of the process, named once
                if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                    raise                                        # under a whole-step capture the error is the capture's (the graph steps aside; the eager retry decides here)
                low = str(e).lower()
                if isinstance(e, getattr(torch, "OutOfMemoryError", MemoryError)) or "out of memory" in low:
                    raise
                self.batched_attn = "flash_bias_attn:per_sample"
                self.batched_attn_event = "apb_refused:%s:%s" % (type(e).__name__, (str(e).splitlines() or [""])[0][:120])
        out = torch.empty((S * N, c), device=qkvg.device, dtype=cd)
        for j in range(S):
            r0, r1 = j * N, (j + 1) * N
            K.flash_bias_attn(q[r0:r1].view(N, h, d).permute(1, 0, 2), k[r0:r1].view(N, h, d).permute(1, 0, 2), v[r0:r1].view(N, h, d).permute(1, 0, 2),
                              bias=bias_i, key_mask=key_mask, gate=g[r0:r1], out=out[r0:r1], out_dtype=cd)
        return out

    def block_batched(self, i, a, S_ln, S_raw, bias_i, key_mask, S, N):
        c, cd = self.c, self.cd
        B = self.blk[i]
        o4 = i * 4 * c; o2 = i * 2 * c
        a_hat = K.ln_modulate(a, scale=S_ln[:, o4:o4 + c], shift=S_ln[:, o4 + c:o4 + 2 * c], out_dtype=cd, rms=self.rms, eps=self.eps_a, mod_period=N)
        qkvg = F.linear(a_hat, B["W_qkvg"], B["b_qkvg"])                                  # [S*N, 4c]
        if B.get("ln_q_w") is not None:      # RF3 kq_norm
            q = K.ln_modulate(qkvg[:, 0:c], weight=B["ln_q_w"], bias=B["ln_q_b"], eps=self.eps_kq, out_dtype=cd)
            k = K.ln_modulate(qkvg[:, c:2 * c], weight=B["ln_k_w"], bias=B["ln_k_b"], eps=self.eps_kq, out_dtype=cd)
        else:
            q = qkvg[:, 0:c]; k = qkvg[:, c:2 * c]
        o = self._attn_batched(qkvg, q, k, bias_i, key_mask, S, N)                        # [S*N, c]
        o = F.linear(o, B["Wo"])
        a1 = K.gate_residual(o, gate=S_raw[:, o2:o2 + c], res=a, out_dtype=torch.float32, gate_period=N)          # a + g1 * attn
        t_in = a if self.structure == "rf3" else a1
        a_hat = K.ln_modulate(t_in, scale=S_ln[:, o4 + 2 * c:o4 + 3 * c], shift=S_ln[:, o4 + 3 * c:o4 + 4 * c], out_dtype=cd, rms=self.rms, eps=self.eps_a, mod_period=N)
        ab = F.linear(a_hat, B["W12"])                                                    # [S*N, 2*hid]
        hb = K.swiglu(ab)
        o = F.linear(hb, B["W3"])
        return K.gate_residual(o, gate=S_raw[:, o2 + c:o2 + 2 * c], res=a1, out_dtype=torch.float32, gate_period=N)

    def forward_batched(self, a, s, bias_laid, key_mask=None):
        S, N, c = a.shape
        if self.batched_route is None:
            self._decide_batched_route()
        if not self.batched_route.startswith("periodic_rows"):  # the serial schedule per sample (same kernels and bits as S serial calls)
            return torch.stack([FusedDiT.forward(self, a[j].contiguous(), s, bias_laid, key_mask) for j in range(S)], 0)
        a = a.reshape(S * N, c)
        if not a.is_contiguous():
            a = a.contiguous()
        S_ln, S_raw = self.cond(s)                               # ONCE per step, [N, .]: shared by the S samples
        for i in range(len(self.blk)):
            a = self.block_batched(i, a, S_ln, S_raw, bias_laid[i], key_mask, S, N)
        return a.view(S, N, c)


# ----------------------------------------------------------------------------- atom transformer (sequence-local attention)
def make_windows(NA, NQ=32, NK=128, dev="cuda"):
    nb = (NA + NQ - 1) // NQ
    q_idx = torch.arange(nb * NQ, device=dev).view(nb, NQ)                      # padded query slots (>= NA -> invalid)
    key0 = torch.arange(nb, device=dev) * NQ + NQ // 2 - NK // 2
    k_idx = key0[:, None] + torch.arange(NK, device=dev)[None, :]               # [nb, NK], may be <0 or >= NA
    k_valid = (k_idx >= 0) & (k_idx < NA)
    q_valid = q_idx < NA
    # graph-safe integer index lists for the final scatter (no boolean-mask indexing -> no host sync)
    flat_valid = torch.arange(NA, device=dev)                                    # valid query slots == atom index (slots are contiguous)
    return q_idx.clamp(max=NA - 1), q_valid, k_idx.clamp(0, NA - 1), k_valid, flat_valid


def atom_block_ref(fam, a_q, s_qq, W, blk, bias_blk, q_valid, k_idx, k_valid, cd, n_head):
    """Stock formulation (xfold/AF3 layout): residual stream in queries layout a_q [nb,32,c]; keys re-gathered from the queries-layout
    stream every block ([nb,128,c], each atom duplicated 4x) exactly like atom_layout.convert(queries_to_keys, ...); dense per-block attention."""
    nb, NQ, c = a_q.shape
    NK = k_idx.shape[1]
    d = c // n_head
    hp = _hp(cd)
    b = W.blocks[blk]
    a_k = a_q.reshape(-1, c)[k_idx] * k_valid[..., None]                       # [nb,128,c]
    s_kk = s_qq.reshape(-1, c)[k_idx] * k_valid[..., None]
    aq_n = _adaln_ref(a_q.reshape(-1, c), s_qq.reshape(-1, c), b["ln_s_w1"], b["W_scale1"], b["b_scale1"], b["W_shift1"], cd).view(nb, NQ, c)
    ak_n = _adaln_ref(a_k.reshape(-1, c), s_kk.reshape(-1, c), b["ln_s_w1"], b["W_scale1"], b["b_scale1"], b["W_shift1"], cd).view(nb, NK, c)
    q = F.linear(aq_n, b["Wq"].to(cd), b["bq"].to(cd)).view(nb, NQ, n_head, d)
    k = F.linear(ak_n, b["Wk"].to(cd)).view(nb, NK, n_head, d)
    v = F.linear(ak_n, b["Wv"].to(cd)).view(nb, NK, n_head, d)
    mask_bias = (~k_valid).to(hp)[:, None, None, :] * (-1e9)                      # [nb,1,1,NK]
    bias = bias_blk.permute(1, 0, 2, 3)                                            # [nb,H,NQ,NK]
    if fam in ("af3t_apb", "protenix"):
        o = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
                                           attn_mask=(bias.to(hp) + mask_bias).to(q.dtype), scale=1.0 / math.sqrt(d))
        o = o.permute(0, 2, 1, 3)
    elif fam == "of3":
        lg = torch.einsum("bqhc,bkhc->bhqk", q * (1.0 / math.sqrt(d)), k) + (bias.to(hp) + mask_bias).to(q.dtype)
        wts = torch.softmax(lg, dim=-1)
        o = torch.einsum("bhqk,bkhc->bqhc", wts, v)
    else:
        lg = torch.einsum("bqhc,bkhc->bhqk", q * (1.0 / math.sqrt(d)), k).to(hp) + bias.to(hp) + mask_bias
        wts = torch.softmax(lg, dim=-1).to(v.dtype)
        o = torch.einsum("bhqk,bkhc->bqhc", wts, v)
    o = o.reshape(nb, NQ, c)
    g = torch.sigmoid(F.linear(aq_n, b["Wg"].to(cd)))
    o = F.linear(g * o, b["Wo"].to(cd))
    zg = torch.sigmoid(F.linear(s_qq.to(cd), b["W_zero1"].to(cd), b["b_zero1"].to(cd)))
    a_q = a_q + (zg * o).to(a_q.dtype) * q_valid[..., None]
    aq_n = _adaln_ref(a_q.reshape(-1, c), s_qq.reshape(-1, c), b["ln_s_w2"], b["W_scale2"], b["b_scale2"], b["W_shift2"], cd).view(nb, NQ, c)
    hdn = F.silu(F.linear(aq_n, b["W1"].to(cd))) * F.linear(aq_n, b["W2"].to(cd))
    o = F.linear(hdn, b["W3"].to(cd))
    zg = torch.sigmoid(F.linear(s_qq.to(cd), b["W_zero2"].to(cd), b["b_zero2"].to(cd)))
    a_q = a_q + (zg * o).to(a_q.dtype) * q_valid[..., None]
    return a_q


def atom_transformer_ref(fam, x, s_q, W, bias_blks, win, cd):
    """x, s_q: flat [NA, c]; returns flat [NA, c]. Gather to queries layout once, scatter back once (as the models do)."""
    q_idx, q_valid, k_idx, k_valid, flat_valid = win
    NA, c = x.shape
    a_q = x[q_idx] * q_valid[..., None]
    s_qq = s_q[q_idx] * q_valid[..., None]
    for i in range(len(W.blocks)):
        b = W.blocks[i]; b["_nhead"] = W.h
        a_q = atom_block_ref(fam, a_q, s_qq, W, i, bias_blks[i], q_valid, k_idx, k_valid, cd, W.h)
    out = a_q.reshape(-1, c).index_select(0, flat_valid)                        # slots 0..NA-1 are the atoms
    return out


class FusedAtomTransformer(FusedDiT):
    """Same fused block as the token transformer but on the flat atom list with the block-sparse window kernel;
    k/v/q projections and the k-side AdaLN run ONCE per atom instead of on the 4x-duplicated gathered keys."""
    def block(self, i, a, S_ln, S_raw, bias_blk_i, atom_mask=None):
        c, cd = self.c, self.cd
        N = a.shape[0]
        B = self.blk[i]
        o4 = i * 4 * c; o2 = i * 2 * c
        a_hat = K.ln_modulate(a, scale=S_ln[:, o4:o4 + c], shift=S_ln[:, o4 + c:o4 + 2 * c], out_dtype=cd)
        qkvg = F.linear(a_hat, B["W_qkvg"], B["b_qkvg"])
        q = qkvg[:, 0:c].view(N, self.h, self.d).permute(1, 0, 2)
        k = qkvg[:, c:2 * c].view(N, self.h, self.d).permute(1, 0, 2)
        v = qkvg[:, 2 * c:3 * c].view(N, self.h, self.d).permute(1, 0, 2)
        o = K.window_attn(q, k, v, bias_blk_i, atom_mask=atom_mask, gate=qkvg[:, 3 * c:4 * c], out_dtype=cd)
        o = F.linear(o, B["Wo"])
        a = K.gate_residual(o, gate=S_raw[:, o2:o2 + c], res=a, out_dtype=torch.float32)
        a_hat = K.ln_modulate(a, scale=S_ln[:, o4 + 2 * c:o4 + 3 * c], shift=S_ln[:, o4 + 3 * c:o4 + 4 * c], out_dtype=cd)
        ab = F.linear(a_hat, B["W12"])
        hb = K.swiglu(ab)
        o = F.linear(hb, B["W3"])
        a = K.gate_residual(o, gate=S_raw[:, o2 + c:o2 + 2 * c], res=a, out_dtype=torch.float32)
        return a
