"""opendde_fpf_ditfast.dit_fast — lever `dit_fused` (+ precision lever `dit_lowp`): the sampler's 24-block token DiffusionTransformer
(diffusion_module.diffusion_transformer) as ONE fused forward.

Per block: AdaLN row kernel -> ONE q|k|v|g GEMM [768->3072] (+q bias) -> the provider-served pair-bias attention `odde_apb_bind.dit_packed` (lever dit_attn_apb:
bias shared over the S samples in-kernel, sigmoid gating fused; REQUIRED by name, there is no other attention path in this module) -> o GEMM ->
[sigmoid-gate * x + fp32 residual + the next AdaLN] row kernel -> ONE a1|a2 GEMM [768->3072] -> SwiGLU row kernel -> b GEMM -> [gate + residual +
the next block's AdaLN] row kernel. The conditioning side of ALL blocks (per block in stock: AdaLN linear_s / linear_nobias_s x2, two output-gate
Linears = 6 x Linear(384->768)) runs as TWO fp32 GEMMs per call with the AdaLN LayerNorm(s) affine folded into the packed columns (s is constant
through the stack); rows are addressed modulo the number of s rows, so a sample-deduplicated s ([1, N, 384], lever cond_dedupe) broadcasts.
The token pair bias is served through the DiT hoist's slot protocol when the hoist is installed (levers/DITFAST odde_addon, `cached(name, producer)`: produced once per item
at the eager record step, replayed inside the sampler graph, poison/recheck by the hoist's own protocol) under slot names `tok.fastbias<i>.float32`
(fp32, 8-aligned row pitch as the fpf_apb row's TMA bias path wants); without the hoist the producer runs per call (stock cost).
Precision (lever dit_lowp = the activation/operand dtype of the a-path GEMMs and of the attention): off -> fp32 storage, cuBLAS under the stock
TF32 flag (upstream's enable_tf32 default: the same GEMM precision class as stock's token-block Linears), attention operands per the provider's
dit cell for the call class (lever dit_attn_apb: the cell picks the operand form); bf16 | fp16 -> those operands in bf16 / fp16 with fp32 accumulation
(fp16: torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is set False). In every cell the residual stream, LayerNorm and softmax
statistics, the conditioning GEMMs and everything outside the 24 token blocks stay fp32.
Graph-capture safe: no host syncs, no data-dependent shapes, no RNG, current-stream launches only. Numerics class TOLERANCE (GEMM shapes change).
"""
import math, os
import torch
import torch.nn.functional as F
from ._plumbing import LeverRefused, StepAside, count_aside, diffusion_module, say, register_exit, odde_hoist
from opt_core.kernels.apb.ditfast import kernels as K                       # the row kernels are the shared core's (opt_core)

NAME = "dit_fused"
REPORT = {"installed": False}
COUNTS = {"calls": 0, "lowp_calls": 0, "asides": 0}                                                # odde: stack forwards entered / of which with 16-bit operands (opendde_opt/ran.py)
C_A, C_S, H, CD, NF = 768, 384, 16, 48, 2


def _hoist():                                                                          # odde: OpenDDE's hoist object
    return odde_hoist()


class FastTokenStack:
    def __init__(self, dm, act_dtype):
        from opendde.model.modules.transformer import DiffusionTransformerBlock       # odde
        self.dm = dm
        self.dt = dm.diffusion_transformer
        self.blocks = list(self.dt.blocks)
        self.nb = len(self.blocks)
        self.act = act_dtype
        self.fused = True
        self.scale = 1.0 / math.sqrt(CD)
        # ---- attention: lever dit_attn_apb (the provider-served token site, odde_apb_bind), REQUIRED by name — this module carries no attention kernel of its own
        try:
            import odde_apb_bind as AB                                       # odde: levers/SAMPLER (on the path: odde_sampler put it there)
        except Exception as e:
            raise LeverRefused(f"{NAME}: requires lever dit_attn_apb — odde_apb_bind not importable ({e!r})")
        wd = AB.word("dit")
        if wd is None or AB.KA.split_word(wd)[0] in ("exact", "faithful"):        # odde: the kit's word of lever dit_attn_apb (fast | big | <row>)
            raise LeverRefused(f"{NAME}: requires lever dit_attn_apb (ODDE_DIT_ATTN=fast | big | <row>; got {os.environ.get('ODDE_DIT_ATTN', '')!r}) — the fused block attention is the provider-served site")
        self._apb = AB
        self.attn_name = f"opt_core.kernels.apb by word {wd} (binding {AB.NAME} {AB.__version__}; install-time selection: {AB.COUNTS['dit'].get('selection')})"
        self.bias_dtype = torch.float32                                   # the attention rows read the fp32 hoisted bias (the dit cells are keyed on the fp32 statement)
        self.calls = 0
        self.range = None                                                # optional abs-max recorder (enable_range()); off unless a caller turns it on
        self.range_names = ["adaln_att(an)", "qkvg", "attn_out_gated(og)", "o_proj(x)", "adaln_ff(an2)", "a1a2(h12)", "swiglu(hb)", "b_proj(xb)", "pair_bias", "residual_A(fp32, never cast)"]
        self.local_bias_cache = None
        dev = next(self.dt.parameters()).device
        for i, blk in enumerate(self.blocks):
            if not isinstance(blk, DiffusionTransformerBlock):
                raise LeverRefused(f"{NAME}: block {i} is {type(blk).__name__}")
            apb, ctb = blk.attention_pair_bias, blk.conditioned_transition_block
            att = apb.attention
            if not apb.has_s or apb.cross_attention_mode or att.linear_g is None or att.num_heads != H or att.c_hidden != CD:
                raise LeverRefused(f"{NAME}: block {i} is not the token-stack shape (has_s={apb.has_s} cross={apb.cross_attention_mode} heads={att.num_heads} c={att.c_hidden})")
            if ctb.n != NF or ctb.c_a != C_A or ctb.c_s != C_S:
                raise LeverRefused(f"{NAME}: block {i} transition n={ctb.n} c_a={ctb.c_a} c_s={ctb.c_s}")
            if att.linear_q.bias is None or att.linear_k.bias is not None or att.linear_o.bias is not None:
                raise LeverRefused(f"{NAME}: block {i} bias layout differs from the pinned stock")
        with torch.no_grad():
            W = lambda lin: lin.weight.detach().float()
            self.w_qkvg, self.b_qkvg, self.w_o, self.w_a12, self.w_b = [], [], [], [], []
            ws_cols, bs_cols, wg_cols, bg_cols = [], [], [], []
            z0 = torch.zeros(C_A, device=dev)
            for blk in self.blocks:
                apb, ctb = blk.attention_pair_bias, blk.conditioned_transition_block; att = apb.attention
                self.w_qkvg.append(torch.cat([W(att.linear_q), W(att.linear_k), W(att.linear_v), W(att.linear_g)], 0).to(act_dtype).contiguous())
                self.b_qkvg.append(torch.cat([att.linear_q.bias.detach().float(), z0, z0, z0], 0).to(act_dtype).contiguous())
                self.w_o.append(W(att.linear_o).to(act_dtype).contiguous())
                self.w_a12.append(torch.cat([W(ctb.linear_nobias_a1), W(ctb.linear_nobias_a2)], 0).to(act_dtype).contiguous())
                self.w_b.append(W(ctb.linear_nobias_b).to(act_dtype).contiguous())
                wa, wc = apb.layernorm_a.layernorm_s.weight.detach().float(), ctb.adaln.layernorm_s.weight.detach().float()
                ws_cols.append(torch.cat([W(apb.layernorm_a.linear_s) * wa[None, :], W(apb.layernorm_a.linear_nobias_s) * wa[None, :],
                                          W(ctb.adaln.linear_s) * wc[None, :], W(ctb.adaln.linear_nobias_s) * wc[None, :]], 0))
                bs_cols.append(torch.cat([apb.layernorm_a.linear_s.bias.detach().float(), z0, ctb.adaln.linear_s.bias.detach().float(), z0], 0))
                wg_cols.append(torch.cat([W(apb.linear_a_last), W(ctb.linear_s)], 0))
                bg_cols.append(torch.cat([apb.linear_a_last.bias.detach().float(), ctb.linear_s.bias.detach().float()], 0))
            self.w_s = torch.cat(ws_cols, 0).contiguous()      # [nb*4*768, 384] fp32 (TF32 GEMM like stock's own Linear)
            self.b_s = torch.cat(bs_cols, 0).contiguous()
            self.w_g = torch.cat(wg_cols, 0).contiguous()      # [nb*2*768, 384]
            self.b_g = torch.cat(bg_cols, 0).contiguous()
            self.eps_a = float(self.blocks[0].attention_pair_bias.layernorm_a.layernorm_a.eps)
            self.eps_s = float(self.blocks[0].attention_pair_bias.layernorm_a.layernorm_s.eps)
        self.param_bytes = sum(t.numel() * t.element_size() for L in (self.w_qkvg, self.b_qkvg, self.w_o, self.w_a12, self.w_b) for t in L) + \
            sum(t.numel() * 4 for t in (self.w_s, self.b_s, self.w_g, self.b_g))

    # ---------------------------------------------------------------- pair bias for block i (hoist protocol when installed)
    def _bias(self, i, z, enable_efficient_fusion, extra_attn_bias=None):                 # odde: + OpenDDE's extra_attn_bias (structural pair attention bias)
        apb = self.blocks[i].attention_pair_bias
        act = self.bias_dtype

        def producer():
            from opendde.model.utils import permute_final_dims                               # odde
            if enable_efficient_fusion:
                weight = (apb.linear_nobias_z.weight * apb.layernorm_z.weight[None, :])[:, :, None, None]
                bias = F.conv2d(z, weight)                                   # stock: [.., 16, N, N] from the normalized, permuted z
            else:
                bias = permute_final_dims(apb.linear_nobias_z(apb.layernorm_z(z)), [2, 0, 1])
            eb = extra_attn_bias                                                              # odde: verbatim transformer.py AttentionPairBias.standard_multihead_attention:1330-1335
            if eb is not None:
                while len(eb.shape) < len(bias.shape) - 1:
                    eb = eb.unsqueeze(dim=0)
                if len(eb.shape) == len(bias.shape) - 1:
                    eb = eb.unsqueeze(dim=-3)
                bias = bias + eb.to(dtype=bias.dtype, device=bias.device)
            while bias.dim() > 3 and bias.shape[0] == 1:                                      # odde: [16, N, N] for the kernel (leading broadcast dims dropped)
                bias = bias[0]
            if bias.dim() != 3:                                                                # a per-sample pair bias (the kernel shares one [16, N, N] bias over the samples): this call
                raise StepAside("per_sample_bias")                                             # steps aside by name to the stock forward (caught in forward)
            n = bias.shape[-1]
            if bias.dtype == act and n % 8 == 0 and bias.is_contiguous():
                return bias
            pitch = (n + 7) // 8 * 8                                         # 8-element row pitch: the SDPA kernels take the view without a re-pad
            buf = torch.empty(tuple(bias.shape[:-1]) + (pitch,), dtype=act, device=bias.device)[..., :n]
            buf.copy_(bias)
            return buf
        Hh = _hoist()
        if Hh is not None and getattr(Hh, "installed", False):
            return Hh.cached(f"tok.fastbias{i}.{str(act).split('.')[-1]}", producer)
        if self.local_bias_cache is not None:                                # a local per-block cache standing in for the hoist (never set in the kit path)
            if i not in self.local_bias_cache:
                self.local_bias_cache[i] = producer()
            return self.local_bias_cache[i]
        return producer()

    # ---------------------------------------------------------------- the stack
    def _rec(self, j, t):
        if self.range is not None:
            torch.maximum(self.range[j], t.detach().abs().amax().float(), out=self.range[j])

    def enable_range(self):
        """Optional recorder: running abs-max of every low-precision operand, device-side (graph-safe); used when sizing the fp16 word."""
        self.range = torch.zeros(len(self.range_names), device=self.w_o[0].device)
        return self.range

    def forward(self, a, s, z, n_queries=None, n_keys=None, inplace_safe=False, chunk_size=None, enable_efficient_fusion=False, extra_attn_bias=None):
        """Call forms the fused stack does not serve STEP ASIDE BY NAME — counted under COUNTS (asides, aside_<word>) and served by
        the module's stock forward (DiffusionTransformer.forward, the class method this install shadows) with the original arguments; never a raise."""
        kw = dict(n_queries=n_queries, n_keys=n_keys, inplace_safe=inplace_safe, chunk_size=chunk_size, enable_efficient_fusion=enable_efficient_fusion)
        if extra_attn_bias is not None:
            kw["extra_attn_bias"] = extra_attn_bias
        word = None
        if n_queries is not None or n_keys is not None:                                        # local (windowed) attention requested on the token stack
            word = "local_attention"
        else:
            lead = a.shape[:-3]; S, N = int(a.shape[-3]), int(a.shape[-2]); Ss = int(s.shape[-3]) if s.dim() >= 3 else -1
            if a.shape[-1] != C_A or s.shape[-1] != C_S or s.dim() < 3 or s.shape[-2] != N or (Ss != S and Ss != 1) or math.prod(lead) != 1:
                word = "shapes"
        if word is None:
            try:
                return self._fused(a, s, z, inplace_safe=inplace_safe, chunk_size=chunk_size, enable_efficient_fusion=enable_efficient_fusion, extra_attn_bias=extra_attn_bias)
            except StepAside as e:                                                             # found past the entry checks (the pair bias layout): undo this call's counters
                word = e.word; self.calls -= 1; COUNTS["calls"] -= 1
                if self.act != torch.float32:
                    COUNTS["lowp_calls"] -= 1
        count_aside(COUNTS, word)
        return type(self.dt).forward(self.dt, a, s, z, **kw)

    @torch.autocast("cuda", enabled=False)                      # odde: the stack runs its own precision plan (fp32 stream, low-precision operands only per its word) whatever the ambient autocast is (upstream runs predict() under torch.autocast); inputs are cast to fp32 at entry below
    def _fused(self, a, s, z, inplace_safe=False, chunk_size=None, enable_efficient_fusion=False, extra_attn_bias=None):   # odde: + extra_attn_bias
        self.calls += 1; COUNTS["calls"] += 1
        if self.act != torch.float32:
            COUNTS["lowp_calls"] += 1                                                          # odde: lever dit_lowp's counter
        lead = a.shape[:-3]; S, N = int(a.shape[-3]), int(a.shape[-2]); Ss = int(s.shape[-3]); self.last_s_rows = (Ss, N)   # (1, N) when cond_dedupe serves one conditioning row set, (5, N) stock
        M, Ns = S * N, Ss * N
        act = self.act
        A = a.reshape(M, C_A).float().contiguous().clone()                   # fp32 residual stream (own storage: updated in place below)
        s2 = s.reshape(Ns, C_S).float()
        s_hat = F.layer_norm(s2, (C_S,), None, None, self.eps_s)             # LN(s) without affine (the affine is folded into w_s)
        XS = torch.addmm(self.b_s, s_hat, self.w_s.t())                       # [Ns, nb*3072] fp32: AdaLN scale|shift for both AdaLNs of every block
        GS = torch.addmm(self.b_g, s2, self.w_g.t())                          # [Ns, nb*1536] fp32: attention gate | transition gate of every block
        biases = [self._bias(i, z, enable_efficient_fusion, extra_attn_bias) for i in range(self.nb)]   # odde
        cond = []
        for i in range(self.nb):
            o1 = i * 4 * C_A; o2 = i * 2 * C_A
            cond.append((XS[:, o1:o1 + C_A], XS[:, o1 + C_A:o1 + 2 * C_A], XS[:, o1 + 2 * C_A:o1 + 3 * C_A], XS[:, o1 + 3 * C_A:o1 + 4 * C_A],
                         GS[:, o2:o2 + C_A], GS[:, o2 + C_A:o2 + 2 * C_A]))
        an = K.adaln(A, cond[0][0], cond[0][1], act, self.eps_a)             # block 0 attention AdaLN: [M, 768] act
        for i in range(self.nb):
            x1, x2, y1, y2, g_att, g_ff = cond[i]
            # --- attention half (an = AdaLN_i(A) already computed)
            qkvg = torch.addmm(self.b_qkvg[i], an, self.w_qkvg[i].t())        # [M, 3072] act: q|k|v|g (+ q bias)
            og = self._apb.dit_packed(qkvg, biases[i], S, N, out_dtype=act)   # [M, 768] act: attention (bias shared over S) * sigmoid(g) through the provider's row for this class (lever dit_attn_apb)
            self._rec(0, an); self._rec(1, qkvg); self._rec(2, og); self._rec(8, biases[i])
            x = torch.mm(og, self.w_o[i].t())                                 # [M, 768] act
            self._rec(3, x)
            an2 = K.resgate_adaln(g_att, x, A, y1, y2, act, self.eps_a)       # A += sigmoid(linear_a_last(s)) * x ; an2 = transition AdaLN(A)
            # --- transition half
            h12 = torch.mm(an2, self.w_a12[i].t())                            # [M, 3072] act
            hb = K.swiglu(h12, act)                                           # [M, 1536] act
            self._rec(4, an2); self._rec(5, h12); self._rec(6, hb)
            xb = torch.mm(hb, self.w_b[i].t())                                # [M, 768] act
            self._rec(7, xb); self._rec(9, A)
            if i + 1 < self.nb:
                an = K.resgate_adaln(g_ff, xb, A, cond[i + 1][0], cond[i + 1][1], act, self.eps_a)   # A += gate*xb ; next block's attention AdaLN
            else:
                K.resgate(g_ff, xb, A, out=A)                                 # last block: A += sigmoid(linear_s(s)) * linear_b(..)
        return A.view(*lead, S, N, C_A)


def _strip_token_overrides(dm):
    n = 0
    for blk in dm.diffusion_transformer.blocks:
        apb, ctb = blk.attention_pair_bias, blk.conditioned_transition_block
        for obj, attr in ((blk, "forward"), (apb.layernorm_a, "forward"), (ctb.adaln, "forward"), (apb.attention, "_wrap_up"), (apb, "standard_multihead_attention")):
            if attr in obj.__dict__:
                del obj.__dict__[attr]; n += 1
    return n


def install(model):
    """Kit hook entry (odde_sampler.install, after the DiT hoist and lever dit_attn_apb are installed). Env: ODDE_DIT_FUSED=1 engages
    dit_fused; ODDE_DIT_LOWP = off | bf16 | fp16 is lever dit_lowp's word. Returns the report dict; prints the DITFAST marker line."""
    dm = diffusion_module(model)
    dt = dm.diffusion_transformer
    if "forward" in dt.__dict__ and getattr(dt.__dict__["forward"], "_fpf_lever", None) == NAME:
        return REPORT
    if os.environ.get("ODDE_DIT_FUSED", "0") != "1":                                        # odde
        raise LeverRefused(f"{NAME}: install called with ODDE_DIT_FUSED={os.environ.get('ODDE_DIT_FUSED')!r} (expected 1)")
    lowp = os.environ.get("ODDE_DIT_LOWP", "off")
    if lowp not in ("off", "bf16", "fp16"):
        raise LeverRefused(f"dit_lowp: ODDE_DIT_LOWP={lowp!r} unknown (off | bf16 | fp16)")
    act = {"off": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[lowp]
    if act == torch.float16:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False   # fp16 GEMMs accumulate in fp32 (stock runs no fp16 GEMM; nothing else is affected)
    stripped = _strip_token_overrides(dm)                                     # the DiT hoist's instance overrides on the 24 token blocks (subsumed sites; census re-bases)
    st = FastTokenStack(dm, act); _STACKS.append(st)
    fwd = lambda *a, **k: st.forward(*a, **k)
    fwd._fpf_lever = NAME
    dt.forward = fwd
    dt._opendde_fpf_ditfast = st
    hoist = _hoist_installed()
    REPORT.update(installed=True, blocks=st.nb, act=str(st.act), lowp=lowp, attention=st.attn_name, bias_slots="hoist" if hoist else "local",
                  strip=stripped, packed_param_MB=round(st.param_bytes / 2**20, 1))
    register_exit(NAME, lambda: {"stack_calls": st.calls, "blocks_x_calls": st.calls * st.nb, "act": str(st.act), "lowp": lowp, "attention": st.attn_name,
                                 "s_rows_last": getattr(st, "last_s_rows", None), "bias_slots": "hoist" if _hoist_installed() else "local"})
    say(f"DITFAST:on(blocks={st.nb} act={st.act} lowp={lowp} attn={st.attn_name} bias_slots={'hoist' if hoist else 'local'} strip={stripped} qkvg=768x3072 a12=768x3072 packed={REPORT['packed_param_MB']}MB)")
    return REPORT


_STACKS = []


def _hoist_installed():                                                                # odde
    return odde_hoist() is not None

