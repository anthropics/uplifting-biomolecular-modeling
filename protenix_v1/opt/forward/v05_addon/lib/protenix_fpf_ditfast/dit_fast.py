"""protenix_fpf_ditfast.dit_fast — lever `dit_fused` (+ precision lever `dit_lowp`): the sampler's 24-block token DiffusionTransformer
(diffusion_module.diffusion_transformer) as ONE fused forward.

Per block: AdaLN row kernel -> ONE q|k|v|g GEMM [768->3072] (+q bias) -> fused pair-bias attention `fpf_apb.dit_apb` (lever dit_attn:
bias shared over the S samples in-kernel, sigmoid gating fused; REQUIRED by name, there is no other attention path in this module) -> o GEMM ->
[sigmoid-gate * x + fp32 residual + the next AdaLN] row kernel -> ONE a1|a2 GEMM [768->3072] -> SwiGLU row kernel -> b GEMM -> [gate + residual +
the next block's AdaLN] row kernel. The conditioning side of ALL blocks (per block in stock: AdaLN linear_s / linear_nobias_s x2, two output-gate
Linears = 6 x Linear(384->768)) runs as TWO fp32 GEMMs per call with the AdaLN LayerNorm(s) affine folded into the packed columns (s is constant
through the stack); rows are addressed modulo the number of s rows, so a sample-deduplicated s ([1, N, 384], lever cond_dedupe) broadcasts.
The token pair bias is served through the kit's DiT-hoist slot protocol when the hoist is installed (dit_hoist.HOIST.cached: produced once per item
at the eager record step, replayed inside the sampler graph, poison/recheck by the hoist's own protocol) under slot names `tok.fastbias<i>.float32`
(fp32, 8-aligned row pitch as fpf_apb's TMA bias path wants); without the hoist the producer runs per call (stock cost).
Precision (lever dit_lowp = the activation/operand dtype of the a-path GEMMs and of the attention): off -> fp32 storage, cuBLAS under the stock
TF32 flag (Protenix's own enable_tf32 default: the same GEMM precision class as stock's token-block Linears), attention operands per fpf_apb's
dit_attn fp32 cell (tf32x3, or fp16 when its dit_attn_fp16 lever is on); bf16 | fp16 -> those operands in bf16 / fp16 with fp32 accumulation
(fp16: torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is set False). In every cell the residual stream, LayerNorm and softmax
statistics, the conditioning GEMMs and everything outside the 24 token blocks stay fp32.
Graph-capture safe: no host syncs, no data-dependent shapes, no RNG, current-stream launches only. Numerics class TOLERANCE (GEMM shapes change).
"""
import math, os
import torch
import torch.nn.functional as F
from ._plumbing import LeverRefused, diffusion_module, say, register_exit
from opt_core.kernels.apb.ditfast import kernels as K

NAME = "dit_fused"
REPORT = {"installed": False}
C_A, C_S, H, CD, NF = 768, 384, 16, 48, 2


def _hoist():
    try:
        import dit_hoist
        return dit_hoist.HOIST
    except Exception:
        return None


class FastTokenStack:
    def __init__(self, dm, act_dtype):
        from protenix.model.modules.transformer import DiffusionTransformerBlock
        self.dm = dm
        self.dt = dm.diffusion_transformer
        self.blocks = list(self.dt.blocks)
        self.nb = len(self.blocks)
        self.act = act_dtype
        self.fused = True
        self.scale = 1.0 / math.sqrt(CD)
        # ---- attention provider: lever dit_attn (fpf_apb), REQUIRED by name — this module carries no attention kernel of its own
        if os.environ.get("PTX_DIT_ATTN", "0") != "1":
            raise LeverRefused(f"{NAME}: requires lever dit_attn (PTX_DIT_ATTN=1) — the fused block attention is protenix_fpf_apb.dit_apb")
        try:
            import protenix_fpf_apb as fpf_apb
            from protenix_fpf_apb import install as _apb_install
        except Exception as e:
            raise LeverRefused(f"{NAME}: requires lever dit_attn — protenix_fpf_apb not importable ({e!r})")
        st_ = fpf_apb.STATE.get("dit_attn", {})
        prec = "fp16" if (fpf_apb.STATE.get("dit_attn_fp16", {}).get("on") or os.environ.get("PTX_DIT_ATTN_FP16", "0") == "1") else "fp32"
        cell = dict(st_.get("cell") or _apb_install._cell("dit_attn", prec))   # the SAME per-card cell own install uses (opd word for fp32 inputs, bias TMA)
        if cell.get("bias_tma", True) and not getattr(fpf_apb.apb_triton, "_HAS_TMA", True):
            cell["bias_tma"] = False
        self._apb = fpf_apb
        self._apb_opd = cell["opd"]
        self._apb_cfg = None if cell.get("bias_tma", True) else dict(BIAS_TMA=False)
        self.attn_name = f"protenix_fpf_apb.dit_apb[{getattr(fpf_apb, '__version__', '?')}; fp32-input opd={cell['opd']}; bias_tma={cell.get('bias_tma', True)}]"
        self.bias_dtype = torch.float32                                   # fpf_apb reads the hoisted bias in fp32 (the dtype its dit_attn cell takes)
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
    def _bias(self, i, z, enable_efficient_fusion):
        apb = self.blocks[i].attention_pair_bias
        act = self.bias_dtype

        def producer():
            from protenix.model.utils import permute_final_dims
            if enable_efficient_fusion:
                weight = (apb.linear_nobias_z.weight * apb.layernorm_z.weight[None, :])[:, :, None, None]
                bias = F.conv2d(z, weight)                                   # stock: [.., 16, N, N] from the normalized, permuted z
            else:
                bias = permute_final_dims(apb.linear_nobias_z(apb.layernorm_z(z)), [2, 0, 1])
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
        if self.local_bias_cache is not None:                                # stand-alone stand-in for the hoist (never set in the kit path)
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

    def forward(self, a, s, z, n_queries=None, n_keys=None, inplace_safe=False, chunk_size=None, enable_efficient_fusion=False):
        if n_queries is not None or n_keys is not None:
            raise LeverRefused(f"{NAME}: local attention requested on the token stack")
        self.calls += 1
        lead = a.shape[:-3]; S, N = int(a.shape[-3]), int(a.shape[-2]); Ss = int(s.shape[-3]); self.last_s_rows = (Ss, N)   # (1, N) when cond_dedupe serves one conditioning row set, (5, N) stock
        if a.shape[-1] != C_A or s.shape[-1] != C_S or s.shape[-2] != N or (Ss != S and Ss != 1) or math.prod(lead) != 1:
            raise LeverRefused(f"{NAME}: shapes a={tuple(a.shape)} s={tuple(s.shape)}")
        M, Ns = S * N, Ss * N
        act = self.act
        A = a.reshape(M, C_A).float().contiguous().clone()                   # fp32 residual stream (own storage: updated in place below)
        s2 = s.reshape(Ns, C_S).float()
        s_hat = F.layer_norm(s2, (C_S,), None, None, self.eps_s)             # LN(s) without affine (the affine is folded into w_s)
        XS = torch.addmm(self.b_s, s_hat, self.w_s.t())                       # [Ns, nb*3072] fp32: AdaLN scale|shift for both AdaLNs of every block
        GS = torch.addmm(self.b_g, s2, self.w_g.t())                          # [Ns, nb*1536] fp32: attention gate | transition gate of every block
        biases = [self._bias(i, z, enable_efficient_fusion) for i in range(self.nb)]
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
            og = self._apb.dit_apb(qkvg, biases[i], S, N, gate=True, out_dtype=act,
                                   opd=(self._apb_opd if act == torch.float32 else None), cfg=self._apb_cfg)   # [M, 768] act: attention (bias shared over S) * sigmoid(g), fused (lever dit_attn)
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
    """Kit hook entry (called after the model is built and the sampler graph / DiT hoist / attention levers are installed). Env: PTX_DIT_FAST=1 engages
    dit_fused; PTX_DIT_LOWP = off | bf16 | fp16 is lever dit_lowp's word. Returns the report dict; prints the DITFAST marker line."""
    dm = diffusion_module(model)
    dt = dm.diffusion_transformer
    if "forward" in dt.__dict__ and getattr(dt.__dict__["forward"], "_fpf_lever", None) == NAME:
        return REPORT
    if os.environ.get("PTX_DIT_FAST", "0") != "1":
        raise LeverRefused(f"{NAME}: install called with PTX_DIT_FAST={os.environ.get('PTX_DIT_FAST')!r} (expected 1)")
    lowp = os.environ.get("PTX_DIT_LOWP", "off")
    if lowp not in ("off", "bf16", "fp16"):
        raise LeverRefused(f"dit_lowp: PTX_DIT_LOWP={lowp!r} unknown (off | bf16 | fp16)")
    act = {"off": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[lowp]
    if act == torch.float16:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False   # fp16 GEMMs accumulate in fp32 (stock runs no fp16 GEMM; nothing else is affected)
    stripped = _strip_token_overrides(dm)                                     # dit_hoist (or another lever's) instance overrides on the 24 token blocks (subsumed sites; census re-bases)
    st = FastTokenStack(dm, act); _STACKS.append(st)
    fwd = lambda *a, **k: st.forward(*a, **k)
    fwd._fpf_lever = NAME
    dt.forward = fwd
    dt._protenix_fpf_ditfast = st
    hoist = _hoist_installed()
    REPORT.update(installed=True, blocks=st.nb, act=str(st.act), lowp=lowp, attention=st.attn_name, bias_slots="hoist" if hoist else "local",
                  strip=stripped, packed_param_MB=round(st.param_bytes / 2**20, 1))
    register_exit(NAME, lambda: {"stack_calls": st.calls, "blocks_x_calls": st.calls * st.nb, "act": str(st.act), "lowp": lowp, "attention": st.attn_name,
                                 "s_rows_last": getattr(st, "last_s_rows", None), "bias_slots": "hoist" if _hoist_installed() else "local"})
    say(f"DITFAST:on(blocks={st.nb} act={st.act} lowp={lowp} attn={st.attn_name} bias_slots={'hoist' if hoist else 'local'} strip={stripped} qkvg=768x3072 a12=768x3072 packed={REPORT['packed_param_MB']}MB)")
    return REPORT


_STACKS = []


def _hoist_installed():
    try:
        import dit_hoist
        return bool(getattr(dit_hoist.HOIST, "installed", False))
    except Exception:
        return False

