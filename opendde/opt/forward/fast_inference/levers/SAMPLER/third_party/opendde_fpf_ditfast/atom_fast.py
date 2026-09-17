"""opendde_fpf_ditfast.atom_fast — lever `atom_fused`: both atom transformers of the
sampler (diffusion_module.atom_attention_encoder/decoder .atom_transformer, 3 AtomTransformerBlocks each, c_atom=128, 4 heads x 32, 32-query x
128-key local windows) as fused stacks: one entry kernel (AdaLN_a + AdaLN_kv of block 0), per block ONE q|g GEMM [128->256] (+q bias) and ONE
k|v GEMM, the provider-served local attention `odde_apb_bind.atom_views` (lever atom_attn_apb: one launch per call, sigmoid gating fused; REQUIRED by name),
o GEMM, [gate*x + fp32 residual + transition AdaLN] kernel, ONE a1|a2 GEMM [128->512], SwiGLU kernel, b GEMM, [gate + residual + the next block's
two AdaLNs] kernel. All per-block conditioning (3 AdaLN scale/shift pairs with sigmoid applied, 2 sigmoid output gates: functions of the
step-invariant atom conditioning c) and the local pair bias (from the step-invariant p) are produced ONCE PER ITEM through the DiT-hoist slot
protocol (slots `atomfast.<enc|dec>.cond<i>` / `.locbias<i>`; without the hoist they are produced per call).
Precision: fp32 storage, cuBLAS under the stock TF32 flag, attention operands per the atom_attn cell (tf32rn); residual stream and LayerNorm
statistics fp32. (FastAtomStack takes an activation dtype argument so a reduced-precision word can be measured by constructing the stack
directly; the kit install below has no such switch.) Graph-capture safe.
Numerics class TOLERANCE (GEMM shapes change; op-level vs stock at TF32 distance).
"""
import math, os
import torch
import torch.nn.functional as F
from ._plumbing import LeverRefused, count_aside, diffusion_module, say, register_exit, odde_hoist
from opt_core.kernels.apb.ditfast import atom_kernels as K2                 # the row kernels are the shared core's (opt_core)

NAME = "atom_fused"
REPORT = {"installed": False}
COUNTS = {"calls": 0, "asides": 0}                                                                 # odde: stack forwards entered (opendde_opt/ran.py)
C, H, CD, NF = 128, 4, 32, 2


def _hoist():                                                                          # odde: OpenDDE's hoist object (installed) or None
    return odde_hoist()


class FastAtomStack:
    def __init__(self, owner_name, atom_transformer, act_dtype):
        from opendde.model.modules.transformer import DiffusionTransformerBlock       # odde
        self.name = owner_name                     # "enc" | "dec"
        self.at = atom_transformer
        self.dt = atom_transformer.diffusion_transformer
        self.blocks = list(self.dt.blocks)
        self.nb = len(self.blocks)
        self.act = act_dtype
        self.n_queries, self.n_keys = atom_transformer.n_queries, atom_transformer.n_keys
        self.calls = 0
        self.range = None                                                # optional abs-max recorder (enable_range()); off unless a caller turns it on
        self.range_names = ["adaln_a(an)", "adaln_kv(kvn)", "qg", "kv", "attn_out_gated(og)", "o_proj(x)", "adaln_ff(an2)", "a1a2(h12)", "swiglu(hb)", "b_proj(xb)", "residual_A(fp32, never cast)"]
        self.attn_op = None
        # ---- attention: lever atom_attn_apb (the provider-served windowed site, odde_apb_bind), REQUIRED by name — no local-attention code of our own
        try:
            import odde_apb_bind as AB                                       # odde: levers/SAMPLER
        except Exception as e:
            raise LeverRefused(f"{NAME}: requires lever atom_attn_apb — odde_apb_bind not importable ({e!r})")
        wa = AB.word("atom")
        if wa is None:                                                       # odde: the kit's word of lever atom_attn_apb (fast | big | <row>)
            raise LeverRefused(f"{NAME}: requires lever atom_attn_apb (ODDE_ATOM_ATTN=fast | big | <row>) — the fused block attention is the provider-served site")
        self._apb = AB
        self.attn_name = f"opt_core.kernels.apb by word {wa} (binding {AB.NAME} {AB.__version__}; install-time selection: {AB.COUNTS['atom'].get('selection')})"
        for i, blk in enumerate(self.blocks):
            if not isinstance(blk, DiffusionTransformerBlock):
                raise LeverRefused(f"{NAME}: {owner_name} block {i} is {type(blk).__name__}")
            apb, ctb = blk.attention_pair_bias, blk.conditioned_transition_block; att = apb.attention
            if not (apb.has_s and apb.cross_attention_mode and att.num_heads == H and att.c_hidden == CD and att.linear_g is not None
                    and ctb.n == NF and ctb.c_a == C and att.linear_q.bias is not None and att.linear_k.bias is None):
                raise LeverRefused(f"{NAME}: {owner_name} block {i} is not the pinned atom-block shape")
        with torch.no_grad():
            W = lambda lin: lin.weight.detach().float()
            z0 = torch.zeros(C, device=W(self.blocks[0].attention_pair_bias.attention.linear_q).device)
            self.w_qg, self.b_qg, self.w_kv, self.w_o, self.w_a12, self.w_b = [], [], [], [], [], []
            for blk in self.blocks:
                att, ctb = blk.attention_pair_bias.attention, blk.conditioned_transition_block
                self.w_qg.append(torch.cat([W(att.linear_q), W(att.linear_g)], 0).to(act_dtype).contiguous())
                self.b_qg.append(torch.cat([att.linear_q.bias.detach().float(), z0], 0).to(act_dtype).contiguous())
                self.w_kv.append(torch.cat([W(att.linear_k), W(att.linear_v)], 0).to(act_dtype).contiguous())
                self.w_o.append(W(att.linear_o).to(act_dtype).contiguous())
                self.w_a12.append(torch.cat([W(ctb.linear_nobias_a1), W(ctb.linear_nobias_a2)], 0).to(act_dtype).contiguous())
                self.w_b.append(W(ctb.linear_nobias_b).to(act_dtype).contiguous())
            self.eps = float(self.blocks[0].attention_pair_bias.layernorm_a.layernorm_a.eps)
        self.scale = 1.0 / math.sqrt(CD)

    # -------------------------------------------------- per-item conditioning (hoist slots; producers = the stock expressions on c [1, N_atom, 128])
    def _cond(self, i, s):
        blk = self.blocks[i]; apb, ctb = blk.attention_pair_bias, blk.conditioned_transition_block

        def mk():
            with torch.no_grad():
                s2 = s.reshape(-1, C).float()
                la, lk, ad = apb.layernorm_a, apb.layernorm_kv, ctb.adaln
                rows = [torch.sigmoid(la.linear_s(la.layernorm_s(s2))), la.linear_nobias_s(la.layernorm_s(s2)),
                        torch.sigmoid(lk.linear_s(lk.layernorm_s(s2))), lk.linear_nobias_s(lk.layernorm_s(s2)),
                        torch.sigmoid(apb.linear_a_last(s2)),
                        torch.sigmoid(ad.linear_s(ad.layernorm_s(s2))), ad.linear_nobias_s(ad.layernorm_s(s2)),
                        torch.sigmoid(ctb.linear_s(s2))]
                return torch.stack(rows, 0).contiguous()          # [8, N_atom, 128] fp32: sc_a sh_a sc_kv sh_kv g_att sc_t sh_t g_ff
        Hh = _hoist()
        t = Hh.cached(f"atomfast.{self.name}.{i}.cond", mk) if Hh is not None else mk()
        return [t[j] for j in range(8)]

    def _bias(self, i, p):
        apb = self.blocks[i].attention_pair_bias

        def mk():
            from opendde.model.utils import permute_final_dims                           # odde
            with torch.no_grad():
                return permute_final_dims(apb.linear_nobias_z(apb.layernorm_z(p)), [3, 0, 1, 2]).contiguous()   # [.., 4, n_trunks, 32, 128] fp32
        Hh = _hoist()
        return Hh.cached(f"atomfast.{self.name}.{i}.locbias", mk) if Hh is not None else mk()

    def _attention(self, qg, kv, bias, S, NA):
        """q|g and k|v GEMM outputs [M, 256] -> gated attention output [M, 128] via the provider-served windowed row on [S, NA, H, CD] views (no copies)."""
        q = qg[:, :C].view(S, NA, H, CD); g = qg[:, C:].view(S, NA, H, CD)
        k = kv[:, :C].view(S, NA, H, CD); v = kv[:, C:].view(S, NA, H, CD)
        o = self._apb.atom_views(q, k, v, bias, g, n_queries=self.n_queries, n_keys=self.n_keys, out_dtype=qg.dtype)   # [S, NA, H, CD]; 1/sqrt(CD) scale + sigmoid(g) in the row (lever atom_attn_apb)
        return o.reshape(S * NA, C)

    def _rec(self, j, t):
        if self.range is not None:
            torch.maximum(self.range[j], t.detach().abs().amax().float(), out=self.range[j])

    def enable_range(self):
        """Optional recorder: running abs-max of every low-precision operand (graph-safe)."""
        self.range = torch.zeros(len(self.range_names), device=self.w_o[0].device)
        return self.range

    @torch.autocast("cuda", enabled=False)                      # odde: the stack runs its own precision plan (fp32 stream, low-precision operands only per its word) whatever the ambient autocast is (upstream runs predict() under torch.autocast); inputs are cast to fp32 at entry below
    def forward(self, q, c, p, inplace_safe=False, chunk_size=None):
        word = None                                                                     # odde: call forms the fused stack does not serve STEP ASIDE BY NAME to the
        if p.dim() < 4 or q.dim() < 3 or tuple(p.shape[-3:-1]) != (self.n_queries, self.n_keys):   # module's stock forward (AtomTransformer.forward), counted under COUNTS; never a raise
            word = "window"
        else:
            lead = q.shape[:-3]; S, NA = int(q.shape[-3]), int(q.shape[-2])
            if q.shape[-1] != C or c.shape[-1] != C or c.shape[-2] != NA or math.prod(lead) != 1 or math.prod(c.shape[:-2]) != 1 or math.prod(p.shape[:-4]) != 1:
                word = "shapes"
        if word is not None:
            count_aside(COUNTS, word)
            return type(self.at).forward(self.at, q, c, p, inplace_safe=inplace_safe, chunk_size=chunk_size)
        self.calls += 1; COUNTS["calls"] += 1                                          # odde
        n_blocks, n_queries, n_keys = p.shape[-4:-1]
        M = S * NA; act = self.act
        A = q.reshape(M, C).float().contiguous().clone()
        cond = [self._cond(i, c) for i in range(self.nb)]
        biases = [self._bias(i, p) for i in range(self.nb)]
        sca, sha, sck, shk = cond[0][0:4]
        an, kvn = K2.adaln2(A, sca, sha, sck, shk, act, self.eps, kv=True)
        for i in range(self.nb):
            sca, sha, sck, shk, g_att, sct, sht, g_ff = cond[i]
            qg = torch.addmm(self.b_qg[i], an, self.w_qg[i].t())          # [M, 256]: q | g(raw)
            kv = torch.mm(kvn, self.w_kv[i].t())                           # [M, 256]: k | v
            og = self._attention(qg, kv, biases[i], S, NA)                # [M, 128] act (lever atom_attn)
            self._rec(0, an); self._rec(1, kvn); self._rec(2, qg); self._rec(3, kv); self._rec(4, og)
            x = torch.mm(og.to(act) if og.dtype != act else og, self.w_o[i].t())
            self._rec(5, x)
            an2 = K2.resgate_adaln2(g_att, x, A, sct, sht, None, None, act, self.eps, ln=True, kv=False)   # A += g_att*x ; transition AdaLN
            h12 = torch.mm(an2, self.w_a12[i].t())                         # [M, 512]
            hb = K2.swiglu2d(h12, act)                                     # [M, 256]
            xb = torch.mm(hb, self.w_b[i].t())                             # [M, 128]
            self._rec(6, an2); self._rec(7, h12); self._rec(8, hb); self._rec(9, xb); self._rec(10, A)
            if i + 1 < self.nb:
                nsca, nsha, nsck, nshk = cond[i + 1][0:4]
                an, kvn = K2.resgate_adaln2(g_ff, xb, A, nsca, nsha, nsck, nshk, act, self.eps, ln=True, kv=True)   # A += g_ff*xb ; next block's AdaLNs
            else:
                K2.resgate_adaln2(g_ff, xb, A, ln=False)
        return A.view(*lead, S, NA, C)


def _strip_atom_overrides(at):
    n = 0
    for blk in at.diffusion_transformer.blocks:
        apb, ctb = blk.attention_pair_bias, blk.conditioned_transition_block
        for obj, attr in ((blk, "forward"), (apb.layernorm_a, "forward"), (apb.layernorm_kv, "forward"), (ctb.adaln, "forward"), (apb.attention, "_wrap_up"),
                          (apb, "local_multihead_attention"), (apb, "forward"), (ctb, "forward"),
                          (apb.linear_a_last, "forward"), (ctb.linear_s, "forward"), (apb.layernorm_z, "forward"), (apb.linear_nobias_z, "forward")):
            # instance-level overrides of the DiT hoist on these blocks: this stack owns the whole block (its own hoist slots
            # atomfast.<enc|dec>.<i>.cond / .locbias carry the step-invariant tensors), so the per-submodule producers must not run or record
            if attr in obj.__dict__:
                del obj.__dict__[attr]; n += 1
        for aln in (apb.layernorm_a, getattr(apb, "layernorm_kv", None), ctb.adaln):        # odde: OpenDDE's hoist (levers/DITFAST odde_addon.py) overrides one level deeper on
            if aln is None:                                                                  # these blocks (AdaLN layernorm_s / linear_s / linear_nobias_s forwards): this
                continue                                                                     # stack owns the blocks, its own slots carry those tensors
            for sub_ in (getattr(aln, "layernorm_s", None), getattr(aln, "linear_s", None), getattr(aln, "linear_nobias_s", None)):
                if sub_ is not None and "forward" in sub_.__dict__:
                    del sub_.__dict__["forward"]; n += 1
    return n


def install(model):
    """Kit hook entry (odde_sampler.install). Env: ODDE_ATOM_FUSED=1 engages atom_fused (fp32 activations; no precision switch)."""
    dm = diffusion_module(model)
    if os.environ.get("ODDE_ATOM_FUSED", "0") != "1":                                       # odde
        raise LeverRefused(f"{NAME}: install called with ODDE_ATOM_FUSED={os.environ.get('ODDE_ATOM_FUSED')!r} (expected 1)")
    act = torch.float32
    rep = {}
    for owner, mod in (("enc", dm.atom_attention_encoder.atom_transformer), ("dec", dm.atom_attention_decoder.atom_transformer)):
        if "forward" in mod.__dict__ and getattr(mod.__dict__["forward"], "_fpf_lever", None) == NAME:
            continue
        stripped = _strip_atom_overrides(mod)                               # the DiT hoist's instance overrides on these blocks (subsumed sites)
        st = FastAtomStack(owner, mod, act); _STACKS.append(st)
        fwd = (lambda st_: (lambda *a, **k: st_.forward(*a, **k)))(st)
        fwd._fpf_lever = NAME
        mod.forward = fwd; mod._opendde_fpf_ditfast = st
        rep[owner] = {"blocks": st.nb, "strip": stripped}
    hoist = _hoist() is not None and getattr(_hoist(), "installed", False)
    REPORT.update(installed=True, act=str(act), stacks=rep, attention=_STACKS[-1].attn_name if _STACKS else None, cond_slots="hoist" if hoist else "local")
    register_exit(NAME, lambda: {"stack_calls": {st_.name: st_.calls for st_ in _STACKS}, "blocks_x_calls": sum(st_.calls * st_.nb for st_ in _STACKS), "act": str(act),
                                 "attention": _STACKS[-1].attn_name if _STACKS else None})
    say(f"ATOMFAST:on(enc={rep.get('enc', {}).get('blocks')} dec={rep.get('dec', {}).get('blocks')} act={act} attn={REPORT['attention']} cond_slots={'hoist' if hoist else 'local'} strip={sum(v['strip'] for v in rep.values())})")
    return REPORT


_STACKS = []
