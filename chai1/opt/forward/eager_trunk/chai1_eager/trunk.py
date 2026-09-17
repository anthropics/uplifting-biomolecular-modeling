# Architecture and weights: Chai-1 (chai_lab 0.6.1), Copyright 2024 Chai Discovery, Inc., Apache-2.0 — this file re-expresses the
# exported TorchScript graph `models_v2/trunk.pt` as structured eager nn.Modules (same op order, same dtype casts).
"""Structured eager Chai-1 trunk.

Numerics policy of the export (mirrored here exactly):
  * interface tensors bf16; every LayerNorm runs in fp32 (`x.to(float32)` first, eps 1e-5 except OPM ln_out eps 0.1);
  * every Linear/EinMix runs in bf16 (fp32 weight cast to bf16 at use; activations cast to bf16);
  * einsum / SDPA / softmax(dtype=fp32) exactly as traced; masks applied with masked_fill(-10000 / 0);
  * row-wise ops are chunked exactly like the trace (Transition: ceil(numel*expansion/2^30) chunks on dim -2; OPM: MSA depth in
    slices of 4096; pair-weighted averaging: slices of 8192) so GEMM shapes match the scripted module.
State-dict keys are identical to `torch.jit.load('trunk.pt').state_dict()` (1398 tensors)."""
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

BF = torch.bfloat16
F32 = torch.float32
TWO30 = 1 << 30

# ---- runtime switches -------------------------------------------------------------------------------------------------------
CFG = dict(
    precast_bf16=True,      # cache bf16 copies of fp32 weights (the cast is deterministic -> bitwise identical; saves one cast kernel per weight use)
    record_ranges=False,    # emit torch.profiler record_function ranges per op class
    trimul_impl=None,       # None = stock eager; else callable(module, z, mask) -> update | NotImplemented (NotImplemented = the statement runs)
    triattn_impl=None,      # None = stock eager; else callable(module, zn_bf16, mask) -> cat(out dirs) before linear_out | NotImplemented (the statement runs)
    ln_impl=None,           # None = the export's LayerNorm statements; else callable(kind, x, c, w, b, eps) -> tensor | NotImplemented (chai1_opt.pairtrack
                            # `exactln`, chai1_exactln.serve: the shared core's LayerNorm provider row for the mode's tier word — under `exact` an ATen
                            # replica of the SAME statement, bit for bit, the bf16 casts fused; NotImplemented = the statement runs)
    msa_impl=None,          # None = the export's MSA-module statement on every padded MSA row; else callable(module, s, z, msa_input_feats, msa_mask, pair_mask)
                            # -> z | NotImplemented (chai1_opt.pairtrack `msa_pad`, chai1_eager.msa_kernels: the SAME statement on the leading row blocks that
                            # carry any mask position — the export's own 4096 / 8192-row slice grid kept, so every GEMM of a kept slice is the whole
                            # statement's GEMM; bit-compared per shape class at its first call; NotImplemented = the statement runs whole)
    transition_impl=None,   # None = the export's Transition statement; else callable(module, x) -> y | NotImplemented (chai1_opt.pairtrack `transition`,
                            # chai1_eager.transition_core: the statement bound BY WORD to opt_core.kernels.transition — exact: the provider's exact-class rows fed
                            # the statement's own LayerNorm output and bit-compared per class at the first call; fast / big: the provider's fast-class rows, LayerNorm fused)
)
_BFCACHE = {}


def bfw(p):
    """fp32 parameter -> bf16 tensor (cached when CFG['precast_bf16']).

    The cache is keyed by id(p); every entry HOLDS `p` and is served only when `entry[0] is p`, and only long-lived nn.Parameter
    objects are inserted. Hazard this guards against: a per-call temporary (e.g. an unbind view of a weight) passed through an
    id()-keyed cache has its id recycled by CPython after the call, so a later lookup could return another tensor's bf16 copy
    (wrong weights, run-to-run nondeterministic output — invisible when the operand it multiplies is all-masked zeros). Per-call
    weight views are therefore cached under their Parameter's id by the module itself (OuterProductMean._weights_bf16, _scaled_out_w)."""
    if p.dtype == BF:
        return p
    if not CFG["precast_bf16"]:
        return p.to(BF)
    k = id(p)
    e = _BFCACHE.get(k)
    if e is not None and e[0] is p and e[1].device == p.device:
        return e[1]
    t = p.detach().to(BF)
    if isinstance(p, nn.Parameter):          # stable id for the module's lifetime; the strong ref pins it anyway
        _BFCACHE[k] = (p, t)
    return t


def clear_cache():
    _BFCACHE.clear()


def ln_bf16(x, c, w=None, b=None, eps=1e-5):
    """The export's LayerNorm statement on a bf16 activation feeding a bf16 Linear: ``F.layer_norm(x.to(float32), (c,), w, b, eps).to(bfloat16)``
    (fp32 arithmetic and affine on the widened tensor, the result cast for the following EinMix).  ``CFG['ln_impl']`` may serve it (same bits, one
    kernel); ``NotImplemented`` from it (or no plug) runs the statement itself."""
    f = CFG["ln_impl"]
    if f is not None:
        r = f("bf16", x, c, w, b, eps)
        if r is not NotImplemented:
            return r
    return F.layer_norm(x.to(F32), (c,), w, b, eps).to(BF)


def ln_f32(x, c, w=None, b=None, eps=1e-5):
    """``F.layer_norm(x, (c,), w, b, eps)`` on an fp32 tensor (TriMul's affine-free output LayerNorms); ``CFG['ln_impl']`` may serve it (same bits)."""
    f = CFG["ln_impl"]
    if f is not None:
        r = f("f32", x, c, w, b, eps)
        if r is not NotImplemented:
            return r
    return F.layer_norm(x, (c,), w, b, eps)


def rng(name):
    return torch.profiler.record_function(name) if CFG["record_ranges"] else contextlib.nullcontext()


def n_chunks_for(x, w_ab):
    """Trace formula: ceil(numel(x) * (out_features // in_features) / 2^30) (computed with floor-divides by -2^30 and -1)."""
    a = x.numel() * (w_ab.shape[0] // w_ab.shape[1])
    return (a // -TWO30) // -1


# ---- blocks -----------------------------------------------------------------------------------------------------------------
class Transition(nn.Module):                      # TransitionLayer_v1
    def __init__(self, c, hidden):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c)
        self.linear_no_bias_ab = nn.Linear(c, 2 * hidden, bias=False)
        self.linear_out = nn.Linear(hidden, c, bias=False)
        self.c = c

    def forward(self, x):
        f = CFG["transition_impl"]
        if f is not None:
            r = f(self, x)                                    # a binding may decline a class (NotImplemented): the statement below
            if r is not NotImplemented:
                return r
        return self.forward_statement(x)

    def forward_statement(self, x):
        """The export's statement: per row chunk (ceil(numel * 2h/c / 2^30) chunks on dim -2) LayerNorm (fp32, affine) -> bf16, ONE [c -> 2h] projection
        a|b, silu(a) * b, the [h -> c] projection; chunks concatenated."""
        with rng("transition"):
            n = n_chunks_for(x, self.linear_no_bias_ab.weight)
            outs = []
            for ch in torch.chunk(x, n, -2):
                xn = ln_bf16(ch, self.c, self.layer_norm.weight, self.layer_norm.bias)      # LN(ch.to(f32)).to(bf16), served fused when plugged
                ab = F.linear(xn, bfw(self.linear_no_bias_ab.weight))
                del xn
                a, b = torch.chunk(ab, 2, -1)
                prod = F.silu(a).mul_(b)
                del a, b, ab
                outs.append(F.linear(prod, bfw(self.linear_out.weight)))
                del prod
            return outs[0] if len(outs) == 1 else torch.concatenate(outs, -2)


class TriangleMultiplication(nn.Module):          # TriangleMultiplicativeUpdateFastv2r (outgoing+incoming merged)
    def __init__(self, c, d=None):
        super().__init__()
        d = d or c
        self.c, self.d = c, d
        self.layernorm_z_in = nn.LayerNorm(c)
        self.merged_linear_p = nn.Linear(c, 4 * d, bias=False)          # [a1 | b1 | a2 | b2]
        self.merged_linear_g = nn.Linear(c, 4 * d + c, bias=False)      # [gates(4d) | out_gate(c)]
        self.linear_z_out = nn.Linear(d, c, bias=False)

    def forward(self, z, mask):
        """returns the residual update (caller adds)."""
        with rng("trimul"):
            if CFG["trimul_impl"] is not None:
                r = CFG["trimul_impl"](self, z, mask)      # adapters return NotImplemented for shapes they do not serve (e.g. template blocks, c=64)
                if r is not NotImplemented:
                    return r
            c = self.c
            zn = ln_bf16(z, c, self.layernorm_z_in.weight, self.layernorm_z_in.bias)      # the two `.to(bf16)` reads of one fp32 LN result are one bf16 tensor (same values)
            p = F.linear(zn, bfw(self.merged_linear_p.weight))
            g = torch.sigmoid(F.linear(zn, bfw(self.merged_linear_g.weight)))
            del zn
            ab = torch.mul(p, g[..., :-c])
            del p
            out_gate = g[..., -c:].clone()     # keep only the C out-gate channels alive (values identical)
            del g
            a1b1, a2b2 = torch.chunk(ab, 2, -1)
            a1b1 = a1b1.masked_fill(torch.bitwise_not(mask.unsqueeze(3)), 0)
            a2b2 = a2b2.masked_fill(torch.bitwise_not(mask.transpose(1, 2).unsqueeze(3).contiguous()), 0)
            del ab
            a1, b1 = torch.chunk(a1b1, 2, -1)
            a2, b2 = torch.chunk(a2b2, 2, -1)
            with rng("trimul.einsum"):
                x1 = torch.einsum("... i k d, ... j k d -> ... i j d", a1, b1)
                del a1, b1, a1b1
                x2 = torch.einsum("... k i d, ... k j d -> ... i j d", a2, b2)
                del a2, b2, a2b2
            x = torch.add(ln_f32(x1.to(F32), self.d), ln_f32(x2.to(F32), self.d))
            del x1, x2
            x = F.linear(x.to(BF), bfw(self.linear_z_out.weight))
            return torch.mul(x, out_gate)


class TriangleAttention(nn.Module):               # TriangleAttentionUpdate_v2a (starting+ending, 4 heads)
    H = 4

    def __init__(self, c, dh):
        super().__init__()
        H = self.H
        self.c, self.dh = c, dh
        self.pair2b = nn.Linear(c, 2 * H, bias=False)
        self.pair2qkvg1 = nn.Linear(c, 4 * H * dh, bias=False)
        self.pair2qkvg2 = nn.Linear(c, 4 * H * dh, bias=False)
        self.linear_out = nn.Linear(2 * H * dh, c, bias=False)
        self.out_scalers = nn.Parameter(torch.ones(c))

    def forward(self, z, mask):
        """returns the residual update (caller adds)."""
        with rng("triattn"):
            B, N = mask.shape[0], mask.shape[1]
            H, dh, c = self.H, self.dh, self.c
            zn = ln_bf16(z, c)                        # affine-free LN; every consumer reads `zn.to(bf16)` (the transposed read: transpose and cast commute exactly)
            out = CFG["triattn_impl"](self, zn, mask) if CFG["triattn_impl"] is not None else NotImplemented   # a plug may decline (NotImplemented): the own statement
            if out is NotImplemented:
                b = F.linear(zn, bfw(self.pair2b.weight))
                b = b.reshape(B, N, N, 2, H).permute(0, 3, 4, 1, 2).reshape(B, 2, H, 1, N, N)
                b = b.masked_fill(torch.bitwise_not(mask.reshape(B, 1, 1, 1, N, N)), -10000)
                b = b.reshape(B, 2, H, N, N).permute(1, 0, 2, 3, 4).reshape(2, B * H, 1, N, N)
                outs = []
                for d, (W, zin) in enumerate(((self.pair2qkvg1.weight, zn), (self.pair2qkvg2.weight, zn.transpose(1, 2)))):
                    x = F.linear(zin, bfw(W))
                    x = x.reshape(B, N, N, H, 4, dh).permute(4, 0, 3, 1, 2, 5).reshape(4, B * H, N, N, dh)
                    q, k, v, g = torch.unbind(x)
                    with rng("triattn.sdpa"):
                        o = F.scaled_dot_product_attention(q, k, v, b[d])
                    del q, k, v
                    o = torch.mul(o, torch.sigmoid(g))
                    del g, x
                    outs.append(o.reshape(B, H, N, N, dh).permute(0, 2, 3, 1, 4).reshape(B, N, N, H * dh))
                    del o
                del zin
                out = torch.cat(outs, -1)
                del outs
            W = torch.mul(self.linear_out.weight, self.out_scalers[0:].unsqueeze(1))
            return F.linear(out, W.to(BF) if not CFG["precast_bf16"] else _scaled_out_w(self, W))


def _scaled_out_w(mod, W):
    p = mod.linear_out.weight                 # long-lived Parameter -> stable key; entry pins it and is checked with `is` (see bfw)
    k = ("outw", id(p))
    e = _BFCACHE.get(k)
    if e is not None and e[0] is p and e[1].device == W.device:
        return e[1]
    t = W.to(BF)
    _BFCACHE[k] = (p, t)
    return t


class AttentionPairBias(nn.Module):              # single-track attention with pair bias (16 heads x 24)
    def __init__(self, c_s=384, c_z=256, H=16, dh=24):
        super().__init__()
        self.H, self.dh, self.c_s, self.c_z = H, dh, c_s, c_z
        self.single_layer_norm = nn.LayerNorm(c_s)
        self.pair_layer_norm = nn.LayerNorm(c_z)
        self.pair_linear = nn.Linear(c_z, H, bias=False)
        self.attention = nn.Module()
        self.attention.query_bias = nn.Parameter(torch.zeros(H, dh))
        self.attention.input2qkvg = nn.Module()
        self.attention.input2qkvg.weight = nn.Parameter(torch.zeros(c_s, 4, H, dh))
        self.attention.output_proj = nn.Module()
        self.attention.output_proj.weight = nn.Parameter(torch.zeros(H, dh, c_s))

    def forward(self, s, z, pair_mask, single_mask):
        """returns masked residual update (caller adds)."""
        with rng("single_attn"):
            B, N = pair_mask.shape[0], pair_mask.shape[1]
            H, dh = self.H, self.dh
            sn = ln_bf16(s, self.c_s, self.single_layer_norm.weight, self.single_layer_norm.bias)
            zb = ln_bf16(z, self.c_z, self.pair_layer_norm.weight, self.pair_layer_norm.bias)
            bias = F.linear(zb, bfw(self.pair_linear.weight)).permute(0, 3, 1, 2)
            bias = bias.masked_fill(torch.bitwise_not(pair_mask.reshape(B, 1, N, N)), -10000)
            qkvg = torch.einsum("dfa,aebc->edbfc", sn, bfw(self.attention.input2qkvg.weight))
            q, k, v, g = torch.unbind(qkvg)
            q = torch.add(q, self.attention.query_bias.reshape(1, H, 1, dh))
            with rng("single_attn.sdpa"):
                o = F.scaled_dot_product_attention(q.to(BF), k, v, bias)
            o = torch.mul(o, torch.sigmoid(torch.add(g, 1)))
            out = torch.einsum("ecbd,cda->eba", o, bfw(self.attention.output_proj.weight))
            return torch.mul(out, single_mask.unsqueeze(-1))


class PairformerBlock(nn.Module):
    def __init__(self, c_z=256, c_s=384, with_single=True, tri_dh=64):
        super().__init__()
        self.transition_pair = Transition(c_z, 2 * c_z)
        self.triangle_multiplication = TriangleMultiplication(c_z)
        self.triangle_attention = TriangleAttention(c_z, tri_dh)
        self.with_single = with_single
        if with_single:
            self.transition_single = Transition(c_s, 2 * c_s)
            self.attention_pair_bias = AttentionPairBias(c_s, c_z)

    def forward(self, s, z, pair_mask, single_mask=None):
        # all three pair updates read the block-input z (parallel branches), exactly as traced
        z1 = torch.add(z, self.triangle_multiplication(z, pair_mask))
        z2 = torch.add(z1, self.triangle_attention(z, pair_mask))
        z_out = torch.add(z2, self.transition_pair(z))
        if not self.with_single:
            return s, z_out
        s1 = torch.add(s, self.attention_pair_bias(s, z, pair_mask, single_mask))
        s_out = torch.add(s1, self.transition_single(s))
        return s_out, z_out


class Pairformer(nn.Module):
    def __init__(self, n_blocks, c_z, c_s=384, with_single=True, tri_dh=64):
        super().__init__()
        self.blocks = nn.ModuleList([PairformerBlock(c_z, c_s, with_single, tri_dh) for _ in range(n_blocks)])

    def forward(self, s, z, pair_mask, single_mask=None):
        for blk in self.blocks:
            s, z = blk(s, z, pair_mask, single_mask)
        return s, z


class TemplateEmbedder(nn.Module):
    def __init__(self, c_z=256, c_t=64):
        super().__init__()
        self.c_t = c_t
        self.proj_in = nn.Sequential(nn.LayerNorm(c_z), nn.Linear(c_z, c_t, bias=False))
        self.pairformer = Pairformer(2, c_t, with_single=False, tri_dh=32)
        self.template_layernorm = nn.LayerNorm(c_t)
        self.proj_out = nn.Sequential(nn.ReLU(), nn.Linear(c_t, c_z, bias=False))

    def forward(self, z, template_input_feats, template_input_masks, token_pair_mask):
        with rng("template"):
            B, N = token_pair_mask.shape[0], token_pair_mask.shape[1]
            n_templates = torch.sum(torch.any(template_input_masks, [-2, -1]), [1])
            p = F.linear(ln_bf16(z, z.shape[-1], self.proj_in[0].weight, self.proj_in[0].bias), bfw(self.proj_in[1].weight))
            template_mask = template_input_masks.__and__(token_pair_mask.reshape(B, 1, N, N))
            outs = []
            for t in range(template_input_feats.shape[1]):
                zt = torch.add(p, template_input_feats[0:].select(1, t))
                mt = template_mask[0:].select(1, t)
                _, zt = self.pairformer(None, zt, mt)
                outs.append(zt)
            u = torch.stack(outs, 1)
            u = F.layer_norm(u.to(F32), (self.c_t,), self.template_layernorm.weight, self.template_layernorm.bias)
            u = torch.mul(u, template_mask.unsqueeze(-1))
            u = torch.sum(u, [1], False, dtype=F32)
            u = torch.div(u, torch.clamp_min(n_templates, 1).reshape(B, 1, 1, 1))
            out = F.linear(torch.relu(u).to(BF), bfw(self.proj_out[1].weight))
            return torch.add(z, out)


class OuterProductMean(nn.Module):
    CH = 4096

    def __init__(self, c_m=64, c_z=256, g=8, e=8):
        super().__init__()
        self.weight_ab = nn.Parameter(torch.zeros(2, g, e, c_m))
        self.ln_out = nn.LayerNorm(g * e * e, eps=0.1)
        self.linear_out = nn.Linear(g * e * e, c_z)
        self.c_m = c_m

    def _weights_bf16(self):
        """(wa, wb) bf16 slices of weight_ab — same values as the traced `unbind(weight_ab.contiguous()) -> .to(bf16)`;
        cached under the Parameter's (stable) id with a same-object (`is`) check when CFG['precast_bf16'] (see bfw)."""
        p = self.weight_ab
        if CFG["precast_bf16"]:
            k = ("opm_ab", id(p))
            e = _BFCACHE.get(k)
            if e is not None and e[0] is p and e[1][0].device == p.device:
                return e[1]
        wa, wb = torch.unbind(p.detach().contiguous())
        pair = (wa.to(BF), wb.to(BF))
        if CFG["precast_bf16"]:
            _BFCACHE[("opm_ab", id(p))] = (p, pair)
        return pair

    def forward(self, msa, msa_mask):
        with rng("msa.opm"):
            B, S, N = msa.shape[0], msa.shape[1], msa.shape[2]
            wa, wb = self._weights_bf16()          # cached under the Parameter's stable id: per-call unbind temporaries must never enter the id()-keyed _BFCACHE (see bfw)
            acc = None
            for s0 in range(0, S, self.CH):
                ch, mk = msa[:, s0:s0 + self.CH], msa_mask[:, s0:s0 + self.CH]
                xb = ln_bf16(ch, self.c_m)                                                  # LN(ch.to(f32)).to(bf16); the mask's zeros commute with the cast exactly
                xb.masked_fill_(torch.bitwise_not(mk.reshape(B, mk.shape[1], N, 1)), 0)
                with rng("msa.opm.einsum"):
                    A = torch.einsum("abc,defc->abdef", wa, xb)
                    Bm = torch.einsum("abc,defc->abdef", wb, xb)
                    del xb
                    op = torch.einsum("abcde,afcdg->cegabf", A, Bm)
                    del A, Bm
                op = op.reshape(B, N, N, -1)
                acc = torch.add(op, 0) if acc is None else torch.add(acc, op)
                del op
            x = ln_bf16(acc, acc.shape[-1], self.ln_out.weight, self.ln_out.bias, 0.1)
            return F.linear(x, bfw(self.linear_out.weight), bfw(self.linear_out.bias))


class MSAPairWeightedAveraging(nn.Module):
    CH = 8192

    def __init__(self, c_m=64, c_z=256, H=8, dv=32):
        super().__init__()
        self.H, self.dv, self.c_m, self.c_z = H, dv, c_m, c_z
        self.layernorm_msa = nn.LayerNorm(c_m)
        self.linear_msa2vg = nn.Linear(c_m, 2 * H * dv, bias=False)
        self.layernorm_pair = nn.LayerNorm(c_z)
        self.linear_pair = nn.Linear(c_z, H, bias=False)
        self.linear_out_no_bias = nn.Linear(H * dv, c_m, bias=False)

    def forward(self, msa, z, pair_mask, msa_mask):
        with rng("msa.pwa"):
            B, S, N = msa.shape[0], msa.shape[1], msa.shape[2]
            H, dv = self.H, self.dv
            y = ln_bf16(z, self.c_z, self.layernorm_pair.weight, self.layernorm_pair.bias)
            b = F.linear(y, bfw(self.linear_pair.weight)).permute(0, 3, 1, 2)
            b = b.masked_fill(torch.bitwise_not(pair_mask.reshape(B, 1, N, N)), -10000)
            w = torch.softmax(b, -1, dtype=F32).to(BF)
            outs = []
            for s0 in range(0, S, self.CH):
                ch, mk = msa[:, s0:s0 + self.CH], msa_mask[:, s0:s0 + self.CH]
                Sc = ch.shape[1]
                y0 = ln_bf16(ch, self.c_m, self.layernorm_msa.weight, self.layernorm_msa.bias)
                vg = F.linear(y0, bfw(self.linear_msa2vg.weight)).reshape(B, Sc, N, 2, H, dv).permute(3, 0, 1, 2, 4, 5)
                del y0
                v, g = torch.unbind(vg)
                del vg
                g = torch.sigmoid(g)
                v = v.masked_fill(torch.bitwise_not(mk).reshape(B, Sc, N, 1, 1), 0)
                with rng("msa.pwa.einsum"):
                    o = torch.einsum("abcd,aedbf->aecbf", w, v)
                del v
                o = torch.mul(g, o).reshape(B, Sc, N, H * dv)
                del g
                outs.append(F.linear(o, bfw(self.linear_out_no_bias.weight)))
                del o
            return torch.cat(outs, 1)


class MSAModule(nn.Module):
    def __init__(self, n_blocks=4, c_m=64, c_z=256, c_s=384):
        super().__init__()
        self.linear_s2m = nn.Linear(c_s, c_m, bias=False)
        self.outer_product_mean = nn.ModuleList([OuterProductMean(c_m, c_z) for _ in range(n_blocks)])
        self.msa_pair_weighted_averaging = nn.ModuleList([MSAPairWeightedAveraging(c_m, c_z) for _ in range(n_blocks - 1)])
        self.msa_transition = nn.ModuleList([Transition(c_m, 4 * c_m) for _ in range(n_blocks - 1)])
        self.pair_transition = nn.ModuleList([Transition(c_z, 4 * c_z) for _ in range(n_blocks)])
        self.triangular_multiplication = nn.ModuleList([TriangleMultiplication(c_z) for _ in range(n_blocks)])
        self.triangular_attention = nn.ModuleList([TriangleAttention(c_z, 64) for _ in range(n_blocks)])
        self.c_m = c_m

    def forward(self, s, z, msa_input_feats, msa_mask, pair_mask):
        f = CFG["msa_impl"]
        if f is not None:
            r = f(self, s, z, msa_input_feats, msa_mask, pair_mask)      # a plug may decline (NotImplemented): the statement on every row
            if r is not NotImplemented:
                return r
        return self.forward_rows(s, z, msa_input_feats, msa_mask, pair_mask)

    def forward_rows(self, s, z, msa_input_feats, msa_mask, pair_mask, s_opm=None):
        """The export's statement on the MSA rows handed in; ``s_opm`` (None = all of them, else a multiple of ``OuterProductMean.CH``) = the leading
        rows the outer-product means read (chai1_eager.msa_kernels.msa_pad: the rows past it are whole all-masked 4096-row slices, exact zeros there)."""
        with rng("msa_module"):
            B, N = pair_mask.shape[0], pair_mask.shape[1]
            msa = torch.add(msa_input_feats, F.linear(s.to(BF), bfw(self.linear_s2m.weight)).reshape(B, 1, N, self.c_m))
            for i in range(len(self.outer_product_mean)):
                if s_opm is None:
                    z = torch.add(z, self.outer_product_mean[i](msa, msa_mask))
                else:
                    z = torch.add(z, self.outer_product_mean[i](msa[:, :s_opm], msa_mask[:, :s_opm]))
                if i < len(self.msa_pair_weighted_averaging):
                    msa_t = torch.add(msa, self.msa_transition[i](msa))
                    msa = torch.add(msa_t, self.msa_pair_weighted_averaging[i](msa, z, pair_mask, msa_mask))
                    del msa_t
                z1 = torch.add(z, self.triangular_multiplication[i](z, pair_mask))
                z2 = torch.add(z1, self.pair_transition[i](z))
                z = torch.add(z2, self.triangular_attention[i](z2, pair_mask))
            return z


class Trunk(nn.Module):
    def __init__(self, n_blocks=48, c_z=256, c_s=384):
        super().__init__()
        self.template_embedder = TemplateEmbedder(c_z, 64)
        self.msa_module = MSAModule(4, 64, c_z, c_s)
        self.pairformer_stack = Pairformer(n_blocks, c_z, c_s, with_single=True, tri_dh=64)
        self.token_single_recycle_proj = nn.Sequential(nn.LayerNorm(c_s), nn.Linear(c_s, c_s, bias=False))
        self.token_pair_recycle_proj = nn.Sequential(nn.LayerNorm(c_z), nn.Linear(c_z, c_z, bias=False))
        self.c_z, self.c_s = c_z, c_s

    @staticmethod
    def _recycle(seq, x):
        return F.linear(ln_bf16(x, x.shape[-1], seq[0].weight, seq[0].bias), bfw(seq[1].weight))

    def forward(self, token_single_trunk_initial_repr, token_pair_trunk_initial_repr, token_single_trunk_repr, token_pair_trunk_repr,
                msa_input_feats, msa_mask, template_input_feats, template_input_masks, token_single_mask, token_pair_mask, crop_size=None, **unused):
        z = torch.add(token_pair_trunk_initial_repr, self._recycle(self.token_pair_recycle_proj, token_pair_trunk_repr))
        s = torch.add(token_single_trunk_initial_repr, self._recycle(self.token_single_recycle_proj, token_single_trunk_repr))
        z = self.template_embedder(z, template_input_feats, template_input_masks, token_pair_mask)
        z = torch.add(z, self.msa_module(s, z, msa_input_feats, msa_mask, token_pair_mask))   # trunk-level residual around the whole MSA module (as traced)
        with rng("pairformer_stack"):
            s, z = self.pairformer_stack(s, z, token_pair_mask, token_single_mask)
        return s, z


def load_trunk(state_dict, device="cuda"):
    """state_dict: the scripted module's state_dict (fp32). Returns Trunk in eval mode on `device` (weights stay fp32)."""
    m = Trunk()
    missing, unexpected = m.load_state_dict({k: v for k, v in state_dict.items()}, strict=False)
    assert not unexpected, unexpected[:10]
    assert not missing, missing[:10]
    m = m.to(device) if device is not None else m
    m.eval().requires_grad_(False)
    clear_cache()
    return m
