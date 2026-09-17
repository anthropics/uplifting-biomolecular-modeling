"""boltz_atom.py — the Boltz-2 atom-attention levers (boltz 2.2.1): the diffusion module's AtomAttentionEncoder / AtomAttentionDecoder (3 + 3
windowed atom-transformer layers, 32 query x 128 key atoms, 4 heads x 32, pair bias) and the atom<->token glue, run inside every one of the 200
denoiser steps. Runtime patches on the AtomDiffusion INSTANCE the model builds (instance attributes shadow the class forwards, so the kit's
class-level sampler levers — boltz_graph_patch (CUDA-graph step) and boltz_dit_hoist (step-invariant hoist) — compose unchanged around and inside
these); nothing in the installed `boltz` package is edited; the trunk's input-embedder AtomAttentionEncoder (bf16 autocast, once per predict) is
NOT touched — only `structure_module.score_model.atom_attention_{encoder,decoder}` (fp32, matmul precision 'highest', autocast off).

Units (BOLTZ_ATOM=<unit>[,<unit>...]; each unit one registry lever; `apply()` prints state=on|off|skipped by name; off = the stock statements):

  keys   [exact class]  lever atom_keys_gather.  `single_to_keys` — the one-hot einsum 'b j i d, j k -> b k i d' over the [2K, 8K] fp32 indexing
         matrix that gathers each query window's 128 key atoms (6x per step on the AdaLN output b, O(M^2 * D)) — served by a gather kernel
         (boltz_atom_kernels.gather_keys): out[b,k,s,:] = x[b, 32k-48+s, :] inside the atom axis, +0.0 outside, every zero written +0.0.
         Bitwise to stock BECAUSE: the matrix is one-hot-or-empty by construction from integer indices (encodersv2.get_indexing_matrix), the
         sampler runs fp32 with float32_matmul_precision 'highest' (IEEE products x*1.0 = x, x*0.0 = +-0, accumulated from +0.0 so the sum is x
         exactly and an empty column is +0.0; cuBLAS never emits -0.0 there, the gather canonicalises -0.0 -> +0.0), and the operand is finite
         (precondition: a non-finite x poisons the stock GEMM's whole output channel via inf*0 = NaN but only its own slot in a gather — the model's
         AdaLN output is finite; documented, tested). Installed by handing the sampler a `diffusion_conditioning` dict whose `to_keys` is the
         gather (the dict Boltz2.forward builds per call is copied, never mutated); every caller of to_keys inside sample() — stock layers, the
         hoist's cache build — is served. The conditioning-time uses of to_keys (AtomEncoder.forward, once per predict) stay stock.
  glue   [exact class]  lever atom_glue_hoist.  The encoder/decoder bodies VERBATIM with the step-invariant statements hoisted to the first
         (eager) denoiser call of each sample(): feats['atom_to_token'].float(), its repeat_interleave(m), the mean normalisation
         atom_to_token / (sum + 1e-6) (the encoder's bmm(mean^T, q_to_a) stays THE SAME cuBLAS call on the same strided operand: 1/(n+1e-6)
         weights are a real reduction, not a gather); and the decoder's broadcast bmm(atom_to_token[Bm,M,N] one-hot fp32, a_to_q[Bm,N,128]) served
         by a row gather by token index (gather_rows; the same one-hot argument: exactly one 1.0 per atom row or none, +0.0 canonical, finite
         operand). Cooperates with boltz_dit_hoist: when its cache holds the repeated conditioning `c_m` for this call's `c`, that object is the
         one handed to the AtomTransformer (the hoist's identity protocol), else c.repeat_interleave(m) as stock.
  fused  [fast class]   lever atom_fused.  The encoder and decoder as 8 + 8 kernel launches per step (boltz_atom_kernels): per 3-layer atom
         transformer 1 + 3x2 Triton kernels — AdaLN+[q|k|v|g] projection; windowed attention per (32-query window, head) with the pair bias read
         in place from the [B,K,32,128,12] conditioning tensor (bf16 or fp32 — no expansion, no cast, no mask tensor: key validity from the atom
         pad mask and the window arithmetic) and the sigmoid gate; out-projection + gated residual + AdaLN + SwiGLU transition + gated residual
         fused with the NEXT layer's AdaLN+projection (or the encoder's relu(Linear 128->768) tail / the decoder's LayerNorm+Linear(128->3) tail) —
         plus a deterministic segmented-mean kernel for the atom->token aggregation and a gather-add prologue for the token->atom broadcast. The
         per-atom conditioning of the 6 layers (AdaLN sigmoid-scale / bias x2, the two sigmoid output gates: stock expressions, fp32) is computed
         once per sample() into static buffers refreshed in place (CUDA-graph safe: fixed addresses per shape; never built during capture).
         Numerics: fp32 throughout, every dot at input_precision='ieee' unless the precision lever says otherwise; differences to stock are
         summation order (LayerNorm / softmax / GEMM reductions), i.e. ~1e-6 relative per layer — tolerance class.
  BOLTZ_ATOM_GEMM=ieee|tf32|tf32x3|bf16  lever atom_gemm (fast tier only; absent = ieee): the operand precision of the fused kernels' dots
         (bf16: bf16 operands + bf16 inter-kernel activations, fp32 accumulate / LayerNorm / softmax / residual stream).

Composition and CUDA-graph safety: the sample() epoch (a counter bumped by an instance-level sample wrapper that defers to whatever
AtomDiffusion.sample is at call time — stock, boltz_graph_patch's, boltz_dit_hoist's — exactly as boltz2_opt.phase does) keys the refresh; the
first denoiser call of a sample() is eager under every sampler of this kit, so the refresh (torch ops, allocations on a new shape) never runs
under capture (asserted by name if it would); the captured step bakes this module's static buffers and reads them on replay; a new item of the
same shape refreshes them in place at its own eager first step. Buffers of other shapes are dropped when a new shape arrives; for inputs at or
above boltz_graph_patch's release threshold they are dropped when sample() returns (as the graph and the hoist cache are).

Programmatic: `import boltz_atom as BA; BA.apply("keys,glue" | "keys,fused", gemm="ieee")` before the model is built (hooks AtomDiffusion.__init__),
or `BA.install(model.structure_module, units, gemm)` on a built model; `BA.STATS` / `BA.report()` the census.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional

import torch

import boltz_atom_kernels as K

UNITS = ("keys", "glue", "fused")
LEVER_OF = {"keys": "atom_keys_gather", "glue": "atom_glue_hoist", "fused": "atom_fused"}
GEMM_LEVER = "atom_gemm"
STATS: Dict[str, Any] = {"units": [], "gemm": "ieee", "release_rule": "graph", "epoch": 0, "installed": 0, "refreshes": 0, "refresh_s": [], "shape_drops": 0, "releases": 0,
                         "calls": {"enc_fused": 0, "dec_fused": 0, "enc_glue": 0, "dec_glue": 0, "keys": 0, "rows_gather": 0},
                         "fallback": {}, "errors": {}, "hoist_c_m": {"hit": 0, "miss": 0}, "last": {}}
_EPOCH = {"n": 0}
_INST: Dict[int, "State"] = {}
_APPLIED = {"init_hooked": False, "units": (), "gemm": "ieee"}


def _log(*a):
    print("[boltz_atom]", *a, file=sys.stderr, flush=True)


def _count(kind: str, key: str, n: int = 1):
    d = STATS[kind]; d[key] = d.get(key, 0) + n


# =============================================================== per-instance state ===============================================================
class State:
    """Static buffers + packed weights of one AtomDiffusion instance."""

    def __init__(self, diff, units, gemm):
        self.units = tuple(units); self.gemm = gemm
        self.filled = None                    # (epoch, key) of the last conditioning refresh
        self.key = None                       # (B, M, NTOK, device) of the resident buffers
        self.bufs: Dict[tuple, torch.Tensor] = {}
        self.w = None                         # packed weights (built lazily on the module's device)
        self.hoist_ok = False

    def buf(self, name, shape, dtype, device):
        k = (name, tuple(shape), dtype, str(device))
        t = self.bufs.get(k)
        if t is None:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"boltz_atom: static buffer {name}{tuple(shape)} requested during CUDA-graph capture — the eager first denoiser step of this sample() did not build it")
            t = torch.empty(tuple(shape), dtype=dtype, device=device)
            self.bufs[k] = t
        return t

    def drop_other_shapes(self, key):
        if self.key is not None and self.key != key:
            self.bufs = {}; self.filled = None
            STATS["shape_drops"] += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.key = key

    def release(self):
        self.bufs = {}; self.filled = None; self.key = None


def _state(diff) -> State:
    st = _INST.get(id(diff))
    if st is None:
        st = State(diff, _APPLIED["units"], _APPLIED["gemm"]); _INST[id(diff)] = st
    return st


# ================================================================= packed weights =================================================================
def _pack_layer(layer, device):
    apb = layer.pair_bias_attn
    f = lambda t: t.detach().to(device=device, dtype=torch.float32).contiguous()  # noqa: E731
    W1 = torch.cat([apb.proj_q.weight, apb.proj_k.weight, apb.proj_v.weight, apb.proj_g.weight], 0)     # [512,128]
    lw = {"W1T": f(W1.t()), "BQ": f(apb.proj_q.bias), "WOT": f(apb.proj_o.weight.t()),
          "WSGT": f(layer.transition.swish_gate[0].weight.t()), "WABT": f(layer.transition.a_to_b.weight.t()), "WBAT": f(layer.transition.b_to_a.weight.t()),
          "eps": float(layer.adaln.a_norm.eps), "eps2": float(layer.transition.adaln.a_norm.eps)}
    lw["bf16"] = {k: v.to(torch.bfloat16) for k, v in lw.items() if torch.is_tensor(v) and k != "BQ"}
    return lw


def _weights(st: State, net, device):
    if st.w is not None and st.w["device"] == str(device):
        return st.w
    enc, dec = net.atom_attention_encoder, net.atom_attention_decoder
    f = lambda t: t.detach().to(device=device, dtype=torch.float32).contiguous()  # noqa: E731
    w = {"device": str(device),
         "enc": [_pack_layer(l, device) for l in enc.atom_encoder.diffusion_transformer.layers],
         "dec": [_pack_layer(l, device) for l in dec.atom_decoder.diffusion_transformer.layers],
         "WR": f(enc.r_to_q_trans.weight),                                   # [128,3]
         "WATT": f(enc.atom_to_token_trans[0].weight.t()),                   # [128,768]
         "LNW": f(dec.atom_feat_to_atom_pos_update[0].weight), "LNB": f(dec.atom_feat_to_atom_pos_update[0].bias),
         "eps_out": float(dec.atom_feat_to_atom_pos_update[0].eps),
         "WPOS": f(dec.atom_feat_to_atom_pos_update[1].weight),              # [3,128]
         }
    w["WATT_bf16"] = w["WATT"].to(torch.bfloat16)
    st.w = w
    return w


def _lw_for(lw, hoist, prec):
    """Kernel weight dict of one layer: packed weights (bf16 copies under the bf16 precision lever) + the hoisted conditioning tensors."""
    d = dict(lw["bf16"]) if prec == "bf16" else {k: v for k, v in lw.items() if torch.is_tensor(v)}
    d["BQ"] = lw["BQ"]
    d.update(hoist)
    return d


# ============================================================ per-sample() hoists ============================================================
def _unrepeat_c(c, B, m):
    """The conditioning c on the un-repeated basis [B, M, 128] fp32: the encoder may hand it repeated (stock c_skip: [B*m, M, 128])."""
    if c.shape[0] == B:
        return c.float()
    if c.shape[0] == B * m:
        return c[::m].float()
    raise RuntimeError(f"boltz_atom: conditioning batch {c.shape[0]} is neither B={B} nor B*m={B * m}")


def _layer_hoists(layer, c):
    """The step-invariant per-atom conditioning of one DiffusionTransformerLayer, the stock expressions on c [B,M,128] fp32:
    AdaLN sigmoid(s_scale(s_norm(c))) and s_bias(s_norm(c)) (attention and transition), the two sigmoid output gates."""
    sn = layer.adaln.s_norm(c)
    S1 = torch.sigmoid(layer.adaln.s_scale(sn)); B1 = layer.adaln.s_bias(sn)
    G1 = layer.output_projection(c)
    sn2 = layer.transition.adaln.s_norm(c)
    S2 = torch.sigmoid(layer.transition.adaln.s_scale(sn2)); B2 = layer.transition.adaln.s_bias(sn2)
    G2 = layer.transition.output_projection(c)
    return {"S1": S1, "B1": B1, "G1": G1, "S2": S2, "B2": B2, "G2": G2}


def _refresh(st: State, net, feats, c, m, need_fused: bool, need_glue: bool):
    """Once per sample() epoch (first, eager denoiser call): fill the static buffers for this (B, M, NTOK) from feats and c."""
    B, M = feats["atom_pad_mask"].shape
    NT = feats["atom_to_token"].shape[-1]
    dev = feats["atom_pad_mask"].device
    key = (int(B), int(M), int(NT), str(dev))
    ep = STATS["epoch"]
    if st.filled == (ep, key):
        return
    if torch.cuda.is_available() and dev.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("boltz_atom: conditioning refresh requested during CUDA-graph capture — the first denoiser call of this sample() was not eager")
    t0 = time.time()
    st.drop_other_shapes(key)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        a2t = feats["atom_to_token"]
        padf = feats["atom_pad_mask"].float()
        st.buf("pad", (B, M), torch.float32, dev).copy_(padf)
        # token index per atom (one-hot rows -> index; empty rows -> -1) and the one-hot check by construction
        a2tf = a2t.float()
        rs = a2tf.sum(-1)
        onehot = bool(((a2t == 0) | (a2t == 1)).all().item()) and bool(((rs == 0) | (rs == 1)).all().item())
        st.onehot = onehot
        tok = torch.where(rs > 0.5, a2tf.argmax(-1), torch.full_like(rs, -1, dtype=torch.long)).to(torch.int32)
        st.buf("tok", (B, M), torch.int32, dev).copy_(tok)
        if need_fused:
            cnt = a2tf.sum(1)                                                                     # [B, NT] atoms per token (fp32, as stock's sum(dim=1))
            st.buf("wtok", (B, NT), torch.float32, dev).copy_(1.0 / (cnt + 1e-6))
            order = torch.argsort(torch.where(tok >= 0, tok, torch.full_like(tok, NT)).long(), dim=1, stable=True).to(torch.int32)   # atoms grouped by token, stable
            st.buf("order", (B, M), torch.int32, dev).copy_(order)
            off = torch.zeros(B, NT + 1, dtype=torch.int32, device=dev)
            off[:, 1:] = torch.cumsum(cnt.round().to(torch.int32), dim=1).to(torch.int32)
            st.buf("off", (B, NT + 1), torch.int32, dev).copy_(off)
            cu = _unrepeat_c(c, B, m).contiguous()
            for part, layers in (("enc", net.atom_attention_encoder.atom_encoder.diffusion_transformer.layers),
                                 ("dec", net.atom_attention_decoder.atom_decoder.diffusion_transformer.layers)):
                for i, layer in enumerate(layers):
                    for name, val in _layer_hoists(layer, cu).items():
                        st.buf((part, i, name), (B, M, 128), torch.float32, dev).copy_(val)
        if need_glue:
            a2tm = a2tf.repeat_interleave(m, 0) if m > 1 else a2tf                              # value-identical to stock's repeat (m == 1: a copy of the same values)
            st.buf("a2t_f", (B * m, M, NT), torch.float32, dev).copy_(a2tm)
            st.buf("a2t_mean", (B * m, M, NT), torch.float32, dev).copy_(a2tm / (a2tm.sum(dim=1, keepdim=True) + 1e-6))
            st.buf("tok_m", (B * m, M), torch.int32, dev).copy_(tok.repeat_interleave(m, 0) if m > 1 else tok)
            st.glue_m = m
    st.filled = (ep, key)
    STATS["refreshes"] += 1; STATS["refresh_s"].append(round(time.time() - t0, 4)); STATS["refresh_s"] = STATS["refresh_s"][-50:]


def _hoist(st, part, i):
    B, M = st.key[0], st.key[1]
    g = lambda n: st.bufs[((part, i, n), (B, M, 128), torch.float32, st.key[3])]  # noqa: E731
    return {n: g(n) for n in ("S1", "B1", "G1", "S2", "B2", "G2")}


def _b(st, name, shape, dtype):
    return st.bufs[(name, tuple(shape), dtype, st.key[3])]


# ================================================================ fused (fast tier) ================================================================
def _fused_transformer(st, part, w, x, qkv, g, ob, bias, m, R, M, Kw, prec, tail):
    """Layers 1..3 of one atom transformer after the first kernel wrote x/qkv/g: 3x (windowed attention, out+transition[+next AdaLN/proj | tail])."""
    B = st.key[0]
    pad = _b(st, "pad", (B, M), torch.float32)
    if bias.dim() != 5 or bias.shape[1] != Kw or bias.shape[2] != 32 or bias.shape[3] != 128:
        raise RuntimeError(f"boltz_atom: pair bias of shape {tuple(bias.shape)} is not [B,K,32,128,L*H]")
    bias = bias.contiguous()
    LD = bias.shape[-1]
    layers = w[part]
    for i in range(3):
        lw = _lw_for(layers[i], _hoist(st, part, i), prec)
        K.launch_win_attn(qkv=qkv, g=g, bias=bias, ld=LD, loff=4 * i, bmult=m, pad=pad, ob=ob, R=R, M=M, K=Kw, cmult=m, prec=prec)
        if i < 2:
            nlw = _lw_for(layers[i + 1], _hoist(st, part, i + 1), prec)
            K.launch_out_transition(ob=ob, x=x, lw=lw, epi=1, nlw=nlw, qkv=qkv, g=g, R=R, M=M, cmult=m, prec=prec, eps=layers[i]["eps2"])
        else:
            tail(lw)


def _encoder_fused(self, feats, q, c, atom_enc_bias, to_keys, r=None, multiplicity=1):
    st = self._boltz_atom_state; net = self._boltz_atom_net
    m = int(multiplicity)
    _refresh(st, net, feats, c, m, need_fused=True, need_glue=False)
    B, M = st.key[0], st.key[1]; NT = st.key[2]
    dev = q.device; prec = st.gemm
    w = _weights(st, net, dev)
    R = B * m * M; Kw = M // 32
    act = torch.bfloat16 if prec == "bf16" else torch.float32
    x = torch.empty((R, 128), dtype=torch.float32, device=dev)
    qkv = torch.empty((R, 384), dtype=act, device=dev); g = torch.empty((R, 128), dtype=act, device=dev); ob = torch.empty((R, 128), dtype=act, device=dev)
    qa = torch.empty((R, 768), dtype=torch.float32, device=dev)
    a = torch.empty((B * m, NT, 768), dtype=torch.float32, device=dev)
    q0 = _unrepeat_c(q, B, m).contiguous()                       # q [B,M,128] (stock repeats it; the kernel indexes the un-repeated rows)
    pos = r.reshape(R, 3).contiguous().float()
    lw0 = _lw_for(w["enc"][0], _hoist(st, "enc", 0), prec)
    K.launch_adaln_qkvg(x=x, pro=1, q0=q0, pos=pos, wr=w["WR"], lw=lw0, qkv=qkv, g=g, R=R, M=M, cmult=m, ntok=NT, prec=prec, eps=w["enc"][0]["eps"])
    watt = w["WATT_bf16"] if prec == "bf16" else w["WATT"]

    def tail(lw):
        K.launch_out_transition(ob=ob, x=x, lw=lw, epi=2, watt=watt, qa=qa, R=R, M=M, cmult=m, prec=prec, eps=w["enc"][2]["eps2"])

    _fused_transformer(st, "enc", w, x, qkv, g, ob, atom_enc_bias, m, R, M, Kw, prec, tail)
    K.launch_segmean(qa=qa, order=_b(st, "order", (B, M), torch.int32), off=_b(st, "off", (B, NT + 1), torch.int32), wtok=_b(st, "wtok", (B, NT), torch.float32),
                     a=a, Bm=B * m, M=M, ntok=NT, mult=m)
    STATS["calls"]["enc_fused"] += 1
    q_out = x.view(B * m, M, 128)
    c_skip = c if m == 1 else c.repeat_interleave(m, 0)          # stock hands the decoder c repeated; shape-identical for m == 1
    return a, q_out, c_skip, to_keys


def _decoder_fused(self, a, q, c, atom_dec_bias, feats, to_keys, multiplicity=1):
    st = self._boltz_atom_state; net = self._boltz_atom_net
    m = int(multiplicity)
    _refresh(st, net, feats, c, m, need_fused=True, need_glue=False)
    B, M = st.key[0], st.key[1]; NT = st.key[2]
    dev = q.device; prec = st.gemm
    w = _weights(st, net, dev)
    R = B * m * M
    with torch.autocast("cuda", enabled=False):
        a_to_q = self.a_to_q_trans(a.float()).contiguous()       # [B*m, NT, 128] — the token-level GEMM stays cuBLAS
    act = torch.bfloat16 if prec == "bf16" else torch.float32
    x = torch.empty((R, 128), dtype=torch.float32, device=dev)
    qkv = torch.empty((R, 384), dtype=act, device=dev); g = torch.empty((R, 128), dtype=act, device=dev); ob = torch.empty((R, 128), dtype=act, device=dev)
    rup = torch.empty((R, 3), dtype=torch.float32, device=dev)
    qskip = q.reshape(R, 128).contiguous()
    lw0 = _lw_for(w["dec"][0], _hoist(st, "dec", 0), prec)
    K.launch_adaln_qkvg(x=x, pro=2, qskip=qskip, a2q=a_to_q, tok=_b(st, "tok", (B, M), torch.int32), lw=lw0, qkv=qkv, g=g, R=R, M=M, cmult=m, ntok=NT, prec=prec,
                        eps=w["dec"][0]["eps"])

    def tail(lw):
        K.launch_out_transition(ob=ob, x=x, lw=lw, epi=3, lnw=w["LNW"], lnb=w["LNB"], wpos=w["WPOS"], rup=rup, R=R, M=M, cmult=m, prec=prec,
                                eps=w["dec"][2]["eps2"], eps_out=w["eps_out"])

    _fused_transformer(st, "dec", w, x, qkv, g, ob, atom_dec_bias, m, R, M, M // 32, prec, tail)
    STATS["calls"]["dec_fused"] += 1
    return rup.view(B * m, M, 3)


# ================================================================ glue (exact tier) ================================================================
def _hoist_c_m(c, m):
    """boltz_dit_hoist's repeated conditioning for THIS c when its cache is live (its identity protocol: the AtomTransformer levels key on it)."""
    DH = sys.modules.get("boltz_dit_hoist")
    if DH is not None:
        try:
            C = DH._active()
            if C is not None and getattr(C, "m", None) == m and C.dcf.get("c") is c and ("c_m",) in C.t:
                STATS["hoist_c_m"]["hit"] += 1
                return C.t[("c_m",)]
        except Exception:  # noqa: BLE001 — a sibling lever's internals: any surprise = the stock statement below, counted
            pass
        STATS["hoist_c_m"]["miss"] += 1
    return c.repeat_interleave(m, 0)


def _encoder_glue(self, feats, q, c, atom_enc_bias, to_keys, r=None, multiplicity=1):
    """AtomAttentionEncoder.forward (encodersv2.py:449-489) verbatim, with the atom_to_token cast / repeat / mean-normalisation hoisted."""
    st = self._boltz_atom_state; net = self._boltz_atom_net
    m = int(multiplicity)
    _refresh(st, net, feats, c, m, need_fused=False, need_glue=True)
    B, N, _ = feats["ref_pos"].shape
    atom_mask = feats["atom_pad_mask"].bool()
    if self.structure_prediction:
        q = q.repeat_interleave(multiplicity, 0)
        r_to_q = self.r_to_q_trans(r)
        q = q + r_to_q
    c = _hoist_c_m(c, m)
    atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
    q = self.atom_encoder(q=q, mask=atom_mask, c=c, bias=atom_enc_bias, multiplicity=multiplicity, to_keys=to_keys)
    Bm, M, NT = st.key[0] * m, st.key[1], st.key[2]
    with torch.autocast("cuda", enabled=False):
        q_to_a = self.atom_to_token_trans(q).float()
        atom_to_token_mean = _b(st, "a2t_mean", (Bm, M, NT), torch.float32)
        a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)                             # the stock call on the stock (transposed-view) operand
    a = a.to(q)
    STATS["calls"]["enc_glue"] += 1
    return a, q, c, to_keys


def _decoder_glue(self, a, q, c, atom_dec_bias, feats, to_keys, multiplicity=1):
    """AtomAttentionDecoder.forward (encodersv2.py:540-569) verbatim, with the one-hot broadcast bmm served by a row gather."""
    st = self._boltz_atom_state; net = self._boltz_atom_net
    m = int(multiplicity)
    _refresh(st, net, feats, c, m, need_fused=False, need_glue=True)
    Bm, M, NT = st.key[0] * m, st.key[1], st.key[2]
    with torch.autocast("cuda", enabled=False):
        a_to_q = self.a_to_q_trans(a.float())
        if st.onehot:
            a_to_q = K.gather_rows(a_to_q, _b(st, "tok_m", (Bm, M), torch.int32)); STATS["calls"]["rows_gather"] += 1
        else:                                                                                     # never by construction (featurizer one_hot); the stock statement, counted by name
            _count("fallback", "a2t_not_onehot")
            a_to_q = torch.bmm(_b(st, "a2t_f", (Bm, M, NT), torch.float32), a_to_q)
    q = q + a_to_q.to(q)
    atom_mask = feats["atom_pad_mask"]
    atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
    q = self.atom_decoder(q=q, mask=atom_mask, c=c, bias=atom_dec_bias, multiplicity=multiplicity, to_keys=to_keys)
    r_update = self.atom_feat_to_atom_pos_update(q)
    STATS["calls"]["dec_glue"] += 1
    return r_update


# ================================================================ keys (exact tier) ================================================================
class GatherToKeys:
    """`to_keys` served by the gather kernel: x [B, K*32, Dw] -> [B, K, 128, Dw] (single_to_keys' contract; K from the input)."""
    __slots__ = ("W", "H", "inner")

    def __init__(self, inner=None, W=32, H=128):
        self.W, self.H, self.inner = W, H, inner

    def __call__(self, x):
        B, N, Dw = x.shape
        if self.W != 32 or self.H != 128 or N % 32:
            _count("fallback", "keys_shape")
            return self.inner(x)
        if _CHECK_FINITE[0] and not bool(torch.isfinite(x).all()):                             # test hook only (a device sync): never on in a run
            raise RuntimeError("boltz_atom: non-finite operand handed to the keys gather")
        STATS["calls"]["keys"] += 1
        return K.gather_keys(x, N // 32)


_CHECK_FINITE = [False]


def _wrap_to_keys(tk):
    if tk is None or isinstance(tk, GatherToKeys):
        return tk
    W = H = None
    kw = getattr(tk, "keywords", None)
    if isinstance(kw, dict):
        W, H = kw.get("W"), kw.get("H")
    if (W, H) != (32, 128):
        _count("fallback", f"keys_window_{W}x{H}")
        return tk
    return GatherToKeys(inner=tk, W=W, H=H)


# ============================================================= install / sample wrapper =============================================================
_RELEASE = {"rule": "graph"}     # graph: boltz_graph_patch's token threshold decides (the sampler group's rule); sample: every sample() releases the static buffers at its
                                 # return (the memory row: nothing of the fused kernels resident under the next prediction's trunk peak; the eager first step rebuilds them)


def set_release(rule: str) -> None:
    if rule not in ("graph", "sample"):
        raise RuntimeError(f"boltz_atom: release rule {rule!r} is not a value (graph | sample)")
    _RELEASE["rule"] = rule; STATS["release_rule"] = rule


def _release_due(n_tokens: int) -> bool:
    if _RELEASE["rule"] == "sample":
        return True
    BGP = sys.modules.get("boltz_graph_patch")
    if BGP is not None and hasattr(BGP, "release_due"):
        try:
            return bool(BGP.release_due(n_tokens))
        except Exception:  # noqa: BLE001
            return False
    return False


def _make_sample(sm, inner):
    def boltz_atom_sample(*a, **k):
        _EPOCH["n"] += 1; STATS["epoch"] = _EPOCH["n"]
        st = _state(sm)
        if "keys" in st.units:
            dc = k.get("diffusion_conditioning")
            if isinstance(dc, dict) and "to_keys" in dc:
                dc2 = dict(dc); dc2["to_keys"] = _wrap_to_keys(dc["to_keys"])
                k = dict(k); k["diffusion_conditioning"] = dc2
        try:
            return inner(*a, **k) if inner is not None else type(sm).sample(sm, *a, **k)
        finally:
            feats = k.get("feats")
            n_tok = int(feats["token_pad_mask"].shape[-1]) if isinstance(feats, dict) and torch.is_tensor(feats.get("token_pad_mask")) else 0
            rel = _release_due(n_tok)
            if rel:
                st.release(); STATS["releases"] += 1
            STATS["last"] = {"units": list(st.units), "gemm": st.gemm, "epoch": STATS["epoch"], "refreshes": STATS["refreshes"], "released": rel,
                             "calls": dict(STATS["calls"]), "fallback": dict(STATS["fallback"]), "hoist_c_m": dict(STATS["hoist_c_m"]),
                             "resident_mb": round(sum(t.numel() * t.element_size() for t in st.bufs.values()) / 2**20, 1)}
    boltz_atom_sample._boltz_atom = True
    boltz_atom_sample._inner = inner
    return boltz_atom_sample


def new_epoch() -> int:
    """Declare a new sample() epoch by hand (tests / direct score-model calls outside sample()): the next encoder call re-hoists."""
    _EPOCH["n"] += 1; STATS["epoch"] = _EPOCH["n"]
    return _EPOCH["n"]


def install(sm, units=None, gemm=None) -> Dict[str, Any]:
    """Install on a built AtomDiffusion instance: the sample epoch wrapper (+ keys), and per unit the encoder/decoder instance forwards."""
    units = tuple(units) if units is not None else _APPLIED["units"]
    gemm = gemm or _APPLIED["gemm"]
    bad = [u for u in units if u not in UNITS]
    if bad:
        raise ValueError(f"boltz_atom: unknown unit(s) {bad}; units are {UNITS}")
    if gemm not in K.PRECS:
        raise ValueError(f"boltz_atom: gemm precision {gemm!r} not in {K.PRECS}")
    if ("fused" in units or "keys" in units or "glue" in units) and not K.triton_ok():
        raise RuntimeError(f"boltz_atom: refused — triton unavailable ({K._TRITON['why']})")
    st = _state(sm); st.units = units; st.gemm = gemm
    cur = sm.__dict__.get("sample")
    if not getattr(cur, "_boltz_atom", False):
        sm.__dict__["sample"] = _make_sample(sm, cur)
    net = sm.score_model
    enc, dec = net.atom_attention_encoder, net.atom_attention_decoder
    for mod in (enc, dec):                                       # plain instance attributes (nn.Module.__setattr__ would register `net` as a submodule: a cycle)
        mod.__dict__["_boltz_atom_state"] = st; mod.__dict__["_boltz_atom_net"] = net
    import types
    if "fused" in units:
        if not getattr(enc, "structure_prediction", True):
            raise RuntimeError("boltz_atom: refused — the fused encoder serves the structure module's encoder (structure_prediction=True) only")
        enc.__dict__["forward"] = types.MethodType(_encoder_fused, enc); dec.__dict__["forward"] = types.MethodType(_decoder_fused, dec)
        st.bodies = "fused"
    elif "glue" in units:
        enc.__dict__["forward"] = types.MethodType(_encoder_glue, enc); dec.__dict__["forward"] = types.MethodType(_decoder_glue, dec)
        st.bodies = "glue"
    else:
        enc.__dict__.pop("forward", None); dec.__dict__.pop("forward", None); st.bodies = "stock"
    STATS["installed"] += 1; STATS["units"] = list(units); STATS["gemm"] = gemm
    return {"units": list(units), "gemm": gemm, "bodies": st.bodies}


def uninstall(sm) -> None:
    net = sm.score_model
    for mod in (net.atom_attention_encoder, net.atom_attention_decoder):
        mod.__dict__.pop("forward", None); mod.__dict__.pop("_boltz_atom_state", None); mod.__dict__.pop("_boltz_atom_net", None)
    cur = sm.__dict__.get("sample")
    if getattr(cur, "_boltz_atom", False):
        if cur._inner is not None:
            sm.__dict__["sample"] = cur._inner
        else:
            sm.__dict__.pop("sample", None)
    _INST.pop(id(sm), None)


def parse_units(text: Optional[str]):
    raw = [t.strip().lower() for t in (text or "").split(",") if t.strip()]
    if raw in (["off"], ["0"]):
        return ()
    bad = [t for t in raw if t not in UNITS]
    if bad:
        raise ValueError(f"BOLTZ_ATOM={text!r}: {bad} are not units (the units are {'|'.join(UNITS)})")
    return tuple(u for u in UNITS if u in raw)


def apply(units=None, gemm=None):
    """Arm: every AtomDiffusion built after this call is installed with `units` (env BOLTZ_ATOM when None) at GEMM precision `gemm`
    (env BOLTZ_ATOM_GEMM, default ieee). Empty units = disarm (instances built later stay stock). Returns the units armed."""
    if units is None:
        units = os.environ.get("BOLTZ_ATOM", "")
    if isinstance(units, str):
        units = parse_units(units)
    gemm = (gemm or os.environ.get("BOLTZ_ATOM_GEMM") or "ieee").strip().lower()
    if gemm not in K.PRECS:
        raise ValueError(f"BOLTZ_ATOM_GEMM={gemm!r}: not one of {K.PRECS}")
    _APPLIED["units"] = tuple(units); _APPLIED["gemm"] = gemm
    STATS["units"] = list(units); STATS["gemm"] = gemm
    from boltz.model.modules.diffusionv2 import AtomDiffusion
    if units and not _APPLIED["init_hooked"]:
        stock_init = AtomDiffusion.__init__

        def __init__(self, *a, **k):
            stock_init(self, *a, **k)
            if _APPLIED["units"]:
                install(self, _APPLIED["units"], _APPLIED["gemm"])

        AtomDiffusion._boltz_atom_stock_init = stock_init
        AtomDiffusion.__init__ = __init__
        _APPLIED["init_hooked"] = True
    _log("armed units =", ",".join(units) or "(none)", "| gemm =", gemm)
    return tuple(units)


def report() -> Dict[str, Any]:
    return {"units": list(STATS["units"]), "gemm": STATS["gemm"], "levers": [LEVER_OF[u] for u in STATS["units"]] + ([GEMM_LEVER] if STATS["gemm"] != "ieee" else []),
            "installed": STATS["installed"], "refreshes": STATS["refreshes"], "calls": dict(STATS["calls"]), "fallback": dict(STATS["fallback"]),
            "errors": dict(STATS["errors"]), "hoist_c_m": dict(STATS["hoist_c_m"]), "shape_drops": STATS["shape_drops"], "releases": STATS["releases"],
            "triton": K._TRITON, "cfg": K.cfg(), "last": dict(STATS["last"])}
