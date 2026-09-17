"""Lever atom_sdpa (class fast: tolerance) — the WINDOWED atom attention of the diffusion denoiser through torch's fused SDPA kernel.

Stock: Attention.forward (attention.py L42-100) serves every attention of the model.  The atom transformers of the denoiser's AtomEncoder /
AtomDecoder (diffusion_transformer.py AtomTransformerStack / AtomTransformerBlock / CrossAttention; 3 blocks per stack, c_atom 96 = 2 heads x 48)
call it with RANK-5 operands: a_q [B, N, W, Lq=56, 96] (the 4-residue x 14-atom query windows), a_k [B, N, W, Lk=168, 96] (the 12-residue key
windows, an overlapping unfold view), mask [B, 1, W, Lq, Lk] (bool: chain / residue-distance / padding), pair_bias [B, 1, W, H, Lq, Lk] fp32.
After the head split the SDPA operands are rank 6, which torch serves with the MATH backend: q / k / v copied contiguous, the [B, N, W, H, Lq, Lk]
fp32 logits materialised (bmm), `+ bias`, _safe_softmax (softmax + isneginf + all + where passes), a second bmm, the output re-arranged.
This lever: the same q / k / v linears (stock modules, stock precision), viewed [B, N, W, H, L, D] without a copy; the stock two bias statements
summed ONCE into a buffer whose key extent is padded to a multiple of 16 (the fused kernels' alignment rule; the [..., :Lk] view is what is
read — no torch-side pad copy, ever); then per (b, n) ONE fused call F.scaled_dot_product_attention(q[b,n], k[b,n], v[b,n], attn_mask=bias[b, 0|n])
over the batch of W windows under the mem-efficient | cuDNN | flash pin (fp32 operands -> the mem-efficient kernel: fp32 in / out, 3xTF32
tensor-core products, online softmax — never the [.., Lq, Lk] logits in HBM), outputs stacked back to [B, N, W, Lq, H*D] (the stock rearrange's
one copy).  The N (sample) loop is what keeps the bias a real-stride tensor: the stock bias is shared by the N samples (stride 0 over N), which one
folded (N*W) batch cannot express without materialising N copies.  Numerics: re-associated softmax and 3xTF32 products -> tolerance class,
deterministic; rank-3 (Pairformer), rank-4 (DiT) and use_high_precision calls keep the statement below this
wrapper BY NAME (`rank` / `high_precision`), a key window lead other than (B, N, W) or a bias that is not [B, 1|N, W, 1|H, 1|Lq, Lk] likewise
(`kv_lead` / `bias_form`).  Levers on the same attribute compose in install (row) order: what this lever does not serve reaches the
forward below it unchanged.  AFO_ATOM_SDPA=0: installed, every call `disabled` (the kit's AFO_* convention); MODEL_OPT_LEVERS_OFF=atom_sdpa
removes it from the row.  Under denoiser_graph the fused launches are captured like any other kernel (census = warm-up + capture calls)."""
from __future__ import annotations

import os

from . import Installed, rebind, size_gated

NAME = "LOCAL.atlasfold.atom_sdpa"
ENV = "AFO_ATOM_SDPA"
TARGET = "atlasfold.model.network.attention"
EXPECTED = ("disabled", "rank", "high_precision", "kv_lead", "bias_form")   # `no_fused_backend` is NOT expected (a torch whose fused kernels decline the call: stock statement, gate refused)
ALIGN = 16                                                                    # key extent of the bias buffer padded to this (elements): every stride the fused kernels check is then aligned
_STATE = {"override": None}


def enabled() -> bool:
    if _STATE["override"] is not None:
        return bool(_STATE["override"])
    return os.environ.get(ENV, "1") != "0"


def bench_arm(label: str) -> None:
    """In-process override of AFO_ATOM_SDPA (not a kit switch): label 'A' = the statement below (lever inert), any other label = the fused SDPA."""
    _STATE["override"] = (label != "A")


bench_arm.state = lambda: {"atom_sdpa": enabled()}   # type: ignore[attr-defined]


PROBE_SHAPE = (8, 2, 56, 168, 48)                                              # W, H, Lq, Lk, D of one windowed fp32 call (the production geometry, a few windows)


def probe() -> dict:
    """The route ``check --mode fast`` prints: does a fused SDPA backend accept the fp32 [W, H, 56, 48] x [W, H, 168, 48] call with a [W, H, 56, 168]
    additive bias read from a 16-aligned buffer on this device?  {ok, backend|reason}; ok None without a CUDA device (nothing to route on)."""
    try:
        import torch
        import torch.nn.functional as Fn
        from torch.nn.attention import sdpa_kernel, SDPBackend
    except Exception as e:  # noqa: BLE001
        return {"kernel": "sdpa_fused_fp32_window", "ok": None, "reason": f"torch.nn.attention unavailable: {type(e).__name__}"}
    if not torch.cuda.is_available():
        return {"kernel": "sdpa_fused_fp32_window", "ok": None, "routed": None, "resolved": None, "reason": "no_cuda"}
    W, H, Lq, Lk, D = PROBE_SHAPE
    try:
        q = torch.randn(W, H, Lq, D, device="cuda"); k = torch.randn(W, H, Lk, D, device="cuda"); v = torch.randn_like(k)
        Lkp = -(-Lk // ALIGN) * ALIGN
        bias = torch.randn(W, H, Lq, Lkp, device="cuda")[..., :Lk]
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]):
            Fn.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        torch.cuda.synchronize()
        return {"kernel": "sdpa_fused_fp32_window", "ok": True, "routed": True, "resolved": "torch.sdpa[efficient|cudnn]", "reason": None}
    except Exception as e:  # noqa: BLE001 — no fused backend takes the fp32 windowed call: atom_sdpa would count `no_fused_backend` (the statement below, gate refused); say so here first
        return {"kernel": "sdpa_fused_fp32_window", "ok": False, "routed": False, "resolved": None, "reason": f"{type(e).__name__}: {str(e)[:160]}"}


def windowed_sdpa(Fn, q6, k6, v6, bias6, pin):
    """q6 [B,N,W,H,Lq,D], k6 / v6 [B,N,W,H,Lk,D] (views), bias6 [B, 1|N, W, H, Lq, Lk] (aligned view) -> [B, N, W, Lq, H*D] (one copy)."""
    import torch
    B, N, W, H, Lq, D = q6.shape
    Nb = bias6.shape[1]
    rows = []
    with pin:
        for b in range(B):
            outs = [Fn.scaled_dot_product_attention(q6[b, n], k6[b, n], v6[b, n], attn_mask=bias6[b, n if Nb == N else 0]) for n in range(N)]
            rows.append(torch.stack(outs, 0))                                   # [N, W, H, Lq, D]
    out = rows[0].unsqueeze(0) if B == 1 else torch.stack(rows, 0)              # [B, N, W, H, Lq, D]
    return out.permute(0, 1, 2, 4, 3, 5).reshape(B, N, W, Lq, H * D)           # == stock's rearrange('... h lq d -> ... lq (h d)')


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    import torch
    import torch.nn.functional as Fn
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from opt_core.counters import Ledger
    try:
        att = importlib.import_module(TARGET)
    except Exception as e:  # noqa: BLE001
        return Installed("atom_sdpa", False, reason=f"import:{TARGET}:{type(e).__name__}")
    cls = att.Attention
    stock = cls.forward                                                         # Attention.forward at install time: upstream's, or the wrapper of a lever installed earlier on this attribute
    ledger = Ledger(NAME, impl=f"torch.sdpa(fused,windowed)@{torch.__version__.split('+')[0]}", origin="kit", expected=EXPECTED)
    FUSED = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]

    def forward(self, a_q, a_k, mask, pair_bias):
        if not enabled():
            ledger.fallback("disabled")
            return stock(self, a_q, a_k, mask, pair_bias)
        if a_q.dim() != 5 or a_k.dim() != 5:                                   # rank 3 (Pairformer) / rank 4 (DiT) calls: not this lever's statement
            ledger.fallback("rank")
            return stock(self, a_q, a_k, mask, pair_bias)
        if self.use_high_precision:
            ledger.fallback("high_precision")
            return stock(self, a_q, a_k, mask, pair_bias)
        B, N, W, Lq, _ = a_q.shape
        Lk = a_k.shape[-2]
        if tuple(a_k.shape[:3]) != (B, N, W):
            ledger.fallback("kv_lead")
            return stock(self, a_q, a_k, mask, pair_bias)
        H, D = self.num_heads, self.head_dim
        mb = None if mask is None else ((~mask).unsqueeze(-3))                  # [B, 1|N, W, 1, 1|Lq, Lk] bool (the stock statement's operand, before the dtype copy)
        pb = pair_bias
        ok = (mb is None or (mb.dim() == 6 and mb.shape[0] == B and mb.shape[1] in (1, N) and mb.shape[2] == W and mb.shape[4] in (1, Lq) and mb.shape[5] == Lk)) and \
             (pb is None or (pb.dim() == 6 and pb.shape[0] == B and pb.shape[1] in (1, N) and pb.shape[2] == W and pb.shape[3] in (1, H) and pb.shape[4] in (1, Lq) and pb.shape[5] == Lk)) and \
             not (mb is None and pb is None)
        if not ok:
            ledger.fallback("bias_form")
            return stock(self, a_q, a_k, mask, pair_bias)
        q = self.linear_q(a_q)                                                  # the stock linears (stock precision words apply to them, not to this lever)
        k, v = self.linear_k(a_k), self.linear_v(a_k)
        q6 = q.view(B, N, W, Lq, H, D).permute(0, 1, 2, 4, 3, 5)                # [B, N, W, H, Lq, D] views, last dim contiguous
        k6 = k.view(B, N, W, Lk, H, D).permute(0, 1, 2, 4, 3, 5)
        v6 = v.view(B, N, W, Lk, H, D).permute(0, 1, 2, 4, 3, 5)
        Nb = max(1 if mb is None else mb.shape[1], 1 if pb is None else pb.shape[1])
        Lkp = -(-Lk // ALIGN) * ALIGN
        buf = torch.empty((B, Nb, W, H, Lq, Lkp), dtype=q.dtype, device=q.device)
        bias6 = buf[..., :Lk]                                                   # strides (.., H*Lq*Lkp, Lq*Lkp, Lkp, 1): aligned whatever Lk is
        if mb is not None and pb is not None:
            torch.add(mb.to(q.dtype) * (-self.inf), pb.to(q.dtype), out=bias6)  # the stock two statements, summed once into the aligned buffer
        elif pb is not None:
            bias6.copy_(pb.to(q.dtype).expand(B, Nb, W, H, Lq, Lk))
        else:
            bias6.copy_((mb.to(q.dtype) * (-self.inf)).expand(B, Nb, W, H, Lq, Lk))
        try:
            out = windowed_sdpa(Fn, q6, k6, v6, bias6, sdpa_kernel(FUSED))
        except RuntimeError:                                                    # no fused backend takes the call on this torch / device: the statement below, by name (the gate refuses it)
            ledger.fallback("no_fused_backend")
            return stock(self, a_q, a_k, mask, pair_bias)
        ledger.serve(f"W{W}xN{N}x{Lq}x{Lk}")
        return out
    forward.__qualname__ = "Attention.forward[atlasfold_opt:atom_sdpa]"
    rebind(cls, "forward", forward, stock)
    return Installed("atom_sdpa", True, lines=[lambda: ledger.line(tag)], gates=[size_gated(ledger)],
                     facts={"impl": getattr(ledger, "impl", None), "env": os.environ.get(ENV, "1"), "ledger": ledger})
