"""Synthetic pair-stack modules for the tests and the GPU proof of ``opt_core.mem.chunk``: the OpenFold-shaped statements of a pair
transition, a triangle attention (starting / ending node, with the row-chunked path stock engines run above their thresholds), a
triangle multiplication (outgoing / incoming), a confidence statement over pair logits and an MSA
transition. Small, engine-free, exact statements: the chunked forms are held to these with torch.equal. Not part of the package."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from opt_core.mem import chunk as C


class PairTransition(nn.Module):
    """LayerNorm → two up-projections (SwiGLU) → down-projection, per position; ``forward(x)`` is the un-chunked statement."""

    def __init__(self, c: int, n: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, n * c, bias=False)
        self.fc2 = nn.Linear(c, n * c, bias=False)
        self.fc3 = nn.Linear(n * c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        return self.fc3(F.silu(self.fc1(h)) * self.fc2(h))


class TriangleAttention(nn.Module):
    """Triangle attention over the rows of ``x [..., I, J, C]``: LayerNorm, a triangle bias from every (j, k) pair, q/k/v/gate projections,
    softmax over the key axis (fp32), gating, output projection. ``forward(x, mask, chunk_size)`` is the stock statement: the bias from a
    LayerNorm'd copy of the whole pair, then the core over all query rows (``chunk_size=None``) or over row blocks of the LayerNorm'd copy."""

    def __init__(self, c: int, c_hidden: int, no_heads: int, starting: bool, inf: float = 1e9):
        super().__init__()
        self.c_hidden, self.no_heads, self.starting, self.inf = c_hidden, no_heads, starting, inf
        self.ln = nn.LayerNorm(c)
        self.linear_bias = nn.Linear(c, no_heads, bias=False)
        self.linear_q = nn.Linear(c, c_hidden * no_heads, bias=False)
        self.linear_k = nn.Linear(c, c_hidden * no_heads, bias=False)
        self.linear_v = nn.Linear(c, c_hidden * no_heads, bias=False)
        self.linear_g = nn.Linear(c, c_hidden * no_heads)
        self.linear_o = nn.Linear(c_hidden * no_heads, c)

    def bias_from_ln(self, x_ln: torch.Tensor) -> torch.Tensor:
        return self.linear_bias(x_ln).float()                                   # [..., R, J, H]

    def attend(self, x_ln_rows: torch.Tensor, bias_full: torch.Tensor, mask_rows: Optional[torch.Tensor]) -> torch.Tensor:
        H, D = self.no_heads, self.c_hidden
        lead = x_ln_rows.shape[:-1]
        q = self.linear_q(x_ln_rows).view(*lead, H, D)
        k = self.linear_k(x_ln_rows).view(*lead, H, D)
        v = self.linear_v(x_ln_rows).view(*lead, H, D)
        logits = torch.einsum("...rjhd,...rkhd->...rhjk", q, k) * (1.0 / math.sqrt(D))
        logits = logits + bias_full.permute(*range(bias_full.dim() - 3), -1, -3, -2).unsqueeze(-4)   # [..., 1, H, J, K]
        if mask_rows is not None:
            logits = logits + (self.inf * (mask_rows - 1)).to(logits.dtype)[..., :, None, None, :]   # [..., R, 1, 1, K]
        p = torch.softmax(logits.float(), dim=-1).to(q.dtype)
        o = torch.einsum("...rhjk,...rkhd->...rjhd", p, v)
        o = o * torch.sigmoid(self.linear_g(x_ln_rows)).view(*lead, H, D)
        return self.linear_o(o.reshape(*lead, H * D))

    def parts(self) -> C.TriAttnParts:
        return C.TriAttnParts(layer_norm=self.ln, bias=self.bias_from_ln, attention=self.attend)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, chunk_size: Optional[int] = None) -> torch.Tensor:
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = None if mask is None else mask.transpose(-1, -2)
        x_ln = self.ln(x)
        bias = self.bias_from_ln(x_ln)
        if chunk_size is None:
            out = self.attend(x_ln, bias, mask)
        else:
            n = x_ln.shape[-3]
            out = torch.empty_like(x_ln)
            for i0 in range(0, n, chunk_size):
                i1 = min(n, i0 + chunk_size)
                out[..., i0:i1, :, :] = self.attend(x_ln[..., i0:i1, :, :], bias, None if mask is None else mask[..., i0:i1, :])
        if not self.starting:
            out = out.transpose(-2, -3)
        return out


class TriangleMultiplication(nn.Module):
    """Outgoing (``...ikc,...jkc->...ijc``) or incoming (``...kic,...kjc->...ijc``) triangle multiplication with gated, masked operands and
    an output LayerNorm over the hidden channels; ``forward(x, mask)`` is the stock statement."""

    def __init__(self, c: int, c_hidden: int, outgoing: bool):
        super().__init__()
        self.outgoing, self.c_hidden = outgoing, c_hidden
        self.ln_in = nn.LayerNorm(c)
        self.p_a, self.g_a = nn.Linear(c, c_hidden), nn.Linear(c, c_hidden)
        self.p_b, self.g_b = nn.Linear(c, c_hidden), nn.Linear(c, c_hidden)
        self.ln_out = nn.LayerNorm(c_hidden)
        self.linear_z = nn.Linear(c_hidden, c)
        self.linear_g = nn.Linear(c, c)
        self.eq = "...ikc,...jkc->...ijc" if outgoing else "...kic,...kjc->...ijc"

    def operand(self, x_ln: torch.Tensor, mask: Optional[torch.Tensor], which: str) -> torch.Tensor:
        p, g = getattr(self, f"p_{which}"), getattr(self, f"g_{which}")
        a = torch.sigmoid(g(x_ln)) * p(x_ln)
        return a if mask is None else a * mask.unsqueeze(-1).to(a.dtype)

    def product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum(self.eq, a, b)

    def finish(self, prod: torch.Tensor, x_ln: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear_g(x_ln)) * self.linear_z(self.ln_out(prod))

    def parts(self) -> C.TriMulParts:
        return C.TriMulParts(layer_norm_in=self.ln_in, operand_a=lambda h, m: self.operand(h, m, "a"), operand_b=lambda h, m: self.operand(h, m, "b"),
                             product=self.product, finish=self.finish, outgoing=self.outgoing)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_ln = self.ln_in(x)
        return self.finish(self.product(self.operand(x_ln, mask, "a"), self.operand(x_ln, mask, "b")), x_ln)


class ConfidenceStatement(nn.Module):
    """Per-pair logits ``[..., I, J, bins]`` → expected error (softmax over bins × bin centres), a contact probability (the first ``contact_bins``
    bins) and the per-row alignment term (mean over J of ``1 / (1 + (e / d0)^2)``); ``forward`` adds the one across-row finishing statement (the
    max over rows), which stays outside the chunk."""

    def __init__(self, bins: int, max_error: float = 32.0, d0: float = 8.0, contact_bins: int = 8):
        super().__init__()
        step = max_error / bins
        self.register_buffer("centers", torch.arange(bins, dtype=torch.float32) * step + step / 2)
        self.d0, self.contact_bins = d0, contact_bins

    def per_rows(self, logits_rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        p = torch.softmax(logits_rows.float(), dim=-1)
        err = (p * self.centers).sum(-1)
        contact = p[..., : self.contact_bins].sum(-1)
        tm_row = (1.0 / (1.0 + (err / self.d0) ** 2)).mean(-1)
        return {"err": err, "contact": contact, "tm_row": tm_row}

    def forward(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        d = dict(self.per_rows(logits))
        d["tm"] = d["tm_row"].max(dim=-1).values
        return d


def finish_confidence(rows: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """The finishing statement on the assembled per-row outputs (the across-row max)."""
    d = dict(rows)
    d["tm"] = d["tm_row"].max(dim=-1).values
    return d


_PAIR_FORWARD = PairTransition.forward


class MSATransition(PairTransition):
    """The same statement on an MSA-shaped ``[..., S, N, C]`` tensor (per position); its own ``forward`` (bound to the un-patched body at
    import) so that a lever patched on ``PairTransition`` does not reach it through inheritance."""

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return _PAIR_FORWARD(self, m)


def seeded(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
