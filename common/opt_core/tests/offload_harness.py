"""The ``host_park`` test launcher over ``tests/synthetic_pair.py`` — the ONE synthetic pair stack of the mem tests. No pair statement is
defined here: the transition and the triangle attention are ``synthetic_pair``'s; this file holds (a) the lever's own exactness-class
probes, which are not pair statements — ``ScaleShift`` (LayerNorm + affine: GEMM-free, the bit-exact anchor), ``RowGate`` / ``ColGate``
(a mean over the whole non-streamed axis: separable by ONE dim only — the wrong-dim negative test, and the fp32-reduction ``band``
class), ``ParamBank`` (a small linear beside a large resident parameter bank: module parking); (b) the composition ``SynthStack``;
(c) the streamed form of every op class (:func:`ops`: whole-tensor statement beside its ``host_park`` form, with the exact label and
its reason) — the trunk runners :func:`run_reference` (everything resident) and :func:`run_offloaded` (z / z_init parked, every op
streamed by its separable dim, the head module parked) and the per-op exactness phase of the GPU proof all compose from it. The
triangle attention (starting node) streams as the engines' row-chunked stock path does: a resident ``[..., N, N, H]`` bias filled
from row blocks through the ``HostTensor.rows`` accessor, then row attention against it through ``stream_blocks``. Not part of the
package."""
from __future__ import annotations

from typing import Callable, List, Tuple

import torch
from torch import nn

from synthetic_pair import PairTransition, TriangleAttention  # noqa: F401  (the pair statements; re-exported for the tests)


class ScaleShift(nn.Module):
    """LayerNorm over C, then a per-channel affine: per position, GEMM-free — the bit-exact anchor under streaming by either dim."""

    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.gamma = nn.Parameter(torch.ones(c) * 1.5)
        self.beta = nn.Parameter(torch.zeros(c) + 0.25)

    def forward(self, z):
        return self.ln(z) * self.gamma + self.beta


class RowGate(nn.Module):
    """out[..., i, j, :] = z[..., i, j, :] * sigmoid(W mean_j z[..., i, j, :]) — depends on the whole row i (separable by rows only);
    the mean over the non-streamed axis is an fp32 reduction whose order the kernel picks by shape: ``band``, not bit-exact."""

    def __init__(self, c: int):
        super().__init__()
        self.lin = nn.Linear(c, c)

    def forward(self, z):
        return z * torch.sigmoid(self.lin(z.mean(dim=-2, keepdim=True)))


class ColGate(nn.Module):
    """The transpose of RowGate: depends on the whole column j (separable by cols only); ``band`` for the same reason."""

    def __init__(self, c: int):
        super().__init__()
        self.lin = nn.Linear(c, c)

    def forward(self, z):
        return z * torch.sigmoid(self.lin(z.mean(dim=-3, keepdim=True)))


class ParamBank(nn.Module):
    """z + (z W^T) * bank[0]: a small linear plus a parameter bank of ``bank_rows × c`` that is resident for the call and unused past row 0."""
    def __init__(self, c: int, bank_rows: int):
        super().__init__()
        self.lin = nn.Linear(c, c)
        self.bank = nn.Parameter(torch.randn(bank_rows, c) * 0.01)

    def forward(self, z):
        return z + self.lin(z) * self.bank[0]


class SynthStack(nn.Module):
    """``n_blocks`` × [transition (residual) · triangle attention, starting node (residual) · row gate · col gate · scale-shift · recycle
    add] and a ParamBank head. ``n`` is the transition's hidden factor (``n=1``: hidden = C), ``c_hidden`` / ``no_heads`` the attention's."""

    def __init__(self, c: int = 8, n: int = 1, c_hidden: int = 4, no_heads: int = 2, n_blocks: int = 2, bank_rows: int = 64):
        super().__init__()
        self.c, self.no_heads = c, no_heads
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({"trans": PairTransition(c, n), "attn": TriangleAttention(c, c_hidden, no_heads, starting=True),
                                              "row": RowGate(c), "col": ColGate(c), "ss": ScaleShift(c)}))
        self.head = ParamBank(c, bank_rows)


def attn_block_for(n: int, no_heads: int, transient_bytes: int = 1 << 30) -> int:
    """Rows per attention block so the fp32 logits ``[block, H, N, N]`` stay under ``transient_bytes`` (1 … 32); the reference's
    ``chunk_size`` and the streamed block are the same value, so both arms run the same per-block arithmetic."""
    return max(1, min(32, transient_bytes // max(1, no_heads * n * n * 4)))


REASONS = {
    "transition": "GEMM-bearing (M = block*N vs N*N): declared measured",
    "tri_attn_start": "LN + bias GEMM per row block (M = block*N) and the row attention against the resident bias: declared measured",
    "row_gate": "mean over the whole non-streamed axis: an fp32 reduction whose order depends on the shape — band by construction",
    "col_gate": "mean over the whole non-streamed axis: an fp32 reduction whose order depends on the shape — band by construction",
    "scale_shift": "LayerNorm over C + affine, GEMM-free: bitwise",
    "recycle_add": "elementwise add of a parked row block: bitwise",
    "head": "GEMM-bearing head (M = block*N): declared measured",
}


def stream_attn(m: TriangleAttention, park, src: str, dst: str, attn_block: int) -> None:
    """The starting-node triangle attention over a parked pair ``src`` into ``dst``: pass 1 fills a device-resident ``[..., N, N, H]``
    bias from row blocks (``HostTensor.rows`` accessor: no write-back), pass 2 streams ``attn_block`` rows at a time against it
    (``stream_blocks``, residual added on the block). The reference is ``z + m(z, None, chunk_size=attn_block)``."""
    zt = park.get(src)
    lead, n = zt.shape[:-3], zt.shape[-3]
    bias = torch.empty(*lead, n, n, m.no_heads, dtype=torch.float32, device=park.device)
    step = park.s.block
    for a in range(0, n, step):
        e = min(n, a + step)
        rows = zt.rows(a, e)
        bias[..., a:e, :, :] = m.bias_from_ln(m.ln(rows))
        del rows
    park.stream_blocks(lambda b, a, e: b + m.attend(m.ln(b), bias, None), src, "rows", block=attn_block, out=dst, exact="measured",
                       reason=REASONS["tri_attn_start"])
    del bias


def ops(stack: SynthStack, park, attn_block: int, block_index: int = 0) -> List[Tuple[str, str, Callable, Callable]]:
    """Every op class of one stack block as ``(name, dim, whole(z_dev) -> z_dev, streamed(src, dst) -> None)``; the streamed form's exact
    label and reason are declared inside. ``recycle_add`` reads the parked ``z_init``; ``head`` needs the parked head module resident."""
    blk = stack.blocks[block_index]
    zi = park.get("z_init")
    S = park.stream_blocks
    return [
        ("transition", "rows", lambda z: z + blk["trans"](z),
         lambda src, dst: S(lambda b, a, e: b + blk["trans"](b), src, "rows", out=dst, exact="measured", reason=REASONS["transition"])),
        ("transition", "cols", lambda z: z + blk["trans"](z),
         lambda src, dst: S(lambda b, a, e: b + blk["trans"](b), src, "cols", out=dst, exact="measured", reason=REASONS["transition"])),
        ("tri_attn_start", "rows", lambda z: z + blk["attn"](z, None, attn_block),
         lambda src, dst: stream_attn(blk["attn"], park, src, dst, attn_block)),
        ("row_gate", "rows", lambda z: blk["row"](z),
         lambda src, dst: S(lambda b, a, e: blk["row"](b), src, "rows", out=dst, exact="band", reason=REASONS["row_gate"])),
        ("col_gate", "cols", lambda z: blk["col"](z),
         lambda src, dst: S(lambda b, a, e: blk["col"](b), src, "cols", out=dst, exact="band", reason=REASONS["col_gate"])),
        ("scale_shift", "rows", lambda z: blk["ss"](z),
         lambda src, dst: S(lambda b, a, e: blk["ss"](b), src, "rows", out=dst, exact="bitwise", reason=REASONS["scale_shift"])),
        ("scale_shift", "cols", lambda z: blk["ss"](z),
         lambda src, dst: S(lambda b, a, e: blk["ss"](b), src, "cols", out=dst, exact="bitwise", reason=REASONS["scale_shift"])),
        ("recycle_add", "rows", lambda z: z + zi.to_device(),
         lambda src, dst: S(lambda b, a, e: b + zi.rows(a, e), src, "rows", out=dst, exact="bitwise", reason=REASONS["recycle_add"])),
        ("head", "rows", lambda z: stack.head(z),
         lambda src, dst: S(lambda b, a, e: stack.head(b), src, "rows", out=dst, exact="measured", reason=REASONS["head"])),
    ]


TRUNK_ORDER = ("transition", "tri_attn_start", "row_gate", "col_gate", "scale_shift", "recycle_add")     # the trunk uses the rows form of the transition / scale-shift


@torch.no_grad()
def run_reference(stack: SynthStack, z, z_init, attn_block: int):
    """The un-levered path: everything resident, every op on the whole tensor (the attention on the stock row-chunked path with
    ``chunk_size=attn_block``). Returns the final z (device)."""
    for blk in stack.blocks:
        z = z + blk["trans"](z)
        z = z + blk["attn"](z, None, attn_block)
        z = blk["row"](z)
        z = blk["col"](z)
        z = blk["ss"](z)
        z = z + z_init
    return stack.head(z)


@torch.no_grad()
def run_offloaded(stack: SynthStack, park, attn_block: int) -> str:
    """The levered path over the tensors ALREADY parked as ``"z"`` and ``"z_init"`` (the caller parks them and drops its device references
    first — the lever never frees a tensor it did not allocate): every op streamed by its separable dim through the same streamed forms
    :func:`ops` lists, the head parked with residency for its streamed application. Returns the final parked name; the host bytes of
    ``park.get(name).host`` are the result."""
    zt = park.get("z")
    head = park.park_module(stack.head, "head", hooks=False)
    park.park_like("tmp", zt.shape, zt.dtype)
    cur, other = "z", "tmp"
    for bi in range(len(stack.blocks)):
        forms = {(name, dim): streamed for name, dim, _, streamed in ops(stack, park, attn_block, bi)}
        for name in TRUNK_ORDER:
            dim = "cols" if name == "col_gate" else "rows"
            forms[(name, dim)](cur, other)
            cur, other = other, cur
    with head.resident():
        forms[("head", "rows")](cur, other)
    cur, other = other, cur
    return cur
