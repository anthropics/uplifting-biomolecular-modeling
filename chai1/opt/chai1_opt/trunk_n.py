"""chai1_opt.trunk_n — the eager trunk at the live-token extent (lever ``trunk_n``, fast / big; tolerance-class).

Upstream pads every input to the next of its crops (256 / 384 / 512 / 768 / 1024 / 1536 / 2048 tokens; the padding trails the live tokens and
``token_single_mask`` is False there) and every exported component runs at the padded crop.  The kit's trunk is the structured eager trunk
(``chai1_eager.trunk``: shape-generic — the same modules at any token count), so this lever runs each trunk call (all recycles) on the leading
``N' = min(crop, ceil(n_live / GRID) * GRID)`` token positions: the ten token-indexed inputs of ``Trunk.forward`` are narrowed to ``N'`` (single [B, N, .]
on dim 1; pair / masks on dims 1 and 2; the MSA features and mask on their token dim; the template features and masks on both token dims) and the two
outputs (the single and pair trunk representations, bf16) are padded back to the crop with zeros before they return to upstream's loop — the token
embedders, the diffusion module and the confidence head (per-crop transpiles of the exports) run at the stock crop unchanged, and every consumer of
the trunk representations masks the padded positions (token attention keys, atom<->token aggregation, the confidence head's pair operations).
Numerics: tolerance-class — the reduction extents of the trunk's own statements (triangle-multiplication K, softmax lengths, the outer-product mean's
token GEMMs) change with N', so outputs differ from the crop-padded run within the fast tier's tolerance (the class of ``msa_rows``); never on exact.
GRID = 64: the tile multiple of the trunk's kernels (fpf_trimul_v4 cells, SDPA); the triangle-attention provider serves the crops its table lists and names
its row (or the cuDNN statement) at other N' — counted on its own LEVER line.

Every call books ``served`` (shape ``N<crop>-><N'>``) or a declared step-aside: ``no_pad`` (N' == crop: nothing to cut), ``layout`` (the live tokens
are not the leading positions, or an input's token dims are not the crop's — the trunk runs at the crop)."""
import sys
from typing import Optional

__all__ = ["NAME", "GRID", "STRATEGY", "EXPECTED_FALLBACKS", "new_ledger", "TrunkN", "reclass"]
NAME = "trunk_n"
GRID = 64
STRATEGY = "LOCAL.chai1.trunk_n"
EXPECTED_FALLBACKS = ("no_pad", "layout")
TAG = "[chai1-opt]"
# Trunk.forward's token-indexed inputs -> the dims that index tokens
TOKEN_DIMS = {
    "token_single_trunk_initial_repr": (1,), "token_single_trunk_repr": (1,), "token_single_mask": (1,),
    "token_pair_trunk_initial_repr": (1, 2), "token_pair_trunk_repr": (1, 2), "token_pair_mask": (1, 2),
    "msa_input_feats": (2,), "msa_mask": (2,),
    "template_input_feats": (2, 3), "template_input_masks": (2, 3),
}


def new_ledger(expected=EXPECTED_FALLBACKS):
    from opt_core.counters import Ledger

    class TrunkNLedger(Ledger):                       # a run whose every call had nothing to cut (inputs at a crop boundary) holds the gate: declared words only
        def gate(self, name=None, *, require_served=False):
            return Ledger.gate(self, name, require_served=require_served)

    return TrunkNLedger(STRATEGY, impl="chai1_opt.trunk_n", origin="kit", expected=tuple(expected))


class TrunkN:
    """The plan per trunk call: ``narrow(kw) -> kw'`` and ``pad(out) -> out`` (None from ``plan`` = the call runs at the crop, word booked)."""

    def __init__(self, ledger, grid: int = GRID):
        self.ledger, self.grid = ledger, int(grid)
        self.calls = {"served": 0, "no_pad": 0, "layout": 0}
        self.last = None                                # (crop, N', n_live) of the last served call
        self._mask_ref = None                           # (mask tensor, n_live, leading?) — one host read per distinct mask object (the recycles of a fold share it)

    chai1_opt_lever = NAME

    def describe(self) -> dict:
        return {"grid": self.grid, "last": (f"N{self.last[0]}->{self.last[1]} live={self.last[2]}" if self.last else "none"), "calls": dict(self.calls)}

    def _live(self, mask):
        ref = self._mask_ref
        if ref is not None and ref[0] is mask:
            return ref[1], ref[2]
        m = mask if mask.dim() == 1 else mask.reshape(-1, mask.shape[-1]).any(0)       # [N]: live over the batch
        n_live = int(m.sum().item())
        leading = bool(m[:n_live].all().item()) if n_live else True
        self._mask_ref = (mask, n_live, leading)
        return n_live, leading

    def plan(self, kw: dict):
        import torch
        mask = kw.get("token_single_mask")
        pm = kw.get("token_pair_mask")
        if mask is None or pm is None or not torch.is_tensor(mask) or mask.dim() != 2:
            self.ledger.fallback("layout"); self.calls["layout"] += 1
            return None
        N = int(mask.shape[1])
        n_live, leading = self._live(mask)
        n2 = min(N, max(self.grid, -(-n_live // self.grid) * self.grid))
        if n2 >= N:
            self.ledger.fallback("no_pad"); self.calls["no_pad"] += 1
            return None
        if not leading:
            self.ledger.fallback("layout"); self.calls["layout"] += 1
            return None
        for k, dims in TOKEN_DIMS.items():                                             # every token dim present must be the crop's
            v = kw.get(k)
            if v is None:
                continue
            if not torch.is_tensor(v) or any(d >= v.dim() or int(v.shape[d]) != N for d in dims):
                self.ledger.fallback("layout"); self.calls["layout"] += 1
                return None
        return _Plan(self, N, n2, n_live)


class _Plan:
    def __init__(self, owner, N, n2, n_live):
        self.owner, self.N, self.n, self.n_live = owner, N, n2, n_live

    def narrow(self, kw: dict) -> dict:
        out = dict(kw)
        for k, dims in TOKEN_DIMS.items():
            v = kw.get(k)
            if v is None:
                continue
            for d in dims:
                v = v.narrow(d, 0, self.n)
            out[k] = v.contiguous()
        return out

    def pad(self, outs):
        import torch.nn.functional as F
        s, z = outs[0], outs[1]
        p = self.N - self.n
        s = F.pad(s, (0, 0, 0, p))                      # [B, N', c_s]      -> [B, N, c_s]
        z = F.pad(z, (0, 0, 0, p, 0, p))                # [B, N', N', c_z]  -> [B, N, N, c_z]
        o = self.owner
        o.ledger.serve(f"N{self.N}->{self.n}"); o.calls["served"] += 1; o.last = (self.N, self.n, self.n_live)
        return (s, z) + tuple(outs[2:])


def reclass(tw, impl: TrunkN):
    """Give the installed trunk wrapper (``chai1_eager.stack.EagerTrunkWrapper`` or a subclass another lever made of it) a forward that narrows the
    call to the plan's N' and pads the outputs back; cooperative with ``chai1_opt.big``'s BigTrunk (either order: both call their base's forward)."""
    base = type(tw)
    if getattr(base, "_chai1_trunk_n", False):
        return tw

    class TrunkNWrapper(base):
        _chai1_trunk_n = True

        def forward(self, crop_size, *, return_on_cpu=False, move_to_device=None, **kw):
            if move_to_device is not None:
                kw = {k: (v.to(move_to_device) if hasattr(v, "to") else v) for k, v in kw.items()}
            plan = impl.plan(kw)
            if plan is None:
                return base.forward(self, crop_size, return_on_cpu=return_on_cpu, move_to_device=None, **kw)
            out = base.forward(self, crop_size, return_on_cpu=False, move_to_device=None, **plan.narrow(kw))
            out = plan.pad(out)
            return tuple(o.cpu() for o in out) if return_on_cpu else out

    TrunkNWrapper.__name__ = TrunkNWrapper.__qualname__ = "TrunkNWrapper"
    tw.__class__ = TrunkNWrapper
    return tw
