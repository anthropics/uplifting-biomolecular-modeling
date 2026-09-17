"""Lever relpos_lazy (class exact): stock computes the two relative-position one-hot tensors once per predict and keeps them in the batch dict
for the whole call — batch["seq_rel_pos"] fp32 [B, L, L, 73] (1.2 GB at 2,048 tokens, 4.9 GB at 4,096; model.py:356-362) and the windowed
batch["atom_rel_pos"] fp32 [B, W, 32, 14, 128, 14, 14] (~1.4 MB per token: 2.9 GB at 2,048) — although each is read at exactly two /
one call sites (trunk: model.py:384 `z += self.z_rel_pos(batch["seq_rel_pos"])` per pass; diffusion conditioning: diffusion_transformer.py:161
and atom_attention.py:67, once per predict).  This lever stores the producing modules + their (small, integer) inputs instead and materializes
the identical tensor inside each consumer call, freeing it right after: the one-hots are transient instead of resident through sampling and the
confidence head.  Same producer code on the same inputs -> the same tensor (bit-identical by construction); the consumers run unmodified on it.
Sites: AtlasFold(.compute_rel_pos_encoding) in model.py / model_multimer.py (producer), AtlasFold.run_trunk's z_rel_pos call via
LinearNoBias.forward, PairConditioning.forward, AtomAttentionStack.forward (consumers)."""
from __future__ import annotations

import torch

from opt_core.counters import Ledger

from . import Installed

NAME = "LOCAL.atlasfold.relpos_lazy"


class LazyFeat:
    """Stands in a batch dict slot for a tensor that `fn()` reproduces exactly; `unsqueeze` etc. materialize.  The producer is replayed under
    the autocast state that was current when stock would have produced the tensor (model.predict runs under the runner's bf16 autocast;
    the diffusion consumers run with autocast disabled — AtomRelativePositionEncoding's `aatype @ atom_rel_pos` matmul is autocast-sensitive)."""
    __slots__ = ("fn", "nbytes_hint", "ac")

    def __init__(self, fn, nbytes_hint=0, device_type="cuda"):
        self.fn, self.nbytes_hint = fn, nbytes_hint
        try:
            enabled = torch.is_autocast_enabled(device_type)
            dtype = torch.get_autocast_dtype(device_type)
        except TypeError:                                   # older torch signatures
            enabled = torch.is_autocast_enabled() if device_type == "cuda" else torch.is_autocast_cpu_enabled()
            dtype = torch.get_autocast_gpu_dtype() if device_type == "cuda" else torch.get_autocast_cpu_dtype()
        self.ac = (device_type, enabled, dtype)

    def materialize(self) -> torch.Tensor:
        dev, enabled, dtype = self.ac
        with torch.autocast(dev, dtype=dtype, enabled=enabled):
            return self.fn()

    # the stock consumers call these on the batch entry directly
    def unsqueeze(self, dim):
        return self.materialize().unsqueeze(dim)

    def to(self, *a, **k):
        return self.materialize().to(*a, **k)


def install(mode: str, tag: str, ctx: dict):
    from atlasfold.model import model as M
    from atlasfold.model.network import diffusion_transformer as DT
    from atlasfold.model.network import atom_attention as AA
    from atlasfold.model.network.primitives import linear as LIN
    try:
        from atlasfold.model import model_multimer as MM
    except Exception:  # noqa: BLE001
        MM = None
    ledger = Ledger(NAME, impl="producer-replay", origin="kit")
    restores = []

    def make_compute(cls):
        stock = cls.compute_rel_pos_encoding

        def compute_rel_pos_encoding(self, batch):
            enc_s, enc_a = self.seq_rel_pos_encoding, self.atom_rel_pos_encoding
            keys_s = {k: batch[k] for k in ("res_idx", "asym_id", "entity_id", "sym_id") if k in batch}
            keys_a = {k: batch[k] for k in ("res_idx", "asym_id", "seq_mask", "aatype") if k in batch}
            dev = batch["res_idx"].device.type
            batch["seq_rel_pos"] = LazyFeat(lambda: enc_s(keys_s), device_type=dev)
            batch["atom_rel_pos"] = LazyFeat(lambda: enc_a(keys_a), device_type=dev)
            ledger.serve("L%d" % int(batch["res_idx"].shape[-1]))
        compute_rel_pos_encoding.__wrapped_stock__ = stock
        cls.compute_rel_pos_encoding = compute_rel_pos_encoding
        restores.append(lambda: setattr(cls, "compute_rel_pos_encoding", stock))

    make_compute(M.AtlasFold)
    if MM is not None and hasattr(MM, "AtlasFold_Multimer") and hasattr(MM.AtlasFold_Multimer, "compute_rel_pos_encoding"):
        make_compute(MM.AtlasFold_Multimer)

    # consumer 1: z_rel_pos is a LinearNoBias called with the batch entry (trunk, once per pass)
    lin_cls = LIN.LinearNoBias
    stock_lin = lin_cls.forward

    def lin_forward(self, input, *a, **k):
        if isinstance(input, LazyFeat):
            input = input.materialize()
        return stock_lin(self, input, *a, **k)
    lin_forward.__wrapped_stock__ = stock_lin
    own = "forward" in lin_cls.__dict__
    lin_cls.forward = lin_forward
    restores.append((lambda: setattr(lin_cls, "forward", stock_lin)) if own else (lambda: delattr(lin_cls, "forward")))

    # consumer 2: PairConditioning reads batch["seq_rel_pos"] and torch.cat's it (diffusion conditioning, once per predict)
    stock_pc = DT.PairConditioning.forward

    def pc_forward(self, batch, z, *a, **k):
        v = batch.get("seq_rel_pos")
        if isinstance(v, LazyFeat):
            b = dict(batch); b["seq_rel_pos"] = v.materialize()
            return stock_pc(self, b, z, *a, **k)
        return stock_pc(self, batch, z, *a, **k)
    pc_forward.__wrapped_stock__ = stock_pc
    DT.PairConditioning.forward = pc_forward
    restores.append(lambda: setattr(DT.PairConditioning, "forward", stock_pc))
    # consumer 3: AtomAttentionStack.forward calls batch["atom_rel_pos"].unsqueeze(1) -> LazyFeat.unsqueeze materializes (no patch needed)
    return Installed("relpos_lazy", True, lines=[lambda: ledger.line(tag)], gates=[ledger.gate], facts={"restores": len(restores)})
