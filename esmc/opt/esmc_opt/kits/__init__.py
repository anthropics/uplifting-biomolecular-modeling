"""ESMC kits — exact levers over the stock `esm` package forward — and the shared pieces every kit imports (one home each,
never re-implemented):

    KIT_ENV                                  the kits' shared marker variable: set by every kit's apply(), asserted absent by the stock arm
    make_inputs(batch, length, seed)         the seeded canonical-residue unit batch (<cls> … <eos>, ids 4..23; no padding => the
                                             fused/default attention path)
"""
from __future__ import annotations

KIT_ENV = "ESMC_KIT"                       # set => a kit is in force; the stock reference process refuses to run with it

RESIDUE_IDS = (4, 24)      # tokenizer.json: L A G V S E R T I D P K Q N F Y M H W C = 4..23
CLS_ID, PAD_ID, EOS_ID, MASK_ID = 0, 1, 2, 32


def make_inputs(batch: int, length: int, seed: int = 0, device="cuda"):
    """Seeded unit batch of canonical residues with <cls>/<eos> (no padding => the fused/default attention path)."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + batch * 4099 + length)
    ids = torch.randint(RESIDUE_IDS[0], RESIDUE_IDS[1], (batch, length), generator=g)
    ids[:, 0] = CLS_ID
    ids[:, -1] = EOS_ID
    return ids.to(device)
