"""Input builders and the stock forward call shared by the E1 kits and their tests.

A masked-marginal batch is what the scorer makes (``E1/scorer.py``): B rows of ONE parent of length L, each row the parent
with one ``?`` (mask) — all rows the same length, so the batch carries no padding. The forward is the stock route
(``predictor.py``: ``torch.no_grad`` + ``torch.autocast('cuda', bfloat16)``; fp32 weights loaded by ``tools/score.py``).
"""
from __future__ import annotations

import random

AA = "ACDEFGHIKLMNPQRSTVWY"
MASK = "?"                     # the tokenizer's mask character (tokenizer.py)


def make_rows(B: int, L: int, seed: int = 0) -> tuple[str, list[str]]:
    """(parent, rows): a seeded random parent of length L and B masked-marginal rows of it (mask position i*7919 mod L)."""
    rng = random.Random(seed * 1000003 + L)
    parent = "".join(rng.choice(AA) for _ in range(L))
    rows = [parent[:(i * 7919) % L] + "?" + parent[(i * 7919) % L + 1:] for i in range(B)]
    return parent, rows


def batch_kwargs(rows: list[str], device: str = "cuda") -> dict:
    """The batch the stock batch preparer builds for these rows (token ids, within-seq / global position ids, sequence ids)."""
    from E1.batch_preparer import E1BatchPreparer
    return E1BatchPreparer().get_batch_kwargs(rows, device=device)


def stock_forward(model, batch: dict):
    """The stock forward on a prepared batch -> (logits fp32 (B, T, V), embeddings (B, T, d))."""
    import torch
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        out = model(input_ids=batch["input_ids"], within_seq_position_ids=batch["within_seq_position_ids"],
                    global_position_ids=batch["global_position_ids"], sequence_ids=batch["sequence_ids"],
                    past_key_values=None, use_cache=False, output_attentions=False, output_hidden_states=False)
    return out.logits, out.embeddings


def load_stock_model(size: str, snapshot: str | None = None, device: str = "cuda", dtype=None):
    """The model: fp32 weights from the pinned snapshot (``tools/score.py`` L43 ``dtype=torch.float``), eval mode;
    the weights sha, the dependency stack and the hub kernel revision asserted first (engines.e1.kits.pins).
    ``dtype`` (default None = the CLI's ``torch.float``) is the stock option the README's interactive route uses: the bf16 load
    ``from_pretrained(snapshot, dtype=torch.bfloat16).to(device).eval()``."""
    import torch
    from E1.modeling import E1ForMaskedLM
    from . import pins
    stack = pins.stack_report()
    w = pins.assert_weights(size, snapshot)
    model = E1ForMaskedLM.from_pretrained(w["snapshot"], dtype=torch.float if dtype is None else dtype).to(device)
    model.eval()
    kern = pins.kernel_revision_report()
    return model, {"weights": w, "stack": stack, "kernel": kern, "load_dtype": str(torch.float if dtype is None else dtype)}


def make_rows_mixed(B: int, L: int, seed: int = 0):
    """B rows of DIFFERENT lengths (L, L-8, L-16, ...; one parent each, one ``?`` each) so ``batch_kwargs`` pads with -1 —
    the input that exercises a kit's padded-batch (fallback) path. Returns (parents, rows)."""
    import random
    rng = random.Random(seed)
    parents, rows = [], []
    for i in range(B):
        n = max(L - 8 * i, 16)
        parent = "".join(rng.choice(AA) for _ in range(n))
        pos = rng.randrange(n)
        parents.append(parent)
        rows.append(parent[:pos] + MASK + parent[pos + 1:])
    return parents, rows


def parse_shape(token: str):
    """'16x256' -> (16, 256, False); '8x256m' -> (8, 256, True) (mixed lengths / padded)."""
    mixed = token.endswith("m")
    b, l = token.rstrip("m").split("x")
    return int(b), int(l), mixed



