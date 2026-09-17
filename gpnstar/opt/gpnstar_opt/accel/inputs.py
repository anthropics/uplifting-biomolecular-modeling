"""Deterministic, realistic GPN-Star input batches for the identity self-check and the engage tests.

GPN-Star consumes a tokenised whole-genome alignment window:
    input_ids      (B, L, T)  int64  target row(s) (T=1 = human), centre position masked with '?' (=5) for VEP
    source_ids     (B, L, N)  int64  all N aligned species (column 0 = target species)
    target_species (B, T)     int64  index of the target species (0)
Vocabulary: "-ACGT?" -> 0..5.

For the hg38 V100 checkpoint the upstream repository's real 128 bp x 100-species test fixture
(tests/fixtures/hg38_chr6_31575665_31575793_multiz100way.npz) is tiled/cropped. For other species counts no
small public fixture exists, so alignment-like tokens are synthesised (and labelled synthetic wherever used).
"""

from __future__ import annotations

import numpy as np
import torch

VOCAB = "-ACGT?"
MASK_ID = VOCAB.index("?")
GAP_ID = VOCAB.index("-")
NUC_IDS = [VOCAB.index(c) for c in "ACGT"]


def load_fixture_tokens(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        return z["gpn_star_v100_tokens"].astype(np.int64)  # (128, 100)


def _tile_to_length(tokens: np.ndarray, L: int) -> np.ndarray:
    """Crop (centre) or tile a (L0, N) alignment to length L."""
    L0 = tokens.shape[0]
    if L <= L0:
        start = (L0 - L) // 2
        return tokens[start : start + L]
    reps = int(np.ceil(L / L0))
    return np.concatenate([tokens] * reps, axis=0)[:L]


def synth_alignment(L: int, N: int, rng: np.random.Generator, human: np.ndarray | None = None) -> np.ndarray:
    """Alignment-like tokens: column 0 = human-like sequence; other species copy it with
    substitutions (15%) and gaps (15%)."""
    if human is None:
        human = rng.choice(NUC_IDS, size=L)
    else:
        human = _tile_to_length(human[:, None], L)[:, 0]
    msa = np.repeat(human[:, None], N, axis=1)
    u = rng.random((L, N))
    sub = u < 0.15
    gap = (u >= 0.15) & (u < 0.30)
    msa[sub] = rng.choice(NUC_IDS, size=int(sub.sum()))
    msa[gap] = GAP_ID
    msa[:, 0] = human
    return msa.astype(np.int64)


def make_batch(
    B: int,
    L: int,
    N: int,
    *,
    fixture_tokens: np.ndarray | None = None,
    seed: int = 0,
    mask_center: bool = True,
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    """Build a deterministic batch. Row b is the fixture (or synthetic MSA) circularly shifted by a
    seed-determined offset with a few random substitutions, so rows differ but stay realistic."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(B):
        if fixture_tokens is not None and fixture_tokens.shape[1] == N:
            base = _tile_to_length(fixture_tokens, L).copy()
            if b > 0:
                base = np.roll(base, int(rng.integers(0, L)), axis=0)
                flip = rng.random(base.shape) < 0.02
                base[flip] = rng.choice(NUC_IDS + [GAP_ID], size=int(flip.sum()))
                base[:, 0] = np.where(base[:, 0] == GAP_ID, rng.choice(NUC_IDS, size=L), base[:, 0])
        else:
            human = fixture_tokens[:, 0] if fixture_tokens is not None else None
            base = synth_alignment(L, N, rng, human=human if b == 0 else None)
        rows.append(base)
    msa = np.stack(rows)  # (B, L, N)
    source_ids = msa.copy()
    input_ids = msa[:, :, :1].copy()
    if mask_center:
        c = L // 2
        input_ids[:, c, 0] = MASK_ID
        source_ids[:, c, 0] = MASK_ID  # VEPInference.prepare masks both
    batch = {
        "input_ids": torch.from_numpy(input_ids).to(device).contiguous(),
        "source_ids": torch.from_numpy(source_ids).to(device).contiguous(),
        "target_species": torch.zeros((B, 1), dtype=torch.int64, device=device),
    }
    return batch


def llr_scores(logits: torch.Tensor, center: int) -> torch.Tensor:
    """VEP-style scores from logits (B, L, T, V): all alt-minus-ref logit differences at the masked
    centre among A,C,G,T -> (B, 4, 4)."""
    nuc = logits[:, center, 0, :][:, NUC_IDS].float()  # (B, 4)
    return nuc[:, None, :] - nuc[:, :, None]
