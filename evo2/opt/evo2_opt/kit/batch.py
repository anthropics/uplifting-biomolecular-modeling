"""S1: ``evo2.scoring.prepare_batch`` (the token batch every ``score_sequences`` call builds before the forward) with each row built as one
numpy int64 array instead of a Python list of numpy scalars handed to ``torch.tensor``: the same (input_ids, seq_lengths) — vortex's
CharLevelTokenizer tokenizes a string to its UTF-8 bytes, a row is ``[eod_id] * prepend_bos + bytes + [pad_id] * (max_len - len(seq))``
as torch.long on ``device``, rows concatenated along dim 0 — for a fraction of the host time, which the device otherwise waits out between
two scoring forwards. Any other tokenizer object takes the stock function."""
from __future__ import annotations

import numpy as np
import torch

import evo2.scoring as _scoring
from vortex.model.tokenizer import CharLevelTokenizer

LEVER = "S1_prepare_batch_numpy_rows"
_ORIG = {"prepare_batch": _scoring.prepare_batch}


def _char_level(tokenizer) -> bool:
    return isinstance(tokenizer, CharLevelTokenizer) and type(tokenizer).tokenize is CharLevelTokenizer.tokenize


def prepare_batch(seqs, tokenizer, prepend_bos: bool = False, device: str = "cuda:0"):
    if not _char_level(tokenizer) or not all(isinstance(s, str) for s in seqs):
        return _ORIG["prepare_batch"](seqs, tokenizer, prepend_bos=prepend_bos, device=device)
    seq_lengths = [len(seq) for seq in seqs]
    max_seq_length = max(seq_lengths)
    bos = np.full(int(prepend_bos), tokenizer.eod_id, dtype=np.int64)
    input_ids = []
    for seq in seqs:
        ids = np.frombuffer(seq.encode("utf-8"), dtype=np.uint8).astype(np.int64)          # CharLevelTokenizer.tokenize(seq), as one array
        padding = np.full(max_seq_length - len(seq), tokenizer.pad_id, dtype=np.int64)
        row = np.concatenate([bos, ids, padding])
        input_ids.append(torch.from_numpy(row).to(device).unsqueeze(0))
    input_ids = torch.cat(input_ids, dim=0)
    return input_ids, seq_lengths


def install() -> str:
    if _scoring.prepare_batch is not prepare_batch:
        _ORIG["prepare_batch"] = _scoring.prepare_batch
        _scoring.prepare_batch = prepare_batch
    return LEVER


def remove() -> None:
    _scoring.prepare_batch = _ORIG["prepare_batch"]


def is_installed() -> bool:
    return _scoring.prepare_batch is prepare_batch
