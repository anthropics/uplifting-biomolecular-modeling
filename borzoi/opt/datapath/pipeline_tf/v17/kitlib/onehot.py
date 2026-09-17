"""Exact LUT one-hot for baskerville.dna.dna_1hot (the input term of the variant loop).

The stock (baskerville dna.py:39-92) walks the 524,288-bp window in a Python loop per allele. ``vcf.dna_length_1hot`` (vcf.py:495-514)
calls it as ``dna.dna_1hot(seq, n_uniform=True)`` -> float16 (len, 4): A/C/G/T -> a unit row, every other byte -> 0.25 per
column. This module computes the same array by a (256, 4) lookup table: the values 0, 1 and 0.25 are exact in float16, the
row for every byte value is the stock's row for that byte, so the output is bit-identical by construction (tests/).
``seq_len`` trimming / padding and the ``n_uniform=False`` bool form are covered exactly too; the one path the table cannot
reproduce verbatim (``n_sample`` = random draws for N) is delegated to the stock function untouched.
"""
from __future__ import annotations

import numpy as np

try:                                   # the stock function: the fallback for every path this module does not cover
    from baskerville import dna as _stock_dna
    _STOCK_DNA_1HOT = _stock_dna.dna_1hot
except Exception:                      # pragma: no cover — baskerville absent (CPU tests build their own reference)
    _stock_dna = None
    _STOCK_DNA_1HOT = None

_LUT_F16 = np.full((256, 4), 0.25, dtype="float16")          # n_uniform: every non-ACGT byte -> 0.25 (dna.py:85-86)
for _i, _nt in enumerate(b"ACGT"):
    _LUT_F16[_nt] = 0.0
    _LUT_F16[_nt, _i] = 1.0
_LUT_BOOL = np.zeros((256, 4), dtype="bool")                  # bool form (n_uniform False, n_sample False): non-ACGT -> all 0
for _i, _nt in enumerate(b"ACGT"):
    _LUT_BOOL[_nt, _i] = True


def dna_1hot(seq: str, seq_len: int = None, n_uniform: bool = False, n_sample: bool = False):
    """Drop-in for ``baskerville.dna.dna_1hot``: bit-identical output, table lookup instead of the per-base loop.

    Covered exactly: ``seq_len is None`` (the route's call: ``dna_1hot(seq, n_uniform=True)``) and ``seq_len <= len(seq)``
    (trim) for both the float16 (n_uniform) and bool forms; ``seq_len > len(seq)`` pads with zero rows (bool) / zero rows
    (float16: the stock leaves the padding rows at 0, never 0.25 — dna.py:73-86 sets only positions inside the sequence).
    ``n_sample=True`` draws random bases for N in the stock -> delegated verbatim (not reproducible by a table)."""
    if n_sample and not n_uniform:
        if _STOCK_DNA_1HOT is None:
            raise RuntimeError("n_sample requires the stock baskerville.dna.dna_1hot")
        return _STOCK_DNA_1HOT(seq, seq_len=seq_len, n_uniform=n_uniform, n_sample=n_sample)
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim: seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2
    seq = seq.upper()
    codes = np.frombuffer(seq.encode("latin-1"), dtype=np.uint8)
    lut = _LUT_F16 if n_uniform else _LUT_BOOL
    body = lut[codes]
    if seq_start == 0 and len(seq) == seq_len:
        return np.ascontiguousarray(body)
    out = np.zeros((seq_len, 4), dtype=lut.dtype)
    out[seq_start: seq_start + len(seq)] = body
    return out


def install() -> dict:
    """Point ``baskerville.dna.dna_1hot`` at the LUT for THIS process (``vcf.dna_length_1hot`` resolves ``dna.dna_1hot`` at
    call time). Returns the stamp for the run record. Process-local by design — never an import hook on the stock command."""
    if _stock_dna is None:
        raise RuntimeError("baskerville is not importable; the kit entry scripts run inside the stock environment")
    _stock_dna.dna_1hot = dna_1hot
    return {"kit_onehot": "LUT", "replaced": "baskerville.dna.dna_1hot", "stock_id": hex(id(_STOCK_DNA_1HOT))}
