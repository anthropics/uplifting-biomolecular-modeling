"""CPU tests of the pipeline_tf v17 kit (no TF, no baskerville needed): the LUT one-hot against a literal transcription of the
stock loop (baskerville dna.py:39-92), the chunk plan on a synthetic strand transform, and the chunked post's byte equality with the
stock on a synthetic write_snp_len-shaped statistic."""
import os
import sys

import numpy as np
import pytest
from scipy.sparse import dok_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(os.path.dirname(HERE), "v17")
sys.path.insert(0, FROZEN)
from kitlib import onehot, sad_post  # noqa: E402


def stock_dna_1hot(seq, seq_len=None, n_uniform=False, n_sample=False):
    """baskerville @544073b dna.py:39-92 transcribed (the CPU test's stand-in for the real function)."""
    if seq_len is None:
        seq_len = len(seq); seq_start = 0
    else:
        if seq_len <= len(seq):
            seq_trim = (len(seq) - seq_len) // 2; seq = seq[seq_trim: seq_trim + seq_len]; seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2
    seq = seq.upper()
    seq_code = np.zeros((seq_len, 4), dtype="float16") if n_uniform else np.zeros((seq_len, 4), dtype="bool")
    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A": seq_code[i, 0] = 1
            elif nt == "C": seq_code[i, 1] = 1
            elif nt == "G": seq_code[i, 2] = 1
            elif nt == "T": seq_code[i, 3] = 1
            else:
                if n_uniform: seq_code[i, :] = 0.25
    return seq_code


@pytest.mark.parametrize("seq", ["ACGTNacgtnXRYK-.", "N" * 33, "ACG", "acgt" * 257, "", "ACGTACGTACGTACGT"])
@pytest.mark.parametrize("kw", [dict(n_uniform=True), dict(), dict(seq_len=8, n_uniform=True), dict(seq_len=8), dict(seq_len=40, n_uniform=True), dict(seq_len=40), dict(seq_len=7, n_uniform=True)])
def test_onehot_bit_identical(seq, kw):
    if not seq and "seq_len" not in kw:
        pytest.skip("empty sequence without seq_len")
    x = stock_dna_1hot(seq, **kw); y = onehot.dna_1hot(seq, **kw)
    assert x.dtype == y.dtype and x.shape == y.shape
    assert np.array_equal(x.view(np.uint8), y.view(np.uint8))


def test_onehot_random_long():
    rng = np.random.default_rng(1)
    seq = "".join(rng.choice(list("ACGTNacgtn"), size=20000))
    x = stock_dna_1hot(seq, n_uniform=True); y = onehot.dna_1hot(seq, n_uniform=True)
    assert np.array_equal(x.view(np.uint8), y.view(np.uint8)) and y.flags["C_CONTIGUOUS"]


def _strand_transform(pairs):
    """pairs: list of ('single' | 'pair') -> (T, TS) csr exactly as borzoi_sad.py:189-200 builds it."""
    rows = []
    for p in pairs:
        rows += ["s"] if p == "single" else ["+", "-"]
    T = len(rows); TS = len(pairs)
    m = dok_matrix((T, TS)); sti = 0
    for ti, r in enumerate(rows):
        m[ti, sti] = True
        if r == "s" or r == "-":
            sti += 1
    return m.tocsr(), T, TS


def test_chunk_plan_contiguous_and_covering():
    rng = np.random.default_rng(2)
    pairs = list(rng.choice(["single", "pair"], size=301))
    csr, T, TS = _strand_transform(pairs)
    for n in (1, 2, 3, 6, 8, 50):
        plan = sad_post.chunk_plan(csr, n)
        assert plan[0][:1] == (0,) and plan[-1][1] == T and plan[0][2] == 0 and plan[-1][3] == TS
        for (ilo, ihi, lo, hi), (jlo, jhi, klo, khi) in zip(plan, plan[1:]):
            assert ihi == jlo and hi == klo
        for ilo, ihi, lo, hi in plan:        # every input column of the chunk maps inside [lo, hi)
            sub = csr[ilo:ihi]
            assert sub.indices.min() >= lo and sub.indices.max() < hi


def _write_snp_len_like(ref_preds, alt_preds, sad_out, si, sad_stats):
    """A write_snp_len-shaped statistic (elementwise ops + per-column sums, float16 rows) for the CPU test."""
    rl = np.log2(ref_preds + 1); al = np.log2(alt_preds + 1)
    if "SAD" in sad_stats:
        sad_out["SAD"][si] = np.clip(alt_preds.sum(axis=0) - ref_preds.sum(axis=0), np.finfo(np.float16).min, np.finfo(np.float16).max).astype("float16")
    if "logSAD" in sad_stats:
        sad_out["logSAD"][si] = np.clip(al.sum(axis=0) - rl.sum(axis=0), np.finfo(np.float16).min, np.finfo(np.float16).max).astype("float16")
    if "D2" in sad_stats:
        sad_out["D2"][si] = np.clip(np.sqrt(np.power(alt_preds - ref_preds, 2).sum(axis=0)), np.finfo(np.float16).min, np.finfo(np.float16).max).astype("float16")


def _untransform_like(preds, tdf):
    scale = np.expand_dims(np.array(tdf["scale"]), axis=0)
    preds = preds / scale
    cs = np.expand_dims(np.array(tdf["clip_soft"]), axis=0)
    preds = np.where(preds > cs, cs + (preds - cs) ** 2, preds)
    m = np.array([s.find("_sqrt") != -1 for s in tdf["sum_stat"]])
    preds[:, m] = preds[:, m] ** (4 / 3)
    return preds * scale


def test_chunked_post_bit_identical():
    import pandas as pd
    rng = np.random.default_rng(3)
    pairs = list(rng.choice(["single", "pair"], size=257))
    csr, T, TS = _strand_transform(pairs)
    tdf = pd.DataFrame({"scale": rng.uniform(0.1, 3, size=T), "clip_soft": rng.choice([32, 384, 768], size=T).astype("int64"),
                        "sum_stat": rng.choice(["sum", "sum_sqrt", "mean"], size=T)})
    L = 512
    ref = rng.gamma(0.5, 8.0, size=(L, T)).astype(np.float32); alt = (ref * rng.normal(1.0, 0.05, size=(L, T))).astype(np.float32)
    stats = ["SAD", "logSAD", "D2"]
    r = _untransform_like(ref, tdf) * csr; a = _untransform_like(alt, tdf) * csr
    ref_out = {s: np.empty((1, TS), dtype="float16") for s in stats}
    _write_snp_len_like(r, a, ref_out, 0, stats)
    for threads in (1, 2, 5, 8):
        cp = sad_post.ChunkedPost(tdf, csr, stats, _untransform_like, _write_snp_len_like, threads)
        out = {s: np.empty((1, TS), dtype="float16") for s in stats}
        cp.write_variant(ref, alt, out, 0)
        cp.pool.shutdown()
        for s in stats:
            assert np.array_equal(ref_out[s][0].view(np.uint8), out[s][0].view(np.uint8)), (threads, s)


def test_pool_size_form():
    """v17's pool_size() form depends on the runtime probe's outcome (hard-count skip | probe ok | probe refused) — every
    form names floor=2, margin=2; the exact branch is environment-dependent, so only the shared shape is asserted."""
    p = sad_post.pool_size()
    assert p["threads"] >= 2 and p["threads"] <= max(2, p["cpu_quota"])
    assert p["form"].startswith("max(2, ") and "probe_table" in p


def test_applies_option_set():
    class O: pass
    o = O(); o.targets_file = "t"; o.sad_stats = ["SAD", "logSAD", "D2", "logD2"]
    assert sad_post.applies(o, True, False)
    assert not sad_post.applies(o, False, False)          # no strand sums
    assert not sad_post.applies(o, True, True)            # sum_length (write_snp path)
    o.sad_stats = ["SAD", "REF"]; assert not sad_post.applies(o, True, False)
    o.sad_stats = ["SAD"]; o.targets_file = None; assert not sad_post.applies(o, True, False)

