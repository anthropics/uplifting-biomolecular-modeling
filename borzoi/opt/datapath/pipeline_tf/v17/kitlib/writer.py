"""THE WRITER of the per-variant host post (``write_snp_len``, calico/borzoi @5c93582 borzoi_sad.py:408-523): the stock's statistics,
expression for expression, computing ONLY the intermediates the requested statistics use — the stock's bits by construction. The stock
computes every intermediate unconditionally (two log2 passes, two sqrt passes, six column sums, six diff/abs arrays) whatever ``--stats``
asks for. numpy never fuses or reassociates elementwise loops and each per-column reduction (sum / argmax over axis 0) sees the identical
column bytes, so dropping an unused pass cannot change a byte of any emitted statistic (``tests/test_pipeline_tf.py`` holds the chunked
form of this writer, byte for byte, to a stock-shaped reference computed on the whole arrays).

The writer has the stock's signature ``(ref_preds, alt_preds, sad_out, si, sad_stats)`` and is called by the chunked post on each column
chunk exactly as the stock loop calls ``write_snp_len`` on the whole arrays; a requested statistic outside ``WRITER_STATS`` sends the
whole call to the stock function (the supported surface stays the stock's). One writer, no switch: its name and tier are stamped and
printed on the kit's line."""
from __future__ import annotations

import numpy as np

WRITER = "exact"                     # the one writer's name (stamped, printed on the KIT_POST line, read by the package by AST)
#: the statistics the writer implements (the stock write_snp_len's per-column set; anything else -> the stock function)
WRITER_STATS = ("SAD", "SADlog", "logSAD", "sqrtSAD", "SAX", "D1", "logD1", "sqrtD1", "D2", "logD2", "sqrtD2", "JS", "logJS")   # = sad_post.PER_COLUMN_STATS; REF / ALT (full arrays) never reach the chunked post
TIER = "tier 1 (the stock's bits by construction: the stock's expressions, only the requested passes)"
_F16_LO, _F16_HI = np.finfo(np.float16).min, np.finfo(np.float16).max



def _clip16(x):
    return np.clip(x, _F16_LO, _F16_HI)


def write_snp_len_exact(ref_preds, alt_preds, sad_out, si, sad_stats):
    """THE WRITER: the stock write_snp_len's statistics, expression for expression (the stock's own calls — ``np.power(x, 2)``, never
    ``**`` or ``np.square``: the bits are the ufunc's), each intermediate computed once and only when a requested statistic uses it."""
    seq_length, num_targets = ref_preds.shape
    stats = set(sad_stats)
    need_log = bool(stats & {"logSAD", "logD1", "logD2", "logJS"})
    need_sqrt = bool(stats & {"sqrtSAD", "sqrtD1", "sqrtD2"})
    need_diff = bool(stats & {"SAX", "D1", "D2"})
    ref_preds_log = alt_preds_log = ref_preds_sqrt = alt_preds_sqrt = None
    if need_log:
        ref_preds_log = np.log2(ref_preds + 1)
        alt_preds_log = np.log2(alt_preds + 1)
    if need_sqrt:
        ref_preds_sqrt = np.sqrt(ref_preds)
        alt_preds_sqrt = np.sqrt(alt_preds)
    if stats & {"SAD", "SADlog"}:
        ref_preds_sum = ref_preds.sum(axis=0)
        alt_preds_sum = alt_preds.sum(axis=0)
    if "logSAD" in stats:
        ref_preds_log_sum = ref_preds_log.sum(axis=0)
        alt_preds_log_sum = alt_preds_log.sum(axis=0)
    if "sqrtSAD" in stats:
        ref_preds_sqrt_sum = ref_preds_sqrt.sum(axis=0)
        alt_preds_sqrt_sum = alt_preds_sqrt.sum(axis=0)
    if need_diff:
        altref_diff = alt_preds - ref_preds
    if stats & {"SAX", "D1"}:
        altref_adiff = np.abs(altref_diff)
    if stats & {"logD1", "logD2"}:
        altref_log_diff = alt_preds_log - ref_preds_log
    if "logD1" in stats:
        altref_log_adiff = np.abs(altref_log_diff)
    if stats & {"sqrtD1", "sqrtD2"}:
        altref_sqrt_diff = alt_preds_sqrt - ref_preds_sqrt
    if "sqrtD1" in stats:
        altref_sqrt_adiff = np.abs(altref_sqrt_diff)
    # compare reference to alternative via sum subtraction
    if "SAD" in stats:
        sad = alt_preds_sum - ref_preds_sum
        sad = _clip16(sad)
        sad_out["SAD"][si] = sad.astype("float16")
    if "SADlog" in stats:
        sad_log = np.log2(alt_preds_sum + 1) - np.log2(ref_preds_sum + 1)
        sad_log = _clip16(sad_log)
        sad_out["SADlog"][si] = sad_log.astype("float16")
    if "logSAD" in stats:
        log_sad = alt_preds_log_sum - ref_preds_log_sum
        log_sad = _clip16(log_sad)
        sad_out["logSAD"][si] = log_sad.astype("float16")
    if "sqrtSAD" in stats:
        sqrt_sad = alt_preds_sqrt_sum - ref_preds_sqrt_sum
        sqrt_sad = _clip16(sqrt_sad)
        sad_out["sqrtSAD"][si] = sqrt_sad.astype("float16")
    # compare reference to alternative via max subtraction
    if "SAX" in stats:
        max_i = np.argmax(altref_adiff, axis=0)
        sax = altref_diff[max_i, np.arange(num_targets)]
        sad_out["SAX"][si] = sax.astype("float16")
    # L1 norm of difference vector
    if "D1" in stats:
        sad_d1 = altref_adiff.sum(axis=0)
        sad_d1 = _clip16(sad_d1)
        sad_out["D1"][si] = sad_d1.astype("float16")
    if "logD1" in stats:
        log_d1 = altref_log_adiff.sum(axis=0)
        log_d1 = _clip16(log_d1)
        sad_out["logD1"][si] = log_d1.astype("float16")
    if "sqrtD1" in stats:
        sqrt_d1 = altref_sqrt_adiff.sum(axis=0)
        sqrt_d1 = _clip16(sqrt_d1)
        sad_out["sqrtD1"][si] = sqrt_d1.astype("float16")
    # L2 norm of difference vector
    if "D2" in stats:
        altref_diff2 = np.power(altref_diff, 2)
        sad_d2 = np.sqrt(altref_diff2.sum(axis=0))
        sad_d2 = _clip16(sad_d2)
        sad_out["D2"][si] = sad_d2.astype("float16")
    if "logD2" in stats:
        altref_log_diff2 = np.power(altref_log_diff, 2)
        log_d2 = np.sqrt(altref_log_diff2.sum(axis=0))
        log_d2 = _clip16(log_d2)
        sad_out["logD2"][si] = log_d2.astype("float16")
    if "sqrtD2" in stats:
        altref_sqrt_diff2 = np.power(altref_sqrt_diff, 2)
        sqrt_d2 = np.sqrt(altref_sqrt_diff2.sum(axis=0))
        sqrt_d2 = _clip16(sqrt_d2)
        sad_out["sqrtD2"][si] = sqrt_d2.astype("float16")
    if "JS" in stats:
        from scipy.special import rel_entr
        # normalized scores
        pseudocounts = np.percentile(ref_preds, 25, axis=0)
        ref_preds_norm = ref_preds + pseudocounts
        ref_preds_norm /= ref_preds_norm.sum(axis=0)
        alt_preds_norm = alt_preds + pseudocounts
        alt_preds_norm /= alt_preds_norm.sum(axis=0)
        # compare normalized JS
        ref_alt_entr = rel_entr(ref_preds_norm, alt_preds_norm).sum(axis=0)
        alt_ref_entr = rel_entr(alt_preds_norm, ref_preds_norm).sum(axis=0)
        js_dist = (ref_alt_entr + alt_ref_entr) / 2
        sad_out["JS"][si] = js_dist.astype("float16")
    if "logJS" in stats:
        from scipy.special import rel_entr
        # normalized scores
        pseudocounts = np.percentile(ref_preds_log, 25, axis=0)
        ref_preds_log_norm = ref_preds_log + pseudocounts
        ref_preds_log_norm /= ref_preds_log_norm.sum(axis=0)
        alt_preds_log_norm = alt_preds_log + pseudocounts
        alt_preds_log_norm /= alt_preds_log_norm.sum(axis=0)
        # compare normalized JS
        ref_alt_entr = rel_entr(ref_preds_log_norm, alt_preds_log_norm).sum(axis=0)
        alt_ref_entr = rel_entr(alt_preds_log_norm, ref_preds_log_norm).sum(axis=0)
        log_js_dist = (ref_alt_entr + alt_ref_entr) / 2
        sad_out["logJS"][si] = log_js_dist.astype("float16")



def make_writer(stock_write_snp_len):
    """The writer callable for this process + its stamp. A requested statistic outside WRITER_STATS sends the call to the stock function."""

    def writer(ref_preds, alt_preds, sad_out, si, sad_stats):
        if any(s not in WRITER_STATS for s in sad_stats):
            return stock_write_snp_len(ref_preds, alt_preds, sad_out, si, sad_stats)
        return write_snp_len_exact(ref_preds, alt_preds, sad_out, si, sad_stats)
    writer.stamp = {"writer": WRITER, "tier": TIER, "stats_implemented": WRITER_STATS}
    writer.__name__ = f"write_snp_len_{WRITER}"
    return writer


def line(stamp: dict) -> str:
    """The one line the kit prints at the first post: the writer and its tier, said once."""
    return f"pipeline_tf v16 post writer {stamp['writer']!r} — {stamp['tier']}"
