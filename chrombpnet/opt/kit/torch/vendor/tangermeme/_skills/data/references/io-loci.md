# Loading data with tangermeme.io

`tangermeme.io` turns genomic files into the tensors the rest of the library
consumes. The central footgun is that `extract_loci` returns a **different number
of elements depending on which arguments you set** — unpacking blindly is the #1
mistake.

## extract_loci — the variable return

```python
from tangermeme.io import extract_loci

extract_loci(
    loci,                 # BED/narrowPeak path, DataFrame, or list of them
    sequences,            # FASTA path, pyfaidx.Fasta, or {chrom: tensor} dict
    signals=None,         # list of bigWig paths/objects -> OUTPUT signal tensor
    in_signals=None,      # list of bigWig -> INPUT signal tensor (e.g. controls)
    chroms=None,
    in_window=2114,       # input window length (sequence + in_signals)
    out_window=1000,      # output window length (signals)
    max_jitter=0,         # EXPANDS windows for downstream jittering; not applied here
    min_counts=None, max_counts=None, target_idx=0,
    n_loci=None, summits=False,
    alphabet=['A','C','G','T'], ignore=['N'],
    exclusion_lists=None,
    return_mask=False,
    verbose=False,
)
```

### Return order (this is the footgun)

The result is a list assembled in this fixed order, and a **bare object** (not a
list) is returned when only one element is present:

1. `X` — one-hot sequences `(n, len(alphabet), in_window)`, dtype **int8** —
   **always present**
2. `y` — output signals `(n, n_signals, out_window)` (single signal still gets a
   signal axis; order = file order) — only if `signals` is given
3. `X_in` — input signals — only if `in_signals` is given
4. `mask` — kept-locus boolean tensor — only if `return_mask=True`

**Leave `X` as int8.** `predict`, `deep_lift_shap` and `saturation_mutagenesis`
upcast each batch to the model's dtype, so the int8 blob is the whole memory win.
Only `.float()` it for `pisa`, which doesn't upcast, or when you call `model(X)`
yourself.

So unpack to match exactly what you requested:

```python
X = extract_loci(loci, fasta)                                  # 1 -> bare tensor
X, y = extract_loci(loci, fasta, signals=bws)                  # 2
X, y, mask = extract_loci(loci, fasta, signals=bws, return_mask=True)  # 3
X, y, X_in, mask = extract_loci(loci, fasta, signals=bws,
                                in_signals=ctrls, return_mask=True)     # 4
```

Get the count wrong and you will silently bind a tensor to the wrong variable.

### Why returned rows may not match input loci

Loci are dropped when they fall off chromosome ends (after jitter), sit on
chromosomes not in `chroms`, fail `min_counts`/`max_counts` (measured on
`signals[target_idx]`), or hit an `exclusion_lists` region. **Use
`return_mask=True` whenever you need to align results back to the input rows** —
the mask tells you which loci survived.

### Multiple loci files are interleaved, not concatenated

Passing a list of BED/narrowPeak files **interleaves** them round-robin until the
shortest is exhausted, then appends the remainder — it does not concatenate them in
order. `n_loci` then truncates that interleaved list. Useful for mixing a
locus-of-interest with genomic background; surprising if you expected file-order
concatenation.

### Window semantics

`in_window` (sequence + `in_signals`) and `out_window` (`signals`) are centered on
each locus, independent of the locus's own width. `summits=True` centers on the
narrowPeak summit instead of the region midpoint. `max_jitter` *expands* both
windows so a downstream data generator can jitter cheaply — it does not jitter the
returned data itself.

## read_meme — motif PWMs

```python
from tangermeme.io import read_meme
motifs = read_meme("motifs.meme")        # dict: name -> PWM tensor (4, length)
```

Wraps the memelite reader. Note: FIMO/Tomtom scanning moved out of tangermeme to
`memesuite-lite` — use that package for motif scanning.

## read_vcf — variants

```python
from tangermeme.io import read_vcf
vcf = read_vcf("variants.vcf")           # pandas DataFrame, comments stripped
```

Feed variants into `tangermeme.variant_effect.*` to score substitution / deletion
/ insertion effects.

## Related references

[notebook-walkthrough.md](notebook-walkthrough.md) (loading is step 3 of the
end-to-end flow), [model-wrapping.md](model-wrapping.md) (the
`(batch, channels, length)` layout the loaded `X` follows),
[motif-effects.md](motif-effects.md) (consuming the loaded sequences).
