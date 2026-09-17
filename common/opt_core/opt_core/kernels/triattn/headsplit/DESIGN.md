# kernels/triattn/headsplit — design note (exact by construction)

Not a new attention kernel: a CALL PATTERN around cuEquivariance's public `triangle_attention` plus a head-major variant of the fused
triangle-attention prologue, hosted by a kit lever (`LEVER = "triatt_headsplit_exact"`, switch `PTX_TRIATT_HEADSPLIT=1`,
`MODEL_OPT_LEVERS_OFF` word `triatt_headsplit_exact`; module `__version__ = "1.0.2"`).

## What it changes

In the exact tier's triangle-attention statement (prologue → cuEquivariance `triangle_attention` → epilogue), above `N_SPLIT` tokens the
attention core is called ONCE PER HEAD instead of once for all H heads, and the per-head outputs are concatenated into the `[R, H, J, D]`
tensor the epilogue reads (one extra pass over `o`).

Why: the library's kernel re-reads the fp32 pair bias `[H, N, N]` per row tile; once the all-heads bias (H·N²·4 bytes) exceeds a fraction of
the device's L2 it streams from HBM on every tile, while one head's slice (N²·4 bytes) stays L2-resident.  Every CTA of the library kernel
computes exactly what it computes in the full call — same operands, same arithmetic, same reduction order — so the outputs are bitwise
equal; nothing arch-specific is involved.

Zero-copy: the library's public wrapper copies any non-contiguous q/k/v/bias, and a head slice of the prologue's `[I, H, J, D]` q is not
contiguous.  `prologue_hm.py` is therefore the fused prologue (`kernels/fpf_mkpf`: z read once in its native layout, LayerNorm in registers,
q|k|v|g projections + fp32 pair bias) with ONE change: q, k, v are STORED head-major (`[H, I, J, D]` storage returned as `[I, H, J, D]`-shaped
views).  The LayerNorm rows and the projection chunks are the `fpf_mkpf` Triton jit helpers themselves (imported, not copied), launched with
the same cell's tile configuration, so every stored value is the value the standard prologue stores — only its address differs.  After it,
each head's (row chunk of) q/k/v and `bias[:, :, h]` are contiguous and go to the library untouched (`zc` route); a caller whose q/k/v are not
head-major takes the `copy` route (the library copies the per-head slices itself).  Row chunking, padded buffers (re-interpreted head-major,
pads kept zero) and the lean prologue mode compose with the split.

## Envelope and guards

* Split iff `H > 1` and key length `N > n_split(device) = max(N_SPLIT_FLOOR = 1024, ⌊sqrt(L2_FRACTION · L2_bytes / 32)⌋)` with `L2_FRACTION = 0.6`
  — i.e. when the 8-head fp32 bias exceeds that fraction of the device L2 (`torch.cuda.get_device_properties().L2_cache_size`, fixed at
  `install()`); at or below the floor the statement is left as it is.
* Shape check: the first call of every new (site, rows, H, N, D, mask) shape runs the split AND the full call on the call's own operands;
  `torch.equal` → the shape is served, else `Refused` by name (never a silent fallback).  `TESTED_CC = ((9, 0),)`; any other device engages and
  is NAMED `untested` in `stats()` / the marker — the per-shape check is the guard everywhere.
* Routes outside the envelope are declared and counted, not hidden: `pro_passthru`, `<site>_single`; counters {zc, copy, single, pro_hm,
  pro_pass, heads, chk_eq, chk_ne} for the kit's lever line, full per-site counters and `shapecheck {shapes, eq, ne}` in `report()`.

## Numerics

Exactly the library op's (whatever kernel cuEquivariance dispatches on the device): the lever adds no arithmetic.  Deterministic where the
library is.

## Known limits

Engages only inside the exact block path that calls cuEquivariance; needs the library importable.  Memory: one extra `[R, H, J, D]` pass for the
concatenation; the head-major prologue allocates the same bytes as the standard one.  `n_split` assumes 8 heads × 4-byte bias in its L2
formula (the AF3-family triangle attention); a device that exposes no `L2_cache_size` is assumed 50 MB and named so in `stats()["device"]`.
