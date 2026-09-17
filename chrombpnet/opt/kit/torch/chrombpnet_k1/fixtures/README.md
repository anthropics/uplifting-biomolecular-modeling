# fixtures

`l40s_counts_head_64rows.npz` — synthetic test data for `../../test_counts_form_cpu.py` (the sm_89 counts-head dense order). No model
weights or model outputs are in this file: every array is drawn from `numpy.random.default_rng(20260916)` in this order — for `nobias`
(C = 512) then `bias` (C = 128): `<head>_gap_out` = |normal(0, 1)| of shape (64, C) (pooled activations are non-negative), `<head>_W` =
normal(0, 0.02) of shape (C,), `<head>_b` = normal(0, 0.02) of shape (1,), all float32. `<head>_logcounts` (64,) are the reference outputs:
Dense(C→1) computed by an independent float32 NumPy implementation of the order `chrombpnet_k1.kernels.dense_groups_bfly` documents —
per-element products (one rounding each), a stride-halving butterfly inside each contiguous group of 128 elements, the group partials
accumulated sequentially, the bias added last:

```python
def ref_dense_groups_bfly(g, w, b, group=128):            # g (N, C) float32, w (C,) float32, b float32 scalar
    prod = (g * w.reshape(1, -1)).astype(np.float32); acc = None
    for s0 in range(0, g.shape[1], group):
        p = prod[:, s0:s0 + group].copy(); s = p.shape[1]
        while s > 1:
            s //= 2; p = (p[:, :s] + p[:, s:2 * s]).astype(np.float32)
        acc = p[:, 0] if acc is None else (acc + p[:, 0]).astype(np.float32)
    return (acc + np.float32(b)).astype(np.float32)
```

That reference was checked once against 64 rows per head of dense-layer outputs captured from the stock model on an sm_89 card (64/64
bitwise for C = 512 and C = 128; the same rows summed as one 512-wide group agree in only 41/64), so the test keeps its meaning — the kit's
`dense_groups_bfly` must equal the sm_89 order bitwise and a different group size must not — without shipping any captured data. On this
file the full-group order agrees with the reference in 17/64 rows for C = 512 (the fixture discriminates).
