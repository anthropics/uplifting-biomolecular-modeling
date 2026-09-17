# fpf_trimul_v4 — TriangleMultiplication (outgoing | incoming) as three launches per call

A portable Triton member of the FlashPairformer pairformer-kernel family: fused LN_in + dual gated projection prologue writing
channel-major bf16 planes (K1) | one cuBLAS strided-batched contraction on the zero-padded planes | fused LN_out + output projection + gate
(+ residual) epilogue (K3); a leading batch is one launch set. Tolerance class: the rounding points of the cuEquivariance op under bf16
autocast, not bit-identical to it. Design, numerics and known limits: `DESIGN.md`. Licence and credits: `NOTICE`. Version:
`fpf_trimul_v4.__version__`.

## Entries
**Module path** — `fpf_trimul_v4.trimul:fn`, the stock `TriangleMultiplication{Outgoing,Incoming}.forward` signature
(`fn(module, z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative="torch")`), e.g. as the
FlashPairformer small-N plug `FPF_SMALLN_TRIMUL_FAST_FN=fpf_trimul_v4.trimul:fn`. Served cell: CUDA, `triangle_multiplicative == "cuequivariance"`,
`module.c_z == module.c_hidden == 256`, z bf16 under bf16 autocast, `[N,N,256]` or `[B,N,N,256]`, mask None / `[N,N]` / `[1|B,N,N]`, N >= 101
(`FPF_TRIMUL_V4_NMIN`), a launch cell for the device. Any other call takes the stock forward for that call, counted by reason
(`trimul.COUNTS['fallback']`). A lever that cannot run (no cell and no SAFE cell, warm probe refused, kernel error) RAISES — the caller's mode
refuses by name; an out-of-memory is torch's own error, never rerouted. With `inplace_safe=True, _add_with_inplace=True` the return value is
z + update as a NEW tensor (z is not mutated); otherwise the update alone.

**Module-agnostic** — `fpf_trimul_v4.generic` (any module layout: you pass tensors; c_z in {128, 256}, hidden D in {128, 256}, optional biases):
```python
from fpf_trimul_v4 import generic as G
w = G.pack_weights(ln_in_w=.., ln_in_b=.., w_ag=.., w_ap=.., w_bg=.., w_bp=..,        # four [D, C] matrices (a-gate, a-proj, b-gate, b-proj) + optional b_ag/b_ap/b_bg/b_bp [D]
                   ln_out_w=.., ln_out_b=.., w_o=.., w_og=.., b_o=None, b_og=None,      # w_o [C, D] (+[C]), w_og [C, C] (+[C])
                   cache_owner=module)                                                 # optional: cached on module._fpf_cache (pack ONCE, not per call)
# fused projection instead:  G.pack_weights(..., w_proj=W4[4D, C], b_proj=b4 or None, proj_split=("ap","bp","ag","bg"), ...)  (row-block order of your fused Linear)
ok, why = G.supported(z, mask, weights=w)                                                # never raises; keep your stock path for ok == False
out = G.trimul(z, mask, direction="outgoing" | "incoming", weights=w, residual=False, pad=16)   # NEW tensor; residual=True returns z + update fused in the epilogue
```
Math served (x = LN_in(z)): `a = sigmoid(x W_ag^T + b_ag) * (x W_ap^T + b_ap) * mask`, `b` likewise; `X_ij = sum_k a_ik b_jk` (outgoing) / `sum_k a_ki b_kj`
(incoming) per hidden channel; `out = sigmoid(x W_og^T + b_og) * (LN_out(X) W_o^T + b_o)` (+ z). z bf16, or fp32 z accepted (pointer-load K1, fp32 LayerNorm
input, fp32 output; the GEMM operands stay bf16 — a precision substitution for fp32 / TF32 trunks, to be gated in-model by the consumer). Outside the
served cell `G.TrimulUnsupported(reason)` is raised before any launch (`G.COUNTS['unsupported']`). Also: `G.trimul_packed` (hot path with a
pre-packed weight set), `G.workspace_bytes(N, D, C, elem_out, pad, B)` (the per-call transient bytes, a formula), `G.reference_torch` (the math
above in a chosen dtype: the contract to test a new module layout against), `G.probe` / `G.probe_all_classes` (the warm numerics probe, once per
process and shape class, stamped under `OPT_CORE_VERDICT_DIR`).

**Inside opt_core** — row `v4` of the provider face `opt_core.kernels.trimul` (kits bind tier words there); the row-block pair stack
`opt_core.mem.rowpair.trimul_fused` reads `table.select` for its launch cells.

## Launch cells
`table.json` is the one table: rows keyed `'<cc M.m>|<triton M.m>'` (exact) then `'<cc M.m>|*'`, each with the (256, 256) no-bias K1 / K3 cells
and per-shape cells under `overrides` (`k1_C<C>_D<D>[_bias]`, `table.resolve_cfg`). A kit's own table (`FPF_TRIMUL_V4_CELLS`) serves the parts
`table.json` lists under `kit_keys`. A capability without an admitted row runs the SAFE cell of `opt_core.kernels.safe_settings`, named on
stderr; with neither, the module path takes the stock forward (`no-cell`) and `generic` raises. `trimul.cells_sha()` identifies the table in the
printed lines.

## Environment
| variable | meaning |
|---|---|
| `FPF_TRIMUL_V4_NMIN` | token floor of the served cell (default 101) |
| `FPF_TRIMUL_V4_CELLS` | path of a kit's own cells table (read at the `kit_keys` parts) |
| `FPF_TRIMUL_V4_BATCH=loop` | serve a leading batch as B single-plane launch sets instead of one launch set |
| `FPF_TRIMUL_V4_STOCK_ROUND` | engineering: overrides `stock_round` (the bf16 rounding of the gated output before the fp32 residual add; on by default, as the stock op rounds) |
| `FPF_TRIMUL_V4_CFG` | engineering: a literal `{k1, k3}` cell in place of the table |
| `FPF_TRIMUL_V4_ALLOW_DEFAULT_CELLS=1` | engineering: admit table rows whose status word is not a served status |
| `OPT_CORE_VERDICT_DIR` | directory of the warm-probe verdict stamps |

## Printed lines (stderr)
`[fpf_trimul_v4] cell on <device>: table.json[<key>] (...) -> SERVE {...}` once per device (or the `safe settings served (...)` line of
`opt_core/pair_fused:trimul`, or `cell=none: <why>` when the lever cannot run); `[fpf_trimul_v4 probe] C=.. D=.. <bias> <dtype>: ...` once per shape
class; `[fpf_trimul_v4] FIRST CALL served: {...}`; one line per fallback reason; `[fpf_trimul_v4.generic] BATCH LOOP FALLBACK by name: ...` when a
batch exceeds a launch-set limit; `COUNTS {...}` from `trimul` and `generic` at exit.

## Tests
`tests/test_generic.py` (the generic entry against `reference_torch`), `tests/test_cells_eq.py` (host-descriptor cells vs the pointer /
device-descriptor kernels: byte equality on one stack); the core's `tests/test_trimul_v4_*.py`.
