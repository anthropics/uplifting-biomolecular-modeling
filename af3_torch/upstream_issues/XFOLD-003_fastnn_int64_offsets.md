# XFOLD-003 — the fastnn Triton kernels: 64-bit row / head offsets (`xfold/fastnn/{gated_linear_unit,layer_norm,attention}.py`)

**Defect.** The three kernels form every element offset in 32-bit integers and scale a row (or batch·head) index by a stride:

| kernel | statement | tensor it indexes | wraps (`index · stride > 2³¹ − 1`) past |
|---|---|---|---|
| `gated_linear_unit.py` `_glu_kernel` | `out_ptr + stride_om * offs_cm` (store), `x_ptr + offs_am * stride_xm` (load) | the wrapper flattens every leading dimension into rows: the pairformer pair transition's output `[N², 512]`, input `[N², 128]` (also `[N², 256]` / `[N², 64]` elsewhere) | **N = 2048 tokens** (`2048² · 512 = 2³¹`), 4096 for the input |
| `layer_norm.py` `_layer_norm_fwd_fused` | `X += row * N`, `Y += row * N` (`row = program_id(0)`) | one program per row: the diffusion pair conditioning's `[N², 267]`, every pair LayerNorm's `[N², 128]` | 2836 tokens (`⌈2³¹ / 267⌉` rows), 4096 |
| `attention.py` `_attention_core` | `off_hz * stride_qh` (q / k / v loads), `off_hz * stride_oh` (store), `off_hz_bias * (N_CTX * N_CTX)` | triangle attention: q `[N, 4, N, 32]`, `off_hz < 4N`, `stride_qh = 32N` | 4096 tokens (bias: 23,170) |

A wrapped tile is addressed `2³²` elements below its place. When that address is unmapped the launch faults (`CUDA error: an illegal memory
access was encountered`, reported at whatever call synchronises next); when it is mapped — the caching allocator's layout decides, so the same
input faults in one process and finishes in another — the stores land inside other live tensors and the true rows keep `torch.empty`'s
contents: the fold finishes and is wrong. Measured on H100 (torch 2.13 / triton 3.7), the GLU kernel alone in a fresh process beside a 6 GiB
tensor of sevens: N = 2048 correct and the sentinel untouched; 2078 finished with 63,375,360 sentinel elements overwritten (= 123,780 wrapped
rows × 512) and the last rows of the output wrong; 2254 finished with 453,740,544 overwritten (= 886,212 × 512); 2472 faulted. The eager path
(`--nofastnn`) is not affected.

**Patch** (`XFOLD-003_fastnn_int64_offsets.diff`, four statements). Promote the index to int64 before it is scaled: `_glu_kernel` —
`offs_am = (… % M).to(tl.int64)`, `offs_cm = (…).to(tl.int64)`; `_layer_norm_fwd_fused` — `row = tl.program_id(0).to(tl.int64)`;
`_attention_core` — `off_hz = tl.program_id(1).to(tl.int64)` (its bias / mask / output offsets follow from it). Nothing else changes — tile
schedule, K loops, accumulators, epilogues, masks, the GLU autotune configurations and key — so every output element is produced by the same
arithmetic in the same order: outputs equal the unpatched kernels' byte for byte wherever those addressed inside their bound, and are correct
past it. The remaining int32 terms are per-row column offsets and `N_CTX * N_CTX` (wraps past 46,340 tokens). Cost: an int64 multiply-add per
tile row; GLU and LayerNorm kernel times at 1400 tokens are within run-to-run noise of the unpatched kernels' (CHANGES.md 0.2.10 has the numbers).

**In the kit.** Applied in the port (`opt/forward/af3t/af3_torch/xfold/fastnn/`), every mode, `off` included — `stock/PINS.json`
`upstream.port.changed_files`, STOCK.md 'the port' (3). The fastnn kernels run under `off` and `exact` (the stock CLI's `--fastnn` default);
`fast` / `big` run the eager primitives with the kit's own kernels and never launch them.
