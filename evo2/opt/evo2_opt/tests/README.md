# evo2_opt tests (CPU; no GPU, no upstream packages needed; the two specdec files skip without torch)

    cd opt && python -m unittest discover -s evo2_opt/tests -t .

| file | covers |
|---|---|
| `test_activation_gate.py` | the gate behind the ACTIVE / NOT ACTIVE line: refused only for no CUDA device or `evo2` / `vtx` absent or off their pins; an unlisted GPU, a library off the stack's pins, Transformer Engine absent are named on the line; `enable('off')` is the stock, any other mode name raises |
| `test_autoload.py` | the `.pth` hook: nothing installed under `EVO2_OPT` unset / `off`; one finder under `exact`, fired once by the `evo2` import only; any other value exits 2 at that import |
| `test_kfft_shapes.py` | `kit/kshapes.py`, vortex-kernels route: the FFT chains' sizes, the spectrum / output storage shared by both chains, resident bytes at one shape, the `hcm` FFT-conv call predicate, filter-group rows |
| `test_kit_gate.py` | the kit's call-surface gate: every (batch, length) on the kit path, inference-params / padded calls on the stock path counted and named once, one shape's stores resident at a time, the `gated` wrapper, the hook rule (hooked model → stock path; a hook on a folded projection → refused) |
| `test_lkeyed.py` | `kit/lkeyed.py`: the length-keyed caches hold one length at a time across every registered store; a batch change keeps them; hold / clear before the first write |
| `test_pins.py` | `stock/PINS.json` as the package reads it: tree root and `EVO2_OPT_HOME`, the stack by Transformer Engine presence, the schema the readers rely on |
| `test_rmsnorm_rule.py` | `kit/rmsnorm_rule.py`: ATen's partial-thread count for the bf16 row norm at the `evo2_7b` and `evo2_40b` widths (the fused RMSNorm kernel's reduction-order parameter); other widths are not served |
| `test_route_driver.py` | `route/evo2_route.py`: FASTA reading, the modes' constructions (`use_kernels=True` for the one-device stock, the constructor defaults for `evo2_40b`) and the mode vocabulary, `scores.jsonl` / `run.json` written from one `score_sequences` call on a fake `evo2`, the mode / `EVO2_OPT` agreement as a usage error before any model loads |
| `test_specdec_rule.py` | `gen/specdec/rule.py`: accept / residual / bonus against exact rational arithmetic; the emitted token is distributed as p (top-k-truncated p, q with zeros, p == q, one-hot p, a near-tie); greedy; draws go through torch's generator |
| `test_specdec_state.py` | `gen/specdec/core.py`: `StateStack` snapshot / per-layer record / commit restore the slots' values in place (the tensors stay the same objects), flattened or not; depth overflow is named; the multi-token hook steps a cascade once per position |
| `test_torchconv_shapes.py` | `kit/kshapes.py`, torch-conv route: the long-filter build's channel tiles cover every channel once in order, the inverse transform's view fits the full-row spectrum storage, the medium filter's rows per group |

The levers' bit-identity with the stock is a property of the GPU kernels (the same ops in the same order on the device) and is not asserted by these CPU tests.
