# The kit's patched `rf3` files — sampler CUDA graph + step-invariant hoist

Five files of RoseTTAFold3's `rf3` package (RosettaCommons foundry @ `4010e3e2e`), carried whole under `patched/rf3/` and installed over the
patched interpreter's copy of `rf3` by `install.sh` (the kit's `install` verb runs it; the stock interpreter is never touched):

| file | change |
|---|---|
| `rf3/graph_flags.py` | new: the two run-time flags below and the per-roll-out hoist cache (`hoist_begin` / `hoist_get` / `hoist_end`) |
| `rf3/diffusion_samplers/inference_sampler.py` | `RF3_CUDAGRAPH=1`: the 200-step denoising roll-out captured as one CUDA graph per token count and replayed step by step (noise drawn up front with the stock generator calls, in stock order); `RF3_CUDAGRAPH=0`: the stock loop, verbatim |
| `rf3/model/RF3_structure.py` | `RF3_HOIST=1`: the denoiser's pair conditioning and single-conditioning prefix computed once per roll-out and reused by every step (they do not depend on the step, the noise level or the coordinates — `docs/INVARIANCE_PROOF.md`) |
| `rf3/model/layers/af3_diffusion_transformer.py` | the same hoist for the atom-attention encoder's prefix and every attention block's pair-bias projection (24 token-level, 6 atom-level); under `RF3_GRAPH_SAFE_OPS` the host-synchronising statements of the atom encoder/decoder (a data-dependent size, a boolean-mask gather, an assertion) replaced by capture-safe tensor ops with identical values |
| `rf3/loss/loss.py` | under `RF3_GRAPH_SAFE_OPS`, the batch identity matrix built on the device instead of copied from the host (same values) |

Every edit is marked `[rf3_cudagraph]` or `[xattempt_hoist]` in the file and keeps the stock statement in its `else` branch.

Flags (environment of the fold process, read once by `rf3/graph_flags.py`): `RF3_CUDAGRAPH` = `1` graph (the value when unset) | `0` stock
loop; `RF3_HOIST` = `1` hoist | `0` stock statements (the value when unset); the capture-safe op rewrites are on exactly when the graph is on and the
capture warm-up is 3 eager steps (`graph_flags.py`, no variable); the runtime lever `warm` (`graph_flags.set_levers(warm=True)`, default off — the FPF adapter's arm step `@L1.warm` sets it) keeps 3 steps before the process's first capture and runs 1 before every later one (the warm-up touches only the static clones and consumes no draw: the roll-out's values do not depend on the count; `GRAPH_STATS` counts captures and warm-up steps). The kit's modes set `RF3_CUDAGRAPH=1 RF3_HOIST=1`; `--mode off` runs the stock
interpreter, where none of these files is installed.

```bash
bash install.sh              # classify the 5 target files of the active interpreter's rf3 (stock at the pin | these files), back up, install, byte-check
bash install.sh --status     # per-file state;  --check: exit 0 iff installed;  --uninstall: restore the backups
```

Layout: `install.sh`; `patched/rf3/...` (the five files); `public_inputs/1brs_tiles.json` (barnase–barstar, PDB 1BRS, tiled 1:1 … 4:4 =
199–796 tokens, single sequence — the kit's `warm` input and the README's example); `docs/INVARIANCE_PROOF.md` (what is hoisted, from which
code path, and why it cannot depend on the step); `LICENSE`, `NOTICE`, `LICENSE_NOTES.md`, `LICENSE_foundry_BSD-3-Clause.md`.

Note on outputs: foundry sets a prediction's summary `ranking_score` to about -99.85 when its clash/validity gate fires, and on low-confidence
inputs (the 1BRS 3:3 and 4:4 tiles) that gate can fire in one run and not in another of the same configuration, stock included — read
pTM / ipTM / pLDDT rather than `ranking_score` on such inputs. Stock's own numerics are not bitwise run-to-run (CUDA atomics in `scatter_mean`).
