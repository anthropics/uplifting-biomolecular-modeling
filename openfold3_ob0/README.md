# OpenFold3 (OpenBind-0) — optimization kit

Drop-in modes that make stock OpenFold3 0.5.0 (`run_openfold predict` with the OpenBind-0 checkpoint) faster and lighter on
GPU memory. You call `run_openfold predict` exactly as before; the kit adds a `--mode`:

- `off` — stock OpenFold3, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs; `--n_gpu P` splits one `big` prediction across P GPUs of one host.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs, up to 5,000 tokens on one GPU · `--n_gpu P` splits `big` across P GPUs of one host.

## Setup

Pick ONE way to get the pinned stack — stock OpenFold3 0.5.0 and all it needs (pins: the 'Stack' section of `STOCK.md`):
**A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA GPU of compute capability ≥ 8.0, driver ≥ 570.

Route C additionally needs:

- a released CPython 3.11, gcc and the CUDA 12.8 toolkit.

Type the first block from the directory that holds `openfold3_ob0/` and `common/`:

```bash
# A — Docker (preferred)
docker build -f openfold3_ob0/environment/Dockerfile -t openfold3_ob0-kit:dev .
docker run --rm -it --gpus all -v /weights:/weights -v $PWD/out:/kit/openfold3_ob0/out openfold3_ob0-kit:dev bash   # cwd /kit/openfold3_ob0; out/ = $PWD/out
# B — Apptainer / Singularity (the whole Setup for B): converts A's image (needs A once; no Docker daemon at run time)
apptainer build openfold3_ob0.sif openfold3_ob0/environment/apptainer.def
mkdir -p out jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --bind <weights dir>:/weights --bind "$PWD/out":/kit/openfold3_ob0/out openfold3_ob0.sif "$@"; }   # kit = B's ./run.sh
export OPENFOLD3_OB0_CKPT=/weights/of3-ob-2025-06-30-174k.pt && kit install --weights /weights   # /weights: the checkpoint; fetched or checked; read-only ok
kit check --config h100 --mode fast              # Run lines: kit <command> … · --config a100|h200|b200|b300 too
# C — instead of A or B, on your own host (venv): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`. Under **A** you
are now in the container shell, which opens in `/kit/openfold3_ob0` (its `out/` is the mounted `$PWD/out`). Under **C** you are
in your activated environment inside `openfold3_ob0/`; STOCK.md's Stack section already ran the install line, so continue at
`export`. Run:

```bash
[ -f run.sh ] || cd openfold3_ob0                # no-op once inside · C via §Stack: resume at export
bash run.sh install --weights /weights             # --weights: 2.3 GB if absent, else hash-check; read-only ok
export OPENFOLD3_OB0_CKPT=/weights/of3-ob-2025-06-30-174k.pt   # required on every route: names the 2.3 GB checkpoint
bash run.sh check --config h100 --mode fast         # dry run: mode/GPU gate, pin, digest · A100: --config a100
```

What the blocks assume:

- **Weights.** `OPENFOLD3_OB0_CKPT` is required on every route and names the checkpoint `of3-ob-2025-06-30-174k.pt`;
  `install --weights <dir>` fetches it into `<dir>` when absent, otherwise hash-checks it (read-only is fine). Under B,
  `<weights dir>` is your weights directory on the host, bound to `/weights`.
- **GPU cards.** `--config h100|a100|h200|b200|b300` loads `configs/<card>.env` on `pred` / `check` / `warm`.
- **Pin check.** It refuses another `openfold3` version, an editable checkout or an edited file, by name (exit 3); other
  library versions are named on the `ACTIVE` / `LEVER` lines, not refused.
- **No network.** Fetch what `stock/PINS.json` names (wheel, source tree, CCD, checkpoint ≈ 2.3 GB), then
  `bash run.sh install --wheel FILE --src TARBALL --ccd FILE`.

## Run

```bash
Q=stock/src/examples/example_inference_inputs/query_ubiquitin.json   # upstream's example query
bash run.sh pred --config h100 --mode off   --query-json $Q --output-dir out/off   --use-msa-server false   # stock, in a clean subprocess
bash run.sh pred --config h100 --mode exact --query-json $Q --output-dir out/exact --use-msa-server false
bash run.sh pred --config h100 --mode fast  --query-json $Q --output-dir out/fast  --use-msa-server false   # the default mode
bash run.sh pred --config h100 --mode big --query-json $Q --output-dir out/big --use-msa-server false
bash run.sh pred --config h100 --mode big --n_gpu 2 --query-json $Q --output-dir out/big_x2 --use-msa-server false   # 2, 4 or 8 GPUs of one host
```

**Options.** `pred` is one process and one `run_openfold predict` call (under `big`, queries below and above its size
threshold — `OF3O_MIN_TOKENS`, 1,401 polymer tokens on H100 and A100 — run as two calls).

- Upstream's own flags (`--use-msa-server`, `--num-diffusion-samples`, …) pass through.
- `--runner-yaml FILE` (repeatable) is an overlay on the mode's configuration, `off` included, with the mode's execution keys
  winning and every override printed.
- The kit's own flags: `--mode`, `--n_gpu`, `--config`, `--det`, `--z-dtype`, `--conf-dtype`, `--graphs-max-tokens`,
  `--allow-template-drop` (`--ckpt` is upstream's `--inference-ckpt-path`).
- `. configs/h100.env; OPENFOLD3_OB0_OPT=<mode> run_openfold predict …` engages a mode from the unchanged stock command line.

**Outputs.** They land where stock writes them, `<output-dir>/<query>/seed_<s>/…` (rank logs under `<output-dir>/_tp/` with
`--n_gpu`).

**What a run prints** (stderr, prefixed `[openfold3_ob0-opt…]`).

- `check` prints `DRY-RUN mode=<m> …` (exit 0).
- A kit mode prints `ACTIVE mode=<m> line=… levers_requested=…` — the `ACTIVE` line, naming the mode and the optimizations
  engaged (the kit calls its individually switchable optimizations 'levers'). At exit it prints one `LEVER name=… state=…`
  line per optimization and, last, `pred --mode <m>: exit rule -> <rc> (…)`.
- `--mode off` prints `stock subprocess: …` in place of the `ACTIVE` line.
- If a mode cannot engage, the command exits 3 and names the reason (`NOT ACTIVE: <reason>`); it never falls back to stock
  silently.
- Under `fast`/`big` the exit census `[opt_core] CELLS …` may print `NAMED_FALLBACK:apb … served=sdpa:auto` lines:
  call classes outside the measured table (here the 5-sample diffusion batch) run PyTorch SDPA by name; these are
  reports, not errors, and the outputs are unaffected.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | the prediction failed (incl. incomplete output) |
| 2 | usage error |
| 3 | a mode, optimization or pin requirement not met (named) |
| 5 | declared templates dropped by upstream (`--allow-template-drop` accepts, exit 0) |

**First run.** A first kit-mode run spends about 20 s compiling Triton kernels (`bash run.sh warm --config h100 --mode fast --out DIR`
does it ahead): compare modes on a second run. The first run in a fresh cache directory also performs a one-time numerical self-check
of the fused kernels on your GPU (seconds on H100; up to ~30 s on A100) before serving them; later runs skip it.

**Where the gain is.** The kit's gain is in prediction time per input. Each `pred` call also spends ≈ 45 s starting Python and
loading the 2.3 GB checkpoint in every mode, so the single-query example above gains only a little under `fast` over `off` on
an H100 — the gain grows with the queries and seeds per call.

## Modes

- `off` — stock `run_openfold predict` on the stock configuration (`STOCK.md`), in a clean subprocess with nothing of the kit
  importable.
- `exact` — fast model construction, the diffusion pair cache, the sampler step in a CUDA graph, memoised casts and masks,
  bit-exact LayerNorm / triangle / transition kernels admitted after an in-process equality proof; outputs identical to `off`
  under `--det 1`. Use it when outputs must not move.
- `fast` (default) — `exact` plus fused Triton kernels for the pair stack, flash attention in the trunk and the diffusion
  transformer, and bf16 in the diffusion roll-out, the trunk hand-off and the confidence phase; bf16 re-association, within
  stock's seed-to-seed variation. Use it for throughput.
- `big` — numerics as `fast`. From `OF3O_MIN_TOKENS` polymer tokens up, pair stack and templates run in row blocks (confidence
  heads from their own gate) and the pair update is streamed from host memory; below the gate it is exactly `fast`.
  `--n_gpu 2|4|8` row-shards it over P GPUs. Use it when `fast` lacks memory.

## Known upstream issues

| ID | symptom in upstream 0.5.0 | flag / behaviour |
|---|---|---|
| `OB0-001` | offline template fetch fails; upstream predicts untemplated, exit 0 | each kit route: `TEMPLATES DROPPED …`, exit 5 (`--allow-template-drop`: 0) |
| `OB0-002` | chunk-size tuner raises when the confidence stack's call form changes mid-process | `fast`: handled by `tuner_guard`; `off` / `exact` unaffected |

Both are notes under `upstream_issues/`; this kit registers no `--upstream-fix` ID. Related upstream behaviour:
`--use-templates false` on a query that still carries template keys fails inside upstream (`UnpicklingError`) — remove the
keys instead.

## Notes

- **Cards.** Every optimization engages on compute capability 8.0 or newer (an untested card or library version is named on
  the `ACTIVE` / `LEVER` lines); below 8.0 a mode exits 3 naming the reason.
  - **A100** (`--config a100`): under `exact` the transition kernel, not yet proven equal on an A100 at this stack, runs as the
    module's own code (named).
  - **H200** (`--config h200`): optimizations as on H100; under `exact` a kernel unproven on H200 runs the module's code, named
    on its `LEVER` line.
- **A mode without some optimizations.** `MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]` runs a mode without the named ones (names
  as printed on the `LEVER` lines); the `ACTIVE` line then carries `levers_off=<names>`.
- **Deterministic runs.** `--det 1`: the deterministic recipe on any mode (deterministic torch / cuDNN algorithms,
  `CUBLAS_WORKSPACE_CONFIG`); `exact` then equals `off` bit for bit.
