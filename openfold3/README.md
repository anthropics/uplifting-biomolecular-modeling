# OpenFold3 — optimization kit

Drop-in modes that make stock OpenFold3 0.4.1 (`run_openfold predict`) faster and lighter on GPU memory. You call
`run_openfold predict` exactly as before; the kit adds a `--mode`:

- `off` — stock OpenFold3, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs; `--n_gpu P` splits one `big` prediction across P GPUs of one host.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs, up to 6,000 tokens on one GPU · `--n_gpu P` splits `big` across P GPUs of one host.

## Setup

Pick ONE way to get the pinned stack — stock OpenFold3 0.4.1 and everything it needs (the 'Stack' section of `STOCK.md`
lists every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- a Linux x86-64 host with an NVIDIA GPU whose driver supports CUDA 12.8 (≥ 570).

Route C additionally needs:

- the CUDA 12.8 toolkit and a C compiler;
- an unmodified `openfold3` 0.4.1 install: the pin check accepts nothing else (exit 3 otherwise).

Type the first block from the directory that holds `openfold3/` and `common/`:

```bash
# A — Docker (recommended: the whole pinned stack, stock and this kit in one image)
docker build -f openfold3/environment/Dockerfile -t openfold3-kit:dev .      # other cards: --build-arg DS4SCI_ARCHS=… (Notes)
docker run --rm -it --gpus all -v /weights:/weights -v "$PWD/out":/kit/openfold3/out openfold3-kit:dev bash   # the shell opens in /kit/openfold3
# B — Apptainer / Singularity (the whole Setup for B): build the .sif once where Docker runs (converts A's image); the cluster then needs only Apptainer
apptainer build openfold3-kit.sif openfold3/environment/apptainer.def    # from local openfold3-kit:dev (per card: A's DS4SCI_ARCHS)
W="<dir holding of3-p2-155k.pt>"                                             # your weights directory on the host
mkdir -p out jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --bind "$W":/weights/openfold3 --bind "$PWD/out":/kit/openfold3/out openfold3-kit.sif "$@"; }   # kit = B's ./run.sh; Run's out/… lands in ./out
export OPENFOLD3_CKPT=/weights/openfold3/of3-p2-155k.pt && kit install --weights /weights/openfold3   # /weights/openfold3: ckpt fetched or checked; read-only ok
kit check --config h100 --mode fast                          # cards: --config a100|h200|b200|b300 on every kit line
#     B keeps its compile cache in ./jit (seeded once from the image, then reused) and a small checksum memo of the weights under ~/.cache/openfold3_opt.
# C — instead of A or B, on your own host (venv): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`. Under **A** you
are now in the container shell, which opens in `/kit/openfold3`. Under **C** you are in your activated environment inside
`openfold3/`; STOCK.md's Stack section already ran the install line, so continue at `export`. Run:

```bash
[ -f run.sh ] || cd openfold3                               # no-op once inside · C via §Stack: resume at export
bash run.sh install --weights /weights/openfold3               # --weights: fetch if absent, else hash-check; read-only ok
export OPENFOLD3_CKPT=/weights/openfold3/of3-p2-155k.pt      # required: the checkpoint file, every route
bash run.sh check --config h100 --mode fast                    # dry run, rc 0: a DRY-RUN line and a WEIGHTS line
```

What the blocks assume:

- **Weights.** `OPENFOLD3_CKPT` is required on every route and names the checkpoint file `of3-p2-155k.pt`.
  `install --weights <dir>` fetches it into `<dir>` when absent, otherwise hash-checks it (read-only is fine). Under B, `W` is
  your weights directory on the host, bound to `/weights/openfold3`.
- **Other cards at build time.** `docker build` takes `--build-arg DS4SCI_ARCHS=…` for a card other than H100/H200 (Notes);
  the `.sif` is converted from the local `openfold3-kit:dev` image, so it serves the same cards.
- **GPU cards.** `--config h100|a100|h200|b200|b300` loads `configs/<card>.env` (`MODEL_OPT_TARGET_GPU`, `OPENFOLD_CACHE`, the
  compile caches under `MODEL_OPT_JIT_ROOT` when set); every command takes it.
- **Compile cache.** A kit mode compiles its kernels once per machine; the cache persists under `MODEL_OPT_JIT_ROOT`, and
  `bash run.sh warm --config h100 --mode fast --out out/warm` pays it ahead of time. Each process captures its CUDA graphs at
  start. Route B keeps its compile cache in `./jit` (filled on the first run, then reused) and a small checksum memo of the
  weights under `~/.cache/openfold3_opt`. The first run in a fresh cache directory also performs a one-time numerical
  self-check of the fused kernels on your GPU (seconds on H100; up to ~30 s on A100) before serving them; later runs skip it.

## Run

```bash
Q=stock/src/examples/example_inference_inputs/query_ubiquitin.json   # upstream's example; under B it resolves inside the image
F="--use-msa-server false --use-templates false"                     # it carries no MSA or template files: fold it offline
# Under B (kit <command> …) your own --query-json and --output-dir take absolute host paths under $HOME, $PWD or a bound directory.
bash run.sh pred --config h100 --mode off   --query-json $Q --output-dir out/off   $F   # stock, in a clean subprocess
bash run.sh pred --config h100 --mode exact --query-json $Q --output-dir out/exact $F
bash run.sh pred --config h100 --mode fast  --query-json $Q --output-dir out/fast  $F   # the default when no mode is named
bash run.sh pred --config h100 --mode big --query-json $Q --output-dir out/big $F
bash run.sh pred --config h100 --mode big --n_gpu 2 --query-json $Q --output-dir out/big_x2 $F   # 2, 4 or 8 GPUs of one host
```

**Options.** `pred` makes one `run_openfold predict` call (two when a `big` query set straddles its size gate; P ranks under
`--n_gpu P`). It accepts:

- upstream's `--query-json`, `--output-dir`, `--num-model-seeds`, `--num-diffusion-samples`, `--use-msa-server`,
  `--use-templates`, `--inference-ckpt-path` and `--runner-yaml` (repeatable, each override printed);
- the kit's `--mode`, `--n_gpu`, `--det`, `--upstream-fix`, `--ckpt`, `--graphs-max-tokens`, `--allow-template-drop`.

For any other `run_openfold predict` option use `OPENFOLD3_OPT=<mode> run_openfold predict --runner-yaml <the mode's YAML, CHANGES.md> …`,
which engages a mode from the unchanged stock command line.

**Inputs under B.** Your own `--query-json` and `--output-dir` take absolute host paths under `$HOME`, `$PWD` or a bound
directory; upstream's example resolves inside the image.

**Outputs.** They land where stock writes them: here 5 `*_model.cif` and the confidence JSONs under `<output-dir>/ubiquitin/seed_42/`.

**What a run prints** (stderr).

- A kit mode prints `[openfold3-opt] ACTIVE mode=<mode> line=… gpu=… levers_requested=… n_gpu=…` once engaged — the `ACTIVE`
  line, naming the mode and the optimizations engaged (the kit calls its individually switchable optimizations 'levers'). At
  the end it prints `exit mode=… levers_applied=…`, with one `LEVER name=… state=…` line per optimization.
- `off` prints `stock: run_openfold predict …` instead.
- Every mode prints a `Model forward time: …` line per item and closes with `pred --mode <m>: exit rule -> <rc> (…)`.
- If a mode cannot engage, the command prints `[openfold3-opt] NOT ACTIVE: <reason>` and exits 3; it never falls back to
  stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | the prediction failed (incl. incomplete outputs) |
| 2 | usage error |
| 3 | not active, or an optimization did not install |
| 5 | (`pred` only) templates dropped |

## Modes

- `off` — stock `run_openfold predict` with the stock configuration (`STOCK.md`) in a clean subprocess, nothing of the kit
  importable; adds only a forward timer.
- `exact` — redundant work skipped or cached (weight init, checkpoint mapping, per-rollout pair and atom caches, a CUDA-graphed
  sampler on small inputs) and bit-exact kernels where the GPU has them; identical to `off` under `--det 1` (the
  deterministic recipe; default `--det 0`). Use it when outputs must not move.
  - Its exact triangle-attention row serves pair stacks of 101 tokens and more. On shorter inputs — the 76-token example
    above included — stock itself takes its small-input attention path, so that call never reaches the row and the
    `LEVER name=triatt_exact` line reads `rows=none` (by design; outputs are stock's own).
- `fast` (default) — `exact` plus fused kernels for the pair stack (opt_core's FlashPairformer cells), diffusion transformer
  and atom attention, CUDA graphs at every size, bf16 trunk and heads; within stock's seed-to-seed variation. Use it for
  throughput.
- `big` — `fast`'s kernels with the pair-track statements in row blocks and confidence heads per row block; numerics as
  `fast`. Small inputs run as `fast` minus its memory-costing optimizations, as its `LEVER` lines note (gates: `CHANGES.md`).
  `--n_gpu 2|4|8` row-shards the pair track over P GPUs. Use it when `fast` runs out of memory.

**Where the gain is.** The kit's gain is in the model's forward (prediction) time per input and grows with input size; on one
small input `exact` and `fast` gain less end to end, start-up and featurisation included, than the forward pass alone does.

## Known upstream issues

| `--upstream-fix <ID>` | what it fixes (one line) | default |
|---|---|---|
| `OF3-001` | `--use-templates true`: templates are preprocessed but never handed to the model; the fix makes the model consume them (covers OF3-004 too) | off |

- Fixes are opt-in on every mode, `off` included, and print `UPSTREAM-FIX <ID> applied`.
- Independently, a query that declares templates and reaches the model with none prints `TEMPLATES DROPPED …` and exits 5 after
  writing its outputs (`--allow-template-drop` accepts it).
- `upstream_issues/` also holds notes OF3-003, OF3-004, OF3-005 (no fix flag; OF3-005 is handled by the optimization
  `tuner_guard` on `fast` / `big`).

## Notes

- **Other cards** (`--config a100|h200|b200|b300`). An image for a card other than H100/H200 is built with
  `--build-arg DS4SCI_ARCHS=…` (STOCK.md, Stack section).
  - H200 has H100's compute capability (9.0), so every mode engages exactly as on H100.
  - `configs/a100.env` lists every A100 difference (launch tiles; one template-stack shape runs upstream's statement, counted
    on its `LEVER` line).
  - Under `exact`, the exit census `[opt_core] CELLS … alerts=<n>` counts the
    `UNCOVERED_CELL:transition|<cc>|…|N=<n> … served=caller:torch_swiglu` lines printed with it: transition calls whose shape
    has no vouched cell on this card (a few small shapes on any card, more on the A100) run upstream's own statement.
    These are reports, not errors; the outputs are unaffected. Under `fast`/`big` the census may also print
    `NAMED_FALLBACK:apb … served=sdpa:auto` lines: call classes outside the measured table (here the 5-sample
    diffusion batch) run PyTorch SDPA by name; these too are reports, not errors.
- **DS4Sci op.** `off`, and `exact` at `--det 0`, need DeepSpeed's DS4Sci op built for the running card, else `STACK REFUSED …`
  before anything runs (exit 3; STOCK.md, Stack section).
- **Deterministic runs.** `--det 1` selects the deterministic recipe on any mode, `off` included; `exact --det 1` equals
  `off --det 1` bit for bit.
- **A mode without some optimizations.** `MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]` runs a mode without the named ones
  (`levers_off=<names>` on the `ACTIVE` line). `--no-compile` is accepted and inert (`compile=none`; nothing here uses
  `torch.compile`).
