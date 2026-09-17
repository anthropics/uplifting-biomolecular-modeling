# Chai-1 — optimization kit

Drop-in modes that make stock Chai-1 0.6.1 (`chai_lab.chai1.run_inference`, the API behind `chai-lab fold`) faster and lighter
on GPU memory. You call Chai-1 exactly as before; the kit adds a `--mode`:

- `off` — stock Chai-1, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
Upstream licence, third-party notices and the upstream citation request: `THIRD_PARTY_NOTICES.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs, up to 2,000 tokens on one GPU.

## Setup

Pick ONE way to get the pinned stack — stock `chai_lab` 0.6.1 and everything it needs (the 'Stack' section of `STOCK.md`
lists every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver ≥ 580 on the host.

Route C additionally needs:

- gcc, the CUDA 13.0 development headers and a released CPython 3.11;
- or, instead, an environment already running `chai_lab` 0.6.1 as released (the pin check names drift: STOCK.md, 'Pin' section).

A and B build from the directory that holds `chai1/` and `common/`; type the first block there:

```bash
# A — Docker (preferred): the pinned stack, stock and this kit in one image; the shell opens in /kit
docker build -f chai1/environment/Dockerfile -t chai1-kit:dev .
docker run --rm -it --gpus all -w /kit -v /weights/chai1:/weights/chai1 -v $PWD/cache:/cache -v $PWD/out:/kit/chai1/out chai1-kit:dev bash
# B — Apptainer (the whole Setup for B): build the .sif where Docker holds A's image (any host), copy it over; type every kit line in this directory
apptainer build chai1-kit.sif chai1/environment/apptainer.def
mkdir -p cache out; kit() { apptainer run --nv --bind /weights/chai1:/weights/chai1 --bind "$PWD/cache":/cache --bind "$PWD/out":/kit/chai1/out chai1-kit.sif "$@"; }   # kit = B's ./run.sh · other cards: --config a100|h200
export CHAI_DOWNLOADS_DIR=/weights/chai1 MODEL_OPT_JIT_ROOT=/cache/jit && kit install --weights /weights/chai1   # /weights/chai1: models_v2/… fetched or checked; read-only ok
kit check --config h100                            # later run.sh lines: kit <command> …; own inputs: absolute paths
# C — instead of A or B, on your own host (venv): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`. Under **A**
you are now in the container shell (it opens in `/kit`), where the image is installed already: `install` only checks the pin
and handles `--weights`. Under **C** you are in your activated environment: `install` installs the kit and the shared core
editable — skip it if STOCK.md's Stack section already ran it — and checks the pin. Run:

```bash
[ -f run.sh ] || cd chai1                         # no-op once inside · C via §Stack: skip install too
export CHAI_DOWNLOADS_DIR=/weights/chai1          # required; must hold models_v2/, *.apkl, esm/
bash run.sh install --weights "$CHAI_DOWNLOADS_DIR"  # --weights: ≈7 GB or hash-check, read-only ok; C: §Stack did
export MODEL_OPT_JIT_ROOT=/cache/jit              # optional, 0.2–0.5 GB; C: any writable dir
bash run.sh check --config h100                      # dry run: pin, weights (sha256), GPU vs target card; no model
bash run.sh warm  --config h100                      # optional, about 6 min per card: compiled step, all crops
```

What the blocks assume:

- **Weights directory.** `CHAI_DOWNLOADS_DIR` is required and must hold `models_v2/`, `*.apkl` and `esm/`; `/weights/chai1`
  is the mount point under A and B. `install --weights "$CHAI_DOWNLOADS_DIR"` fetches the weights when absent, otherwise hash-checks them
  (read-only is fine).
- **Inputs under B.** Your own inputs are absolute paths; the example input in Run is a path inside the image.
- **Expected pin-check lines.** The pin check prints `declared range torch<2.7,>=2.3.1 -> False` and
  `the pinned stack torch2.13.0-cu130-sm90` on every card: both are expected (`STOCK.md`).
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env` (`MODEL_OPT`, `MODEL_OPT_TARGET_GPU`); every command
  below takes it.
- **Compile cache.** `fast` and `big` compile Triton kernels at first use, once per machine. They also compile the denoiser
  step in every new process unless `run.sh warm` has built its packages for this card (all seven crop sizes; `--crops all`
  pre-fills the Triton caches per crop too). A writable `MODEL_OPT_JIT_ROOT` keeps caches and packages between runs and
  containers (optional; under C any writable directory). The first run in a fresh cache directory also performs a one-time
  numerical self-check of the fused kernels on your GPU (seconds on H100; up to ~30 s on A100) before serving them; later
  runs skip it.

## Run

```bash
X=opt/forward/fast_inference/tests/public_inputs/1BRS_1to1.fasta   # barnase–barstar, 199 tokens; under B: path inside the image
bash run.sh pred --config h100 --mode off   --seed 42 --input $X --out_dir out         # stock, in a clean subprocess
bash run.sh pred --config h100 --mode exact --seed 42 --input $X --out_dir out
bash run.sh pred --config h100 --mode fast  --seed 42 --input $X --out_dir out         # the default when no mode is named
bash run.sh pred --config h100 --mode big --seed 42 --input $X --out_dir out
bash run.sh pred --config h100 --mode fast  --seed 42 --input $X --out_dir out --tag fast_1sample --num-diffn-samples 1   # any chai-lab fold option
```

**Options.** `pred` hands `run_inference` `chai-lab fold`'s own options verbatim (`STOCK.md` lists them). The kit's own flags are:

- `--mode`;
- `--input` — a chai FASTA, an items JSON `{"msa_dir": …, "items": [{"fasta": …, "seeds": [0, 1]}]}`, or `PACK.json`
  beside the example FASTA (four inputs of 199–796 tokens);
- `--out_dir`, `--tag` (default: the mode name);
- `--seed` / `--seeds` — a kit mode needs one;
- `--msa_dir`, `--det`, `--no-compile`, `--allow-partial`.

`chai1-opt pred --mode <mode> …` is the same call without `run.sh`. `. configs/h100.env && CHAI1_OPT=<mode> python your_script.py`
engages a mode in any program that calls `run_inference`.

**Outputs.** They land in `<out_dir>/<tag>/<fasta stem>/seed_<s>/` — above: `out/exact/1BRS_1to1/seed_42/` — as upstream's
`pred.model_idx_<k>.cif` and `scores.model_idx_<k>.npz`.

**What a run prints** (stderr).

- `check` prints `[chai1-opt] DRY-RUN mode=<mode> … weights=pinned …` (rc 0).
- A kit mode prints `[chai1-opt] ACTIVE mode=<mode> …` once engaged — the `ACTIVE` line, naming the mode and the
  optimizations engaged (the kit calls its individually switchable optimizations 'levers'). Then one `LEVER name=… state=…`
  line per optimization, `FORWARD item=… forward_s=…` per input, and `[chai1-opt] EXIT … seed_folds_ok=<k>/<n> …` at exit.
- `--mode off` prints `[chai1-opt] NOT ACTIVE mode=off (stock: nothing applied)` and runs stock (rc 0).
- If a mode cannot engage, the command prints `[chai1-opt] NOT ACTIVE: <reason>` and exits 3 before any fold; it never falls
  back to stock silently.
- The `ResourceWarning` lines printed on stderr are harmless.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | a fold failed |
| 2 | usage error |
| 3 | not active |

**First run.** On an H100 or A100 each command above takes about 1–2.5 min in a fresh process (checkpoint reads, first-use
compiles); compare modes on a second input or run.

## Modes

- `off` — stock `run_inference` as released, in a clean subprocess with nothing of the kit on its path.
- `exact` — the exported model as eager PyTorch, weights / ESM embeddings / feature context reused across inputs and seeds,
  the denoiser step's invariant work hoisted and the step graph-replayed; identical to `off` under `--det 1`. Use it when
  outputs must not move.
- `fast` (default) — `exact` plus TF32 products, the shared core's FlashPairformer triangle kernels in the trunk, the trunk
  run at the input's true length, fused pair-biased attention and a compiled denoiser step; within stock's seed-to-seed
  variation. Use it for throughput.
- `big` — `fast` plus row-chunked MSA-module and pairformer computation and no denoiser graph, so chai-lab's largest crop
  size folds on one GPU; numerics as `fast`. Use it when `fast` runs out of memory.

## Notes

- **Where the gain is.** The kit's gain is in prediction time per input (the `FORWARD … forward_s=` line) over many inputs.
  A kit process first spends about 55 s on start-up, checkpoint loads and featurization (longer than stock's), so the single
  small example in Run takes about as long under `exact`, `fast` or `big` as under `off` end to end (65–70 s per process);
  the gain shows from the second input or seed of a process.
- **A100** (`--config a100`): every optimization engages. The compiled step serves only from packages `run.sh warm` built on
  that card (else the eager step runs, `compile=off:card:no_aoti_packages_cc80`); `exact --det 1` equals `off --det 1` up to
  the 1024-token crop there (larger: `STOCK.md`).
- **H200** (`--config h200`): the H100 settings under the target word `H200`; every optimization engages as on H100, same caches.
- **A mode without some optimizations.** `MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]` runs a mode without the named ones (names
  as printed on the `LEVER` lines); the `ACTIVE` line then carries `PARTIAL off=<names>`. `--no-compile` runs `fast` / `big`
  without the compiled denoiser step — same numerics class.
- **Deterministic runs.** `--det 0|1` (default 0): 1 selects the deterministic recipe (deterministic torch and cuDNN
  algorithms, `CUBLAS_WORKSPACE_CONFIG`) on any mode, `off` included. Stock on a GPU is not bitwise run to run, so
  `exact` = `off` is checkable only as `exact --det 1` vs `off --det 1`.
- **Another torch / CUDA build.** Every mode still runs and says so (`[chai1-opt] NOTE STACK …`); `--strict-stack` refuses
  instead (exit 3).
