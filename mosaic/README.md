# mosaic — optimization kit

Drop-in modes that make stock `mosaic` (escalante-bio, commit `70fec525`; Boltz-2 binder hallucination through `joltz`)
faster to start, faster per design step and lighter on GPU memory ('stock' below always means this unmodified upstream
release). The recipe does not change: upstream's `examples/boltz_notebook.py`, as the kit's driver runs it in every mode
(seeded, with binder length and MSA as flags — STOCK.md, 'Stock exceptions'). The kit adds a `--mode`:

- `off` — that recipe with no optimization applied; stock's arithmetic.
- `exact` — stock's arithmetic, faster; each design is bitwise-reproducible per shape and seed once warmed.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs.

One design per GPU (no multi-GPU mode). What each mode's optimizations change: `CHANGES.md`. Exact versions, the
pinned software stack and all variables: `STOCK.md`. How the three setup routes (A — Docker, B — Apptainer, C — Python
venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` stock's arithmetic, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs.

## Setup

Pick ONE way to get the pinned stack — Ubuntu 22.04, CUDA 12.8, Python 3.12, jax 0.10.2, torch 2.7.1+cpu; stock
`mosaic`, `joltz` and the `boltz` fork at their pinned commits (the 'Stack' section of `STOCK.md` lists every pin):
**A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver 550 or newer.

Route C additionally needs:

- a released CPython 3.12 (the block gets one with `uv`), `git` on `PATH` and network access for pip's source builds.
  Once uv is installed, the route-C lines below are the whole recipe; STOCK.md's Stack section has the apt line, the uv
  bootstrap and the same lines written from inside `mosaic/`, annotated — run one or the other.

Type the first block from the directory that holds `mosaic/` and `common/`:

```bash
# A — Docker (recommended: the whole pinned stack, stock and this kit in one image)
docker build -f mosaic/environment/Dockerfile -t mosaic-kit:dev .
docker run --rm -it --gpus all -v /weights:/weights -v /cache:/cache -v "$PWD/runs":/kit/mosaic/runs -w /kit mosaic-kit:dev bash   # shell in /kit for step 2; weights at /weights/mosaic/boltz
# B — Apptainer (the whole Setup for B): build the .sif once where Docker runs (converts A's image); the cluster then needs only Apptainer
apptainer build mosaic-kit.sif mosaic/environment/apptainer.def
mkdir -p runs /your/weights/mosaic /your/cache; kit() { apptainer run --nv --bind /your/weights/mosaic:/weights/mosaic --bind /your/cache:/cache --bind "$PWD/runs":/kit/mosaic/runs mosaic-kit.sif "$@"; }   # kit = B's ./run.sh · other cards: --config a100|h200
export MOSAIC_CACHE_DIR=/weights/mosaic MODEL_OPT_JIT_ROOT=/cache && kit install --weights /weights/mosaic   # /weights/mosaic: boltz/ fetched or checked; read-only ok
kit check --config h100 --mode fast          # each Run line alike: kit <verb> … (runs/ = ./runs)
# C — instead of A or B, on your own host: a venv on a released CPython 3.12 — the whole route-C recipe once uv is installed; STOCK.md §Stack = apt line + uv bootstrap + these lines written from inside mosaic/, annotated (run one or the other)
uv venv --seed --managed-python --python 3.12.1 ~/venvs/mosaic && . ~/venvs/mosaic/bin/activate
export PYTHONHASHSEED=0
python -m pip install $(grep -E '^(pip|setuptools|wheel)==' mosaic/environment/requirements.lock)
python -m pip install --no-deps -r mosaic/environment/requirements.lock   # every pin; the three upstream packages from their commits
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`.
Under **A** you are now in the container shell, which opens in `/kit`. Under **C** you are in your activated
environment: after the route-C lines above, type the whole next block; if you used STOCK.md's Stack section instead, it
already ran the install line from inside `mosaic/`, so continue at the `export` lines. Run:

```bash
[ -f run.sh ] || cd mosaic                    # no-op once inside · C via §Stack: skip install too
bash run.sh install --weights /weights/mosaic   # --weights: ≈5 GB if absent, else hash-check; read-only ok
export MOSAIC_CACHE_DIR=/weights/mosaic      # must contain boltz/boltz2_conf.ckpt + boltz/mols/
export MODEL_OPT_JIT_ROOT=/cache             # compile caches + frozen features; C: any writable dir
bash run.sh check --config h100 --mode fast     # dry run (loads nothing): pins, weights file, cache, GPU
```

What the blocks assume:

- **Weights directory.** `/weights/mosaic` in the blocks (under A the host's `/weights` is mounted at `/weights`; under B
  you bind your own directory there). It must contain `boltz/boltz2_conf.ckpt` and `boltz/mols/`, and `MOSAIC_CACHE_DIR`
  must name it: `check`, `warm` and `design` require it.
  - `install --weights` fetches the files if absent, else hash-checks what is there, so a complete directory may be read-only.
- **Cache root.** `MODEL_OPT_JIT_ROOT` (`/cache` in the blocks; under C any writable directory) holds the compile caches
  and the frozen features. `warm` and `exact` require a cache root.
- **Outputs under A and B.** `runs/` in the container is the host's `./runs`.
- **Pin check** (every route; `stock/check_pins.py`, run by `install` and before every command). It accepts the three
  upstream packages only when pip installed them from their pinned commits (the lock's `git+https://…@<commit>` lines) or
  from the reference archives in `stock/`; a PyPI release, another commit or an editable checkout is refused by name (exit 3).
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env`: `MODEL_OPT_TARGET_GPU`, `MODEL_OPT_STACK_KEY`, and
  the kit's cache root `MOSAIC_OPT_CACHE_ROOT=$MODEL_OPT_JIT_ROOT/<stack key>/mosaic` unless you export
  `MOSAIC_OPT_CACHE_ROOT` yourself. Every command below takes it.
- **First design of a shape compiles.** The first design of a shape (target, binder length, copies) on a GPU type
  compiles for a few minutes and keeps the result under the cache root; `fast` and `big` need nothing else. An A/B image
  built with STOCK.md's optional cache tar seeds an empty `MODEL_OPT_JIT_ROOT` on the first command after the export.
- **`exact` needs a warm-up per shape.** Run `bash run.sh warm --mode exact` once per new shape and GPU type first: two
  design passes (stock, then one that fills the cache and pins the compiler's kernel choices and the featurized input),
  ≈8 min on H100, ≈14 min on A100.
  - An image built with STOCK.md's optional compile-cache tar ships the example's shape warmed for the host classes
    that tar covers — there `warm` on the example says so and exits 2 (it is for your own shapes). Everywhere else the
    first run of each mode compiles once (a few minutes) and keeps the result under the cache root; `exact` wants its
    `warm` there first, as above.
  - Until warmed, `exact` runs as stock and says so (Notes); with no cache root it is refused (exit 3).

## Run

```bash
# input: the example target shipped in the kit (barstar, inlined in opt/mosaic_opt/tools/fetch_public_inputs.py; warm also fetches 1BRS.pdb from RCSB for provenance, skipped offline), an 80-residue binder, single-sequence
bash run.sh design --config h100 --mode off   --out runs                     # stock, in a clean subprocess
bash run.sh warm   --config h100 --mode exact                                # exact only, once per NEW shape; A/B ship this one: skip
bash run.sh design --config h100 --mode exact --out runs
bash run.sh design --config h100 --mode fast  --out runs                     # the default when no mode is named
bash run.sh design --config h100 --mode big --out runs --target-copies 2   # a larger input: two copies of the target
bash run.sh design --config h100 --mode fast  --out runs --target-fasta /path/to/your_target.fasta --msa /path/to/your_target.a3m --epitope 12,15,40-48   # your own files (not shipped)
```

`design` runs the notebook recipe once in a fresh process.

**Inputs** (these flags apply in every mode, `off` included).

- `--target-fasta F` — one record (`--first-record` for a multi-record file);
- `--msa <.csv|.a3m>` — else single-sequence; no server is contacted;
- `--epitope` — 1-based target positions;
- `--binder-length L` (80), `--target-copies N` (1), `--seed S` (0), `--steps1` / `--steps2`.

**Options** (the kit's own flags).

- `--mode`, `--config`;
- `--det 0|1` — 1 adds `PYTHONUNBUFFERED=1` to the design process in every mode so log lines stream live; default 0;
- `--tag T` — the output subdirectory; a rerun with the same tag overwrites it;
- `--allow-partial` — a partial activation, or `check`'s partial plan, is recorded instead of exit 3;
- `check --json` prints the full report.

Other ways to make the same call:

- `mosaic-opt <command> …` is the underlying command (`run.sh` adds `--config`, the pin check and the `--mode` /
  `MOSAIC_OPT` agreement check);
- `MOSAIC_OPT=<mode> python your_script.py` (or `mosaic_opt.enable("<mode>")`) engages a mode's per-step optimizations
  in your own program when `mosaic` is first imported. The fast weight load and the frozen features replace the driver's
  own calls, so they apply under `mosaic-opt design` only.

**Outputs.** `<out>/<tag>/`, by default `<out>/<mode>_<shape>_s<seed>/`, holding `results.json`, `pssm_seed<S>.npz` and
`refold_seed<S>.npz`. `warm` writes under `$MOSAIC_OPT_CACHE_ROOT/<shape>/`.

**What a run prints** (on stderr).

- `check` prints `[mosaic-opt] DRY-RUN mode=<m> … levers=<ids> …` (exit 0; exit 3 when the line carries
  `would_refuse=…`). The kit calls its individually switchable optimizations 'levers'.
- A kit mode prints `[mosaic-opt] ACTIVE mode=<m> route=driver row=<row> levers=<ids> …` — the `ACTIVE` line, naming
  the mode and the optimizations engaged — then one `LEVER name=… state=…` line per per-step optimization (`fast` /
  `big`), and finally `[mosaic-opt] DONE mode=<m> rc=0 … out=<dir>`.
- `off` prints `[mosaic-opt] NOT ACTIVE: mode off: stock mosaic (…)` and then `DONE mode=off rc=0 …` — a report of the
  stock run, not an error.
- If a mode cannot engage, the command exits 3 and `[mosaic-opt] NOT ACTIVE: <reason>` names why; it never falls back
  to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | ok |
| 1 | failed |
| 2 | usage error |
| 3 | not active (or partial without `--allow-partial`) |

## Modes

- `off` — stock: the notebook recipe through the kit driver with no optimization applied, in a clean subprocess
  stripped of every kit and JAX-cache variable (`MOSAIC_CACHE_DIR` kept).
- `exact` — stock's arithmetic with the per-process start-up costs removed (compilation cache with pinned autotune
  results, weight load, frozen features); every process on one GPU type, stack and shape reproduces the warm-up's design
  bitwise at the same seed.
- `fast` (default) — `exact`'s start-up optimizations without the warm step, plus the per-step optimizations of
  CHANGES.md, among them the FlashPairformer triangle kernels of the shared core (`common/opt_core`, the runtime all
  kits share) and a bfloat16 pair track; TF32 / bf16 re-association, small differences from stock. Use it for throughput.
- `big` — `fast` plus grouped pairformer rematerialisation with sub-block remat and row-chunked triangle attention at
  every input size; numerics as `fast`. Use it when `fast` runs out of memory.

## Notes

- **`exact` on a shape never warmed.** `design --mode exact` then prints `shape <key> not warm …: P1,P3 step aside by name`,
  runs as stock compiles and featurizes, and exits 0; `bash run.sh warm --mode exact` with the same shape flags fills the
  cache once. Optimizations that step aside by name are not a partial activation.
- **Partial activation.** An optimization the mode engaged but the driver's manifest does not show applied is recorded
  as `partial=<lever names>` (exit 3 unless `--allow-partial`). Stack-vs-pin and GPU-vs-config differences appear as
  `notes=…` on the `ACTIVE` line.
- **Other cards.** `--config a100` (A100 80GB / 40GB) and `--config h200` (H200; the H100's kernels) engage every
  optimization of every mode as on the H100.
  - A cache made on another card does not serve them (the stack key names the GPU product): the first run of a shape
    compiles once, later runs load it.
  - An input that does not fit ends with the out-of-memory error itself; no other card is configured.
- **Nothing upstream is patched.** The kit patches nothing in mosaic, joltz or boltz: `install` adds the optimization
  modules as a new `mosaic.fast` sub-package stock never imports. `upstream_issues/` holds three write-ups for upstream.
- **Compile times.** The first `fast` / `big` run at a new shape (target, binder length, copies) compiles ≈3 min on
  H100 (stock compiles in every process), kept under `$MOSAIC_OPT_CACHE_ROOT/<shape>/` for later processes; each mode
  compiles once per shape on its first run. An image built with STOCK.md's optional compile-cache tar starts with the
  example's shape compiled and warmed for the host classes that tar covers, so there no mode writes new cache files.
- **Where the gain is.** The per-step gain grows with token count; the example's 169 tokens sit at the low end.
- **Switching optimizations off.** `MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]` runs a mode without the named optimizations
  (names as printed on the `LEVER` lines); the `ACTIVE` line then carries `levers_off=<names>`.
