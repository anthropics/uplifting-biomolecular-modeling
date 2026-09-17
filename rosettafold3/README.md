# RoseTTAFold3 — optimization kit

Drop-in modes that make stock RoseTTAFold3 (`rf3 fold`, RosettaCommons foundry at commit `4010e3e2e`, `rc-foundry`
0.2.1.dev13) faster and lighter on GPU memory. You call `rf3 fold` exactly as before; the kit adds a `--mode`:

- `off` — stock RoseTTAFold3, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — no arithmetic changes (bitwise `off` whenever stock itself is reproducible), faster.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs; `--n_gpu P` splits one `big` prediction across P GPUs of one host.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs (where stock itself is reproducible), faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs, up to 4,000 tokens on one GPU · `--n_gpu P` splits `big` across P GPUs of one host.

## Setup

Pick ONE way to get the pinned stack — stock RoseTTAFold3 at foundry commit `4010e3e2e` on Python 3.12, torch
2.13.0+cu130, cuEquivariance 0.11.1, and everything it needs (the 'Stack' section of `STOCK.md` lists every pin):
**A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

This kit keeps **two Python interpreters**:

- the stock interpreter stays pristine and runs `--mode off`;
- `opt/venv`, derived from it, carries the kit's five `rf3` files (four of upstream's, modified, plus the new
  `rf3/graph_flags.py`) and runs the kit modes.

Every route needs:

- a host NVIDIA driver that runs CUDA 13.0.

Route C additionally needs:

- `git`, a C compiler and `uv` (or any released CPython 3.12; STOCK.md's Stack section).

Type the first block from the directory that holds `rosettafold3/` and `common/`, with your weights in `<weights dir>`:

```bash
# A — Docker (preferred): the whole pinned stack, stock and this kit in one image
docker build -f rosettafold3/environment/Dockerfile -t rosettafold3-kit:dev .
docker run --rm -it --gpus all -w /kit -v <weights dir>:/weights -v $PWD/out:/kit/rosettafold3/out rosettafold3-kit:dev bash   # a shell in /kit; continue below
# B — Apptainer (the whole Setup for B): build the .sif once where Docker runs (converts A's image); the cluster then needs only Apptainer
apptainer build rosettafold3.sif rosettafold3/environment/apptainer.def
mkdir -p out jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --bind <weights dir>:/weights --bind "$PWD/out":/kit/rosettafold3/out rosettafold3.sif "$@"; }   # kit = B's ./run.sh · other cards: --config a100|h200
apptainer exec --bind <weights dir>:/weights rosettafold3.sif /kit/rosettafold3/opt/venv/bin/python -m rosettafold3_opt.weights /weights/rosettafold3   # /weights/rosettafold3: ckpt fetched or checked; read-only ok
export ROSETTAFOLD3_OPT_CKPT=/weights/rosettafold3/rf3_foundry_01_24_latest_remapped.ckpt && kit check --config h100 --mode fast   # each Run line alike: kit pred …
# C — instead of A or B, on your own host: fresh venv, pinned stack in two pip passes, the two variables the image sets — the whole route-C recipe; STOCK.md §Stack = these lines annotated plus the uv bootstrap line (run one or the other)
uv venv --seed --managed-python --python 3.12.1 venv && . venv/bin/activate
grep -v '^#' rosettafold3/environment/requirements.lock > /tmp/s.txt && python -m pip install --no-deps -r /tmp/s.txt
python -m pip install --no-deps --force-reinstall nvidia-cudnn-cu13==9.20.0.48 nvidia-nccl-cu13==2.29.7
export PYTHONHASHSEED=0 CFLAGS=-g0
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`.
Under **A** you are now in the container shell, which opens in `/kit`. Under **C** you are in your activated
environment: after the route-C lines above, type the whole next block; if you used STOCK.md's Stack section instead, it
already ran `cd rosettafold3` and the install line, so continue at the `export` lines. Run:

```bash
[ -f run.sh ] || cd rosettafold3                                 # no-op once inside · C via §Stack: skip install too
bash run.sh install --make-venv --weights /weights/rosettafold3     # --weights: ≈3 GB if absent, else hash-check; read-only ok
export ROSETTAFOLD3_OPT_STOCK_PYTHON="$(command -v python)"      # the pristine interpreter (preset in the image)
export ROSETTAFOLD3_OPT_CKPT=/weights/rosettafold3/rf3_foundry_01_24_latest_remapped.ckpt   # required by pred
bash run.sh check --config h100 --mode fast                         # dry run, no fold: interpreters, pins, GPU, checkpoint
```

What the blocks assume:

- **Weights directory.** `<weights dir>` on the host is mounted at `/weights` under A and B; the checkpoint lives in
  `/weights/rosettafold3/`. `install --weights` fetches it if absent, else hash-checks it, so a complete directory may be
  read-only.
- **The two interpreters.** `ROSETTAFOLD3_OPT_STOCK_PYTHON` names the pristine interpreter (preset in the image);
  `install --make-venv` creates `opt/venv` from it (details: STOCK.md's Stack section).
- **Checkpoint.** `ROSETTAFOLD3_OPT_CKPT` names the checkpoint file; `pred` requires it.
- **Pin check.** `install` and every later command check both interpreters against the pin (`stock/check_pins.py`):
  `rc-foundry` must come from commit `4010e3e2e` (the lock's git line, an editable checkout at that commit, or the
  vendored `stock/foundry-4010e3e2e.tar.gz`), and the five `rf3` files must be stock's bytes before the kit writes its
  own. Anything else is refused with the reason (exit 3).
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env` (interpreters, checkpoint, compile-cache root);
  every command takes it.
- **First run compiles.** The first run of a kit mode compiles its Triton kernels once per machine and input-size class.
  `bash run.sh warm --config h100 [--mode M]` does that ahead of time.
- **Compile cache.** A writable `MODEL_OPT_JIT_ROOT` keeps the compiled kernels; route B's block puts it in `./jit` on
  the host. The first run in a fresh cache directory also performs a one-time numerical self-check of the fused
  kernels on your GPU (seconds on H100; up to ~30 s on A100) before serving them; later runs skip it.

## Run

```bash
J=opt/forward/rf3_xattempt_addon/public_inputs/1brs_tiles.json            # four tiles of PDB entry 1BRS, ships with the kit
bash run.sh pred --config h100 --mode off   --input $J --out_dir out/off      # stock, in a clean subprocess
bash run.sh pred --config h100 --mode exact --input $J --out_dir out/exact
bash run.sh pred --config h100 --mode fast  --input $J --out_dir out/fast     # the default when no mode is named
bash run.sh pred --config h100 --mode big --input $J --out_dir out/big
bash run.sh pred --config h100 --mode big --n_gpu 2 --input $J --out_dir out/big_x2   # 2, 4 or 8 GPUs of one host
```

**Options.** `pred` passes `--input` to `rf3 fold` as `inputs=` verbatim, together with any trailing `key=value`
overrides. The kit's own flags are:

- `--mode`, `--n_gpu`;
- `--seeds S[,S…]` — one fold per seed, into `<out_dir>/seed-<S>/`;
- `--ckpt <file>`;
- `--allow-partial` (see Notes);
- `--log <file>`.

Two other ways to make the same call:

- `rosettafold3-opt pred …` — the same call without `run.sh`;
- `source configs/h100.env && ROSETTAFOLD3_OPT=<mode> opt/venv/bin/python -m rf3.cli fold …` — engages a mode from
  stock's own command line.

**Outputs.** Stock's, under `<out_dir>/<item>/`. The fold's log is `<out_dir>/pred.log` (`--log` changes it).

**What a run prints.**

- In `pred.log`, a kit mode prints `[rosettafold3-opt] ACTIVE mode=<mode> … levers=…` first — the `ACTIVE` line, naming
  the mode and the optimizations engaged (the kit calls its individually switchable optimizations 'levers') — and, last,
  `EXIT mode=<mode> … ok=True` plus one `LEVER name=… state=…` line per optimization.
- The terminal shows `TREE` / `WEIGHTS` lines and `pred PASS row=<mode> … wall_s=…`; under `off` also
  `[rosettafold3-opt stock] ENV-CLEAN ok: …`.
- `check` prints `DRY-RUN mode=<mode> …` plus one `interpreter <stock|opt>: … pins=…:ok` line per interpreter.
- If a mode cannot engage, the terminal shows `[rosettafold3-opt] NOT ACTIVE: <reason>` and the command exits 3; it
  never falls back to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | ok |
| 1 | failed |
| 2 | usage error |
| 3 | not active |
| other | `rf3 fold`'s own exit code |

## Modes

- `off` — stock `rf3 fold` as released, on the pristine interpreter, in a clean subprocess with nothing of the kit importable.
- `exact` — CUDA-graph replay of the sampler and the pairformer stack, hoisted step-invariant conditioning and confidence
  prologue, exact fused transitions; triangle kernels bit-identical to cuEquivariance's. No arithmetic changes: two `off`
  runs already differ slightly (stock's `scatter_mean` uses CUDA atomics); with a deterministic `scatter_mean` in both
  interpreters `exact` matches `off` bit for bit. Use it when outputs must not move.
- `fast` (default) — `exact` plus the FlashPairformer trunk kernels of the shared core (`common/opt_core`, the runtime
  all kits share) and a diffusion-transformer megakernel; bf16 re-association, within stock's seed-to-seed variation.
  A call below a kernel's size gate runs stock's kernel (`levers_size_gated=…`). Use it for throughput.
- `big` — `fast`'s kernels without CUDA graphs, plus the memory optimizations (chunked, offloaded and parked tensors);
  numerics as `fast`. `--n_gpu 2|4|8` row-shards the pair representation over P GPUs of one host (refused by name under
  another mode). Use it when `fast` runs out of memory.

## Notes

- **Where the gain is.** The kit's gain is in GPU prediction time per input. Every call also pays ≈25 s of start-up and
  checkpoint load plus each input's CPU featurization in any mode, so the four-tile example gains less end to end than
  its prediction time does; the gain grows with inputs and seeds per call.
- **Mixed input sizes.** Very different sizes in one call re-capture the CUDA graphs at each size change
  (`tg_fallbacks` on the `EXIT` line).
- **A100 80GB** (`--config a100`; STOCK.md's Stack section): every mode runs on the kernels' compute-capability-8.0
  variants; an operation whose fused kernel has no 8.0 variant runs stock's code, and its `LEVER` line says so.
- **H200** (`--config h200`): the H100 settings and kernels; under `exact` a fused kernel serves only where it is
  bitwise-equal to stock on this card's stack, stock's code elsewhere.
- **Switching optimizations off.** `MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]` runs a mode without the named optimizations
  (names as on the `LEVER` lines; `withheld=<names>` on the `ACTIVE` line).
- **`--no-compile`** (`pred` / `check` / `warm`, any mode) is accepted and reported as `compile=none`: there is no
  `torch.compile` optimization in this kit.
- **Out of memory.** Under `fast` → use `big`; still out of memory → `big --n_gpu P`.
- **`--allow-partial`** (`pred` / `warm`, `big` only) lets a run whose memory optimization ran the stock path on part of
  its units proceed (`PARTIAL allowed`) instead of exiting 3.
