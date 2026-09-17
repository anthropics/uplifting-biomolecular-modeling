# PXDesign — optimization kit

Drop-in modes that make stock PXDesign binder design (commit `f7884413`, `pxdesign infer`, on Protenix `v0.5.0+pxd`)
faster and lighter on GPU memory. You call `pxdesign infer` exactly as before, and no upstream code is edited — the kit
attaches to the loaded model at run time. The kit adds a `--mode`:

- `off` — stock PXDesign, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs.

One GPU per process, as upstream. What each mode changes: `CHANGES.md`. Exact versions, the pinned software stack and
all variables: `STOCK.md`. How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the
top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs.

## Setup

Pick ONE way to get the pinned stack — stock PXDesign and everything it needs (the 'Stack' section of `STOCK.md` lists
every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver that runs CUDA 12.1 (525.60 or newer).

Route C additionally needs:

- a released CPython 3.11 (the block gets one with `uv`), git, gcc and `nvcc` (CUDA 12.1) on `PATH`. The route-C lines
  below are the whole recipe; STOCK.md's Stack section has the same lines annotated — run one or the other.

Type the first block from the directory that holds `pxdesign/` and `common/`:

```bash
# A — Docker (preferred): the whole pinned stack, stock and this kit, in one image
docker build -f pxdesign/environment/Dockerfile -t pxdesign-kit:dev .
docker run --rm -it --gpus all -w /kit -v /weights/pxdesign:/weights/pxdesign -v $PWD/out:/kit/pxdesign/out pxdesign-kit:dev bash   # a shell in /kit for everything below
# B — Apptainer (the whole Setup for B; `kit` below: a shell function = B's bash run.sh): build the .sif once where Docker runs (converts A's image); the cluster then needs only Apptainer
apptainer build pxdesign-kit.sif pxdesign/environment/apptainer.def
mkdir -p out; kit() { apptainer run --nv --bind /weights/pxdesign:/weights/pxdesign --bind "$PWD/out":/kit/pxdesign/out pxdesign-kit.sif "$@"; }   # B's ./run.sh · other cards: replace h100 with a100|h200
kit install --weights /weights/pxdesign   # checkpoint/ + ccd_cache/ fetched or checked; read-only ok
export PXDESIGN_CKPT_DIR=/weights/pxdesign/checkpoint PROTENIX_DATA_ROOT_DIR=/weights/pxdesign/ccd_cache && kit check --config h100 --mode fast   # --config on every verb; Run lines alike: kit design …
# C — instead of A or B, on your own host: a released CPython 3.11, git, gcc and nvcc (CUDA 12.1) on PATH — the whole route-C recipe; STOCK.md §Stack = these lines annotated (run one or the other)
command -v uv >/dev/null || { t=$(mktemp) && curl -LsSf -o "$t" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t" && . "$HOME/.local/bin/env"; } # uv itself, once per user (skipped when present)
uv venv --seed --managed-python --python 3.11 venv && . venv/bin/activate
grep -v '^#' pxdesign/environment/requirements.lock > /tmp/stack.txt && pip install --no-deps --src "$HOME/src" -r /tmp/stack.txt
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …` (the
`kit()` shell function defined above stands in for `bash run.sh`). Under **A** you are now in the container shell, which
opens in `/kit`. Under **C** you are in your activated environment; continue with the whole next block (STOCK.md's Stack
section, if you used it, stops before the install line; already inside `pxdesign/`, the `cd` is a no-op). Run:

```bash
[ -f run.sh ] || cd pxdesign                       # from the dir above; a no-op once inside the kit dir
bash run.sh install --weights /weights/pxdesign       # fetches ≈3.7 GB if absent, else hash-checks; read-only ok
export PXDESIGN_CKPT_DIR=/weights/pxdesign/checkpoint       # pxdesign_v0.1.0.pt + three protenix_*_v0.5.0.pt
export PROTENIX_DATA_ROOT_DIR=/weights/pxdesign/ccd_cache   # the three CCD-cache files (STOCK.md §Pin)
bash run.sh check --config h100 --mode fast           # dry run: GPU, mode, pins; reads no weights
```

What the blocks assume:

- **Weights directory.** `/weights/pxdesign` in the blocks. `install --weights` fetches the files into `checkpoint/` and
  `ccd_cache/` if absent, else hash-checks what is there, so a complete directory may be read-only.
  - `PXDESIGN_CKPT_DIR` names `checkpoint/` (`pxdesign_v0.1.0.pt` plus three `protenix_*_v0.5.0.pt` files);
  - `PROTENIX_DATA_ROOT_DIR` names `ccd_cache/` (the three CCD-cache files; STOCK.md, 'Pin').
- **Routes A and B need no compilation.** The images hold the kit installed (`install` only checks the pins and fetches
  `--weights`) and nothing compiles at run time.
- **What route B writes on the host.** `./out`, plus `~/.triton/autotune`, a small autotune-cache directory that
  upstream's DeepSpeed import creates and probes (`df`) in every mode, `off` included.
- **Fresh output directory per run.** A second run into the same `-o` is upstream's resume (`NOTHING RAN … already dumped`,
  exit 0), so compare modes with a fresh `-o`.
- **Pin check on route C.** Route C installs the three upstream packages from their git URLs at the pinned commits (the
  `stock/*.tar.gz` archives are reference copies). `install` ends with the pin check, which names and refuses (exit 3)
  an upstream package at another commit, modified or absent — the kit modes then refuse too; `off` still runs.
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env` (`h100` is the default; every command takes it).
  It sets `MODEL_OPT_TARGET_GPU`, fills `LAYERNORM_TYPE` and `MODEL_OPT_STACK_KEY` when unset, and refuses when one of
  the two path variables above is unset.
- **First run on route C compiles.** It builds Protenix's LayerNorm CUDA extension with `nvcc` inside the installed
  `protenix` package (about 3 min; every mode, `off` included; `bash run.sh warm` does it ahead of time). The kit modes
  themselves compile nothing.

## Run

```bash
T=opt/forward/hoist/inputs/tasks_3targets.json                 # three shipped tasks · other cards: add --config a100|h200
bash run.sh design --mode off   -i $T -o out/off   --seeds 101    # stock, in a clean subprocess
bash run.sh design --mode exact -i $T -o out/exact --seeds 101
bash run.sh design --mode fast  -i $T -o out/fast  --seeds 101    # the default when no mode is named
bash run.sh design --mode big -i $T -o out/big --seeds 101
bash run.sh design --mode fast  -i $T -o out/n40   --seeds 101,102 --N_sample 40 --N_step 400 --dtype bf16   # upstream's own options, verbatim
```

**Options.** `design` hands `pxdesign infer` its own command line verbatim (`-i/--input`, `-o/--dump_dir`, `--seeds`,
`--N_sample`, … and anything else it accepts; `--load_checkpoint_dir` defaults to `$PXDESIGN_CKPT_DIR`). The kit's own
flags are `--mode` and `--det`. Two other ways to make the same call:

- `pxdesign-opt design --mode <mode> …` — the same call without `run.sh`;
- `PXDESIGN_OPT=<mode> pxdesign infer …` (after `source configs/h100.env`) — engages a mode from the unchanged stock
  command line.

**Outputs.**

- They land where stock writes them: `<dump_dir>/<task>/seed_<seed>/predictions/<task>_sample_<i>.cif`, tasks × seeds ×
  `N_sample` of them — 3 × 1 × 5 = 15 for each of the first four lines.
- Plus `opt_manifest.json` and upstream's `ERR/` (its per-task error directory, empty when every task ran); under `off`
  also `stock_env_proof.json`.
- Each of the first four lines takes about 1.5–2 min on an H100, `off` the longest; on route C the first one also builds
  the LayerNorm extension.

**What a run prints** (on stderr).

- A kit mode prints `[pxdesign-opt] ACTIVE mode=<m> …` once engaged (the `ACTIVE` line) and
  `[pxdesign-opt] DONE designs=<n> … expected=<N>` at the end, exit code 0.
- `off` prints `[pxdesign-opt] STOCK mode=off …` and `[pxdesign-opt stock] DONE … rc=0`.
- `check` prints `[pxdesign-opt] DRY-RUN mode=<m> …` and `PINS pinned=1 …`, exit code 0 (3 when the line carries `would_refuse=…`).
- If a mode cannot engage — no GPU, upstream or the shared core (`common/opt_core`, the runtime all kits share) off its
  pin, part of the mode inapplicable to the loaded model — the command prints `[pxdesign-opt] NOT ACTIVE: <reason>` and
  exits 3; it never falls back to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | done |
| 1 | failed, or fewer designs than expected |
| 2 | usage error |
| 3 | not active |

## Modes

- `off` — stock `pxdesign infer` as released, in a clean subprocess with nothing of the kit importable.
- `exact` — the sampler's step-invariant conditioning (pair path, per-block pair biases, atom-encoder terms,
  padding-mask bias) computed once per `sample_diffusion` call by the stock sub-modules and replayed each step; identical
  to `off` under `--det 1`. Use it when outputs must not move.
- `fast` (default) — `exact` on un-expanded sample rows, plus TF32 matmuls, the single conditioning computed once for
  the identical `N_sample` copies, and two exact memory trims; TF32 / fp32 re-association differences, reproducible under
  `--det 1`. Use it for throughput.
- `big` — `fast` plus the pair-plane passes evaluated slab by slab, so no full pair plane is ever resident; numerics as
  `fast`. Use it when `fast` runs out of memory.

## Notes

- **Where the gain is.** The kit's gain is in design throughput at each mode's own calibrated batch size. Every call also
  pays about half a minute of checkpoint load plus per-task featurization in every mode, so the Run example (three
  ≈200-token tasks, 5 designs each) gains only a little over `off` under `exact` or `fast`, one large task with many
  samples much more — the gain grows with designs per call.
- **A100 / H200.** Add `--config a100` / `--config h200` to every command (`design` included; omitted, the run still
  engages and `notes=` names the H100 target). Every mode engages there exactly as on the H100, on compute capability 8.0
  as on 9.0; any other card or torch build engages too, named on the `ACTIVE` line (`notes=…`).
- **Determinism.** `--det 1` selects the deterministic recipe (seeds, deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG`)
  in every mode, `off` included; `exact --det 1` equals `off --det 1` bit for bit. `--det 0` (the default) is upstream's
  own seeding.
- **Out of memory.** Under `fast` → use `big`. An out-of-memory error is raised as such in every mode; no mode switches to
  another by itself.
- **One stock setting made explicit.** On every mode, `off` included, `LAYERNORM_TYPE` is set from `--use_fast_ln` before
  Protenix reads it, so the LayerNorm that upstream's own flag asks for is the one that runs (STOCK.md, 'Stock exceptions').
