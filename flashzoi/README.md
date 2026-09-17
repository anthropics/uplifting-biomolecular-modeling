# Flashzoi — optimization kit

Drop-in modes that make stock Flashzoi faster — `borzoi-pytorch` 0.5.1: `Borzoi.from_pretrained("johahi/flashzoi-replicate-{0..3}")`
and `predict_tracks`. You call it exactly as before; the kit adds a `--mode` to its `pred` command, or one environment
variable, `FLASHZOI_OPT`, for your own script:

- `off` — stock Flashzoi, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster. **The default.**

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock.

## Setup

Pick ONE way to get the pinned stack — stock `borzoi-pytorch` 0.5.1 and everything it needs: Debian 12, CUDA 12.4, gcc,
Python 3.11, torch 2.5.1, triton 3.1.0, flash-attn 2.7.0.post2 (the 'Stack' section of `STOCK.md` lists every pin):
**A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver 550 or newer on the host;
- a weights directory, written `/weights/flashzoi` in the blocks (the Hugging Face hub cache that `install --weights` fills).

Type the first block from the directory that holds `flashzoi/`:

```bash
# A — Docker (preferred): the whole pinned stack, stock and this kit in one image
docker build -f flashzoi/environment/Dockerfile -t flashzoi-kit:dev .
docker run --rm -it --gpus all -v /weights/flashzoi:/weights/flashzoi -v "$PWD/out":/kit/flashzoi/out flashzoi-kit:dev bash   # the shell opens in /kit/flashzoi
# B — Apptainer (the whole Setup for B): build the .sif once where Docker runs (converts A's image); the cluster then needs only Apptainer
apptainer build flashzoi-kit.sif flashzoi/environment/apptainer.def
mkdir -p out in jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --bind /weights/flashzoi:/weights/flashzoi --bind "$PWD/out":/kit/flashzoi/out --bind "$PWD/in":/kit/flashzoi/in flashzoi-kit.sif "$@"; }   # kit = B's ./run.sh · /weights/flashzoi = the HF hub cache
export FLASHZOI_WEIGHTS=/weights/flashzoi && kit install --weights /weights/flashzoi   # /weights/flashzoi: HF cache fetched or checked; read-only ok
kit check --config h100                                  # each Run line: kit pred … · cards: --config a100|h200
# C — instead of A or B, on your own host: about 3.3 GB of wheels — the whole route-C recipe; STOCK.md §Stack = these lines annotated (run one or the other)
uv venv --seed --managed-python --python 3.11 fz-env && . fz-env/bin/activate                             # uv's own CPython 3.11; no uv yet? see STOCK.md §Stack
grep -v -E '^(#|borzoi-pytorch==)' flashzoi/environment/requirements.lock > /tmp/stack.txt && pip install --no-deps -r /tmp/stack.txt
pip install --no-deps flashzoi/stock/borzoi_pytorch-0.5.1-py3-none-any.whl
export PYTHONHASHSEED=0 CFLAGS=-g0                                                                         # the two variables the image sets
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`.
Under **A** you are now in the container shell, which opens in `/kit/flashzoi`. Under **C** you are in your activated
environment: after the route-C lines above you are still in the directory holding `flashzoi/`, so type the whole block;
if you followed STOCK.md's Stack section instead, you are inside `flashzoi/` with `install` already done — continue at
`export`. Run:

```bash
[ -f run.sh ] || cd flashzoi                     # no-op once inside · C via §Stack: skip install too
bash run.sh install --weights /weights/flashzoi     # kit + pin check; fetches, digest-checks the 4 replicates
export FLASHZOI_WEIGHTS=/weights/flashzoi        # the one required variable
bash run.sh check --config h100 --mode exact        # dry run: one DRY-RUN line, exit 0; no weights read
```

What the blocks assume:

- **Weights directory.** `--weights` fills `models--johahi--flashzoi-replicate-{0,1,2,3}/snapshots/<revision>/` under
  that directory and keeps files already there: only their digests are checked and nothing is written, so a read-only
  cache of that layout works as it is. `FLASHZOI_WEIGHTS` is the one required variable.
- **Outputs.** `out/…` in the Run commands is `./out` on the host in all three routes (A mounts it there; B's `kit` binds
  `./out` and `./in` of the directory it was defined in).
- **Run's first line under B.** The numpy line that writes the example window runs as
  `apptainer exec flashzoi-kit.sif python -c "…"`, reading the image's copy
  `/kit/flashzoi/opt/forward/kits_v1_25/canary_window_0.npz` instead of the relative path.
- **Route C with an existing install.** Route C accepts `borzoi-pytorch` 0.5.1 installed with the pinned wheel's bytes
  (from PyPI, the wheel in `stock/`, or an unpacked copy). The pin check (`stock/check_pins.py`, run by every command)
  compares the install file for file with that wheel and refuses another version, an edited file or a checkout of another
  commit by name (exit 3). Any other difference in the stack is printed as a NOTE and the run proceeds; a kit mode repeats
  it as `drift=[stack …]` on its `ACTIVE` line.
- **GPU cards.** `--config h100|h200|a100` loads `configs/<card>.env` (`HF_HUB_CACHE`, `HF_HOME`, `HF_HUB_OFFLINE`,
  `MODEL_OPT_STACK_KEY`, `TRITON_CACHE_DIR`); every command below takes it.
- **Compile cache.** The first kit-mode run on a machine compiles the kit's Triton kernels; `bash run.sh warm --config h100`
  does it ahead of time. Later runs load them from `TRITON_CACHE_DIR`, which `MODEL_OPT_JIT_ROOT=<dir>` places under a
  directory you keep (route B's block sets it to `./jit`). Each new batch size's kernel launches are recorded once per
  process as a CUDA graph, then replayed.

## Run

```bash
mkdir -p in && python -c "import numpy as np; np.save('in/w0.npy', np.load('opt/forward/kits_v1_25/canary_window_0.npz')['x'])"   # the shipped window: one (4, 524288) one-hot, rows A,C,G,T
bash run.sh pred --config h100 --mode off   --input in --out out/off       # stock, in a clean subprocess
bash run.sh pred --config h100 --mode exact --input in --out out/exact     # the default when no mode is named
. configs/h100.env && FLASHZOI_OPT=exact python your_script.py          # own borzoi_pytorch code; or flashzoi_opt.enable("exact")
```

**What `pred` does.** It runs the stock call — the four replicates through `predict_tracks` under `torch.autocast("cuda")`,
one window per call — on every `<item>.npy` in `--input` (uint8 or float32). `flashzoi-opt pred …` is the same call
without `run.sh`.

**Options** (they work in both modes):

- `--items a,b` — windows by file stem;
- `--tracks 0-99,4100` — upstream's `slices`; `@file` = one entry per line;
- `--det` — the deterministic recipe (STOCK.md);
- `--jobs FILE` — `input<TAB>out` per line: a kit mode serves every job from one process, loading the weights once; `off`
  runs one stock subprocess per job and exits with the worst job's code;
- `--allow-partial` — kit modes: accept a run in which not every optimization engaged (see exit code 3).

**Outputs.** `<out>/<item>.npy` — float32 `(1, 4, 6144, 7611)` = window, replicate, bin, human track; 748 MB with every
track — plus `rows.jsonl` and `opt_manifest.json`.

**What each command prints** (stderr):

- `check` → `[flashzoi-opt] DRY-RUN mode=exact variant=- gpu=… cc=… components=… knobs=… stack_key=… would_refuse=none`;
  exit 3, not 0, when `would_refuse` names a reason. `stack_key` here is `<compute capability>|<Triton version>`, not the
  cache directory's `MODEL_OPT_STACK_KEY`.
- A kit mode → `[flashzoi-opt] ACTIVE mode=exact variant=- gpu=<name> cc=<capability> components=<levers> (partial: none)`
  once engaged (`components=` lists the optimizations; the kit calls them 'levers'). `drift=[…]` is appended when the GPU
  or stack is outside the pins; every optimization still engages.
- At the end a kit mode prints `[flashzoi-opt] EXIT mode=exact items=<n> ok=<n> failed=0 wall=<s>s partial=none`. A script
  under `FLASHZOI_OPT` prints the same `ACTIVE` / `EXIT` pair; its `EXIT` tally counts `pred` items only.
- `off` → stock's own lines: `[flashzoi-stock] stock environment proof PASS file=…`, `… ready replicates=4 …`, one
  `… pred <item> …` per window, `… EXIT items=<n> ok=<n> failed=0 wall=<s>s`.
- If a mode cannot engage, the command prints `[flashzoi-opt] NOT ACTIVE: <reason>` and exits 3 (2 when the reason is a
  missing setting, e.g. `FLASHZOI_WEIGHTS` unset). It never falls back to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | done |
| 1 | a window or the install failed |
| 2 | not active because a setting is missing (e.g. `FLASHZOI_WEIGHTS` unset) |
| 3 | not active (`NOT ACTIVE: <reason>`); also a `PARTIAL` run — not every optimization engaged, outputs kept (`--allow-partial` accepts it); also `check` when `would_refuse` names a reason |

**First run.** The first `exact` run on a machine also compiles the Triton kernels — compare modes from the second run on.

## Modes

- `off` — stock `predict_tracks` over the four replicates as released, in a clean subprocess with nothing of the kit
  importable. `pred` only.
- `exact` (default) — all thirteen optimizations (`CHANGES.md`): fused Triton stem, norm and activation sites, decoder
  and head; fp16 pre-cast; CUDA-graph replay; pinned outputs. Attention stays upstream's FlashAttention-2. Outputs
  identical to `off`.

## Notes

- **Where the gain is.** The kit's gain is in forward time per call over many windows. Run's single-window `pred` spends
  most of its wall time on interpreter start and loading the four replicates in every mode, so it takes about as long as
  `off` end to end; the gain grows with windows per process.
- **H200** (`--config h200`): the same optimizations, kernels and pins as H100 (both are pinned by name and compute
  capability 9.0).
- **A100** (`--config a100`): served by the class record `class_records/a100.json` (`device class: by record`); every
  optimization engages.
- **Any other CUDA GPU** engages every optimization as `unpinned` (`drift=[gpu …]` on the `ACTIVE` line).
- **Route B with a directly converted image.** For an image converted straight from the Docker image (not built from
  `environment/apptainer.def`), on a host that binds only `libcuda.so.1`: see STOCK.md's Stack section, `TRITON_LIBCUDA_PATH`.
- **Your own scripts.** A kit-attached model serves every documented upstream call (`model(x)` in all its forms,
  `get_embs_after_crop`, `predict`, `predict_gene_count`, `set_track_subset`, `predict_tracks`). `.to()` / `.half()`
  detach it until the next CUDA forward; a non-CUDA or non-float32 model is served by upstream (`ASIDE` line).
