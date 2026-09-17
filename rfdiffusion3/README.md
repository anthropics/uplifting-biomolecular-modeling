# RFdiffusion3 — optimization kit

Drop-in modes that make stock RFdiffusion3 (`rfd3 design`, RosettaCommons foundry at the commit pinned in `STOCK.md`)
faster. You call RFdiffusion3 exactly as before; the kit adds a `--mode`:

- `off` — stock RFdiffusion3, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.**

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation.

## Setup

This kit is a file overlay on the installed `rfd3` package, so the pinned stack is stock RFdiffusion3 at the pin in
**two** Python environments:

- a pristine one, which runs `--mode off`;
- one overlaid with the kit's ten patched `rfd3/model` files, which runs `exact` and `fast`.

Pick ONE way to get it: **A — Docker**, **B — Apptainer**, or **C — two Python venvs on your own host** (the 'Stack'
section of `STOCK.md` is the complete route-C recipe and lists every pin).

Every route needs:

- an NVIDIA driver for CUDA 13.0 on the host.

Route C additionally needs:

- a released CPython 3.12.1, git, curl, a C compiler, and nvcc 13.0 for the one apex source build (ninja comes from
  pip); the whole recipe takes about half an hour;
- its two exports afterwards: `RFDIFFUSION3_STOCK_PYTHON` and `RFD3_CKPT`.

Type the first block from the directory that holds `rfdiffusion3/` and `common/`:

```bash
# A — Docker (preferred): the pinned stack, both environments (/opt/stock, /opt/kit) and this kit (/kit/rfdiffusion3) in one image
docker build -f rfdiffusion3/environment/Dockerfile -t rfdiffusion3-kit:dev .
docker run --rm -it --gpus all -w /kit -v /path/to/weights:/weights -v $PWD/out:/kit/rfdiffusion3/out rfdiffusion3-kit:dev bash   # a shell in /kit
# B — Apptainer / Singularity (the whole Setup for B): converts A's image (needs it in the local Docker daemon at build time only)
apptainer build rfdiffusion3-kit.sif rfdiffusion3/environment/apptainer.def
mkdir -p out jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --bind /path/to/weights:/weights --bind "$PWD/out":/kit/rfdiffusion3/out rfdiffusion3-kit.sif "$@"; }   # kit = B's ./run.sh · other cards: --config a100|h200
export RFD3_CKPT=/weights/rfd3/rfd3_latest.ckpt && kit install --weights /weights/rfd3   # rfd3/rfd3_latest.ckpt: fetched or checked; read-only ok
kit check --config h100 --mode fast                       # Run lines alike: kit <verb> … with their $PWD/out paths
apptainer exec rfdiffusion3-kit.sif tar -xzf /kit/rfdiffusion3/stock/foundry-4010e3e.tar.gz -C out foundry-4010e3e2/models/rfd3/docs   # Run's example specs, out of the image
# C — instead of A or B, on your own host (two environments on CPython 3.12.1): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`, with
their `$PWD/out` paths. Under **A** you are now in the container shell, which opens in `/kit`; there you need only the
`--weights` fetch, the `RFD3_CKPT` export and `check`. Under **C** you are in the kit environment — the one STOCK.md's
Stack section leaves active after its second pass (the pristine environment needs nothing more). Run:

```bash
[ -f run.sh ] || cd rfdiffusion3                          # from the parent dir; a no-op once inside the kit dir
bash run.sh install --weights /weights/rfd3                  # fetches 2.7 GB if absent, else digest-checks; read-only ok
python -m rfdiffusion3_opt.install apply                  # C only, kit environment; a no-op once §Stack applied it
export RFD3_CKPT=/weights/rfd3/rfd3_latest.ckpt           # the checkpoint file itself; or --ckpt <file> per run
export RFDIFFUSION3_STOCK_PYTHON="${RFDIFFUSION3_STOCK_PYTHON:-$HOME/rfd3-stock/bin/python}"   # pristine python: image preset wins; C: §Stack's path
bash run.sh check --config h100 --mode fast                  # dry run: tree, overlay, pins, GPU; exit 0 = ready, 3 = not
```

What the blocks assume:

- **Weights file.** `RFD3_CKPT` names the checkpoint file itself, `/weights/rfd3/rfd3_latest.ckpt` inside the
  container under A and B (the host directory you mount at `/weights` holds `rfd3/`); or pass `--ckpt <file>` per run.
  `install --weights /weights/rfd3` fetches it when absent and otherwise digest-checks it, so a read-only copy works.
- **Pristine interpreter.** `RFDIFFUSION3_STOCK_PYTHON` names the pristine environment's `python`, which `--mode off`
  uses. The image presets it (the export line keeps a preset value); under C it is the path from STOCK.md's Stack section.
- **Overlay.** `python -m rfdiffusion3_opt.install apply` is for route C only, in the kit environment, and changes
  nothing once the Stack section has applied it. `install` and `check` end with the pin report: `rfd3 tree not pristine`
  in the kit environment (the overlay is in: expected), `pristine` in the stock one.
- **Outputs under A and B.** The Run examples write to `$PWD/out/…`; under A and B `$PWD/out` on the host is mounted at
  `/kit/rfdiffusion3/out`, so results land in `./out` on the host. Route B's last Setup line extracts the Run examples'
  spec files out of the image into `out/`.
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env` (`MODEL_OPT_TARGET_GPU`, `PYTORCH_CUDA_ALLOC_CONF`,
  and compile-cache directories under `MODEL_OPT_JIT_ROOT` when it is set); every command below takes it.
- **Compile cache.** The first run of a kit mode compiles kernels and captures the CUDA graph once per process.
  `bash run.sh warm --config h100 --mode fast` loads the checkpoint under the mode ahead of time, and a writable
  `MODEL_OPT_JIT_ROOT` keeps the caches (route B's block puts it in `./jit` on the host).

## Run

```bash
mkdir -p out && tar -xzf stock/foundry-4010e3e.tar.gz -C out foundry-4010e3e2/models/rfd3/docs      # upstream's example specs and inputs (B: done in Setup)
S=$PWD/out/foundry-4010e3e2/models/rfd3/docs/examples/protein_binder_design.json
bash run.sh design --config h100 --mode off   inputs=$S out_dir=$PWD/out/off   n_batches=1 seed=101    # stock, in the pristine environment
bash run.sh design --config h100 --mode exact inputs=$S out_dir=$PWD/out/exact n_batches=1 seed=101
bash run.sh design --config h100 --mode fast  inputs=$S out_dir=$PWD/out/fast  n_batches=1 seed=101    # the default when no mode is named
bash run.sh design --config h100 --mode exact --det 1 inputs=$S out_dir=$PWD/out/det n_batches=1 seed=101   # deterministic recipe
```

**Options.** `design` passes `rfd3 design` its own `key=value` settings unchanged (`inputs=`, `out_dir=`, `seed=`,
`diffusion_batch_size=`, `n_batches=`, `inference_sampler.*=` …). The kit's own flags are `--mode`, `--ckpt`, `--det` and
`--allow-partial`. In the kit environment, `RFDIFFUSION3_OPT=<mode> rfd3 design …` (after `source configs/h100.env`)
engages a mode from the stock command line itself.

**Outputs.**

- Each line above writes 16 designs where stock writes them (`<out_dir>/<spec>_<key>_<batch>_model_<k>.cif.gz` +
  `.json`), plus `opt_manifest.json`.
- Each takes about 1–2 min a mode on an H100 — 3 min for `fast` on a cold cache: it compiles in every process and at
  every new length (see Notes).
- The hydra search-path and atomworks mirror-path warnings are upstream's own.
- Give each run a fresh `out_dir`. A re-run that finds every design already in `out_dir` does no work — upstream skips
  them, no step runs — and the kit mode exits 5 (`KERNELS refused … engine-never-initialized`).

**What a run prints** (on stderr).

- `check` prints `[rfdiffusion3-opt] DRY-RUN mode=<mode> … gpu=…`; exit 0 means ready, 3 means not.
- A kit mode prints `[rfdiffusion3-opt] ACTIVE mode=<mode> … interpreter=patched(10/10) …`, one `APPLIED …` line per
  optimization, `KERNELS route=<mode> …` and, at exit, `[rfdiffusion3-opt] EXIT pid=… <counters>`.
- `--mode off` prints `[rfdiffusion3-opt] design mode=off …` and `[rfdiffusion3-opt stock] STOCK rc-foundry=… tree=pristine …`
  before upstream's own output.
- If a mode cannot engage, the command prints `[rfdiffusion3-opt] NOT ACTIVE: <reason>` and exits 3 — or 5 when a
  kernel it needs cannot run here, or when the batch × length leaves the dense atom-attention path under `exact` (lower
  `diffusion_batch_size` or use `--mode off`). It never falls back to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | the run failed |
| 2 | usage error |
| 3 | not active, not stock, or partial |
| 4 | overlay state (`install apply`) |
| 5 | kernels refused |

## Modes

- `off` — stock `rfd3 design` as released, in the pristine environment under `python -I`, nothing of the overlay
  installed. Also the way to run what `exact` / `fast` decline: classifier-free guidance, `low_memory_mode=true`, the
  symmetry sampler, `compile_model=true`.
- `exact` — step-invariant tensors cached per roll-out, a fused transition kernel with stock's rounding points, the
  initialiser's pair embedding in row blocks, the denoiser step as a CUDA graph. Designs identical to `off` under
  `--det 1`. Use it when outputs must not move.
- `fast` (default) — `exact` plus `torch.compile` of the denoiser, token attention on SDPA and atom attention through a
  fused gather kernel; inductor / SDPA re-association, within stock's seed-to-seed variation. Use it for throughput.

## Notes

- **A100 / H200** (`--config a100|h200`). Every optimization of every mode engages on both — the H200 (compute
  capability 9.0) exactly as the H100; on the A100 (8.0) the fused transition kernel covers its own range of row counts
  and runs stock's code outside it (reported as `rows_cc=8.0 rows_served=…`). No other card is configured.
- **Determinism.** `--det 1` selects the deterministic recipe (cuBLAS workspace, torch's deterministic algorithms, a
  fixed-order scatter-mean) on every mode, `off` included; `exact --det 1` equals `off --det 1` bit for bit. The
  default `--det 0` gives production numerics, not bitwise run to run even in stock.
- **Stack drift and partial runs.** A GPU, torch or Triton other than the pinned stack is reported on the stderr lines
  and the mode still engages in full. A run whose CUDA-graph step did not serve every roll-out is PARTIAL (exit 3);
  `--allow-partial` (`RFDIFFUSION3_OPT_ALLOW_PARTIAL=1`) records it instead.
- **`RFD3_*` variables.** Leave `RFD3_*` (the kit's switches and upstream's two attention switches) unset under `exact` /
  `fast`: a preset value is rejected, named, with exit 3.
- **Where the gain is.** The kit's gain is in sampling time per roll-out. Each process also pays ≈25 s of start-up,
  and `fast` spends roughly a minute per process in `torch.compile` tracing before sampling starts, even with the
  compile cache filled (the compiled kernels themselves are cached, so nothing is rebuilt; a length it has not met
  compiles its kernels once, cached after): a single roll-out or a short job gains more under `exact` than under
  `fast` end to end, and `fast` overtakes `exact` only with several roll-outs per process.
