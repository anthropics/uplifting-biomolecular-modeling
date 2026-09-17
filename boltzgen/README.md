# BoltzGen — optimization kit

Drop-in modes that make stock BoltzGen 0.3.2 (`boltzgen run`) faster and lighter on GPU memory. You call BoltzGen
exactly as before; the kit adds a `--mode`:

- `off` — stock BoltzGen, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs.

## Setup

Pick ONE way to get the pinned stack — stock BoltzGen 0.3.2 and everything it needs (the 'Stack' section of
`STOCK.md` lists every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver for CUDA 13 on the host (version 580 or newer).

Route C additionally needs:

- `uv` on `PATH` (the route-C comment in the block shows its one-line installer);
- `gcc` on `PATH` — Triton builds its kernel launchers with it at first use.

Type the first block from the directory that holds `boltzgen/` and `common/`:

```bash
# A — Docker (preferred): the whole pinned stack, stock and this kit in one image
docker build -f boltzgen/environment/Dockerfile -t boltzgen-kit:dev .
docker run --rm -it --gpus all -v /weights:/weights -v "$PWD/runs":/kit/boltzgen/runs -w /kit boltzgen-kit:dev bash   # step 2 and Run go in this shell
# B — Apptainer, the whole Setup for B (`kit` below: a shell function = B's bash run.sh); from A's image, no daemon at run time; <weights dir>: host HF cache
apptainer build boltzgen-kit.sif boltzgen/environment/apptainer.def
mkdir -p runs; kit() { apptainer run --nv --bind <weights dir>:/weights/boltzgen --bind "$PWD/runs":/kit/boltzgen/runs boltzgen-kit.sif "$@"; }   # B's ./run.sh
export MODEL_OPT_JIT_ROOT="$PWD/jit" BOLTZGEN_CACHE=/weights/boltzgen && kit install --weights /weights/boltzgen   # /weights/boltzgen: HF cache, fetched/checked; read-only ok
kit check --config h100 --mode fast               # Run: kit design … alike; other cards: --config a100|h200
# C — instead of A or B, on your own host: uv via its installer, sha256-checked first — the typed line of STOCK.md §Stack, which is this same recipe from inside boltzgen/, annotated (run one)
uv venv --seed --managed-python --python 3.11.5 ~/venvs/boltzgen-kit && . ~/venvs/boltzgen-kit/bin/activate
grep -E '^(pip|setuptools|wheel)==' boltzgen/environment/requirements.lock | xargs python -m pip install --no-deps
grep -v -E '^(#|boltzgen==)' boltzgen/environment/requirements.lock > /tmp/stack.txt && python -m pip install --no-deps -r /tmp/stack.txt
python -m pip install --no-deps boltzgen/stock/boltzgen-0.3.2-py3-none-any.whl
export PYTHONHASHSEED=0 CFLAGS=-g0                                        # the image's two settings
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`
(`kit design …`). Under **A** you are now in the container shell, which opens in `/kit`. Under **C** you are in your
activated environment; if you followed STOCK.md's Stack section instead of the route-C lines above, you are already
inside `boltzgen/` and its install line has run, so continue at `export`. Run:

```bash
[ -f run.sh ] || cd boltzgen                      # no-op once inside · C via §Stack: skip install too
bash run.sh install --weights /weights/boltzgen      # fetches ≈8.4 GB if absent, else digest-check; read-only ok
export BOLTZGEN_CACHE=/weights/boltzgen           # both boltzgen HF snapshots inside (layout: STOCK.md §Pin)
bash run.sh check --config h100 --mode fast          # one DRY-RUN line: pins, lever dirs and GPU resolved
```

What the blocks assume:

- **Weights directory.** `/weights/boltzgen` is the path inside the container under A and B (route B binds your host
  Hugging Face cache directory, `<weights dir>`, there); under C use any directory. `install --weights` fetches the
  weights into it when absent and otherwise only digest-checks what is there, so a read-only copy works once filled. It
  holds both BoltzGen Hugging Face snapshots (layout: STOCK.md, 'Pin' section); `BOLTZGEN_CACHE` names it.
- **Outputs under A and B.** The Run examples write under `runs/`: the `docker run` line mounts `$PWD/runs` at
  `/kit/boltzgen/runs` and route B's `kit()` function binds it at the same path, so results land in `./runs` on the host.
- **Route C with an existing environment.** An environment that already has BoltzGen 0.3.2 also works. `install`'s pin
  check (`stock/check_pins.py`) refuses another version, or files that differ from the released wheel's RECORD (a patched
  or editable checkout); it reports the torch / CUDA stack without gating on it.
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env`, which sets `MODEL_OPT_TARGET_GPU`, `HF_HOME`,
  `HF_HUB_OFFLINE` and the compile-cache variables; every command takes it. If you call `python -m boltzgen_opt` or the
  stock CLI directly, `source` that file in the shell first.
- **Compile cache.** The first run of a kit mode compiles its kernels once per machine (seconds); a writable
  `MODEL_OPT_JIT_ROOT` keeps them between runs (route B's block sets it to `$PWD/jit` on the host).
  `bash run.sh warm --config h100 --mode big` compiles the kernels every mode shares ahead of time; `fast`'s own fused
  kernels still compile at its first run (seconds). `warm` designs a single structure, which the batch rule under Run
  keeps out of `exact` and `fast` — hence `--mode big`.

## Run

```bash
X="$PWD/opt/forward/fast_inference/tests/specs/pdl1_ref.yaml"; O="$PWD/runs"      # PD-L1 binder spec shipped with the kit; keep paths absolute
[ -e "$X" ] || { X=/kit/boltzgen/opt/forward/fast_inference/tests/specs/pdl1_ref.yaml; O=/kit/boltzgen/runs; }   # route B: the image's spec, the bound runs dir
bash run.sh design --config h100 --mode off   "$X" --output "$O/off"   --num_designs 16 --seed 0 --diffusion_batch_size 16   # stock; the outputs `exact` repeats
bash run.sh design --config h100 --mode exact "$X" --output "$O/exact" --num_designs 16 --seed 0 --diffusion_batch_size 16
bash run.sh design --config h100 --mode fast  "$X" --output "$O/fast"  --num_designs 16 --seed 0 --diffusion_batch_size 16   # the default when no mode is named
bash run.sh design --config h100 --mode big "$X" --output "$O/big" --num_designs 16 --seed 0
# upstream's per-step overrides go after `--`:
bash run.sh design --config h100 --mode off   "$X" --output "$O/comp"  --num_designs 16 --seed 0 -- --config design compile_pairformer=true
```

**Options.** `design` hands `boltzgen configure` your `boltzgen run` command line verbatim — the spec, `--output` and
every other option; upstream's per-step overrides go after a bare `--`, as in the last line. The kit's own flags are
`--mode` and `--seed`.

The kit modes have three requirements; a run that does not meet them is refused, not run as stock:

- `--seed N` is required (pipeline step *i* runs with seed N + *i*; `off` accepts `--seed` too).
- `exact` and `fast` need a diffusion batch of at least two designs: `--diffusion_batch_size 2` or more, or 100+
  designs, where upstream batches by itself.
- `--devices N` greater than 1, and an in-process data loader (`--num_workers 0`, `debug=true`), run under `--mode off` only.

Without `run.sh`: `python -m boltzgen_opt design …` (or `boltzgen-opt`) is the same call, and
`BOLTZGEN_OPT=<mode> boltzgen run …` engages a mode from the stock command line.

**Outputs.**

- Stock's run directory under `--output`, plus `opt_manifest.json`, `opt_configure.log` and `opt_run.log`.
- A `design` into an existing `--output` runs every step again (upstream sets the old `config/` aside).
- The 16-design example takes about six minutes per kit mode; inverse folding, refolding and analysis follow the design
  step's progress bar.

**What a run prints.**

- `check` prints `[boltzgen-opt] DRY-RUN mode=<m> … gpu=… levers=none` and exits 0. Nothing is applied in a dry run
  (the kit calls its individually switchable optimizations 'levers'). A mode that would be refused ends the line with
  `would-refuse: <reason>` and exits 3.
- `design` ends with `[boltzgen-opt] DESIGNS step=… requested=… produced=… oom_skipped=…` and a verdict line.
- For a kit mode the verdict line is `[boltzgen-opt] ACTIVE mode=<m> form=process … levers=… fallbacks=none`. An earlier
  `form=inproc … after=sitecustomize` line is the model process reporting in once the kit's start-up hook has armed the
  optimizations; it lists `inproc`, the launcher's own optimization, under `unavailable=`.
- For `off` the verdict line is `[boltzgen-opt] NOT ACTIVE: mode off: stock — upstream alone in a pristine process`. It
  has a refusal's prefix but exits 0: each step ran upstream's released code, started by a kit launcher that loads no
  optimization.
- If a mode cannot engage, the command prints `[boltzgen-opt] NOT ACTIVE: <reason>` and exits 3; it never falls back to
  stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | the run failed |
| 2 | usage error |
| 3 | not active: the mode was refused |

## Modes

- `off` — stock `boltzgen configure`, then each pipeline step as released, in its own subprocess. A small kit launcher
  (`boltzgen_opt.stock_design`) starts each step: it checks that the environment carries no kit variable and that no
  optimization module or directory is importable, applies `--seed` when given, and calls upstream's step entry point. No
  optimization code loads and stock's arithmetic is untouched (STOCK.md, 'How stock is run'). Unseeded unless `--seed` is given.
- `exact` — the GPU steps in one seeded process, the design sampler replayed as a CUDA graph, no random weight
  initialisation before the checkpoint loads, attention masks built once per structure, CIFs written by background
  processes. Identical to `off` at the same `--seed` and `--diffusion_batch_size` on every spec, binder specs included.
  Use it when outputs must not move.
- `fast` (default) — `exact` plus diffusion conditioning computed once per step and broadcast over the batch,
  pair-biased attention as one fused bfloat16 call (cuDNN backend; matrix multiplies stay TF32) and fused elementwise
  Triton kernels in the token transformer. bf16 rounding, within stock's seed-to-seed variation. Use it for throughput,
  with several designs per diffusion batch (`--diffusion_batch_size`).
- `big` — `exact` without the graph sampler and the mask hoist, plus row-chunked token-distance features and expandable
  allocator segments. Use it when `exact` or `fast` runs out of memory. Outputs:
  - unconditional specs: `exact`'s structures and files bitwise, except the last digits of one reported confidence
    value (`design_iiptm`, summed in a different order; not a ranking key);
  - structure-conditioned specs: a numeric perturbation of `exact`'s trajectory (see Notes).

## Notes

- **A100 / H200.** `--config a100` / `--config h200` load `configs/<card>.env`; every optimization of every mode engages
  on either card. A card other than the pinned H100 80GB is named on one `[boltzgen-opt] NOTE gpu=…` line per process
  and never refused.
- **Reproducibility.** With `--seed`, every mode repeats its own outputs byte for byte (STOCK.md). `big`'s structures
  differ from `exact`'s only with a target structure (64-row token-distance products round differently under TF32; a
  perturbed trajectory, as under `fast`); its `design_iiptm` digits always differ.
- **Out of memory.** As in stock, a design batch that does not fit is skipped by upstream's own handler and the run
  exits 0 with fewer designs. The `DESIGNS` line counts it (`oom_skipped=`), and kit modes add an
  `[sz] {"event": "oom", …}` line.
