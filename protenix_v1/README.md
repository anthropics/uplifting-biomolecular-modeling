# Protenix v1 — optimization kit

Drop-in modes that make stock Protenix 1.1.0 (`protenix pred`) faster and lighter on GPU memory. You call Protenix
exactly as before; the kit adds a `--mode`:

- `off` — stock Protenix, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs; `--n_gpu P` splits one `big` prediction across P GPUs of one host.

A mode is a fixed set of individually switchable optimizations, each reported by name when the run exits. What each
optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs, up to 4,000 tokens on one GPU · `--n_gpu P` splits `big` across P GPUs of one host.

## Setup

Pick ONE way to get the pinned stack — stock Protenix 1.1.0 and everything it needs (the 'Stack' section of `STOCK.md`
lists every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver that runs CUDA 13.0 (version 580 or newer).

Route C additionally needs:

- Python 3.11, `nvcc` and a C++ compiler; STOCK.md's Stack section is the complete route-C recipe.

Type the first block from the directory that holds `protenix_v1/` and `common/`:

```bash
# A — Docker (recommended: the whole pinned stack, stock and this kit in one image)
docker build -f protenix_v1/environment/Dockerfile -t protenix_v1-kit:dev .
docker run --rm -it --gpus all -w /kit -v /weights/protenix_v1:/weights/protenix_v1 -v "$PWD/out":/kit/protenix_v1/out protenix_v1-kit:dev bash
# B — Apptainer / Singularity (the whole Setup for B): converts A's image (build A first, no daemon afterwards); /weights/protenix_v1 gets checkpoint/ + common/
apptainer build protenix_v1-kit.sif protenix_v1/environment/apptainer.def
mkdir -p out jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --bind /weights/protenix_v1:/weights/protenix_v1 --bind "$PWD/out":/kit/protenix_v1/out protenix_v1-kit.sif "$@"; }   # kit = B's ./run.sh
export PROTENIX_ROOT_DIR=/weights/protenix_v1 && kit install --weights /weights/protenix_v1   # /weights/protenix_v1: fetched or checked; read-only ok
kit check --config h100 --mode fast                 # Run lines: kit <verb> … · other cards: --config a100|h200
apptainer exec protenix_v1-kit.sif cat /kit/protenix_v1/opt/forward/v05_addon/inputs/p995_1brs.json > out/p995_1brs.json   # Run's example out of the read-only image, for kit pred
# C — instead of A or B, on your own host (venv): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`.
Under **A** you are now in the container shell, which opens in `/kit`; the next block runs there as written. Under **C**
you are in your activated environment; if STOCK.md's Stack section set it up, you are already inside `protenix_v1/` with
the install line done, so skip `install` here (type the `mkdir … cp …` line, then continue at `export`).

One thing to know before you run it: **the input file must sit in a writable directory.** Stock Protenix writes an
updated copy of the input next to the `--input` file, so the shipped example is run from a copy in `out/`. Route B's last
line above copies it out of the read-only image; under A and C the `cp` line below does it. Under B, put your own inputs
under `out/` as well, or in another host directory you bind into the container. Run:

```bash
[ -f run.sh ] || cd protenix_v1                     # no-op once inside · C via §Stack: skip install too
mkdir -p out && cp opt/forward/v05_addon/inputs/p995_1brs.json out/   # Run's example on a writable path (stock writes beside it)
bash run.sh install --weights /weights/protenix_v1   # --weights: fetch if absent, else hash-check; read-only ok
export PROTENIX_ROOT_DIR=/weights/protenix_v1      # required, every route: the weights + data root
bash run.sh check --config h100 --mode fast           # dry run: JSON report ("ok": true), then EXIT … rc=0
```

What the blocks assume:

- **Weights directory.** `/weights/protenix_v1` in the blocks. `install --weights DIR` fetches only the files that are
  absent — the 1.5 GB checkpoint and six data files, into `checkpoint/` and `common/` (the list: STOCK.md, 'Variables') —
  and hash-checks the ones present; it writes nothing else there, so a complete directory may be read-only.
- **`PROTENIX_ROOT_DIR`** must name that directory on every route: it is the weights and data root the model reads.
- **Pin check on your own host (route C).** `install` accepts Protenix 1.1.0 installed from the wheel under `stock/` or
  from PyPI (the same file). It refuses, naming the problem, another version, a modified or missing package file, or an
  extra module under `protenix/` or `runner/` (the two packages the kit patches in memory at start-up).
- **GPU cards.** `--config h100|h200|a100` loads `configs/<card>.env`; every command below takes it. The file requires
  `PROTENIX_ROOT_DIR` and fills in, where unset, `LAYERNORM_TYPE`, `TORCH_EXTENSIONS_DIR` / `TRITON_CACHE_DIR` and
  `MODEL_OPT_TARGET_GPU`.
- **First run compiles.** A first run compiles once: stock's fast-LayerNorm CUDA extension with `nvcc` (minutes; already
  built into the images), then a kit mode's Triton kernels and CUDA graphs per input size.
  - Compare modes from the second run on.
  - `bash run.sh warm --config h100 --mode <mode> --out_dir <dir>` compiles ahead of time.
- **Compile cache.** A writable `MODEL_OPT_JIT_ROOT` keeps the compile caches between runs; route B's block puts it in
  `./jit` on the host. The first run in a fresh cache directory also performs a one-time numerical self-check of the
  fused kernels on your GPU (about ten seconds on H100; up to ~30 s on A100) before serving them; later runs skip it.

## Run

```bash
X="--input out/p995_1brs.json"                                   # 1BRS barnase–barstar ×5 (995 tokens), MSAs included
bash run.sh pred --config h100 --mode off   $X --out_dir out/off     # stock, in a clean subprocess
bash run.sh pred --config h100 --mode exact $X --out_dir out/exact
bash run.sh pred --config h100 --mode fast  $X --out_dir out/fast    # the default when no mode is named
bash run.sh pred --config h100 --mode big $X --out_dir out/big
bash run.sh pred --config h100 --mode big --n_gpu 2 $X --out_dir out/big_x2    # 2, 4 or 8 GPUs of one host
```

**Options.** `pred` hands everything else to `protenix pred` verbatim (`--seeds`, `--cycle`, `--sample`,
`--use_template`, …). The kit's own flags are `--config`, `--mode`, `--n_gpu`, `--det` and `--allow-partial`.
Two other ways to make the same call:

- `protenix-v1-opt pred --mode <mode> …` — the same call without `run.sh`;
- `PROTENIX_V1_OPT=<mode> protenix pred …` — engages a mode from the unchanged stock command line.

**Inputs.** The directory of the `--input` file must be writable (stock writes an updated copy beside it; see Setup).

**Outputs.** They land where stock writes them: `<out_dir>/<name>/seed_<seed>/predictions/`.

**What a run prints** (on stderr).

- A kit mode prints `[protenix-v1-opt] ACTIVE mode=<mode> …` when it engages — the `ACTIVE` line, naming the mode and
  the optimizations engaged (the kit calls its individually switchable optimizations 'levers').
- At exit it prints one `LEVER name=… state=…` line per optimization, and `[protenix-v1-opt] EXIT mode=<mode> rc=<code>` last.
- `--mode off` prints `[protenix-v1-opt stock] PROOF {…}` lines from the stock process, and the same `EXIT` line.
- If a mode cannot engage, the command prints `[protenix-v1-opt] NOT ACTIVE: <reason>` and exits 3; it never falls back
  to stock silently. `--allow-partial` accepts a mode with some of its optimizations off.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | ok |
| 1 | failed |
| 2 | usage error |
| 3 | not active |

## Modes

- `off` — stock `protenix pred` as released, in a clean subprocess with nothing of the kit importable.
- `exact` — triangle multiplication, triangle attention, the pair transition and the template pair stack run on kernels
  that reproduce stock's arithmetic, plus a CUDA-graphed diffusion step (its attention kernel on H100 inside the kernel's
  supported shapes, otherwise the stock operation, reported by name). Outputs identical to `off` under `--det 1`.
- `fast` (default) — `exact` plus the FlashPairformer triangle kernels of the shared core (`common/opt_core`, the runtime
  all kits share) in bf16, fused transitions, a fused sampler and MSA-module kernels; within stock's seed-to-seed
  variation. Use it for throughput.
- `big` — `fast`'s kernels without the sampler CUDA graphs, plus upstream's chunked pair path and the kit's memory
  optimizations, sized once per process on the largest input. `--n_gpu 2|4|8` row-shards the pair representation over
  P GPUs of one host. Use it when `fast` runs out of memory.

## Notes

- **A100** (`--config a100`). Admission is by compute capability; a card the kit does not list is admitted with a `NOTE`
  line.
  - On compute capability 8.0 the optimizations without a kernel for that architecture report `reason=card_off` on
    their `LEVER` lines.
  - `gblock` / `xtr` / `tmpl_xtr` serve only the pair sizes the shared core admits there (CHANGES.md).
- **H200** (`--config h200`): the same optimizations as on the H100.
- **`NAMED_FALLBACK` lines.** On any card or mode, a `NAMED_FALLBACK … refused=<row>:<reason> served=<row>` line names a
  kernel shape with no admitted entry in the shared core's kernel tables and the operation that served that call instead.
  It is a report, not an error; these lines are also counted in `alerts=` on the `[opt_core] CELLS` line at exit.
- **Switching optimizations off.** `MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]` runs a mode without the named optimizations
  (names as on the `LEVER` lines); the `ACTIVE` line then carries `ablated=<names>`.
- **Determinism.** `--det 1` selects the deterministic recipe on any mode, `off` included (`opt/protenix_v1_opt/det.py`);
  `exact --det 1` equals `off --det 1` bit for bit.
- **Large inputs under `exact` / `fast`.** An input above the mode's sampler-graph token cap (keyed on the card's memory;
  `PTX_SAMPLER_GRAPH_MAXTOK` overrides it) runs the stock sampler loop: `sg`, `hoist` and `sampler_prep` then read
  `state=skipped reason=above_cap`.
- **Out of memory.** Under `fast` → use `big`; still out of memory → `big --n_gpu P`.
- **Sizing `big`.** `big` sizes its optimizations from `--input`; set `PROTENIX_V1_BIG_SIZE_N_TOKEN=<n>` to give the
  largest input's token count explicitly.
- **Logs under `--n_gpu P`.** Each rank's lines are in `<out_dir>/rowpair/rank<r>.log` (rank 0's also on stderr).
