# AF3-torch (xfold on OpenFold3 weights) — optimization kit

Drop-in modes that make stock xfold — PyTorch AlphaFold 3, `run_alphafold.py`, at the pin in `STOCK.md`, on the converted
OpenFold3-preview2 weights — faster and lighter on GPU memory. Stock here is xfold plus the port to these weights (`STOCK.md`;
'stock' below always means this). You call `run_alphafold.py` exactly as before; the kit adds a `--mode`:

- `off` — stock exactly, through the kit's process chain.
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs; `--n_gpu P` splits one `big` prediction across P GPUs of one host.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs, up to 4,000 tokens on one GPU · `--n_gpu P` splits `big` across P GPUs of one host.

## Setup

Pick ONE way to get the pinned stack (every pin: the 'Stack' section of `STOCK.md`): **A — Docker**, **B — Apptainer**, or
**C — Python venvs on your own host**.

Every route needs:

- a host NVIDIA driver that runs CUDA 13.0 (580 or newer);
- a weights directory: `<weights dir>` in the blocks is the host directory holding `of3_ported_weights.bin.zst` itself, or
  receiving it with `--fetch`.

Route C additionally needs git, gcc / g++, make, zlib1g-dev, zstd, patch and network access (details under 'What the blocks
assume').

Type the first block from the directory that holds `af3_torch/` and `common/`:

```bash
# A — Docker (recommended: carries the whole pinned stack)
docker build -f af3_torch/environment/Dockerfile -t af3_torch-kit:dev .
docker run --rm -it --gpus all -v <weights dir>:/weights/af3_torch -v $PWD/out:/kit/af3_torch/out -w /kit af3_torch-kit:dev bash   # a shell for the steps below
# B — Apptainer / Singularity (the whole Setup for B; `kit` below: a shell function = B's bash run.sh): converts A's image, so build A first
apptainer build af3_torch.sif af3_torch/environment/apptainer.def
mkdir -p out jit; kit() { apptainer run --nv --bind <weights dir>:/weights/af3_torch --bind "$PWD/out":/kit/af3_torch/out af3_torch.sif "$@"; }   # B's ./run.sh
export AF3_TORCH_PARAMS_DIR=/weights/af3_torch && kit install --weights /weights/af3_torch --fetch   # of3_ported_weights.bin.zst made or checked; read-only ok
export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit check --config h100 --mode fast   # Run lines: kit <command> … · other cards: --config a100|h200
# C — instead of A or B, on your own host (venv): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …` (`kit` is the
shell function the block defines, standing in for `bash run.sh`). Under **A** you are now in the container shell, which opens in
`/kit`. Under **C** you are in your activated environment; STOCK.md's Stack section already ran `cd` and the install line, so
continue at `export`. Run:

```bash
[ -f run.sh ] || cd af3_torch                            # no-op once inside · C via §Stack: resume at export
bash run.sh install --weights /weights/af3_torch --fetch    # --fetch: 3.7 GB when empty; else hash check (read-only ok)
export AF3_TORCH_PARAMS_DIR=/weights/af3_torch           # the bind target of <weights dir> (a container path)
bash run.sh check --config h100 --mode fast                 # no launch: pins, interpreters, weights digest, ACTIVE line
```

What the blocks assume:

- **Weights.** `AF3_TORCH_PARAMS_DIR` names the directory holding `of3_ported_weights.bin.zst` — under A and B the container
  path `/weights/af3_torch` that `<weights dir>` is bound to. `install --weights DIR --fetch` downloads and converts the
  checkpoint when the directory is empty; without `--fetch` it hash-checks the file already there (read-only is fine).
- **Route C** is three uv-managed Python 3.12 environments: torch for the model; JAX with the alphafold3 fork built from
  source for featurisation and the writers; one for `run.sh`. The recipe ends by exporting `AF3_TORCH_PY`, `AF3_TORCH_JAX_PY`
  and `AF3_TORCH_JAX_REPO`, which the image presets.
- **Pin check.** `install` also fetches xfold's pinned archive for `bash run.sh stock`. Its pin check runs both interpreters and
  exits 3 naming an unset variable or a package off its pin.
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env`.
- **Compile cache.** A mode's first run at a token length compiles and autotunes kernels under `AF3_TORCH_CACHE_ROOT`, which
  the config sets; `bash run.sh warm --config h100 --mode M|all` does it ahead of time.

## Run

```bash
bash run.sh pred --config h100 --mode off   --json_path inputs/1BRS.json --output_dir out/off     # 1BRS: two protein chains, MSAs inside, ships with the kit
bash run.sh pred --config h100 --mode exact --json_path inputs/1BRS.json --output_dir out/exact
bash run.sh pred --config h100 --mode fast  --json_path inputs/1BRS.json --output_dir out/fast    # the default when no mode is named
bash run.sh pred --config h100 --mode big --json_path inputs/1BRS.json --output_dir out/big
bash run.sh pred --config h100 --mode big --n_gpu 2 --json_path inputs/1BRS.json --output_dir out/big_x2   # 2, 4 or 8 GPUs of one host
```

**Options.** `pred` takes xfold's own flags verbatim (`run_alphafold.py --help`). The kit adds:

- `--num_recycles` / `--diffusion_steps` — the model constructor's arguments; absent = its defaults, 10 and 200;
- `--mode`, `--n_gpu`;
- `--no-compile`, `--allow-partial` (Notes).

`AF3_TORCH_OPT=<mode>` names the mode when `--mode` is absent.

**Outputs.** They follow `run_alphafold.py`'s layout under `<output_dir>/<job>/`: `<job>_model.cif`, `<job>_confidences.json`,
`<job>_ranking_scores.csv`, ….

**What a run prints** (stderr).

- Every command prints `[af3-torch-opt] ACTIVE mode=<mode> lever_set=… levers=… n_gpu=… …` — the `ACTIVE` line, naming the
  mode and the optimizations engaged (the kit calls its individually switchable optimizations 'levers'; `CHANGES.md`
  describes each). `check` prints only that and exits 0; `off` shows `levers=none` and a `STOCK … proof=ok` line.
- A `pred` ends with one `LEVER name=… state=…` line per optimization and `DONE ok=1 rc=0 items=<n>/<n> failed=none …`.
- If a mode cannot engage (pins not met, a variable unset or unknown), the command prints
  `[af3-torch-opt] NOT ACTIVE mode=… reason=<reason>` and exits 3; nothing falls back to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | the prediction failed |
| 2 | usage error |
| 3 | not active |

**First run.** On `inputs/1BRS.json` a first run from an empty cache — route C, a fresh `AF3_TORCH_CACHE_ROOT`, or an
image built without STOCK.md's optional compile-cache tar — takes about a minute to a minute and a half per mode on an
H100; compare modes on a second run. Later runs on the same cache root start immediately.

## Modes

- `off` — stock (xfold plus the port) through the kit's process chain, with no optimization: fp32 weights under bf16 autocast,
  xfold's fastnn Triton kernels (`--nofastnn`: eager ops). `bash run.sh stock <flags>` runs xfold's own `run_alphafold.py`
  instead, in the stock venv that STOCK.md's Stack section builds (`AF3_TORCH_STOCK_PY`).
- `exact` — `off` plus the optimizations that keep every number `off` computes (coordinates and confidences identical; only
  the mmCIF header's run timestamp differs): whole-step CUDA graph with hoisted conditioning, the exact TriMul row, fused
  GLU + projection, and layout / launch / cast-memo optimizations.
- `fast` (default) — bf16 weights; the shared core's FlashPairformer triangle multiplication, triangle attention and
  transition kernels; fused pair-bias attention, MSA-module and diffusion-transformer kernels; a sample-batched, graphed,
  compiled sampler on tile-padded inputs. Use it for throughput.
- `big` — `fast`'s kernels recomposed for the lowest peak memory (no CUDA graph, early frees, row-blocked pair
  conditioning); numerics as `fast`. `--n_gpu 2|4|8` row-shards the pair representation across P GPUs of one host
  (`CHANGES.md`, '`--n_gpu P`'). Use it when `fast` runs out of memory.

Optimizations not named above that appear on `LEVER name=…` lines (every one: `CHANGES.md`):

| `LEVER name=…` | what it is |
|---|---|
| `glu_proj` | `exact`'s fused GLU + projection kernel; `transition` takes its place under `fast` / `big` |
| `prefetch` `write_behind` `feat_par` | the process chain overlaps work: the next input is featurised and finished outputs are written while the GPU runs |
| `autotune_cache` `compile` | Triton autotune results kept under `AF3_TORCH_CACHE_ROOT`; `torch.compile` of the sampler (`--no-compile` drops it) |

## Known upstream issues

In xfold as shipped:

- `XFOLD-001` — the trunk runs one Evoformer pass fewer than AlphaFold 3's `num_recycles + 1`;
- `XFOLD-002` — each step's Gaussian noise draw is overwritten by the bare noise scale;
- `XFOLD-003` — the fastnn kernels' int32 offsets overflow past 2048 tokens (`upstream_issues/XFOLD-003_*`);
- `XFOLD-004` — the sampler's per-step coordinates round to bfloat16 under autocast.

The port corrects all four on every mode, `off` and `bash run.sh stock` included; there is no flag.

## Notes

- **A100 / H200** (`--config a100|h200`). A mode selects the same optimizations on every card; where the shared core builds
  no kernel for the card, that optimization's `LEVER` line says `fallback=<word>` and the stock statement serves.
- **Where the gain is.** The kit's gain is in model (GPU) time per input. On `inputs/1BRS.json` (199 tokens) featurisation
  and the writers add about 20 s in every mode, so that example gains less end to end under `exact` or `fast` than its model
  time does; the gain grows with input size and inputs per call.
- **`PARTIAL refused=<kind> rc=3` after a `DONE ok=1` line.** The outputs are written, but an optimization served the stock
  path outside its declared coverage (its `FALLBACK lever=… expected=0` line names it) or hit an error and switched itself
  off; `--allow-partial` accepts such a run (rc 0, `partial=allowed:…`).
- **A mode without some optimizations.** `MODEL_OPT_LEVERS_OFF=<lever>[,…]` runs a mode without those (`levers_off=…` on the
  `ACTIVE` line; unknown names exit 2); `--no-compile` drops `compile`.
