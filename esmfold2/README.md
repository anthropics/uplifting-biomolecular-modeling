# ESMFold2 — optimization kit

Drop-in modes that make stock ESMFold2 (Biohub `esm` 3.3.0 with its `transformers` 4.57.6 fork;
`ESMFold2InputBuilder().fold`) faster and lighter on GPU memory. You call the library exactly as before; the kit adds a `--mode`:

- `off` — stock ESMFold2, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs; `--n_gpu P` splits one `big` prediction across P GPUs of one host.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs, up to 3,000 tokens on one GPU · `--n_gpu P` splits `big` across P GPUs of one host.

## Setup

Pick ONE way to get stock ESMFold2 at the pin together with this kit (the 'Stack' section of `STOCK.md` lists every
pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver ≥ 580 on the host.

Route C additionally needs:

- `uv`, git, a C++ compiler and the CUDA 13.0 toolkit (three CUDA extensions are compiled);
- the three exports in STOCK.md's Stack section;
- the two Biohub packages at their pinned git commits or from `stock/*.tar.gz` (anything else exits 3).
  STOCK.md's Stack section is the complete route-C recipe.

Type the first block from the directory that holds `esmfold2/` and `common/`:

```bash
# A — Docker (preferred): the whole pinned stack, stock and this kit in one image (A100: add --build-arg STACK=img_esmfold2_a100)
docker build -f esmfold2/environment/Dockerfile -t esmfold2-kit:dev .
docker run --rm -it --gpus all -v /weights/esmfold2:/weights/esmfold2 -v $PWD/out:/kit/esmfold2/out esmfold2-kit:dev bash   # a shell in /kit/esmfold2
# B — Apptainer / Singularity (the whole Setup for B): build the .sif once wherever Docker runs (it converts A's image), then copy the one file; running it needs only Apptainer. In kit(), left of ":" = host directory, right of it and HF_HOME = paths in the image
apptainer build esmfold2-kit.sif esmfold2/environment/apptainer.def      # from esmfold2-kit:dev (A100: A + STACK=img_esmfold2_a100)
mkdir -p out jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --env HF_HOME=/weights/esmfold2 --bind /weights/esmfold2:/weights/esmfold2 --bind "$PWD/out":/kit/esmfold2/out esmfold2-kit.sif "$@"; }   # kit = B's ./run.sh · other cards: --config a100|h200
kit install --weights /weights/esmfold2                     # /weights/esmfold2 (hub/…): fetched or checked; read-only ok
kit check --config h100 --variant fast --mode fast          # your own --input: an absolute path under $HOME or $PWD
# C — instead of A or B, on your own host (venv): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`.
Under **A** you are now in the container shell, which opens in `/kit/esmfold2`. Under **C** you are in your activated
environment; if STOCK.md's Stack section set it up, you are already inside `esmfold2/` with the install line done, so
type the `export` line, skip `install`, and run `check`. Run:

```bash
[ -f run.sh ] || cd esmfold2                                # no-op once inside · C via §Stack: skip install too
export HF_HOME=/weights/esmfold2                            # the weights root; layout: STOCK.md §Pin
bash run.sh install --weights $HF_HOME                         # --weights: ≈27 GB if absent, else hash-check; read-only ok
bash run.sh check --config h100 --variant fast --mode fast     # dry run, exit 0, no model; A100/H200: --config a100|h200
```

What the blocks assume:

- **Weights directory.** `/weights/esmfold2` in the blocks; `HF_HOME` must name it (its `hub/…` layout: STOCK.md,
  'Pin'). `install --weights` fetches the files if absent, else hash-checks what is there, so a complete directory may be
  read-only.
- **Paths under B.** In the `kit()` line, the left side of each `--bind …:…` is your host directory; the right side, and
  `HF_HOME`, are paths inside the image. Give your own `--input` as an absolute path under `$HOME` or `$PWD`.
- **Outputs under A and B.** The `out/…` of the Run examples is the host's `./out`.
- **A100 image.** Build with `--build-arg STACK=img_esmfold2_a100` (route A; route B converts that image).
- **`check`.** It reads the pins from package metadata, digests the weights (≈24 GiB by sha256: a minute or two) and
  prints one `[esmfold2-opt] DRY-RUN …` line; exit 0, no model is loaded.
- **GPU cards.** `--config h100|h200|a100` loads `configs/<card>.env` (`MODEL_OPT_TARGET_GPU`,
  `ESMFOLD2_OPT_REQUIRE_FAST_ENV`, `ESMCFOLD_CCD_PATH`, and `TRITON_CACHE_DIR` when `MODEL_OPT_JIT_ROOT` is set); every
  command below takes it.
- **`--variant fast|full_msa|full_nomsa` is required** (no default, one per process). It picks the model: ESMFold2-Fast,
  or ESMFold2 with / without MSAs (multiple sequence alignments).
- **First run.** The first run of a kit mode compiles its Triton kernels once per machine (an image built with
  STOCK.md's optional compile-cache tar starts with them compiled), and
  `bash run.sh warm --config h100 --variant fast --mode fast` does it ahead of time. Under `--mode big` the first run
  also performs a one-time numerical self-check of its fused kernels (about ten seconds on H100); later runs skip it.
  CUDA graphs are captured per process (seconds).
- **Compile cache.** A writable `MODEL_OPT_JIT_ROOT` keeps the cache; route B's block puts it in `./jit` on the host.

## Run

```bash
X="--variant fast --input opt/forward/fast_inference/tests/w4_public_slice.json"   # three 1BRS complexes (199/707/796 tokens), 2 seeds each
bash run.sh pred --config h100 --mode off   $X --out_dir out/off      # stock; --backend fused = the fused path exact reproduces
bash run.sh pred --config h100 --mode exact $X --out_dir out/exact
bash run.sh pred --config h100 --mode fast  $X --out_dir out/fast     # the default when no mode is named
bash run.sh pred --config h100 --mode big $X --out_dir out/big
bash run.sh pred --config h100 --mode big --n_gpu 2 $X --out_dir out/big_x2    # 2, 4 or 8 GPUs of one host
```

**Options.** `pred` hands the library's fold settings through by name, at the library's defaults
(`--num_loops 20 --num_sampling_steps 200 --num_diffusion_samples 1 --msa_max_depth 1024 …`; STOCK.md). The kit's own
flags are `--mode`, `--variant`, `--device` (default `cuda`), `--n_gpu`, `--seeds`, `--det` and, under `off`,
`--backend` (`--backend fused` = the library's fused path, the configuration `exact` reproduces). Two other ways to make
the same call:

- `esmfold2-opt pred --mode fast …` — the same call without `run.sh`;
- `source configs/h100.env; ESMFOLD2_OPT=fast ESMFOLD2_VARIANT=fast python your_script.py` — engages a mode, offline on
  the pinned weights, in any program that folds through the library.

**Inputs.** `--input` is upstream's prediction-input JSON (`{"sequences": [...]}`, one input or a list). A protein
chain's `"msa"` may name an A3M file (path relative to the JSON), which `full_msa` reads.

**Outputs** (every mode).

- `<out_dir>/cif_all/<id>__<fast|full>__s<seed>_x<k>.cif` and `…_x<k>_pae.npz`, rewritten per run;
- `<out_dir>/pred_rows.jsonl`, appended to.

**What a run prints** (on stderr).

- A kit mode prints `[esmfold2-opt] ACTIVE mode=fast variant=fast … n_gpu=1 …` — the `ACTIVE` line; it ends with the
  engaged optimizations (its output calls each one a `lever`). Then one
  `LEVER name=… state=…` line per optimization, and a closing `[esmfold2-opt] EXIT …` summary (exit 0).
- `--mode off` prints `[esmfold2-opt] NOT ACTIVE: mode off: stock esmfold2 …`, ends with
  `[esmfold2-opt] DONE predictions=<n> items=<m> out_dir=<dir>` and exits 0.
- If a mode cannot engage, the command prints `[esmfold2-opt] NOT ACTIVE: <reason>` and exits 3; it never falls back to
  stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | ok |
| 1 | failed or incomplete |
| 2 | usage error |
| 3 | not active |
| other | under `off`, if the stock subprocess crashes, its own exit code is passed through |

## Modes

- `off` — stock `ESMFold2InputBuilder().fold` as released, in a clean subprocess with nothing of the kit importable.
  `--backend shipped` (the default) is the model as loaded; `--backend fused` adds the library's own
  `set_kernel_backend('fused')` + `set_chunk_size(None)`.
- `exact` — CUDA-graph capture of the trunk, encoders and sampler, per-input caches, hoisted constants and re-plumbed
  kernels with the fused backend's arithmetic; outputs identical to `off --backend fused` under `--det 1`. Use it when
  outputs must not move.
- `fast` (default) — `exact` plus the FlashPairformer triangle-multiplication kernels of the shared core
  (`common/opt_core`, the runtime all kits share) and fused Triton / tensor-core kernels for the pair transition, atom
  transformer, diffusion step and MSA module; bf16 re-association, within stock's seed-to-seed variation. Use it for throughput.
- `big` — `fast`'s kernels with no CUDA graph captured, expandable allocator segments, and the language model streamed
  from host memory from 1,500 tokens; numerics as `fast`. Use it when `fast` runs out of memory.
  - `--n_gpu 2|4|8` row-shards the pair representation over one host's GPUs when one GPU runs out too (for size, not
    speed), computing confidence per row block in bf16 from 1,024 tokens.
  - Under `--mode big` the row-sharded DiT attention runs the `apb_attn` kernel by name; the `UNCOVERED_CELL` line it
    prints is informational.

## Notes

- **Where the gain is.** The kit's gain is in fold time per input. Each process also pays ≈25 s of start-up
  (interpreter, checkpoint load and digest) in every mode, so a single small input gains little end to end, while a
  large input under `fast` gains the most against bare `off` (`--backend shipped`).
- **A100 80GB.** Build with the A100 build-arg from Setup (route C:
  `TORCH_CUDA_ARCH_LIST=8.0 bash …/build_wheels.sh --stack img_esmfold2_a100`) and run with `--config a100`. The
  transition kernel that exists only for compute capability 9.0 stays off (`card=unsupported_card:sm80` on its `LEVER`
  line, `not_for_class=…` on the `DRY-RUN` line); the portable kernels serve, and shared-core kernels with no
  compute-capability-8.0 entry run the stock statement (ids: CHANGES.md).
- **H200.** `--config h200`.
- **`exact` and the pair transition.** The pair transition kernel serves 400+-token inputs on compute capability 9.0
  (H100, H200; not yet A100), else the stock statement runs — same outputs. stderr says so with one `NOTE pair: …` line
  and, on 9.0 cards, `NAMED_FALLBACK:transition|… refused=exact:… served=caller:engine_module …` census lines
  (information, not errors).
- **Switching optimizations off.** `MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]` runs a mode without the named optimizations
  (names as on the `LEVER` lines; `ablate=<names>` on the `ACTIVE` line).
- **Determinism.** `--det 1` (with `--seeds`) selects the deterministic recipe on every mode alike; `exact --det 1`
  equals `off --backend fused --det 1` bit for bit.
- **Fast-path requirement.** Every config sets `ESMFOLD2_OPT_REQUIRE_FAST_ENV=1`: every route, `off` included, exits 3
  by name unless flash-attention and TransformerEngine import (STOCK.md, 'Variables').
- **Large pair planes.** Upstream's fused pair-bias attention kernel indexes with 32-bit offsets and faults once a pair
  plane passes 2³¹−1 elements (2,897 tokens at one diffusion sample). Every kit mode launches it in row blocks inside
  that bound (byte-identical results); bare `off` does not.
  - The proposed upstream patches are in `opt/forward/fast_inference/upstream/`.
