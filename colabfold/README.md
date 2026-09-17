# ColabFold — optimization kit

Drop-in modes that make stock ColabFold 1.6.1 (`colabfold_batch`, with `alphafold-colabfold` 2.3.13 on the AlphaFold2-Multimer
v3 parameters) faster. You call `colabfold_batch` exactly as before; the kit adds a `--mode`:

- `off` — stock ColabFold, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs (bit for bit under deterministic XLA flags, Notes), faster.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — the mode for large inputs: `fast`'s optimizations on one GPU, and with `--n_gpu P` the pair representation sharded
  across P GPUs of one host so that still larger complexes fit.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs, up to 4,000 tokens on one GPU · `--n_gpu P` splits `big` across P GPUs of one host.

## Setup

Pick ONE way to get the pinned stack — stock ColabFold 1.6.1 with `alphafold-colabfold` 2.3.13 and everything it needs (the
'Stack' section of `STOCK.md` lists every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**
(route C's whole recipe is inlined in the first block).

Every route needs:

- an NVIDIA GPU of compute capability 8.0 or newer, with a CUDA 12 driver (550 or newer) on the host;
- a directory for the parameters, `/weights/af2_params` below. `install --weights` fills it with
  `params/params_model_{1..5}_multimer_v3.npz` and the marker `params/download_complexes_multimer_v3_finished.txt`.

Type the first block from the directory that holds `colabfold/` and `common/`:

```bash
# A — Docker (preferred): the whole pinned stack, stock and this kit in one image
docker build -f colabfold/environment/Dockerfile -t colabfold-kit:dev .
docker run --rm -it --gpus all -v /weights:/weights -v $PWD/out:/kit/colabfold/out -w /kit colabfold-kit:dev bash   # a shell in /kit for step 2
# B — Apptainer (the whole Setup for B): build the .sif once where Docker runs (converts A's image); the cluster then needs only Apptainer
apptainer build colabfold-kit.sif colabfold/environment/apptainer.def
mkdir -p out; kit() { apptainer run --nv --bind /weights:/weights --bind "$PWD/out":/kit/colabfold/out colabfold-kit.sif "$@"; }   # kit = B's ./run.sh
export COLABFOLD_OPT_DATA_DIR=/weights/af2_params && kit install --weights /weights/af2_params   # /weights/af2_params: params fetched or checked; read-only ok
kit check --config h100 --mode fast                  # each Run line: kit <command> … · cards: --config a100|h200
# C — instead of A or B, on your own host: ≈6.4 GB — the whole route-C recipe; STOCK.md §Stack = these lines annotated, plus the apt and uv-installer lines (run one or the other)
uv venv --seed --managed-python --python 3.11 venv && . venv/bin/activate
pip install --no-deps -r colabfold/environment/requirements.lock
export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95     # required on your own host (the image sets them; Notes)
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`. Under **A** you
are now in the container shell, which opens in `/kit`. Under **C** you are in the environment the route-C lines just activated
(if you used STOCK.md's Stack section instead, it already ran `install`: skip that line too). Run:

```bash
[ -f run.sh ] || cd colabfold                     # no-op once inside · C via §Stack: skip install too
bash run.sh install --weights /weights/af2_params    # --weights: ≈2 GB if absent, else hash-check; read-only ok
export COLABFOLD_OPT_DATA_DIR=/weights/af2_params # required: the parameters directory (layout: Setup above)
bash run.sh check --config h100 --mode fast          # dry run → `DRY-RUN mode=fast …` · cards: --config a100|h200
```

What the blocks assume:

- **Weights directory.** `COLABFOLD_OPT_DATA_DIR` is required and names the parameters directory (layout above);
  `install --weights` fetches the parameters when absent, otherwise hash-checks them (read-only is fine).
- **Memory variables under C.** `TF_FORCE_UNIFIED_MEMORY=0` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` are required on your own
  host (the image sets them; Notes).
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env` (target card, stack key, cache root); every command
  below takes it.
- **Compile cache.** JAX compiles each new input length once per card — ≈2 min with no message while it does, in every mode,
  `off` included — and keeps the result in its persistent cache (`JAX_COMPILATION_CACHE_DIR`).
  - An image built with STOCK.md's optional compile-cache tar ships that cache: under A it is used in place under
    `/opt/jit_cache`; under B a copy is seeded into `/tmp/model_opt_jit-uid<uid>` (the host's `/tmp` under Apptainer). An image
    built without one compiles each new length on its first run, like route C.
  - A cache relocated with `MODEL_OPT_JIT_ROOT` recompiles once on first use there, then persists.
  - On your own host the cache lives under `COLABFOLD_OPT_JIT_ROOT`, and `bash run.sh warm --config h100 --mode M` compiles the
    example's length ahead of time.

## Run

```bash
A=tests/inputs/1BRS_AD.a3m                                            # ships with the kit: barnase–barstar, two chains
bash run.sh pred --config h100 --mode off   $A out/off                   # stock, in a clean subprocess
bash run.sh pred --config h100 --mode exact $A out/exact
bash run.sh pred --config h100 --mode fast  $A out/fast                  # the default when no mode is named
bash run.sh pred --config h100 --mode big $A out/big                 # one GPU: fast's levers
bash run.sh pred --config h100 --mode big --n_gpu 2 $A out/big_x2    # 2, 4 or 8 GPUs of one host
bash run.sh pred --config h100 --mode fast  $A out/r3 --num-recycle 3 --num-models 1    # colabfold_batch's own options, verbatim, in any mode
```

**Options.** `pred` hands `colabfold_batch` its own command line verbatim — `<input> <results> [options]`. The kit reads only
`--mode`, `--n_gpu` and `--help` here; everything else goes to `colabfold_batch`.

**Inputs.** The input is an `.a3m`, `.fasta`, `.csv` or a directory of them. The example is a single-sequence a3m, so pLDDT
near 35 is expected.

**Outputs.** They land where stock writes them (`<results>/…`); the kit adds no file. `colabfold_batch` skips inputs already
done in an existing `<results>` directory unless given `--overwrite-existing-results`.

**What a run prints** (stderr).

- `--mode off` prints `[colabfold-opt stock] STOCK cli=… proof=ok` and `[colabfold-opt] NOT ACTIVE mode=off (stock: nothing applied)`
  and exits 0 — these two lines only record that nothing of the kit is loaded.
- A kit mode prints `[colabfold-opt] ACTIVE mode=<mode> levers=… gpu=…` once engaged — the `ACTIVE` line, naming the mode and
  the optimizations engaged (the kit calls its individually switchable optimizations 'levers').
- At exit it prints one `LEVER name=<LEVER> state=on calls=<n> fallbacks=<n> …` line per optimization. `served=0` or `calls=0`
  beside `fallback_by=<rule>:<n>` means that optimization does not apply at this input's shapes and the stock operation ran,
  by design (`CHANGES.md` lists the rules).
- If a mode cannot engage — the pin, the install, the GPU or `--n_gpu` not as required — the command prints
  `[colabfold-opt] NOT ACTIVE: <reason>` and exits 3; it never falls back to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | the prediction failed |
| 2 | usage error |
| 3 | not active |

**First run.** A mode's first run at a new length compiles; compare modes on a second run into a fresh `<results>` directory.

## Modes

- `off` — stock `colabfold_batch` as released, in a clean subprocess with nothing of the kit importable.
- `exact` — parameters, features and recycling state stay on the GPU across the recycle loop and outputs are fetched once
  (placement only); `off`'s arithmetic — bit for bit under the deterministic XLA flags (Notes), without which stock itself
  varies run to run. Use it when outputs must not move.
- `fast` (default) — `exact` plus a larger attention sub-batch, the shared core's FlashPairformer triangle kernels (Pallas)
  and fused transition, cuDNN / Pallas MSA attention, template rows embedded once; bf16 re-association, within stock's
  seed-to-seed variation. Use it for throughput.
- `big` — the large-input mode. On one GPU it runs exactly `fast`'s optimizations (the At-a-glance one-GPU size holds for
  both); `--n_gpu 2|4|8` row-shards the pair representation over P GPUs of one host, numerics as `fast`. Use it, with
  `--n_gpu`, when `fast` runs out of memory.
  - `--n_gpu` takes 1, 2, 4 or 8; another value, fewer GPUs visible, or P > 1 under another mode exits 3.

## Notes

- **Where the gain is.** The kit's gain is in prediction time per model. Each call also pays ≈40–50 s of start-up, MSA
  (multiple sequence alignment) processing and first-model set-up in every mode, so two models of one small input take
  about as long in `fast` as in `off` (the one-time compile outweighs the per-model gain there): the gain grows with work
  per call.
- **Model load.** A fresh process still spends ≈25–35 s tracing and loading the model before its first structure (stock
  behaviour, every mode); the compile cache removes only the ≈2 min compilation per new length.
- **Memory environment** (every mode, `off` included). `TF_FORCE_UNIFIED_MEMORY=0` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
  must be in the process environment — the image sets them; on your own host you export them (Setup, C). Left unset,
  `colabfold_batch` selects unified memory, which does not progress on this stack (`STOCK.md`). At `--n_gpu 8` the fraction is
  `0.90` (room for NCCL) unless you export one.
- **A100 / H200** (`--config a100` / `h200` load `configs/<card>.env`): every optimization engages on both. H200 uses the
  H100's kernels, yet its first run compiles ≈2 min per cache root: JAX keys the cache by device name.
- **Other cards** of compute capability ≥ 8.0 run with `notes=…` on the `ACTIVE` line; below 8.0, `fast` and `big` exit 3.
- **Deterministic runs.** Under `XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"`, which you export
  yourself, `exact` equals `off` bit for bit and the compile cache is keyed by that recipe. The kit has no switch of its own
  for this (`--det 1` is a usage error, exit 2).
- **A mode without some optimizations.** `MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]` runs a mode without the named ones (names
  as on the `LEVER` lines); the `ACTIVE` line adds `ablated=<names>`.
