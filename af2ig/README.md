# AF2 initial guess (dl_binder_design) — optimization kit

Drop-in modes that make the `af2_initial_guess` step of nrbennet/dl_binder_design at commit `cafa3853` — AlphaFold-2
re-prediction of designed binder–target complexes — faster and lighter on GPU memory. You call `predict_pdb.py` exactly
as before; the kit adds a `--mode`:

- `off` — stock exactly ('stock' below always means the pinned upstream release).
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs.

Every mode, `off` included, runs the step through `predict_pdb.py`: a PyRosetta-free front end with the same computation as
upstream's `predict.py` (STOCK.md, 'Stock exceptions' section). A directory of binder–target complex PDB files goes in
(the binder is the first chain); one predicted complex PDB file and one score line per design come out; no Rosetta silent files.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs, up to 3,000 tokens on one GPU.

## Setup

Pick ONE way to get the pinned stack — Python 3.11, jax 0.5.3 with the CUDA 12 plugin, dm-haiku 0.0.16, tensorflow-cpu
2.21.0; stock dl_binder_design itself is unpacked from `stock/` at install (the 'Stack' section of `STOCK.md` lists every
pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver ≥ 525.60.13 (CUDA 12).

Route C additionally needs:

- Linux x86_64;
- `uv`, or any released CPython 3.11;
- GNU `tar` and `patch`.

Type the first block from the directory that holds `af2ig/` and `common/`. A cluster that has only the `.sif` file needs no
checkout: start at the `kit()` line, in any directory.

```bash
# A — Docker (preferred): the pinned stack, stock and this kit in one image
docker build -f af2ig/environment/Dockerfile -t af2ig-kit:dev .
docker run --rm -it --gpus all -w /kit -v /weights/af2ig:/weights/af2ig -v "$PWD/out":/kit/af2ig/out af2ig-kit:dev bash   # shell opens in /kit
# B — Apptainer (the whole Setup for B): build the .sif once where Docker runs (converts A's image); the cluster then needs only Apptainer
apptainer build af2ig-kit.sif af2ig/environment/apptainer.def
mkdir -p out; kit() { apptainer run --nv --bind /weights/af2ig:/weights/af2ig --bind "$PWD/out":/kit/af2ig/out af2ig-kit.sif "$@"; }   # kit = B's ./run.sh · other cards: --config a100|h200
export AF2_PARAMS=/weights/af2ig && kit install --weights /weights/af2ig   # /weights/af2ig: params/ fetched or checked; read-only ok
kit check --config h100 --mode fast             # A100/H200: --config a100|h200 · Run lines: kit <command> …
#     with --config <card>, B keeps the compile cache under ~/.cache/af2ig_opt/jit ($HOME is bound; seeded from the image once); MODEL_OPT_JIT_ROOT=<lasting dir> moves it
# C — instead of A or B, on your own host (venv): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`. Under **A** you are
now in the container shell, which opens in `/kit`. Under **C** you are in your activated environment; if you followed
STOCK.md's Stack section it already ran this whole block, so go on to Run. Run:

```bash
[ -f run.sh ] || cd af2ig                       # no-op once inside · C via §Stack: block done, go to Run
bash run.sh install --weights /weights/af2ig       # --weights: 5.6 GB if absent, else hash-check; read-only ok
export AF2IG_DIR=$PWD/dl_binder_design/af2_initial_guess   # route C only: the image presets it
export AF2_PARAMS=/weights/af2ig                           # every route: the directory that holds params/
bash run.sh check --config h100 --mode fast        # dry run: pins, checkout, weights digest, GPU; runs nothing
```

What the blocks assume:

- **No separate upstream install.** On every route `run.sh install` itself unpacks and patches the pinned dl_binder_design
  commit.
- **Pin check under C.** On your own host, `install`'s pin check exits 3 naming any package of STOCK.md's Stack section
  that is missing or at another version.
- **Weights directory.** `AF2_PARAMS` names the directory that holds `params/`, on every route. Parameters already on disk
  as `<dir>/params/params_model_1_ptm.npz`: pass `<dir>` as `--weights` and as `AF2_PARAMS`.
- **Inputs under B.** A `--pdbdir` of your own is an absolute path under `$HOME` or `$PWD`; the example inputs (`$X` in
  Run) ship inside the image.
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env` (`MODEL_OPT_TARGET_GPU`, `AF2IG_OPT_JIT_ROOT`,
  `PYTHONDONTWRITEBYTECODE`); every command below takes it.
- **Compile cache.** The first run of a kit mode at a new residue count compiles its XLA program once per machine.
  `bash run.sh warm --mode M --lengths L1,L2,…` does that ahead of time, and `AF2IG_OPT_JIT_ROOT` keeps the cache between runs.

## Run

```bash
X=opt/forward/af2ig_kit/tests/inputs/pdbs                               # six public binder–target complexes (chain A = binder)
bash run.sh pred --config h100 --mode off   --pdbdir $X --out out/off      # stock, in a clean subprocess
bash run.sh pred --config h100 --mode exact --pdbdir $X --out out/exact
bash run.sh pred --config h100 --mode fast  --pdbdir $X --out out/fast     # the default when no mode is named
bash run.sh pred --config h100 --mode big --pdbdir $X --out out/big
bash run.sh pred --config h100 --mode fast  --pdbdir $X --out out/r5 -- -recycle 5    # predict_pdb.py's own options after --, on every mode
```

**Options.** `pred` hands `predict_pdb.py` everything after `--` verbatim, on every mode. The kit's own flags
(`af2ig-opt pred --help` lists them) are:

- `--mode`, `--det`, `--seed`;
- `--allow-partial` (Notes);
- `--precompile N` — background threads that compile each input length's program ahead of the prediction loop.

`af2ig-opt pred --mode <mode> …` is the same call without `run.sh`; `AF2IG_OPT=<mode>` names the mode when `--mode` is absent.

**Outputs.** They land where `off` writes them: `<out>/pdbs/<tag>_af2pred.pdb`, `<out>/out.sc` and `<out>/check.point`.
A rerun into the same `--out` resumes from `check.point` and predicts only what is missing, so a second (timing) run needs a
fresh `--out` (e.g. `--out out/fast2`).

**What a run prints** (stderr).

- A kit mode prints `[af2ig-opt] ACTIVE mode=<m> … levers=<ids>` before the driver starts — the `ACTIVE` line, naming the
  mode and the optimizations engaged (the kit calls its individually switchable optimizations 'levers'; `CHANGES.md` lists
  them by id).
- It ends with `[af2ig-opt] EXIT … items=<done>/<inputs> …`.
- If a mode cannot engage, the command prints `[af2ig-opt] NOT ACTIVE: <reason>` and exits 3; it never falls back to stock
  silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | failed, or fewer outputs than inputs |
| 2 | usage error |
| 3 | not active |

## Modes

- `off` — stock `predict_pdb.py` with no optimization flag, in a clean subprocess: no `AF2IG_OPT*`, compilation-cache or
  sub-batch variable in its environment; `AF2IG_DIR`, `AF2_PARAMS` and your XLA / JAX variables pass through (STOCK.md,
  'How stock is run').
- `exact` — one host transfer per design, GPU-resident parameters, length-ordered inputs, programs compiled ahead of the
  loop and stored on disk, I/O overlapped with the GPU; outputs identical to `off` under `--det 1`. Use it when outputs
  must not move.
  - Its `ACTIVE` line reads `line=-fast …`: `-fast` is the upstream driver's own flag (the preset these optimizations
    build on), not the kit's `fast` mode.
- `fast` (default) — `exact` plus wider inference and template-attention sub-batches, the shared core's FlashPairformer
  triangle kernels (bf16 TriangleAttention operands), flash attention for the MSA row attention and a re-associated
  OuterProductMean; TF32 products. Use it for throughput.
- `big` — `fast` without the wider template-attention sub-batch, with a larger XLA memory pool; numerics as `fast`. Use it
  when `fast` runs out of memory; a complex that still does not fit is reported by the driver and counted short in `items=`.

**First pass versus repeat passes.** The kit's gain is in steady-state prediction time per design.

- A new complex length is compiled on first use in every mode: once per length under the compile-cache root in the kit
  modes (≈2–4 min per length, CPU-dependent: the six-complex example takes ≈8–13 min on its first pass, then loads), and in
  every process under `off`.
- So a first `fast` pass over new lengths gains little over `off` end to end, and a repeat pass gains the most.
- An image built with STOCK.md's optional compile-cache tar ships the programs that tar holds, and runs it covers load
  them instead of compiling (`loaded=… traced=0` on the `LEVER name=L13` line at exit). An image built without one,
  like route C, compiles every length on first use as above.

## Notes

- **Other cards: A100 and H200** (`--config a100`, `--config h200`). Every optimization of every mode engages on both; the
  H200 has compute capability 9.0 and uses the H100's kernel tables.
  - JAX and the kit key the compile cache by GPU device name, so a card other than the one a cache was compiled on
    compiles once per cache root on its first run, then loads: an H200 against a cache made on an H100 80GB HBM3, and
    equally another variant of one card, e.g. an A100 PCIe against a cache made on an A100-SXM4-80GB. The six-complex
    example takes ≈9–10 min on an H200 or an A100 PCIe.
  - The shared core has float32 tile tables for compute capability 8.0 and 9.0 (A100, H100, H200); on a capability without
    float32 rows there, the two triangle kernels are skipped (`skipped=L10,L11`).
- **A mode without some optimizations.** `MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]` runs a mode without the named ones
  (names as printed on the `LEVER` lines); the `ACTIVE` line then carries `levers_off=<names>`.
- **Deterministic runs.** `--det 1` selects the deterministic recipe (`XLA_FLAGS=--xla_gpu_autotune_level=0` on every
  process); `exact --det 1` equals `off --det 1` bit for bit.
- **An optimization that did not engage** during the run makes it exit 3, with the optimization named on the `EXIT` line
  (`partial=<ids>`); `--allow-partial` keeps the run's own exit code.
