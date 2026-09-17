# Caliby — optimization kit

Drop-in modes that make stock Caliby 0.1 (`caliby.load_model(...).sample` / `.ensemble_sample`) faster, in both
upstream protocols. You call Caliby exactly as before; the kit adds a `--mode` and takes the protocol as `--variant`:

- `single` — design on the input structure (the default variant);
- `ensemble32` — design on the input plus 32 Protpardelle-1c conformers of it.

The modes:

- `off` — stock Caliby, exactly as released ('stock' below always means this unmodified upstream release).
- `fast` (serves `single`) — faster, with small documented numeric differences. **The default.**
- `exact` (serves `ensemble32`) — faster, outputs identical to stock.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `fast` (serves `single`) small documented numeric differences, faster than stock · `exact` (serves `ensemble32`) identical outputs, faster than stock.

## Setup

Pick ONE way to get the pinned stack — stock Caliby 0.1, its two upstream dependencies and everything they need (the
'Stack' section of `STOCK.md` lists every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver that runs CUDA 12.4 (version 550 or newer);
- the weights that `install --weights` fetches (below).

Route C additionally needs:

- a released CPython 3.12;
- for the two kit modes, a C compiler on `PATH` (Triton compiles one kernel on first use).

Type the first block from the directory that holds `caliby/` and `common/`:

```bash
# A — Docker (preferred): the pinned stack, stock and this kit in one image
docker build -f caliby/environment/Dockerfile -t caliby-kit:dev .
docker run --rm -it --gpus all -v /weights/caliby:/weights/caliby -v $PWD/out:/kit/caliby/out caliby-kit:dev bash   # a shell in /kit/caliby; out/… = ./out on the host
# B — Apptainer (the whole Setup for B; `kit` below: a shell function = B's bash run.sh): build the .sif once where Docker runs (converts A's image); the cluster then needs only Apptainer
apptainer build caliby-kit.sif caliby/environment/apptainer.def
mkdir -p out jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --bind /weights/caliby:/weights/caliby --bind "$PWD/out":/kit/caliby/out caliby-kit.sif "$@"; }   # B's ./run.sh · /weights/caliby: caliby/, protpardelle-1c/
export MODEL_PARAMS_DIR=/weights/caliby && kit install --weights /weights/caliby   # /weights/caliby: fetched or hash-checked; read-only ok
kit check --config h100 --mode fast                # Run lines alike: kit <verb> … · cards: --config a100|h200
# C — instead of A or B, on your own host (venv): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`. Under
**A** you are now in the container shell, which opens in `/kit/caliby`. Under **C** you are in your activated
environment inside `caliby/`; STOCK.md's Stack section already ran the install line, so continue at `export`. Run:

```bash
[ -f run.sh ] || cd caliby                       # no-op once inside · C via §Stack: resume at export
bash run.sh install --weights /weights/caliby       # --weights: 0.5 GB if absent, else hash-check; read-only ok
export MODEL_PARAMS_DIR=/weights/caliby          # required: caliby/*.ckpt, protpardelle-1c/, proteinmpnn/
bash run.sh check --config h100 --mode fast         # dry run: "would activate" (rc 0) / "would refuse: …" (rc 3)
```

What the blocks assume:

- **Weights directory.** `MODEL_PARAMS_DIR` is required and names the weights root: `caliby/*.ckpt`,
  `protpardelle-1c/` and `proteinmpnn/` inside (`/weights/caliby` in the container under A and B, the same path on the
  host in these blocks). `install --weights` fetches what is absent and hash-checks what is there, so a read-only copy works.
- **Outputs under A and B.** The Run examples write to `out/…`, which is the mounted `$PWD/out` on the host.
- **Inputs under B.** An input outside `$HOME` and `$PWD` needs its own `--bind` in the `kit()` function.
- **Pin check.** `install` and every command run it (`stock/check_pins.py`): the three upstream packages must be
  pip-installed from the archives in `stock/` or, with `git`, from their repositories at the pinned commits. Any other
  install is refused by name (exit 3); `protpardelle` may be absent unless `ensemble32` is run. A kit mode also refuses
  an installed tree in which the upstream files its optimizations replace differ from the pinned bytes (`NOT STOCK: …`).
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env` — the target GPU, the offline switches and the
  cache locations, each listed in STOCK.md's 'Variables' section; every command below takes it.
- **Compile cache.** A session's first kit-mode run compiles the Triton kernel (≈3 s, once per cache root) and
  captures CUDA graphs per process. `bash run.sh warm --config h100 [--mode M] [--variant V]` does that ahead of time;
  `TRITON_CACHE_DIR` keeps the kernel between runs (route B's block keeps it under `./jit` on the host by setting
  `MODEL_OPT_JIT_ROOT`).
- **`ld: cannot find -lcuda`.** A kit mode that stops with this message (only with a SIF not built from
  `environment/apptainer.def`) has its remedy in STOCK.md's Stack section.

## Run

```bash
X=opt/forward/xattempt_addon/tests/public_inputs/7xhz.cif                                              # one PDB entry that ships with the kit
bash run.sh design --config h100 --mode off   --input $X --out_dir out/off                                # stock, in a clean subprocess
bash run.sh design --config h100 --mode fast  --input $X --out_dir out/fast                               # the default when no mode is named
bash run.sh design --config h100 --mode off   --variant ensemble32 --input $X --out_dir out/off32
bash run.sh design --config h100 --mode exact --variant ensemble32 --input $X --out_dir out/exact32
bash run.sh design --config h100 --mode fast  --input $X --out_dir out/fast8 --num_seqs_per_pdb 8 --batch_size 16 --seed 0   # upstream's keywords pass through
```

**Options.** `design` passes upstream's own keywords through under their own names, only when given (`STOCK.md` lists
each with upstream's default). The kit's own flags are:

- `--mode`, `--variant`;
- `--seed S` — seeds the run;
- `--det 0|1` — `1` = torch's deterministic kernels (STOCK.md, 'How stock is run').

The checkpoint is upstream's `--model_name`: `caliby` by default; `soluble_caliby` or `soluble_caliby_v1` for SolubleCaliby.

Without `run.sh`: `caliby-opt design --mode <mode> …`. In your own script: `. configs/h100.env`, then `CALIBY_OPT=<mode>`
(plus `CALIBY_VARIANT=ensemble32` for `exact`) engages the mode at the first `import caliby`. Write results before the
interpreter exits, since an optimization that could not run ends that process with status 3 from an exit hook.

**Inputs.** `--input` takes `.pdb` / `.cif` files (also gzipped), directories, or `.txt` / `.list` files of paths.

**Outputs** (under `--out_dir`).

- `raw/samples/<stem>_sample<i>.cif` (`ensemble32`: `raw/<stem>/samples/`, conformers under `ensembles/<stem>/`);
- `seq_des_outputs.csv`, `cleaned/`, `timing.json`;
- `opt_manifest.json`, and `stock_env_proof.json` under `off`.

**What a run prints** (on stderr).

- A kit mode prints `[caliby-opt] ACTIVE mode=fast variant=single … gpu=… switches=…` and closes with
  `[caliby-opt] design mode=… rc=0 out=… designs=<n> wall=<s>s`.
- `off` prints `[caliby-opt] STOCK mode=off variant=… switches=none` instead (rc 0).
- If a mode cannot engage, or one of its optimizations cannot run on this machine or input, the command prints
  `[caliby-opt] NOT ACTIVE: <reason>` and exits 3; it never falls back to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | the run failed, or fewer designs were written than requested |
| 2 | usage error: an unknown mode or variant, or `--mode exact --variant single` |
| 3 | not active: the mode was refused |

**First run.** Compare modes from a session's second run; the first one compiles.

## Modes

- `off` — stock `clean_pdbs` → `load_model` → `sample` (or `generate_ensembles` → `ensemble_sample`) in a clean
  subprocess, every kit switch proven absent.
- `fast` (default; `single`) — Potts sampler and parameters kept on the GPU (CUDA graphs, sparse couplings), a fused
  Triton low-complexity penalty, a batch's sequences sampled concurrently, CIFs written in the background.
  Deterministic; last-bit Potts-energy differences from stock.
- `exact` (`ensemble32` only; on `single` it is refused by name, exit 2) — the `fast` optimizations plus deterministic
  tied aggregation, parallel conformer featurisation and a cached, sync-free Protpardelle-1c pass with a faster PDB
  writer. Outputs identical to `off`. `--mode fast --variant ensemble32` runs this same optimization set.

## Notes

- **Where the gain is.** The gain is over whole runs — cleaning, model load and design — of many backbones at several
  sequences each. A single small input is mostly upstream's structure cleaning (≈15 s) and model load (≈2 s) in every
  mode, so the one-sequence example above takes about as long (≈28 s) under `off` and `fast`, and a few sequences on one
  input gain little; the gain grows with inputs and sequences per call.
- **Other cards.** `--config a100` and `--config h200` load `configs/a100.env` / `configs/h200.env`, which set the
  card's `MODEL_OPT_TARGET_GPU` and then load `configs/h100.env`; the same optimizations engage as on H100. The GPU is
  compared with `MODEL_OPT_TARGET_GPU` and a mismatch is reported on the `ACTIVE` line (`MISMATCH: …`), never refused.
- **Out of memory.** Running out of GPU memory ends the run with torch's error in every mode, `off` included — never a
  fallback. `ensemble32` needs several times the memory of `single` on the same input; `--pp_batch_size` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` are the knobs for large assemblies.
- **Symmetry-tied designs.** `--pos_constraint_csv` with `symmetry_pos` takes upstream's dense Potts-parameter step
  inside both kit modes, said once per call as
  `[caliby-opt] LEVER name=CALIBY_FAST_POTTS_PARAMS state=skipped reason=symmetry_dense_J …` (a `LEVER` line reports one
  optimization; the kit calls them 'levers'). That step's outputs and the exit code are unchanged.
