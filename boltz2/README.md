# Boltz-2 — optimization kit

Drop-in modes that make stock Boltz-2 (`boltz` 2.2.1, `boltz predict`) faster and lighter on GPU memory. You call
Boltz-2 exactly as before; the kit adds a `--mode`:

- `off` — stock Boltz-2, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs; `--n_gpu P` splits one `big` prediction across P GPUs of one host.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs, up to 3,000 tokens on one GPU · `--n_gpu P` splits `big` across P GPUs of one host.

## Setup

Pick ONE way to get the pinned stack — stock `boltz` 2.2.1 and everything it needs (the 'Stack' section of `STOCK.md`
lists every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver of the 580 series or newer.

Route C additionally needs:

- a C compiler on `PATH` (Triton compiles with it); no CUDA toolkit;
- the `LD_LIBRARY_PATH` export from STOCK.md's Stack section (the recipe appends it to the venv's `activate`, so later
  shells get it on activation).

Type the first block from the directory that holds `boltz2/` and `common/`:

```bash
# A — Docker (preferred): the whole pinned stack, stock and this kit in one image
docker build -f boltz2/environment/Dockerfile -t boltz2-kit:dev .
docker run --rm -it --gpus all --ipc=host -v /weights/boltz2:/weights/boltz2 -v $PWD/out:/kit/boltz2/out boltz2-kit:dev bash   # a shell in /kit/boltz2
# B — Apptainer / Singularity (the whole Setup for B): converts the image built in A (no Docker daemon at run time)
apptainer build boltz2-kit.sif boltz2/environment/apptainer.def
mkdir -p out jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --bind /weights/boltz2:/weights/boltz2 --bind "$PWD/out":/kit/boltz2/out boltz2-kit.sif "$@"; }   # kit = B's ./run.sh · other cards: --config a100|h200
export BOLTZ_CACHE=/weights/boltz2 && kit install --weights /weights/boltz2   # /weights/boltz2: fetched/checked; read-only ok if complete
kit check --config h100 --mode fast            # each Run line alike: kit <command> …
# C — instead of A or B, on your own host (venv): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`.
Under **A** you are now in the container shell, which opens in `/kit/boltz2`. Under **C** you are in your activated
environment inside `boltz2/`; STOCK.md's Stack section already ran the install line, so continue at `export`. Run:

```bash
bash run.sh install --weights /weights/boltz2     # kit, shared core, pin check; fills and checks the cache
export BOLTZ_CACHE=/weights/boltz2             # required in every shell that runs the commands
bash run.sh check --config h100 --mode fast       # dry run: GPU, pins, cache, the mode's levers
```

What the blocks assume:

- **Weights directory.** `/weights/boltz2` in the blocks is the mount point under A and B; under C use any writable
  directory you choose. It holds `boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `ccd.pkl`, `mols.tar` and `mols/`, and must be
  writable until it is complete (STOCK.md, 'Pin' section). `BOLTZ_CACHE` must name it in every shell that runs the commands.
- **Outputs under A and B.** The Run examples write to `out/…`, which is the mounted `$PWD/out` on the host.
- **Inputs under B.** Give your own `--input` as an absolute path under `$HOME` or `$PWD`; the commands run in
  `/kit/boltz2` inside the image.
- **Route C with an existing environment.** An environment that already has `boltz` 2.2.1 installed unmodified on the
  lock's stack also works. The pin check refuses another version or a modified `boltz` tree (exit 3).
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env`; every command takes it. A GPU below compute
  capability 8.0 is refused.
- **Compile cache.** The first run of a kit mode compiles its kernels once per machine. `bash run.sh warm --config <card> --mode <mode> --out
  DIR` does that ahead of time (note its flag is `--out`; `DIR` receives the bundled inputs' predictions and the worker log).
  `MODEL_OPT_JIT_ROOT` keeps the cache between runs; route B's block puts it in `./jit` on the host. The first run in a fresh cache directory
  also performs a one-time numerical self-check of the fused kernels on your GPU (seconds on H100; up to ~30 s on A100) before serving them;
  later runs skip it.
- **Mode without `--mode`.** `BOLTZ2_OPT=<mode>` names the mode when the flag is absent. Optional variables: `STOCK.md`.

## Run

```bash
Y=inputs/1BRS_barnase_barstar.yaml            # bundled: barnase + barstar, single sequence, 199 tokens
bash run.sh pred --config h100 --mode off   --input $Y --out_dir out/off      # stock, in a clean subprocess
bash run.sh pred --config h100 --mode exact --input $Y --out_dir out/exact
bash run.sh pred --config h100 --mode fast  --input $Y --out_dir out/fast     # the default when no mode is named
bash run.sh pred --config h100 --mode big --input $Y --out_dir out/big
bash run.sh pred --config h100 --mode big --n_gpu 2 --input $Y --out_dir out/big_x2    # 2, 4 or 8 GPUs of one host
```

**Options.** `pred` hands `boltz predict` its own options verbatim (upstream's spelling and defaults;
`python -m boltz2_opt pred --help` lists them). The kit's own flags are:

- `--config`, `--mode`;
- `--input` / `--out_dir` (upstream takes the input positionally);
- `--n_gpu`;
- `--seeds a,b` — every input at every seed in one worker;
- `--allow-partial` — lets a run continue without one named optimization, and says so;
- `--no-compile` — accepted on every command, inert here (no mode uses `torch.compile`);
- `--det 1`, on `off` only — adds `--num_workers 1 --no_kernels`; not part of any identity comparison.

`boltz2-opt pred --mode <mode> …` is the same call without `run.sh`.

**Inputs.** `--input` takes YAML or FASTA files, or a directory of them. `--mode off` takes one path per call, as
`boltz predict` does.

**Outputs.**

- Kit modes write stock's file set per (input, seed) under `<out_dir>/by_seed/<name>/s<seed>/`, plus the worker log
  `<out_dir>/pred_worker.log`.
- `off` writes `boltz predict`'s own `<out_dir>/boltz_results_<name>/` (same files, stock's layout).

**What a run prints.**

- A kit mode prints `[boltz2-opt] ACTIVE mode=<mode> … levers=…` once engaged (the kit calls its individually switchable
  optimizations 'levers'), and ends with one `LEVER name=… state=…` line per optimization and `[boltz2-opt] EXIT mode=<mode> … rc=<rc>`.
- `check` ends with `DRY-RUN ok: …`.
- `--mode off` prints `NOT ACTIVE: mode off: stock …` and runs stock, exiting with `boltz predict`'s own code.
- If a mode cannot engage, the command prints `[boltz2-opt] NOT ACTIVE: <reason>` and exits 3; it never falls back to
  stock silently (`--allow-partial` is the one deliberate exception, and it says which optimization it left out).

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | the prediction failed |
| 2 | usage error |
| 3 | not active: the mode was refused before any work, or the A100 `exact` case under Notes |
| 5 | one of upstream's two cuEquivariance kernels (triangle attention / multiplication) that the mode counts on was absent or fell back mid-run (`KERNELS-REFUSED … cueq_…=<state>`) |

**First run.** A first `pred` per mode on the bundled example takes ½–1 min with the one-time compile; compare timings on a rerun.

## Modes

- `off` — stock `boltz predict` as released, in a clean subprocess with nothing of the kit importable.
- `exact` — Pairformer eval-mode shortcuts, fused pair-track blocks around stock's own kernels, a CUDA-graphed diffusion
  sampler with its step-invariant work hoisted, input prefetch and a background writer. Outputs identical to `off`:
  `pred --mode off` and `pred --mode exact` on the same card with the same `--seed` (stock's kernels on, no `--det`) write
  byte-identical `.cif`, confidence `.json` and `plddt`/`pae` `.npz` files. Use it when outputs must not move.
- `fast` (default) — `exact` plus the shared core's FlashPairformer triangle kernels in bf16 on one resident pair tensor,
  fused MSA-module kernels and a fused bf16 token-transformer step; bf16 re-association, within stock's seed-to-seed
  variation. Use it for throughput.
- `big` — `fast`'s pair track, row-chunked diffusion conditioning, early frees and no CUDA graphs; numerics as `fast`.
  `--n_gpu 2|4|8` row-shards the pair representation over P GPUs of one host (refused by name under another mode or with
  fewer GPUs visible). Use it when `fast` runs out of memory.

## Notes

- **Where the gain is.** The kit's gain is in prediction time. Each call also pays ≈25 s of start-up, checkpoint load
  and featurization in every mode, so the 199-token example gains less end to end than its prediction time does — more
  with larger inputs, or with several inputs or seeds per call.
- **A100** (`--config a100`):
  - `msa_pwa_exact` (`exact`) and `atom_fused` (`big`) are switched off on this card (`card_off=<lever>:<reason>` on the `ACTIVE` line);
  - `exact`'s fused pair blocks serve only the row range their `LEVER` lines name;
  - `exact` completes but ends `NOT ACTIVE` (exit 3) below 64 tokens, at 1,024–1,099 tokens and in narrow bands above
    (STOCK.md), where its pair transition is not bitwise with this card's cuBLAS — use `fast` or `big` there.
- **H200** (`--config h200`): compute capability 9.0 like the H100, so every mode runs exactly as on the H100.
- **`--use_potentials`** switches the graphed sampler off for that run (`LEVER name=rollout state=off reason=use_potentials`).
  An input declaring the affinity property gets upstream's affinity pass, unchanged, after its structure pass.
- **Out of memory.** Under `fast` → use `big`; still out of memory → `big --n_gpu P`.
