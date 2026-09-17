# ColabDesign — optimization kit

Drop-in modes that make BindCraft's binder-hallucination design step faster (`binder_hallucination`, BindCraft at
commit `efb5bfeb` on stock ColabDesign 1.1.3 at commit `e31a56fe`, AlphaFold-Multimer v3 weights). You call the design
step exactly as before — one trajectory per call on one GPU; BindCraft's later redesign, reprediction, relax and filter
stages are not part of the kit. The kit adds a `--mode`:

- `off` — stock, exactly as released ('stock' below always means the unmodified upstream release).
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.** On a single trajectory these
  differences can flip upstream's acceptance filters (e.g. an early stop on the initial-pLDDT check); compare
  acceptance rates over several seeds rather than one run.

A mode is a set of named, individually switchable optimizations; what each one changes: `CHANGES.md`. Exact versions,
the pinned software stack and all variables: `STOCK.md`. How the three setup routes (A — Docker, B — Apptainer,
C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation.

## Setup

Pick ONE way to get the pinned stack — stock ColabDesign 1.1.3 at commit `e31a56fe`, Python 3.10, jax 0.6.0 CUDA 12
build, dm-haiku 0.0.17 (the 'Stack' section of `STOCK.md` lists every pin): **A — Docker**, **B — Apptainer**, or
**C — a Python venv on your own host**.

Every route needs:

- an NVIDIA GPU with a CUDA-12-series driver. The `fast` kernels target compute capability 8.0 or newer; the card is
  named on the `ACTIVE` line, never refused.

Route C additionally needs:

- micromamba (or conda / mamba), git and a C compiler; STOCK.md's Stack section is the complete route-C recipe.

Type the first block from the directory that holds `colabdesign/` and `common/`:

```bash
# A — Docker (preferred): the whole pinned stack, stock and this kit
docker build -f colabdesign/environment/Dockerfile -t colabdesign-kit:dev .
docker run --rm -it --gpus all -v /weights/af2:/weights/af2 -v "$PWD/out":/kit/colabdesign/out -w /kit colabdesign-kit:dev bash   # a shell; the next block runs in it
# B — Apptainer (the whole Setup for B): build the .sif once where Docker runs (converts A's image); the cluster then needs only Apptainer
apptainer build colabdesign.sif colabdesign/environment/apptainer.def
mkdir -p out; kit() { apptainer run --nv --bind /weights/af2:/weights/af2 --bind "$PWD/out":/kit/colabdesign/out colabdesign.sif "$@"; }   # kit = B's ./run.sh · other cards: --config a100|h200
export COLABDESIGN_PARAMS_DIR=/weights/af2 && kit install --weights /weights/af2   # /weights/af2/params/*.npz fetched or checked; read-only ok
kit check --mode fast                       # each Run line alike: kit <verb> … (out/… = ./out)
# C — instead of A or B, on your own host: STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`.
Under **A** you are now in the container shell, which opens in `/kit`; the next block runs in it. Under **C** you are
in your activated environment; if STOCK.md's Stack section set it up, you are already inside `colabdesign/` with the
install line done, so continue at `export`. Run:

```bash
[ -f run.sh ] || cd colabdesign             # no-op once inside · C via §Stack: resume at export
bash run.sh install --weights /weights/af2     # --weights: ≈5 GB if absent, else hash-check; read-only ok
export COLABDESIGN_PARAMS_DIR=/weights/af2  # must contain: params/params_model_{1..5}_multimer_v3.npz
bash run.sh check --mode fast                  # seconds, designs nothing
```

What the blocks assume:

- **Weights directory.** `/weights/af2` in the blocks; it must contain `params/params_model_{1..5}_multimer_v3.npz`.
  `COLABDESIGN_PARAMS_DIR` must name it (`--params-dir DIR` replaces the variable on any command).
  - `install --weights` fetches the files if absent, else hash-checks what is there, so a complete directory may be read-only.
  - `install` also fetches BindCraft's `dssp` executable, which the design step runs, from the BindCraft repository at the
    pinned commit into `stock/src/bindcraft/functions/`, checks its SHA-256 and size against `stock/PINS.json` and stops when
    the bytes differ; a copy already in place is checked, not fetched again.
- **Outputs under A and B.** The `out/…` of the Run examples is the host's `./out`.
- **What `check` does.** It reads package metadata against the pins, names the card and digests the weights (seconds; it
  designs nothing). It prints the mode's `ACTIVE` line and `weights=<file> sha256=… (pinned)` — or `weights=not checked`,
  still exit 0, when no params root is visible.
- **First run compiles.** The first run of a kit mode compiles (XLA) once per machine and token count (≈140 s at the
  example's size). `bash run.sh warm --out DIR [--mode M]` does it ahead of time for the example
  (`--starting-pdb/--chains/--binder-len` for another case).
- **Compile cache.** jax's persistent cache keeps the compiled programs (STOCK.md, 'Variables'). Route B keeps it in
  `/tmp/model_opt_jit-uid<uid>`, the host's `/tmp`; relocated with `MODEL_OPT_JIT_ROOT` it recompiles once, then persists.

## Run

```bash
T=stock/src/bindcraft/example/PDL1.pdb                     # BindCraft's bundled example (chain A, hotspot 56, binder 65)
A="--starting-pdb $T --chains A --binder-len 65 --target-hotspot-residues 56 --seed 0"
bash run.sh design --mode off   $A --out out/off              # stock, in a clean subprocess
bash run.sh design --mode exact $A --out out/exact
bash run.sh design --mode fast  $A --out out/fast             # the default when no mode is named
```

**Options.** `design` hands `binder_hallucination` BindCraft's own settings; the kit's own flag is `--mode`.

- `--starting-pdb`, `--chains`, `--binder-len`, `--target-hotspot-residues` (BindCraft's grammar; omit for none), `--seed`, `--binder-name`;
- `--advanced FILE` / `--filters FILE` (defaults `default_4stage_multimer.json`, `default_filters.json`);
- `--params-dir`.

Other ways to make the same call:

- `colabdesign-opt design --mode <mode> …` — the same call without `run.sh`;
- `COLABDESIGN_OPT=<mode> python your_bindcraft_run.py` — engages a mode in an unchanged script at `import colabdesign`;
- `colabdesign_opt.enable("<mode>")` before the first `mk_afdesign_model(...)` — the same from Python.

**Outputs** (under `--out`). `design.pdb`, `design.fasta`, `trajectory.jsonl`, `run.log`, beside BindCraft's own tree
and `failure_csv.csv`. BindCraft's filters may stop or reject a trajectory (`terminate=terminated:<reason>` on the
`[run]` line; exit code 0). Give each run a fresh `--out`: BindCraft refuses to move a design over an existing one.

**What a run prints** (on stderr).

- A kit mode prints `[colabdesign-opt] ACTIVE mode=<m> … gpu=<card> levers=<a+b+…> skipped=<…> arm=kit …` — the
  `ACTIVE` line, naming the card and the optimizations engaged (the kit calls its individually switchable optimizations
  'levers') — then one `LEVER name=<id> state=on|skipped|off …` line per optimization, and
  `EXIT rc=0 mode=<m> levers=… out=<dir>`.
- `off` prints `ACTIVE mode=off … levers=none … arm=stock …` and `ENV-CLEAN ok: …`, the proof that the stock
  subprocess saw nothing of the kit.
- If a mode cannot engage — ColabDesign off the pin, kit or shared core not installed, an unknown mode word, an
  optimization that served no call — the command exits 3 with `[colabdesign-opt] NOT ACTIVE reason=<reason>`; it never
  falls back to stock silently. One optimization that cannot run on the machine is skipped by name
  (`state=skipped reason=cannot_run`) and the rest of the mode runs.
Compare modes on a second run; the first one compiles (Setup).

**Exit codes.**

| code | meaning |
|---|---|
| 0 | ok |
| 1 | failed |
| 2 | usage error |
| 3 | not active |

## Modes

- `off` — BindCraft's design step on stock ColabDesign as released, in a clean subprocess with nothing of the kit importable.
- `exact` — jax's persistent compilation cache, thread-parallel compilation and persisted compiled executables for the
  design programs, and the recycle features kept on the device between steps; outputs identical to `off` at the same
  seed. Use it when outputs must not move.
- `fast` (default) — `exact` plus the FlashPairformer kernels of the shared core (`common/opt_core`, the runtime all kits
  share: attention, triangle multiplication, Transition), its LayerNorm kernel, and unchunked design executables above
  384 tokens; bf16 re-association, deterministic run to run (CHANGES.md). Use it for throughput.

## Notes

- **Other cards.** `--config a100|h200` only sets a label; the card is read at start-up (`gpu=…` on the `ACTIVE` line).
  - Every optimization engages on the **A100** (compute capability 8.0) and on the **H200** (9.0, like the H100) as on
    the H100.
  - On any card, call shapes the shared core has no kernel for run XLA's own operation; they are counted on the
    `LEVER` line (`fallback_by=cell_xla:n`) and named by the `NAMED_FALLBACK` / `UNCOVERED_CELL` lines of the exit
    census — reports, not errors; the outputs each mode documents are unaffected.
  - A compile cache serves only the GPU product it was made on (the `gpu=` word on the `ACTIVE` line); any other card
    or A100 variant compiles on its first run, once per machine and token count (≈2–4 min), then runs warm.
- **Ablation.** `--mode fast-no-<lever>[-no-<lever>…]` runs `fast` without the named optimizations for A/B attribution
  (`ablated=<names>` on the `ACTIVE` line, `class=ablation`); the names are the `name=` ids of the `LEVER` lines (CHANGES.md).
- **Cold start.** Without the kit every process compiles (≈2 min) and every trajectory re-traces the model. `exact` and
  `fast` pay the compile once per machine and token count and the trace once per configuration (a later process loads
  the stored executables in seconds).
  - So one trajectory per process gains mostly the compile and trace removed, and several in one process gain the
    per-step speed as well.
- **Where the gain is.** The kit's gain is in design-step time at steady state and grows with token count; the example's
  180 tokens sit at the low end.
