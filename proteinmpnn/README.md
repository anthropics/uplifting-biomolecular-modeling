# ProteinMPNN — optimization kit

Drop-in modes that make stock ProteinMPNN at commit `8907e667` (`protein_mpnn_run.py`; weight sets `vanilla` |
`soluble`) faster. You call ProteinMPNN exactly as before; the kit adds a `--mode`:

- `off` — stock ProteinMPNN, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster: the `.fa` bytes and the score / probability arrays, at the same seed.

There is no default mode: every command names one with `--mode off|exact` (or `PROTEINMPNN_OPT=off|exact` in the
environment). No `fast` or `big` mode and no multi-GPU route ship for this model.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock.

## Setup

Pick ONE way to get the pinned stack — stock ProteinMPNN at commit `8907e667` and everything it needs (the 'Stack'
section of `STOCK.md` lists every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver ≥ 550.

Route C additionally needs:

- `git`, `gcc` and `uv` (or another released Python 3.11 — STOCK.md's Stack section). The route-C lines below and that
  section are the same recipe, the latter with notes and the uv installer; follow one or the other.

Type the first block from the directory that holds `proteinmpnn/` and `common/`:

```bash
# A — Docker (preferred): stack + kit in one image, no weights inside
docker build -f proteinmpnn/environment/Dockerfile -t proteinmpnn-kit:dev .
docker run --rm -it --gpus all -v "$PWD/ProteinMPNN":/opt/ProteinMPNN -v "$PWD/out":/kit/proteinmpnn/out proteinmpnn-kit:dev bash
# B — Apptainer / Singularity (the whole Setup for B): converts A's image (no daemon at run time)
apptainer build proteinmpnn.sif proteinmpnn/environment/apptainer.def
mkdir -p ProteinMPNN out jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --bind "$PWD/ProteinMPNN":/opt/ProteinMPNN --bind "$PWD/out":/kit/proteinmpnn/out proteinmpnn.sif "$@"; }   # kit = B's ./run.sh · other cards: --config a100|h200
kit install --weights /opt/ProteinMPNN                  # ./ProteinMPNN: the clone, fetched or checked; read-only ok
kit check --config h100 --mode exact                    # each Run line alike: kit <verb> … (out/… = ./out)
#     this kit's one Triton kernel then caches in jit/ beside out/ (MODEL_OPT_JIT_ROOT above; else a private root in the host's /tmp) — STOCK.md §Stack
# C — instead of A or B, on your own host: a Python 3.11 environment made with uv, then every pinned package from the lock file (STOCK.md §Stack lists these same lines with notes and installs uv first — follow one or the other)
uv venv --seed --managed-python --python 3.11.5 ~/venv-mpnn && . ~/venv-mpnn/bin/activate
python -m pip install --no-deps -r proteinmpnn/environment/requirements-mpnn.lock
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …` (their
`out/…` is `./out` on the host). Under **A** you are now in the container shell, which opens in `/kit/proteinmpnn`.
Under **C** you are in your activated environment; `MPNN_DIR` is unset there, so the clone goes to `$HOME/ProteinMPNN`,
and if you followed STOCK.md's Stack section resume at `check`. Run:

```bash
[ -f run.sh ] || cd proteinmpnn                             # no-op once inside · C via §Stack: resume at check
bash run.sh install --weights "${MPNN_DIR:-$HOME/ProteinMPNN}"   # clones ≈0.2 GB if absent, else hash-checks (read-only ok)
export MPNN_DIR="${MPNN_DIR:-$HOME/ProteinMPNN}"            # image preset wins · C: your clone
bash run.sh check --config h100 --mode exact                   # dry run: mode, GPU, pins · other cards: --config a100|h200
```

What the blocks assume:

- **The clone is stock and holds the weights.** Stock is always your clone of ProteinMPNN at the pinned commit, and the
  weights come with it. `install --weights DIR` makes the clone (or checks the one already there) and refuses a
  repository weight file that is absent or differs from `stock/PINS.json`; a read-only clone works. Every command
  refuses a checkout at another commit (exit 3); a clone without `.git` runs with a note.
- **`MPNN_DIR`.** It names the clone, which must contain `protein_mpnn_run.py`, `.git` and `*_model_weights/`. Routes A
  and B set `MPNN_DIR` themselves and keep the clone in the host directory mounted at `/opt/ProteinMPNN` (`./ProteinMPNN`
  in the blocks); under C the export line sets it, and `install`'s closing `export MPNN_DIR=…` hint is that same line.
- **Outputs under A and B.** The Run examples write to `out/…`, which is the mounted `$PWD/out` on the host.
- **Inputs under B.** The Run example's `--input` is a path inside the container (the clone's own example inputs); your
  own files take absolute host paths (Apptainer binds `$HOME` and `$PWD`).
- **Route C with a different stack.** A stack that differs from the lock still runs and is named on a `STACK` line of
  `check` / `design` (a matching stack prints none). `install` ends with `check_pins: <variant>: checks package pass`
  once the clone holds everything a design pass runs.
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env` (`MODEL_OPT`, `MODEL_OPT_TARGET_GPU`,
  `MODEL_OPT_TARGET_GPU_MEM_MIB`); `design`, `check` and `warm` take it, `install` does not.
- **Compile cache.** The first `exact` run on a machine compiles the worker's one Triton kernel, cached on disk in
  a private per-user root under `${TMPDIR:-/tmp}` unless a compile-cache root is set (route B's block sets `MODEL_OPT_JIT_ROOT` to `./jit` beside `out/`;
  STOCK.md's Stack section has the details). Every run also captures its CUDA graphs at start-up.
  `bash run.sh warm --config h100 --mode exact` does both ahead of time with one small design pass on upstream's example input.
- **Driver-only hosts under B.** The SIF built from `environment/apptainer.def` sets `TRITON_LIBCUDA_PATH` itself; for
  other conversions see STOCK.md's Stack section, last paragraph.

## Run

```bash
X="--input ${MPNN_DIR:-/opt/ProteinMPNN}/inputs/PDB_monomers/pdbs"   # upstream's two-monomer example in the clone
S="--seed 37 --num_seq_per_target 8 --batch_size 8"           # protein_mpnn_run.py's own options, verbatim
bash run.sh design --config h100 --mode off   $X --out out/off   $S     # stock, in a clean subprocess
bash run.sh design --config h100 --mode exact $X --out out/exact $S     # out/exact = out/off byte for byte
bash run.sh design --config h100 --mode exact --variant soluble $X --out out/soluble $S   # the soluble weight set (default: vanilla, as upstream)
```

**Options.** `design` passes `protein_mpnn_run.py` its own options verbatim (upstream's `--seed 0` still means a random
seed). The kit's own flags are:

- `--mode`, `--variant`;
- `--bb_batch`, `--hybrid_gemm`, `--allow-partial` (see Notes);
- `--input` / `--out`, or upstream's `--jsonl_path` | `--pdb_path` and `--out_folder`. `--input` also takes a directory
  of PDB files and parses it first.

Weights kept outside the clone load with upstream's `--path_to_model_weights DIR`.

**Outputs.**

- Where stock writes them: `<out>/seqs/`, `scores/` with `--save_score 1`, `probs/` with `--save_probs 1`, plus
  `opt_manifest.json`.
- On the example each line writes `seqs/5L33.fa` and `seqs/6MRR.fa` (8 designs each) in about 5 s on an H100, and
  `cmp out/off/seqs/5L33.fa out/exact/seqs/5L33.fa` prints nothing.

**What a run prints** (on stderr).

- `check` prints `[proteinmpnn-opt] DRY-RUN mode=exact … match=yes levers=… bb_batch=16 probe=measured at launch` and
  exits 0 (`levers=` lists the optimizations; the kit calls them 'levers').
- `--mode exact` prints `[proteinmpnn-opt] ACTIVE mode=exact variant=vanilla … fallbacks=none` at launch and
  `[proteinmpnn-opt] EXIT … route=worker rc=0 … probe=PASS exit=0` at the end. The `EXIT` line and the exit code are the result.
- `--mode off` prints `[proteinmpnn-opt] NOT ACTIVE: mode off (stock route) …`, stock's own output, then
  `EXIT … route=stock … exit=0`.
- If a mode cannot engage (no CUDA device, a checkout off the pin, a failed start-up probe such as
  `<lever>: PROBE FAIL`), the command prints `NOT ACTIVE: <reason>`, exits 3 and designs nothing; it never falls back
  to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | the run failed |
| 2 | usage error |
| 3 | not active: the mode was refused |

## Modes

- `off` — stock `protein_mpnn_run.py` as released (and the stock parse helper for a PDB directory), in a clean
  subprocess with nothing of the kit importable.
- `exact` — one worker process designing 16 backbones per forward pass under CUDA graphs, the draws replayed from the
  stock RNG stream (the optimizations: `CHANGES.md`). Outputs identical to `off`. Use it for every design run. It
  refuses (exit 3) the passes it does not implement — run those with `--mode off`:
  - `--ca_only`, `--score_only`, `--conditional_probs_only`, `--unconditional_probs_only`;
  - `--tied_positions_jsonl`;
  - `--backbone_noise` above 0.

**Where the gain is.** The kit's gain is in design time over a set of backbones designed in one call. Each call also
pays ≈3.5 s of start-up (interpreter, weights, CUDA set-up) in both modes, so the two-backbone example gains little end
to end; the gain grows with backbones per call.

## Notes

- **A100** (`--config a100` loads `configs/a100.env`; on the 40 GB part also `export MODEL_OPT_TARGET_GPU_MEM_MIB=40960`).
  Every optimization of `exact` engages on compute capability 8.0 (the batched message GEMM runs in groups sized for the
  card, judged by its start-up probe); another card is reported (`gpu=… match=no`), never refused.
- **H200** (`--config h200` loads `configs/h200.env`): the H100 settings with this card's name and memory total; the
  same modes and optimizations as on H100.
- **Start-up probes.** `exact` runs only with all of its optimizations, and two start-up probes decide:
  - `hybrid_gemm: PROBE FAIL` → `REFUSED`, exit 3; `--hybrid_gemm 0` runs without that optimization (`opted_out=hybrid_gemm`);
  - `fused_draw: PROBE FAIL` means Triton could not build or reproduce the draw kernel on this host — under route C
    check `gcc` and the Python headers; under route B with an image not built from `environment/apptainer.def`, the
    `libcuda.so` note. Both are in STOCK.md's Stack section.
- **`--bb_batch K`** (default 16): backbones per forward pass — speed and GPU memory only, never the outputs; lower it
  on a small card or for very long backbones.
- **`--allow-partial`** (`PROTEINMPNN_OPT_ALLOW_PARTIAL=1`) runs the line without optimizations that cannot run here
  (`partial=<levers>`). `--det 0|1` is accepted and inert.
- **Designs per target.** `protein_mpnn_run.py` designs `--num_seq_per_target // --batch_size` whole batches per
  temperature, in both modes; the kit prints one `NOTE designs per target requested=… produced=…` line when the request
  is not a multiple of the batch size.
