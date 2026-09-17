# RFdiffusion-1 — optimization kit

Drop-in modes that make stock RFdiffusion 1.1.0 (`scripts/run_inference.py` at commit `86507b65`) faster. You call
RFdiffusion exactly as before; the kit adds a `--mode`:

- `off` — stock RFdiffusion, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.**

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation.

## Setup

Pick ONE way to get the pinned stack — stock RFdiffusion 1.1.0 and everything it needs (the 'Stack' section of
`STOCK.md` lists every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver for CUDA 12 on the host (version 550 or newer).

Route C additionally needs:

- git, gcc (for `fast`'s Triton compile) and a released CPython 3.11;
- RFdiffusion as an editable checkout (it ships no wheel) — the route-C lines below make one. Those lines are the whole
  route-C recipe; STOCK.md's Stack section is the same lines annotated, so run one or the other.

Type the first block from the directory that holds `rfdiffusion1/` and `common/`:

```bash
# A — Docker (preferred): the pinned stack, stock RFdiffusion at the pin and this kit in one image
docker build -f rfdiffusion1/environment/Dockerfile -t rfdiffusion1-kit:dev .
docker run --rm -it --gpus all -v /weights/rfdiffusion:/weights/rfdiffusion -v $PWD/out:/kit/rfdiffusion1/out -w /kit rfdiffusion1-kit:dev bash   # step 2 and Run go in this shell
# B — Apptainer / Singularity (the whole Setup for B): converts A's image (the build reads it from the local Docker daemon; the SIF then runs without one)
apptainer build rfdiffusion1-kit.sif rfdiffusion1/environment/apptainer.def
mkdir -p out/schedules jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --bind /weights/rfdiffusion:/weights/rfdiffusion --bind "$PWD/out":/kit/rfdiffusion1/out --bind "$PWD/out/schedules":/opt/rfd/schedules rfdiffusion1-kit.sif "$@"; }   # kit = B's ./run.sh · /weights/rfdiffusion: both .pt files
export WEIGHTS=/weights/rfdiffusion && kit install --weights "$WEIGHTS"     # ≈1 GB fetched or checked; read-only ok
kit check --config h100 --mode fast                                          # any --mode; --config h100|a100|h200 · Run: kit <verb> …
#     B only: add hydra.run.dir=out/off/hydra on --mode off lines (upstream's hydra log directory; the SIF itself is read-only)
# C — instead of A or B, on your own host: a fresh venv on a released CPython 3.11 — the whole route-C recipe; STOCK.md §Stack = these lines annotated plus step 2's cd and install (run one or the other)
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl build-essential git   # bare Ubuntu host (root: no sudo); skip what you have
command -v uv >/dev/null || { t=$(mktemp) && curl -LsSf -o "$t" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t" && . "$HOME/.local/bin/env"; } # uv itself, once per user (skipped when present)
uv venv --seed --managed-python --python 3.11.5 rfd1 && . rfd1/bin/activate && pip install --no-deps -r rfdiffusion1/environment/requirements.lock
git clone https://github.com/RosettaCommons/RFdiffusion && git -C RFdiffusion checkout 86507b6538f51fce57b5a72477165f03999ed7ae   # or your existing checkout at this commit
pip install --no-deps -e RFdiffusion/env/SE3Transformer -e RFdiffusion          # upstream's two installs, editable
export PYTHONHASHSEED=0 CFLAGS=-g0 CC=gcc CXX=g++                              # the same environment the image sets
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`. Under
**A** you are now in the container shell, which opens in `/kit`. Under **C** you are in your activated environment;
the first line changes into `rfdiffusion1/` if you are not there yet, and if you followed STOCK.md's Stack section it
already ran `install`, so skip that too. Run:

```bash
[ -f run.sh ] || cd rfdiffusion1             # no-op once inside · C via §Stack: skip install too
export WEIGHTS=/weights/rfdiffusion          # holds Complex_base_ckpt.pt and Base_ckpt.pt
bash run.sh install --weights "$WEIGHTS"        # fetches ≈1 GB if absent, else hash-checks; read-only ok
# export RFD_ROOT=/path/to/RFdiffusion       # only for a checkout pip did not install -e (not A or C above)
bash run.sh check --config h100 --mode fast     # dry run, any --mode: card, versions, checkout vs the pin
```

What the blocks assume:

- **Weights directory.** `WEIGHTS` names the directory holding `Complex_base_ckpt.pt` and `Base_ckpt.pt`
  (`/weights/rfdiffusion` inside the container under A and B, the same path on the host in these blocks).
  `install --weights "$WEIGHTS"` fetches them when absent and otherwise hash-checks them, so a read-only copy works.
- **Outputs under A and B.** The Run examples write to `out/…`, which is the mounted `$PWD/out` on the host.
- **Route B and the read-only image.** Route B's `kit()` also binds `$PWD/out/schedules` over `/opt/rfd/schedules`,
  where upstream writes its schedule cache inside the checkout, and `--mode off` lines need `hydra.run.dir=out/off/hydra`
  added (upstream's hydra log directory). STOCK.md's Stack section explains both.
- **The checkout and the pin check.** `install` adds the kit to the environment and runs the pin check
  (`stock/check_pins.py`), which exits 3 and names the finding when the checkout's files differ from commit `86507b65`
  or torch / dgl are off the pin; a different git HEAD over identical files is only reported. `RFD_ROOT` is needed only
  for a checkout pip did not install with `-e` (the commented line). Optional variables: `STOCK.md`.
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env` (`MODEL_OPT_TARGET_GPU`, `MODEL_OPT_STACK_KEY`,
  `MODEL_OPT_JIT_ROOT`, `TRITON_CACHE_DIR`); `design`, `check` and `warm` take it.
- **Compile cache.** The first run of `fast` compiles its Triton kernels once per machine; `bash run.sh warm --mode fast
  --out_dir DIR` does that ahead of time, and a writable `MODEL_OPT_JIT_ROOT` keeps the cache (route B's block puts it
  in `./jit` on the host).

## Run

```bash
X=(inference.input_pdb=opt/forward/fast_inference/inputs_public/insulin_target.pdb 'contigmap.contigs=[A1-115/0 80-80]' 'ppi.hotspot_res=[A59,A83,A91]' inference.num_designs=4)   # a binder to the bundled insulin-receptor target
bash run.sh design --config h100 --mode off   "${X[@]}" inference.output_prefix=out/off/des             # stock, in a clean subprocess
bash run.sh design --config h100 --mode exact "${X[@]}" inference.output_prefix=out/exact/des
bash run.sh design --config h100 --mode fast  "${X[@]}" inference.output_prefix=out/fast/des            # the default when no mode is named
bash run.sh design --config h100 --mode fast --pack 4 "${X[@]}" inference.output_prefix=out/pack/des   # 4 MPS workers on one GPU (needs nvidia-cuda-mps-control)
```

**Options.** `design` hands `scripts/run_inference.py` its own hydra overrides verbatim (`KEY=VALUE`, `+KEY`, `++KEY`,
`~KEY`). The kit's own flags are `--mode`, `--pack`, `--det` and `--dry-run`. `rfdiffusion1-opt design …` is the same
call without `run.sh`; `RFDIFFUSION1_OPT=<mode>` names the mode when `--mode` is absent.

**Outputs.**

- Where stock writes them: `<prefix>_<i>.pdb` + `.trb` and `traj/`, with `opt_manifest.json` and the run's log beside
  them (`run.log`; under `off`: `stock.log`; with `--pack`: `mps_worker*.log`).
- Give each attempt a fresh `inference.output_prefix` directory. A re-run into an existing prefix skips every design
  whose `.pdb` is already there (upstream's cautious mode, exit 0), and a failed attempt's `run.log` left there makes
  the next run exit 3.

**What a run prints.**

- Every run prints one `[rfdiffusion1-opt] ACTIVE mode=<m> … levers=…` line on stderr; `levers=` lists the optimizations
  engaged by the codes in the table at the top of `CHANGES.md` (the kit calls its optimizations 'levers').
- `check` prints `DRY-RUN …` instead.
- If a mode cannot engage, the command prints `NOT ACTIVE: <reason>` and exits 3; it never falls back to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | the run failed |
| 2 | usage error |
| 3 | not active: the mode was refused |

## Modes

- `off` — stock `scripts/run_inference.py` as released, in a clean subprocess with nothing of the kit importable.
- `exact` — one resident process per call: model loaded once, per-design constants memoised, unused CPU featurisation
  skipped, two einsums on `torch.einsum`, array-based PDB writers, one CUDA graph per forward. Identical to `off` under
  `--det 1`. Use it when outputs must not move.
- `fast` (default) — `exact` plus a dense Triton SE(3)-Transformer layer, Triton LayerNorm and TF32 GEMMs; fp32
  re-association and TF32 rounding, within stock's seed-to-seed variation, run-to-run deterministic. Use it for throughput.

`--pack K` (with `exact` or `fast`) runs K resident workers of the mode on one GPU under CUDA MPS, each on a disjoint
slice of `inference.num_designs` (a multiple of K). It needs `nvidia-cuda-mps-control`.

## Known upstream issues

Write-ups and the proposed upstream changes are in `upstream_issues/`. The kit applies no fix and has no fix flag.

| ID | behaviour in RFdiffusion 1.1.0 | in the kit |
|---|---|---|
| `RFD1-001` | under `inference.deterministic=True` a process's first design is not reproducible at a later position of another process (TorchScript's profiling executor switches to fused kernels after its first calls) | the resident driver keeps stock's position classes, so `exact` = `off` position for position |
| `RFD1-002` | the IGSO(3) schedule cache is written in place, not atomically; processes started together on an empty cache directory can read a partial file | none; run one design first (`warm` does) or give concurrent first runs their own `inference.schedule_directory_path=` |

## Notes

- **Other cards.** `--config a100` (A100) and `--config h200` (H200: compute capability 9.0, the H100's kernel settings)
  load `configs/<card>.env`. The kit reads the card at start-up, sets the base driver's `ALLOW_ANY_GPU=1` on any non-H100
  card and picks the Triton launch geometry by compute capability — every optimization of every mode engages.
- **Determinism.** `--det 1` = upstream's `inference.deterministic=True` on both the stock and the kit side (each design
  seeded with its index); `exact --det 1` equals `off --det 1` bit for bit at the same design position, on one CPU host
  class (STOCK.md, 'How stock is run').
- **Out of memory.** Step down `fast` → `exact` → `off`: a kit mode holds more of the card than stock; past its ceiling
  the driver process ends on a CUDA out-of-memory error (exit 1, `run.log`).
- **Refused by name before anything runs** (run these with `--mode off`):
  - symmetry, cyclic peptides, fold conditioning, sequence / structure inpainting;
  - typed `model.*` / `preprocess.*` / `diffuser.*` (except `partial_T`) / `hydra.*` keys;
  - another `inference.model_runner`, `inference.empty_cache_per_design`, `logging.inputs`.
