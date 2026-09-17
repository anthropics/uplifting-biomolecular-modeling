# Genie 3 — optimization kit

Drop-in modes that make stock Genie 3 at commit `d77ae5ac` (`genie3 generate`) faster and lighter on GPU memory. You
call Genie 3 exactly as before; the kit adds a `--mode`:

- `off` — stock Genie 3, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.**

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation.

## Setup

Pick ONE way to get the pinned stack — stock Genie 3 at commit `d77ae5ac` and everything it needs (the 'Stack' section
of `STOCK.md` lists every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver ≥ 525.60 on the host.

Route C additionally needs:

- git and a C compiler (STOCK.md's Stack section says why and lists the packages).

Type the first block from the directory that holds `genie3/` and `common/`:

```bash
# A — Docker (recommended: the whole pinned stack, stock and this kit in one image)
docker build -f genie3/environment/Dockerfile -t genie3-kit:dev .
docker run --rm -it --gpus all -v /weights:/weights -v $PWD/out:/kit/genie3/out -w /kit genie3-kit:dev bash   # host /weights holds genie3/pretrained/v1/…; shell in /kit
# B — Apptainer (the whole Setup for B): build the .sif once where Docker runs (converts A's image); the cluster then needs only Apptainer
apptainer build genie3-kit.sif genie3/environment/apptainer.def
mkdir -p out jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --bind /weights:/weights --bind "$PWD/out":/kit/genie3/out genie3-kit.sif "$@"; }   # kit = B's ./run.sh · other cards: --config a100|h200
export GENIE3_WEIGHTS=/weights/genie3/pretrained/v1 && kit install --weights /weights/genie3   # /weights/genie3: pretrained/v1 fetched/checked; read-only ok
kit check --config h100 --mode fast                    # Run lines alike: kit <verb> …; off: add --writable-tmpfs
# C — instead of A or B, on your own host (venv): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`. Under
**A** you are now in the container shell, which opens in `/kit`. Under **C** you are in your activated environment;
the first line changes into `genie3/` if you are not there yet, and you skip what STOCK.md's Stack section already ran. Run:

```bash
[ -f run.sh ] || cd genie3                             # from the parent dir; a no-op once inside the kit dir
bash run.sh install --weights /weights/genie3             # --weights: 0.4 GB if absent, else hash-check; read-only ok
export GENIE3_WEIGHTS=/weights/genie3/pretrained/v1    # the dir holding config.yaml and checkpoints/ (every route)
bash run.sh check --config h100 --mode fast              # dry run: mode, GPU, pins, weight digests; no model load
```

What the blocks assume:

- **Weights directory.** `GENIE3_WEIGHTS` names the directory holding `config.yaml` and
  `checkpoints/step=600000.ckpt` — `DIR/pretrained/v1` after `install --weights DIR` (fetched when absent, hash-checked
  otherwise; a read-only copy works). Unset, it means `<checkout>/pretrained/v1`. `install` without
  `--weights` downloads nothing. Under A and B the host directory mounted at `/weights` holds `genie3/pretrained/v1/…`.
- **Outputs under A and B.** The Run examples write to `out/…`, which is the mounted `$PWD/out` on the host.
- **`--mode off` under B.** Add `--writable-tmpfs` to the `apptainer run` options for `off` runs (the block's comment);
  STOCK.md's 'Route B' section explains why and gives the alternative bind.
- **Route C and the checkout.** Route C leaves `GENIE3_ROOT` naming the stock checkout. The pin check compares that
  checkout with the shipped source archive file by file: `install` refuses one whose bytes differ, naming the files
  (exit 3), while a different torch / lightning / numpy is reported, not refused. Other variables: `STOCK.md`.
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env` (sets `MODEL_OPT` and `MODEL_OPT_TARGET_GPU`,
  derives `MODEL_OPT_STACK_KEY` when unset; `GENIE3_ROOT` / `GENIE3_WEIGHTS` pass through). `design`, `check` and
  `warm` take it (and `--mode <mode>`); none requires it (see Notes, 'Cards').
- **What compiles.** Only `fast` compiles: the shared core's Triton kernel once per machine, and the `torch.compile`d
  core once per distinct complex length and batch size (about 20 s and a set of cache files per new length).
  Fixed-length problems compile once; binder problems with a variable binder length — the example's problem draws
  each design's binder length from its range — compile once per length encountered and reuse the cache thereafter,
  so a later pass compiles only the lengths no earlier pass met. `off` and `exact` compile nothing; each kit-mode
  process records CUDA graphs (the denoiser's GPU work, recorded once and replayed each step) in seconds, with no
  compiler involved.
- **Compile cache.** `bash run.sh warm --out_dir DIR` runs one design of the example ahead of time (mode `fast` by
  default: the Triton kernel and that design's length are compiled; `--mode exact` runs it end to end, compiling
  nothing). A writable `MODEL_OPT_JIT_ROOT` keeps the caches; route B's block puts it in `./jit` on the host.

## Run

```bash
X=${GENIE3_ROOT:-/opt/genie3}/examples/binder_design/experiment.yaml   # upstream's example; /opt/genie3 = the A/B image's checkout
bash run.sh design --mode off   --input $X --out_dir out/off      # stock, in a clean subprocess
bash run.sh design --mode exact --input $X --out_dir out/exact
bash run.sh design --mode fast  --input $X --out_dir out/fast     # the default when no mode is named
bash run.sh design --mode fast  --input $X --out_dir out/fast8 --batch_size 8         # generation.dataset.batch_size for this pass
bash run.sh design --mode fast  --out_dir out/s0 -- -c $X --num-shards 2 --shard-id 0  # upstream's generate argv, verbatim after `--`
```

**Options.** `design` hands `genie3 generate` its own command line (`--verbose --log-dir --num-devices --shard-id
--num-shards`), or everything after `--` verbatim. The kit's own flags are:

- `--mode`;
- `--input` | `-c` — the request file;
- `--out_dir`;
- `--n` — designs per problem (the request's `n_sample`);
- `--selections` — comma-separated problem keys of the request's dataset;
- `--seed`, `--batch_size`, `--det`;
- `--tag` — a label for this pass in `opt_manifest.json` and on the `ready` line (default `run`).

Without `run.sh`: `genie3-opt design …` is the same call, and `GENIE3_OPT=<mode> genie3 generate -c <request> …` engages
a mode from the unchanged stock command line (`GENIE3_OPT_AUTOLOAD=0` disables that route).

**Outputs.**

- Where stock writes them: `<out_dir>/<problem>/pdbs/<problem>_<i>.pdb` (unconditional requests: `<out_dir>/pdbs/`),
  plus the composed `request.yaml` and `opt_manifest.json`. Without `--out_dir` the request's own `paths.rootdir` is used.
- A second pass into the same `--out_dir` overwrites PDB files of the same name — except a sharded pass (`--num-shards`)
  whose shard is already marked complete there: it generates nothing and exits 0, as upstream does.

**What a run prints** (on stderr).

- `check` prints `[genie3-opt] check mode=<m> would_activate=True gpu=… pinned_stack=… weights=… line=…` and exits 0.
- A kit mode prints `[genie3-opt] ACTIVE mode=<m> attach=driver tier=<1|2> … gpu=<card>(sm<cc>,<cc>|<triton>) levers=<ids> line=…`,
  then `RUN …`, `ready mode=<m> t=<s> … first=<first PDB>`, and at exit `EXIT pid=… source=timings-json …` (exit 0).
  On the `ACTIVE` line: `tier=1` means outputs byte-identical to stock, `tier=2` means `fast`'s documented differences;
  `sm<cc>` and `<cc>|<triton>` are the card's compute capability and the Triton version, the key of the kernel's tile
  table; `levers=` lists the optimizations engaged (the kit calls them 'levers').
- `--mode off` prints `ACTIVE mode=off attach=stock-cli … levers=none line=genie3 generate …` and
  `STOCK proof=ok pinned=True …`, meaning stock ran as asked.
- If a mode cannot engage, the command prints `[genie3-opt] NOT ACTIVE: <reason>` and exits 3; it never falls back to
  stock silently. A `LEVER lever=<id> state=skipped …` line, by contrast, names a single optimization that did not
  apply to the request.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | the run failed or is incomplete |
| 2 | usage error |
| 3 | not active: the mode was refused |

**First run.** A length's first `fast` run includes compilation: time modes on a repeat run of the same lengths (the
same `--seed`), `fast` with `--batch_size 8`.

## Modes

- `off` — stock `genie3 generate -c <request> --log-dir <out_dir>/logs` from the pinned checkout, in a clean subprocess
  with nothing of the kit importable.
- `exact` — one persistent process, a sync-free DDIM step, CUDA-graph replay of the denoiser core with step-invariant
  featuriser terms hoisted, pair intermediates freed as consumed. PDB files identical to `off` at the same batch size
  and seed. Use it when outputs must not move.
- `fast` (default) — `exact` plus per-design pair-transition chunks, TF32 matmuls, the shared core's fused triangle
  multiplication (bf16 operands) and a `torch.compile`d core inside the graph; within stock's seed-to-seed variation.
  Use it for throughput (`--batch_size 8`).

## Notes

- **Where the gain is.** The kit's gain is in sampling time per design, against stock run at batch 8 with
  `generation.compile` on. A pass also spends ≈10–15 s per process on start-up and model set-up in every mode, so a
  small batch of designs gains less end to end than a large one — the gain grows with designs per call, `fast` ahead
  of `exact`.
- **Cards.** `--config h100|a100|h200` on any command (`check`, `design`, `warm`) loads `configs/<card>.env` (checkout,
  weights, a target-card guard, the cache key). It is optional: without it nothing is loaded and the kit reads the card
  itself at start-up (compute capability and memory), which is why the Run lines carry none; the `check` and `ACTIVE`
  lines name the card found (`gpu=…`), not the config. Every optimization of every mode engages on A100 (compute
  capability 8.0; the triangle-multiplication kernel has its own tiles for it) and on H200 (9.0, as H100). No other
  card is configured.
- **Several GPUs.** Use upstream's dataset shards, `--num-shards M --shard-id K`, one pass per GPU, on every mode (with
  `--seed` / `--det 1`, give each shard its own `--seed`). `--num-devices` > 1, beam search and `predict_sidechain` are
  refused by name on `exact` / `fast` (exit 3); `--mode off` runs them.
- **Batch too large.** A batch that does not fit the card is refused before it runs
  (`[genie3-opt] CAPACITY … verdict=refused … suggest_batch=<b>`, exit 3): pass `--batch_size <b>`.
- **Determinism.** `--det 1` selects the deterministic recipe (`experiment.seed` = the request's, else 0; `--seed`
  overrides; identical on every mode).
