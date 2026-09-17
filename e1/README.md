# E1 — optimization kit

Drop-in modes that make stock Profluent E1 1.0.0 (`python -m E1.tools.score`, `E1Predictor` / `E1Scorer`; model sizes
`150m`, `300m`, `600m`) faster. You call E1 exactly as before; the kit adds a `--mode`:

- `off` — stock E1, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster. **The default.**

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

Built with Profluent-E1. Profluent-E1 is licensed under the Profluent-E1 Clickthrough License Agreement (copy: `stock/src/LICENSE`);
recipients of this kit receive Profluent-E1 under that Agreement. Third-party notices and licence texts: `THIRD_PARTY_NOTICES.md`,
`third_party_licenses/`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock.

## Setup

Pick ONE way to get the pinned stack — stock E1 1.0.0 and everything it needs (the 'Stack' section of `STOCK.md` lists
every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**. The images hold no weights:
those live in a host directory you mount (`/weights` below).

Every route needs:

- an NVIDIA driver ≥ 570 on the host.

Route C additionally needs:

- git and a C compiler; no CUDA toolkit. The route-C lines below are the whole recipe (uv itself: STOCK.md's Stack
  section, line 2); that section is the same lines annotated, so run one or the other.

Type the first block from the directory that holds `e1/`:

```bash
# A — Docker (recommended): stock, the pinned stack and this kit in one image
docker build -f e1/environment/Dockerfile -t e1-kit:dev .
docker run --rm -it --gpus all -v /weights:/weights -v "$PWD"/out:/kit/e1/out -w /kit e1-kit:dev bash   # a shell at /kit; out/ lands on the host
# B — Apptainer (the whole Setup for B): build the .sif once where Docker runs (converts A's image); the cluster then needs only Apptainer; /weights = any host directory, empty at first — install fills /weights/e1/hf_home
apptainer build e1-kit.sif e1/environment/apptainer.def
mkdir -p out jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --bind /weights:/weights --bind "$PWD/out":/kit/e1/out e1-kit.sif "$@"; }   # kit = B's ./run.sh · other cards: --config a100|h200
export HF_HOME=/weights/e1/hf_home && kit install --weights /weights/e1/hf_home   # /weights/e1/hf_home = HF cache: fetch/check, read-only ok
kit check --config h100 --variant 600m           # Run lines: kit <verb> … (example paths resolve in /kit/e1)
# C — instead of A or B, on your own host: fresh venv — the whole route-C recipe (uv itself: STOCK.md §Stack line 2); STOCK.md §Stack = these lines annotated (run one or the other)
uv venv --seed --managed-python --python 3.12 ~/e1-venv && . ~/e1-venv/bin/activate
pip install --no-deps -r e1/environment/requirements.lock            # the whole pinned stack, one pass (STOCK.md §Stack)
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …` (the
example paths resolve inside the image, in `/kit/e1`). Under **A** you are now in the container shell, which opens in
`/kit`. Under **C** you are in your activated environment; the first line changes into `e1/` if you are not there yet. Run:

```bash
[ -f run.sh ] || cd e1                           # from the parent dir; a no-op once inside the kit dir
bash run.sh install --weights /weights/e1/hf_home   # fetches 2.1 GB if absent, else digest-checks; read-only ok
export HF_HOME=/weights/e1/hf_home               # the one variable to set
bash run.sh check --config h100 --variant 600m      # dry run: pins, weights digest, GPU; scores nothing
```

What the blocks assume:

- **Weights directory.** `HF_HOME` is the one variable to set: a Hugging Face cache (`/weights/e1/hf_home` inside the
  container under A and B; route B's `/weights` is any host directory, empty at first). `install --weights DIR` fetches
  the three checkpoints into it with upstream's downloader when absent and otherwise digest-checks them, so a read-only
  copy works, and stages the hub RMSNorm kernel from `stock/` (layout: STOCK.md, 'Pin' section) — after installing the
  kit package and running the pin check. A run without them is a `NOT ACTIVE` exit.
- **Outputs under A and B.** The Run examples write to `out/…`, which is the mounted `$PWD/out` on the host.
- **Pin check under C.** It accepts stock installed from the git URL at the pinned commit (the lock's line) or your own
  checkout of that commit (named on the `ACTIVE` line). It refuses another E1 version or commit, an archive install and a
  stack package off its pin (STOCK.md's Stack section); the runs themselves only name stack drift on their `ACTIVE` line
  (see Notes).
- **What `check` prints.** `[e1-opt] DRY-RUN mode=… variant=… … would_refuse=<reason|none>`; exit 0, or 3 when the mode
  would refuse. It scores nothing.
- **GPU cards.** `--config h100|h200|a100` loads `configs/<card>.env` (`MODEL_OPT`, `MODEL_OPT_TARGET_GPU`,
  `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`); every command takes it.
- **Model size.** `--variant 150m|300m|600m` (or `E1_VARIANT`) names the model size; one size per process.
- **Compile cache.** The first run of a kit mode compiles its kernels (Triton) once per machine;
  `bash run.sh warm --config h100 --variant 600m` does that ahead of time, and a writable `MODEL_OPT_JIT_ROOT` keeps the
  cache between runs (route B's block puts it in `./jit` on the host).

## Run

```bash
I=opt/e1_opt/tests/fixtures/warm; P=$I/parent.fasta; M=$I/mutants.fasta   # a short synthetic parent and its point mutants
bash run.sh score --config h100 --variant 600m --mode off   --parent-path $P --mutants-path $M --output-path out/off/scores.csv    # stock, in a clean subprocess
bash run.sh score --config h100 --variant 600m --mode exact --parent-path $P --mutants-path $M --output-path out/exact/scores.csv  # the default mode
. configs/h100.env && mkdir -p out/env && E1_OPT=exact E1_VARIANT=600m python -m E1.tools.score --model-name "$HF_HOME"/hub/models--Profluent-Bio--E1-600m/snapshots/*/ --parent-path $P --mutants-path $M --output-path out/env/scores.csv
```

**Options.** `score` hands `E1.tools.score` its own command line verbatim (`--max-batch-tokens`, `--scoring-method`,
`--context-path`, `--context-reduction`). The kit's own flags are `--mode`, `--variant` (`--model-name
Profluent-Bio/E1-<size>` is the same choice) and `--det`.

**Without `run.sh`.**

- The last line of the block engages `exact` from the unchanged stock command. It passes the snapshot directory as
  `--model-name`, as `run.sh` does (STOCK.md, 'Pin' section), and creates the output directory, which stock does not.
- In Python, `e1_opt.enable("exact")` before the first `E1Predictor` / `E1Scorer` is built (or `e1_opt.apply(model)`)
  does the same.

**What a run prints.**

- A kit mode prints `[e1-opt] ACTIVE mode=exact variant=… kit=v2.0 kit_mode=eager …` once engaged, then
  `[e1-opt] KIT v2.0 … levers=N/N` with one `[e1-opt] LEVER name=… state=on|off …` line per optimization (the kit calls
  its individually switchable optimizations 'levers') and one `[e1-opt exact] KERNELS …` line naming the accelerators
  the process bound.
- Every `run.sh` verb first prints one `[e1-kit] Built with Profluent-E1 …` attribution line on stderr, naming the licence file
  under `stock/src/`.
- `run.sh score` ends with `[e1-opt] EXIT mode=… variant=… complete=<0|1> kernels_fallback=<none|names> rc=<code>`.
- `--mode off` prints an informational `[e1-opt] NOT ACTIVE: mode off: stock e1 (…)` and exits 0 when the tool wrote
  its `scores.csv`, 1 otherwise.
- If a mode cannot engage, the command prints `[e1-opt] NOT ACTIVE: <reason>` and exits 3; it never falls back to
  stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | the run failed |
| 2 | usage error |
| 3 | not active: the mode was refused |

**First run.** On an H100 `check` takes about 6 s and a session's first `score` about 20 s (checkpoint read and kernel
compile); compare modes on a second run.

## Modes

- `off` — stock `python -m E1.tools.score` as released, in a clean subprocess with nothing of the kit importable and
  every `E1_OPT*` / `MODEL_OPT*` variable stripped.
- `exact` (default) — one optimization set on every forward of the model, single-sequence and `--context-path` jobs
  alike; outputs identical to `off` under `--det 1` (the optimizations: `CHANGES.md`).

## Notes

- **Licence.** This project builds upon Profluent-E1 under the terms of the Profluent-E1 Clickthrough License Agreement.
- **A100** (`--config a100`): one optimization (`attn:A3_flex_kernel_options`) is off on that class, `levers=16/17`
  (`CHANGES.md`). **H200** (`--config h200`): same optimizations as H100. A card of no configured class runs the
  config's optimization set; the `ACTIVE` line says so.
- **Where the gain is.** The kit's gain is in model forward time. Every `score` process also spends ≈15 s on interpreter
  start and checkpoint read in either mode, so the small Run example gains little end to end; the gain grows with
  forward work per process.
- **Determinism.** `--det 1` (or `E1_OPT_DET=1`) selects the deterministic recipe on either mode (`STOCK.md`);
  `exact --det 1` equals `off --det 1` bit for bit. At `--det 0` stock times the RMSNorm kernel's autotune per process,
  so two `off` runs can differ in the last bits.
- **Host drift.** A dependency off its pin or an unlisted card is named on the `ACTIVE` line (`notes=`) and the run
  proceeds (only `install`'s pin check refuses a stack package off its pin). An installed E1 off the pinned release, or
  under `exact` an accelerator that is absent or fell back (`KERNELS` line), is a `NOT ACTIVE` exit.
- **Apptainer on a host that lists only `libcuda.so.1`.** The SIF built from `environment/apptainer.def` handles it
  itself; for any other image see STOCK.md's Stack section, `TRITON_LIBCUDA_PATH`.
- **In your own program.** The kit applies when an `E1Predictor` / `E1Scorer` is constructed on a model already on the
  GPU (`model.to("cuda")` first, as upstream's tools do); one model size per process.
