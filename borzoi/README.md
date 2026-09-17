# Borzoi — optimization kit

Drop-in modes that make stock Borzoi variant scoring faster: `borzoi_sad.py` from calico/borzoi @ `5c93582` on
calico/baskerville @ `544073b`, TensorFlow 2.15. You call `borzoi_sad.py` exactly as before; the kit adds a `--mode`:

- `off` — stock Borzoi, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster. **The default.**

No `fast` or `big` mode ships: Borzoi scores a fixed 524,288-bp window on one GPU.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock.

## Setup

Pick ONE way to get the pinned stack — stock Borzoi and everything it needs (the 'Stack' section of `STOCK.md` lists
every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an x86_64 Linux host whose NVIDIA driver runs CUDA 12.2;
- a host directory for the weights and the genome, written `<weights dir>` in the blocks. It must be writable, because
  pysam writes the genome's `.fai` index beside the genome on first use (the weights are only read). Under A and B it is
  mounted at `/weights`; under C use its own path wherever the blocks say `/weights`.

Route C additionally needs `bedtools` and the two upstream checkouts at their pins, with `BORZOI_DIR` and `PATH` set as
the second block shows; STOCK.md's Stack section is the complete recipe for all of it.

Type the first block from the directory that holds `borzoi/`:

```bash
# first, on the host: make the weights directory and fetch upstream's genome into it (about three gigabytes; pysam adds hg38.ml.fa.fai beside it on first use)
mkdir -p <weights dir>                      # any host path you choose; it stands for that path below
curl -L https://storage.googleapis.com/seqnn-share/helper/dependencies/hg38.ml.fa.gz | gunzip -c > <weights dir>/hg38.ml.fa
# A — Docker (recommended: carries the whole pinned stack, stock and this kit)
docker build -f borzoi/environment/Dockerfile -t borzoi-kit:dev .
docker run --rm -it --gpus all -w /kit -v <weights dir>:/weights -v $PWD/out:/kit/borzoi/out borzoi-kit:dev bash   # a shell in /kit; out/ = $PWD/out on the host
# B — Apptainer (the whole Setup for B): build the .sif once where Docker runs (converts A's image); the cluster then needs only Apptainer
apptainer build borzoi-kit.sif borzoi/environment/apptainer.def
mkdir -p out; kit() { apptainer run --nv --bind <weights dir>:/weights --bind "$PWD/out":/kit/borzoi/out borzoi-kit.sif "$@"; }   # kit = B's ./run.sh · other cards: --config a100|h200
kit install --weights /weights              # /weights: f0/model0_best.h5 fetched if absent or checked
kit check --config h100                     # each Run line alike: kit sad …; out/… = ./out here
# C — instead of A or B, on your own host (venv): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …` on the
host, with `BORZOI_DIR` unset there. Under **A** you are now in the container shell, which opens in `/kit`. Under **C**
you are in your activated environment, in the directory holding `borzoi/` (STOCK.md's Stack section stops before
`install`, so type the whole block). Run:

```bash
[ -f run.sh ] || cd borzoi                  # from the dir above; a no-op once inside the kit dir
mkdir -p out                                # stock creates the -o dirs non-recursively
export BORZOI_DIR="${BORZOI_DIR:-$HOME/src/borzoi}"; export PATH="$BORZOI_DIR/src/scripts:$PATH"   # A: image preset wins · C: §Stack's checkout
bash run.sh install --weights /weights         # + pin check; model0_best.h5 fetched if absent or checked
bash run.sh check --config h100                # dry run: one DRY-RUN line + the lever list, rc 0
```

What the blocks assume:

- **Weights and genome.** No variable names the weights: the weights file, the genome and the targets file are the stock
  command's own arguments (STOCK.md's Variables section lists the optional variables). `install --weights /weights`
  fetches `f0/model0_best.h5` into the weights directory if it is absent, else checks it.
- **Outputs under A and B.** `out/…` in the Run commands is `$PWD/out` on the host (A mounts it; B's `kit` binds `./out`).
- **Route C with an existing install.** `install` refuses (exit 3) an upstream install whose files differ from the pins,
  and only reports another Python or package version (STOCK.md, Stack section).
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env` (`MODEL_OPT`, `MODEL_OPT_TARGET_GPU`,
  `PYTHONDONTWRITEBYTECODE`); `sad` and `check` take it.
- **What `check` does.** It scores nothing and imports no TensorFlow: it reports the entry script, the GPU (from
  `nvidia-smi`), package versions, and any difference from the pins as `notes=`.

## Run

```bash
A="-f /weights/hg38.ml.fa --rc --stats SAD,logSAD,D2,logD2 -t ${BORZOI_DIR:-/opt/borzoi}/examples/targets_human.txt -u"   # borzoi_sad.py's own options
X="${BORZOI_DIR:-/opt/borzoi}/examples/params.json /weights/f0/model0_best.h5 tests/inputs/example_snvs.vcf"              # its positionals; six SNVs ship in tests/inputs/
bash run.sh sad --config h100 --mode off   $A -o out/off   $X     # stock, in a clean subprocess
bash run.sh sad --config h100 --mode exact $A -o out/exact $X     # the default when no mode is named
BORZOI_OPT=exact borzoi_sad.py $A -o out/env $X                # same mode, stock command line unchanged (A/C shell)
```

**Shell definitions.** `A`, `X` and route B's `kit` are definitions of the current shell session: re-type them in a new
login or a job script. The relative `tests/inputs/…` path also works under route B, because the container starts in
`/kit/borzoi`, where the kit's files live.

**Options.** `sad` passes `borzoi_sad.py` its own command line verbatim (three positionals, or upstream's worker form).
The kit's flags go between `sad` and those arguments: `--config` and `--mode` first, then `--det 0|1` and
`--allow-partial` (both under Notes). `borzoi-opt sad --mode <mode> …` is the same call without `run.sh`.

**Outputs.** They land where stock writes them: `<-o dir>/sad.h5`.

**What a run prints** (on stderr):

- A kit-mode run prints `[borzoi-opt] ACTIVE mode=exact route=cli kit=pipeline_tf.v17 writer=exact …` before the job
  and `[borzoi-opt] EXIT mode=exact rc=0 …` after it.
- `--mode off` prints `[borzoi-opt] OFF … pinned=true …`, `[borzoi-opt stock] ENV-CLEAN ok: …` and
  `[borzoi-opt] EXIT mode=off rc=0 stock_proof=ok …`.
- `check` prints `[borzoi-opt] DRY-RUN mode=<mode> …` and the list of the mode's optimizations (the kit's output calls
  them 'levers').
- If a mode cannot engage — or one of its optimizations could not — the command prints `[borzoi-opt] NOT ACTIVE: <reason>`
  and exits 3. It never falls back to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | ok |
| 1 | failed |
| 2 | usage error |
| 3 | not active |

**First run.** The first process of a session also compiles TensorFlow's PTX for the card (about a minute on H100,
cached under `~/.nv/ComputeCache`): compare modes on a second run. Under route A mount the driver's cache into the
container so it survives between containers, e.g. add `-v $HOME/.nv:/root/.nv` to the `docker run` line; otherwise
each new container repeats the compile. Route B keeps it under `$HOME` on the host automatically.

## Modes

- `off` — stock `borzoi_sad.py` as released, in a clean subprocess with nothing of the kit importable.
- `exact` (default) — the forward pass traced once with `tf.function` (no XLA) and run as one graph from the second
  variant on, a lookup-table one-hot, thread-pooled post-processing pipelined behind the next forward, and a copy-free
  forward return. Outputs identical to `off`.

## Notes

- **A100, H200** (`--config a100` / `--config h200`): these load `configs/<card>.env`, the H100 settings with the
  target-GPU word changed; every optimization engages on all three cards. No other card is configured: another card, a
  card that is not the config's target, or a stack differing from `stock/PINS.json` is named on the `ACTIVE` line
  (`notes=…`) and the run proceeds.
- **`--det 1`** selects the deterministic recipe (cuDNN autotuning off, numpy CPU dispatch and BLAS threads pinned;
  STOCK.md): `exact --det 1` equals `off --det 1` bit for bit on any host. At `--det 0` (the default) cuDNN autotunes per
  process, so `exact` matches `off` as closely as two `off` runs match each other.
- **Example input.** `tests/inputs/example_snvs.vcf` holds six illustrative site-level SNV records on hg38 coordinates, used only as an
  input-format example; no sample or genotype columns.
- **Post-processing coverage.** The pipelined post-processing covers `-t <targets with strand_pair>` and `--stats` taken
  from `SAD SADlog logSAD sqrtSAD SAX D1 logD1 sqrtD1 D2 logD2 sqrtD2 JS logJS`. Other option sets run stock's
  post-processing inside the kit entry (a `ROUTED post=stock:<reason>` line; outputs unaffected).
- **`--allow-partial`** (or `BORZOI_OPT_ALLOW_PARTIAL=1`) lets a run in which an optimization could not engage keep the
  job's own exit code instead of 3; the `PARTIAL` line names the optimization.
- **Your own programs.** `BORZOI_OPT=exact <program>` (or `import borzoi_opt; borzoi_opt.enable("exact")`) gives any
  program that imports `baskerville` the forward-call and one-hot optimizations, the program unmodified. Only the pinned
  `borzoi_sad.py` also gets the post-processing optimizations (`CHANGES.md`).
