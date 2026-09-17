# ChromBPNet — optimization kit

Drop-in modes that make stock ChromBPNet 1.0.1 (`chrombpnet pred_bw`) faster. You call `chrombpnet pred_bw` exactly as
before; the kit adds a `--mode`:

- `off` — stock ChromBPNet, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — outputs identical to stock's deterministic run (`off --det 1`), faster.
- `fast` — small, documented numeric differences, faster still. **The default.**

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` outputs identical to stock's deterministic run, faster than stock · `fast` faster still, within stock's seed-to-seed variation.

## Setup

Pick ONE way to get the pinned stack — stock ChromBPNet 1.0.1 and everything it needs (the 'Stack' section of
`STOCK.md` lists every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver that runs CUDA 12.4 (550 or newer) on the host;
- a weights directory and a data directory, written `/weights` and `/data` in the blocks (their contents: the weights
  step and 'What the blocks assume', below).

Route C additionally needs:

- a released CPython 3.8;
- the CUDA 11.2 / cuDNN 8.1 system libraries that TensorFlow 2.8.0 loads (STOCK.md, Stack section);
- about 4.9 GB for the torch stack under `/opt/torch`.

Type the first block from the directory that holds `chrombpnet/`:

```bash
# A — Docker (recommended: the whole pinned stack, stock and this kit in one image)
docker build -f chrombpnet/environment/Dockerfile -t chrombpnet-kit:dev .
docker run --rm -it --gpus all -v /weights:/weights -v /data:/data -v $PWD/out:/kit/chrombpnet/out chrombpnet-kit:dev bash
# B — Apptainer / Singularity (the whole Setup for B): converts the image built in A (no Docker daemon needed at run time)
apptainer build chrombpnet-kit.sif chrombpnet/environment/apptainer.def
mkdir -p out jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --bind /weights:/weights --bind /data:/data --bind "$PWD/out":/kit/chrombpnet/out chrombpnet-kit.sif "$@"; }   # kit = B's ./run.sh · other cards: --config a100|h200
export CHROMBPNET_OPT_WEIGHTS=/weights CHROMBPNET_OPT_DATA=/data && kit install --weights /weights   # /weights/GM12878_ATAC/fold_0: .h5 pair; -cm built, checked
kit check --config h100 --mode fast       # each Run line alike: kit <verb> … (out/… = ./out)
#     --bind left sides: your own directories, anywhere; install writes chrombpnet_recompiled.h5 into the weights dir once; Triton caches: jit/ beside out/ (MODEL_OPT_JIT_ROOT above; unset it for run.sh's own choice), the driver's: ~/.nv
# C — instead of A or B, on your own host (venv): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`.
Do the weights step (below) first. Then, under **A**, you are in the container shell, which opens in `/kit/chrombpnet`
with `out/` mapped to the host. Under **C** you are in your activated environment inside `chrombpnet/`, where `/weights`
and `/data` stand for any two directories of yours; STOCK.md's Stack section already ran the `install` line, so resume at
`export`. Run:

```bash
[ -f run.sh ] || cd chrombpnet            # no-op once inside · C via §Stack: resume at export
bash run.sh install --weights /weights       # kit + pin check; --weights: weights step below
export CHROMBPNET_OPT_WEIGHTS=/weights    # holds GM12878_ATAC/fold_0/ (the weights step)
export CHROMBPNET_OPT_DATA=/data          # hg38 fasta+.fai, chrom sizes, regions.bed, bigWig
bash run.sh check --config h100 --mode fast  # dry run: card, torch stack, pins, mode line; rc 0
```

#### Weights step (once)

The two ENCODE model files are put in place by hand, once:

1. Download the ENCODE model archive ENCFF142IOR
   (`https://www.encodeproject.org/files/ENCFF142IOR/@@download/ENCFF142IOR.tar.gz`).
2. Take `fold_0/model.bias_scaled.fold_0.ENCSR637XSC.h5` and `fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5`
   out of it and save them as `/weights/GM12878_ATAC/fold_0/bias_scaled.h5` and `…/nobias.h5`.
3. `bash run.sh install --weights /weights` (in the block above) then builds `chrombpnet_recompiled.h5` beside them — the
   `-cm` model: the two recombined into one Keras file — and checks the three digests (seconds, on the CPU).

Points to know about this step:

- `/weights` must be writable for this step only; `check` and `pred_bw` then just read it. A read-only site copy of the
  two files is first copied somewhere writable.
- A `MISMATCH` on an older copy of `chrombpnet_recompiled.h5`: delete it and re-run (STOCK.md, Pin section).

What the blocks assume:

- **Data directory.** `/data` holds upstream's tutorial inputs: the hg38 fasta with its `.fai`, the chromosome sizes,
  `regions.bed` and the observed bigWig. `regions.bed` is any BED/narrowPeak file of regions, with no fixed count. The
  download URLs and the `.fai` command are in STOCK.md's Variables section.
- **Outputs.** `out/…` in the Run commands is `$PWD/out` on the host under A and B.
- **Route C with an existing install.** Route C's pin check accepts any install of ChromBPNet 1.0.1 and names drift
  instead of refusing it.
- **GPU cards and the config.** `--config h100|h200|a100` loads `configs/<card>.env`, which sets `MODEL_OPT_TARGET_GPU`
  and derives `CHROMBPNET_OPT_MODEL`, `_GENOME`, `_CHROM_SIZES`, `_REGIONS`, `_BIGWIG` and `_CACHE_TAR` from the two
  roots unless they are already set; every command below takes it.
- **Compile cache.** The first run of a kit mode compiles its kernels (Triton) once per machine;
  `bash run.sh warm --config h100 [--mode M]` does that ahead of time. Route B's block keeps the cache in `./jit` on the host
  (`MODEL_OPT_JIT_ROOT`).

## Run

No input ships with the kit. The commands use upstream's tutorial layout (ENCODE GM12878 ATAC-seq: hg38 fasta,
chromosome sizes, a narrowPeak regions file, the observed bigWig) through the variables the config sets. Your own model
and regions work the same way; a kit mode also reads the component models `bias_scaled.h5` and `nobias.h5` from beside
the `-cm` file.

```bash
source configs/h100.env && mkdir -p out   # A/C: sets $A's variables (B: the [ -d configs ] line)
A="-cm $CHROMBPNET_OPT_MODEL -g $CHROMBPNET_OPT_GENOME -c $CHROMBPNET_OPT_CHROM_SIZES -r $CHROMBPNET_OPT_REGIONS -bw $CHROMBPNET_OPT_BIGWIG -bs 1024"
[ -d configs ] || A="-cm /weights/GM12878_ATAC/fold_0/chrombpnet_recompiled.h5 -g /data/hg38.genome.fa -c /data/hg38.chrom.sizes -r /data/regions.bed -bw /data/ENCFF180XQC.bigWig -bs 1024"   # B host (no configs/): the same line spelled out
bash run.sh pred_bw --config h100 --mode off   $A -op out/off   -os out/off      # stock, in a clean subprocess
bash run.sh pred_bw --config h100 --mode exact $A -op out/exact -os out/exact
bash run.sh pred_bw --config h100 --mode fast  $A -op out/fast  -os out/fast     # the default when no mode is named
CHROMBPNET_OPT=fast chrombpnet pred_bw $A -op out/fast -os out/fast           # a mode on the unchanged stock command line
```

**Options.** `pred_bw` hands `chrombpnet pred_bw` its own arguments verbatim (`-bs 1024` above is stock's batch flag).
The kit's own flags are:

- `--config`, `--mode`;
- `--det` (with `off` only; see Modes);
- `--items FILE` (with `exact` or `fast`) — predicts many regions files in one process: one
  `regions<TAB>output_prefix[<TAB>stats]` line per item, the other stock arguments given once.

`chrombpnet-opt pred_bw --mode <mode> …` is the same call without `run.sh`.

**Outputs.** They land where stock writes them (the `-op` and `-os` prefixes, stock's file names), plus
`opt_manifest.json` in the `-op` directory.

**What a run prints.**

- A kit mode prints `[chrombpnet-opt] ACTIVE mode=<mode> route=<route> kit=<version> gpu=<class> det=<0|1> precision=<fp32|tf32>`
  on stdout when it engages, and `[chrombpnet-opt] EXIT pid=<n> mode=<mode> route=<route> det=<0|1> … rc=<rc> …` on stderr
  when it ends. `rc` 0 = done; a non-zero `rc` after `ACTIVE` = engaged but failed, and the first traceback above it is the cause.
- `--mode off` prints `[chrombpnet-opt] NOT ACTIVE mode=off (stock)` before stock's own output.
- If a mode cannot engage, the command prints `[chrombpnet-opt] NOT ACTIVE mode=<mode> reason=<…>`, exits 3 and runs
  nothing. It never falls back to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | ok |
| 1 | failed or incomplete |
| 2 | usage error |
| 3 | not active |

## Modes

- `off` — stock `chrombpnet pred_bw` as released, in a clean subprocess with nothing of the kit importable. `--det 1`
  adds TensorFlow's determinism settings (seed 0, deterministic ops, TF32 off); `--det 0`, the default, is stock as
  shipped. `--det` goes with `off` only.
- `exact` — the model as fp32 Triton kernels in stock's summation order, inside the kit's pipelined job (featurisation,
  writers and metrics overlapped with the forward). Outputs identical to `off --det 1`, byte for byte. Use it when
  outputs must not move.
- `fast` (default) — `exact` with the convolutions at TF32 on the tensor cores, stock's own precision class as shipped;
  TF32 rounding, not bitwise. Use it for throughput. On cards without a Triton route it runs stock's graph inside the
  same pipeline.

## Notes

- **A100** (`--config a100`):
  - `fast` engages with this card's set of optimizations: `tail` and `native_dilation` stay off by the card's table, as
    the kit's line says; kernels compile on first use.
  - `exact` is refused (`… no bitwise Triton route on cuda-80 …`, exit 3); `off --det 1` is the deterministic run there.
- **H200** (`--config h200`): the same routes and optimizations as H100 in every mode (both are compute capability 9.0;
  the shipped kernel cache applies). `configs/h200.env` differs from `h100.env` in the card name only.
- **`-d <chr …>`** does not run in upstream 1.0.1 (`pred_bw` keeps no region and the bigWig writer raises). `off` fails
  as upstream does; `exact` and `fast` stop at `-d` before predicting (`[pred_bw_fast] REFUSED: -d …` on stderr, exit 1).
  No fix ships.
- **Pin lines.** Every command first prints one line per pin (`pinned` / `DRIFT … (want …)`) and proceeds; only a
  `chrombpnet` off the pinned 1.0.1 stops it (exit 3). A card other than the config's `MODEL_OPT_TARGET_GPU` is noted on
  stderr; the route follows the card found.
- **No optimization switches.** No flag or variable selects an individual optimization. Kit-internal names found in the
  environment are removed for the run and named on stderr as `[chrombpnet-opt] IGNORED names=<N1,N2,…> reason=<…>`.
