# GPN-Star — optimization kit

A drop-in mode that makes stock GPN-Star faster — `gpn` 0.9.0 at its pinned commit: `gpn star vep | logits | embedding`
on the released `songlab/gpn-star-*` checkpoints, primary `gpn-star-hg38-v100-200m`. You run your `gpn star` command
exactly as before; one environment variable, `GPNSTAR_OPT`, is the switch:

- the switch unset (or `off`) — stock GPN-Star, exactly as released ('stock' below always means this unmodified
  upstream release).
- `exact` — the one mode: outputs identical to stock's, bit for bit, under whatever numerics the caller runs (stock's
  own speed flag `--tf32` included), and faster.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, bit for bit under the caller's own numerics flags, faster than stock; the switch unset = stock.

## Setup

Pick ONE way to get the pinned stack — stock `gpn` 0.9.0 at its pinned commit and all it needs: Debian 12, Python 3.13,
torch 2.13.0+cu130, CUDA 13.0 as pip wheels (the 'Stack' section of `STOCK.md` lists every pin): **A — Docker**,
**B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver ≥ 580.65.06 on the host;
- disk: 0.8 GB of weights; the full alignment store (below) adds a 42 GB zip and its ≈64 GB unpacked copy.

Route C additionally needs ≈5 GB for the wheels; no CUDA toolkit.

Type the first block from the directory that holds `gpnstar/` and `common/`:

```bash
# A — Docker (preferred): the whole pinned stack, stock and this kit in one image; weights and alignments stay on the host, mounted
docker build -f gpnstar/environment/Dockerfile -t gpnstar-kit:dev .
docker run --rm -it --gpus all -v /weights:/weights -v /data/msa:/data/msa -v "$PWD"/out:/kit/gpnstar/out -w /kit/gpnstar gpnstar-kit:dev bash  # step 2 in that shell
# B — Apptainer (the whole Setup for B): converts A's image — build the .sif once where Docker runs; the cluster then needs only Apptainer
apptainer build gpnstar-kit.sif gpnstar/environment/apptainer.def
mkdir -p weights msa out; kit() { apptainer run --nv --bind "$PWD/weights":/weights --bind "$PWD/msa":/data/msa --bind "$PWD/out":/kit/gpnstar/out gpnstar-kit.sif "$@"; }   # kit = B's ./run.sh (install | check)
kitx() { apptainer exec --nv --pwd /kit/gpnstar --scratch /kit/gpnstar/trainer_output --bind "$PWD/weights":/weights --bind "$PWD/msa":/data/msa --bind "$PWD/out":/kit/gpnstar/out --env HF_HOME=/weights/gpnstar/hf_home gpnstar-kit.sif "$@"; }   # kitx = your own commands with the same binds: kitx env GPNSTAR_OPT=exact gpn star vep …
export HF_HOME=/weights/gpnstar/hf_home; kit install --weights "$HF_HOME" && kit check      # checkpoint fetched into weights/, else digest-checked; then the one DRY-RUN line
# C — instead of A or B, on your own host: a venv on uv's released CPython 3.13, then the lock (≈5 GB of wheels); no CUDA toolkit (STOCK.md §Stack = this recipe annotated: type one, not both)
uv venv --seed --managed-python --python 3.13 ~/gpnstar-venv && . ~/gpnstar-venv/bin/activate
pip install --no-deps -r gpnstar/environment/requirements.lock        # the whole pinned stack in one pass, stock gpn from stock/
```

Route B (Apptainer) is complete at this point: `kit` runs the kit's `install` and `check`, and `kitx` runs your own
commands with the same binds: type the Run block's variable lines (`X=…`, `M=…`, `ARGS=…`) on the host as they are, then
prefix its command lines — `kitx env GPNSTAR_OPT=exact gpn star vep …` (the kit), `kitx gpn star vep …` (stock) — and run
`cmp` on the host in `./out`. Under B the kit's two small Triton kernels compile on first use into `~/.triton` on the host
(under a second; Apptainer binds `$HOME`); set `TRITON_CACHE_DIR` to relocate them.

The second block is for A and C:

- Under **A** you are now in the container shell, which opens in `/kit/gpnstar`; the image already holds the install, so
  `install` there only stages or digest-checks the checkpoint.
- Under **C** you are in your activated environment: after the route-C lines above you are still in the directory
  holding `gpnstar/`, so type the whole block; if you followed STOCK.md's Stack section instead, you are inside `gpnstar/`
  with `install` already done — continue at `export`.
- The block's last two lines fetch the full alignment store, which real use needs and the shipped example does not
  (under B, prefix them with `kitx`).

Run:

```bash
[ -f run.sh ] || cd gpnstar                           # no-op once inside · C via §Stack: skip install too
bash run.sh install --weights /weights/gpnstar/hf_home   # core + kit (editable), pin check; --weights stages the 0.8 GB checkpoint at its pinned revision, sha256-checked per file
export HF_HOME=/weights/gpnstar/hf_home HF_HUB_OFFLINE=1   # where install --weights put the snapshot; offline: every load resolves to it (upstream's own variables, STOCK.md)
bash run.sh check                                        # one line: [gpnstar-opt] DRY-RUN … would_refuse=none (exit 0), else the reason (exit 3)
hf download songlab/multiz100way 99.zarr.zip --repo-type dataset --local-dir /data/msa   # hg38 100-way alignment store: a 42 GB download
python -m zipfile -e /data/msa/99.zarr.zip /data/msa/100/all.zarr/   # unpacks to ≈64 GB (the zip may then be deleted)
```

What the blocks assume:

- **Weights directory.** `/weights/gpnstar/hf_home` is any writable directory (created if absent). `install --weights`
  stages the 0.8 GB checkpoint there at its pinned revision, sha256-checked per file, and `HF_HOME` then names it;
  `--model KEY` stages another checkpoint (layout and table: STOCK.md, Pin section).
- **What `install` refuses.** It exits 3 naming any `gpn`, `transformers` or pinned-stack package off its pin
  (STOCK.md, Stack section).
- **The alignment store.** It has one array per chromosome, named `1` … `22`, `X`, `Y` (no `chr` prefix). A variant
  table uses those names, with `pos` one-based and `ref` = the alignment's human base at that position.

## Run

Your own `gpn star` command line, unchanged, with the switch in front:

```bash
X=examples/variants.parquet; XM=examples/msa      # shipped: 8 variants + a sparse store cut to their windows
M="$HF_HOME"/hub/models--songlab--gpn-star-hg38-v100-200m/snapshots/0c949f132d35619a3eb188b402848c998a3313ae      # the staged snapshot
ARGS="--input-path $X --msa-path $XM --window-size 128 --model-path $M --per-device-eval-batch-size 64 --tf32"       # stock's own flags; --tf32 is its speed flag
gpn star vep $ARGS --output-path out/stock/scores.parquet                      # stock
GPNSTAR_OPT=exact gpn star vep $ARGS --output-path out/exact/scores.parquet    # the kit: same command, same numerics, same bytes, faster
cmp out/stock/scores.parquet out/exact/scores.parquet && echo identical        # on your card
```

**The example.** Each example line takes ≈30 s on an H100 (almost all of it interpreter start and model load), ends with
`Wrote predictions to out/…/scores.parquet` and writes 8 rows (one `score` per variant). Real use is your variant table
and `--msa-path /data/msa`. `logits` and `embedding` engage the same way.

**`--model-path`** must be the snapshot directory: given a repo id, upstream asks the Hub for the phylogenetic-distance
files, which fails offline.

**From Python instead of the variable.**

- `import gpnstar_opt; gpnstar_opt.enable()` before the model is built does what the variable does.
- `gpnstar_opt.apply(model)` applies the optimizations to a `GPNStarForMaskedLM` you loaded yourself, already on the GPU.
- `gpnstar_opt.status()` returns the activation report.
- `gpnstar_opt.disable()` withdraws the hooks: models built afterwards are stock; a model already carrying the
  optimizations keeps them.

**What a run prints** (stderr; the kit calls its optimizations 'levers'):

- When the tool builds its model: `[gpnstar-opt] ACTIVE mode=exact levers=devconst+constcache+srcgather+unifiedkv+dedup+colattn+fusedattn+graph gpu=… card=h100 mib=… model=…@0c949f13 gpn=6f28c81b torch=… transformers=… tf32=on|off`
  (`tf32` = the caller's matmul setting, which the optimizations follow), and eight `LEVER name=… state=on …` lines.
- Each new batch shape adds `[gpnstar-opt] KV route=dedup|unifiedkv|stock … shape=<B>x<L> first_ms=… reason=kvcheck|small_batch|memory|none compile=off|on`.
  With `compile=on` there is also one `COMPILE levers=eager(kit-disabled) …` line: the caller's code is compiled, the
  kit's model runs eager. The `KV` line prints once more at exit (`shape=all`).
- Small shapes add `[gpnstar-opt] GRAPH state=captured shape=<B>x<L> cost_s=… pool_mib=… graphs=<n>` from the shape's
  third batch in the process on, after two eager batches (or `state=ungraphed … reason=…`). A run with one or two batches
  of a shape, like the example, prints no `GRAPH` line.
- A kit that cannot engage prints `[gpnstar-opt] NOT ACTIVE: <reason>` and the process exits 3 — stock never runs
  silently under the switch.
- `GPNSTAR_OPT` unset or `off`: nothing of the kit is loaded.

## Notes

- **Cards.** H100 80 GB and A100 (80 or 40 GB) are the listed classes (`card=h100|a100 mib=…` on the `ACTIVE` line);
  any other card runs too, named `card=untested(…)`, never refused. No optimization differs by card.
- **Numerics are the caller's.** The optimizations set no precision flag. Plain `gpn star …` is fp32 and the kit under
  it is bitwise to that; `--tf32` (and `--torch-compile`, with which transformers also turns TF32 on) puts stock and kit
  alike in TF32 matmuls, bitwise to each other.
- **What the optimizations do** (`CHANGES.md` has each one). No optimization changes the arithmetic: the same
  cuBLAS/ATen kernels run on the same values.
  - `devconst`, `constcache` — phylogenetic constants kept on the GPU, and per-shape constants cached;
  - `srcgather` — a device-side species gather;
  - `unifiedkv`, `dedup` — the column cross-attention's clade K/V projected once per layer over the distinct alignment rows;
  - `colattn` — that attention run as the same op sequence on operands built once, in the layout its matmuls read;
  - `fusedattn` — where the batch shape's first forward proves it equal, the attention read straight from the distinct
    rows in cuBLAS's own summation order, without materialising K/V at all;
  - `graph` — small batch shapes (up to 16 windows of 128 bp per forward) replayed from one whole-forward CUDA graph per shape.
- **K/V routes.** The route is named per batch shape on the `KV` line; outputs are exact on each of them.
  - Under 512 target tokens (B × L): the unified-K/V projection, by rule.
  - A shape whose reduced route would not fit the memory obtainable runs stock's own projections, decided before
    allocating from the route's own footprint — and since the kit's peak is below stock's at every batch from 64 up,
    that is no batch stock itself fits on one 80 GB card at 128 bp.
  - A shape whose layer-0 K/V check rejects the reduced GEMM settles on `unifiedkv` or `stock` (`CHANGES.md`).
- **`fusedattn` on A100.** `fusedattn` reproduces the summation order cuBLAS runs on the H100 class; on the A100 class its
  per-shape guard steps aside (warned once per shape) and the attention runs on the `colattn` operands, exact either way
  (`CHANGES.md`).
- **Memory.** At batch 64 and above the kit's peak device memory is below stock's (one K-sized transient per layer
  instead of stock's four to five); at batch 1-16 the CUDA-graph pools add 0.1-1.2 GB on top of a footprint of about 1 GB.
