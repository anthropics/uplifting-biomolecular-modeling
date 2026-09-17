# Protenix v2 — optimization kit

Drop-in modes that make stock Protenix 2.0.0 (`protenix pred --model_name protenix-v2`) faster and lighter on GPU
memory. You call Protenix exactly as before; the kit adds a `--mode`:

- `off` — stock Protenix, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs (bitwise under `--det 1`), faster.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs; `--n_gpu P` splits one `big` prediction across P GPUs of one host.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs, up to 3,000 tokens on one GPU · `--n_gpu P` splits `big` across P GPUs of one host.

## Setup

**You need the model checkpoint in hand first.** `install --weights` asks only upstream's own download URL for
`checkpoint/protenix-v2.pt`, and that server currently answers HTTP 403: no checkpoint arrives that way, and no other source is tried.

- Obtain the file from the upstream project named in STOCK.md's 'Pin' section and place it yourself at
  `<weights root>/checkpoint/protenix-v2.pt` (sha256 and size: STOCK.md 'Pin').
- `install --weights <weights root>` fetches the six data caches next to it from upstream and digest-checks all seven files
  (`WEIGHTS OK: 7/7`).

Then get the pinned stack. Pick ONE way — stock Protenix 2.0.0 and everything it needs (the 'Stack' section of
`STOCK.md` lists every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver for CUDA 13.0 (version 580 or newer) on the host.

Route C additionally needs (on a fresh host):

- the CUDA 13.0 toolkit (`nvcc`) and gcc; STOCK.md's Stack section is the complete route-C recipe, run inside `protenix_v2/`.

Type the first block from the directory that holds `protenix_v2/` and `common/`:

```bash
# A. Docker (recommended: the whole pinned stack, stock and this kit in one image)
docker build -f protenix_v2/environment/Dockerfile -t protenix_v2-kit:dev .
docker run --rm -it --gpus all -v /weights:/weights -v $PWD/out:/kit/protenix_v2/out -w /kit protenix_v2-kit:dev bash   # shell in /kit; out/ lands on the host
# B. Apptainer (the whole Setup for B): converts A's image from the local Docker daemon (build A first; no Docker needed at run time); typed on the host
apptainer build protenix_v2.sif protenix_v2/environment/apptainer.def
mkdir -p out jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --bind /path/to/weights_root:/weights/protenix --bind "$PWD/out":/kit/protenix_v2/out protenix_v2.sif "$@"; }   # kit = B's ./run.sh · cards: --config a100|h200|b200|b300
export PROTENIX_ROOT_DIR=/weights/protenix && kit install --weights /weights/protenix   # /weights/protenix: files fetched or checked; read-only ok
kit check --config h100                              # Run lines alike: kit pred …; --det 1: add --writable-tmpfs
# C — instead of A or B, on your own host: STOCK.md 'Stack' is the complete recipe (run it inside protenix_v2/)
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`.
Under **A** you are now in the container shell, which opens in `/kit`. Under **C** you are in your activated
environment; if STOCK.md's Stack section set it up, you are already inside `protenix_v2/` with the install line done, so
continue at `export`. Run:

```bash
[ -f run.sh ] || cd protenix_v2                    # no-op once inside · C via §Stack: resume at export
bash run.sh install --weights /weights/protenix       # --weights: 0.7 GB if absent, else hash-check; read-only ok
export PROTENIX_ROOT_DIR=/weights/protenix         # holds checkpoint/protenix-v2.pt + the common/ caches
bash run.sh check --config h100                       # dry run of the pin, GPU and lever-set gates: rc 0, or 3
```

What the blocks assume:

- **Weights directory.** `/weights/protenix` in the blocks (under B, the host directory bound there); it holds
  `checkpoint/protenix-v2.pt` and the `common/` caches, and `PROTENIX_ROOT_DIR` must name it. `install --weights`
  fetches what is absent and hash-checks what is present, so a complete directory may be read-only.
- **Outputs under A and B.** `out/` in the container is the mounted `$PWD/out` on the host.
- **Route C with an existing environment.** An environment that already runs Protenix 2.0.0 whose files match the
  pinned wheel also works (PyPI `protenix==2.0.0`, the wheel in `stock/`, or the v2.0.0 tag). `install` refuses another
  version or modified files by name; a different Python / torch / triton / cuEquivariance is only reported
  (`STACK not pinned: …`).
- **GPU cards.** `--config h100|h200|a100|b200|b300` loads `configs/<card>.env` (`MODEL_OPT_TARGET_GPU`,
  `MODEL_OPT_STACK_KEY`, and the cache directories under a writable `MODEL_OPT_JIT_ROOT`). `pred`, `check` and `warm`
  take it; `install` does not.
- **First run compiles.** The first run of a kit mode compiles its Triton kernels once per machine, so time the modes
  from a second run. `bash run.sh warm --config h100 --mode <mode>` does the compilation ahead of time.
- **Compile cache.** Route B's block keeps it in `./jit` on the host (`MODEL_OPT_JIT_ROOT`). The first run in a fresh
  cache directory also performs a one-time numerical self-check of the fused kernels on your GPU (seconds on H100; up
  to ~30 s on A100) before serving them; later runs skip it.
- **`--det 1` under B.** Add `--writable-tmpfs` to the `apptainer run` line, as the block's comment says (why: STOCK.md,
  'How stock is run').

## Run

```bash
sed "s#__KIT__#$PWD/opt/forward/flashpairformer#" opt/forward/flashpairformer/public_inputs/p199.json > /tmp/p199.json; J=/tmp/p199.json   # shipped example: PDB 1BRS barnase:barstar
apptainer exec protenix_v2.sif sed "s#__KIT__#/kit/protenix_v2/opt/forward/flashpairformer#" /kit/protenix_v2/opt/forward/flashpairformer/public_inputs/p199.json > $PWD/p199.json; J=$PWD/p199.json   # the same under B, from the host shell
bash run.sh pred --config h100 --mode off   --input $J --out_dir out/off   --model_name protenix-v2   # stock, in a clean subprocess
bash run.sh pred --config h100 --mode exact --input $J --out_dir out/exact --model_name protenix-v2
bash run.sh pred --config h100 --mode fast  --input $J --out_dir out/fast  --model_name protenix-v2   # the default when no mode is named
bash run.sh pred --config h100 --mode big --input $J --out_dir out/big --model_name protenix-v2
bash run.sh pred --config h100 --mode big --n_gpu 2 --input $J --out_dir out/big_x2 --model_name protenix-v2   # P GPUs of one host
```

The first two lines write the shipped example (PDB 1BRS, barnase:barstar) to a writable path — the first under A and C,
the second under B, typed on the host.

**Options.** `pred` hands `protenix pred` its own command line verbatim (`--seeds`, `--cycle`, `--use_msa`,
`--use_template`, `--dtype` …). The kit's own flags are `--config`, `--mode`, `--n_gpu` and `--det`. Two other ways to
make the same call:

- `protenix-opt pred --mode <mode> …` — the same call without `run.sh`;
- `source configs/h100.env; PROTENIX_OPT=<mode> protenix pred …` — engages a mode from the unchanged stock command line
  (one GPU).

**Inputs.** The input's own directory must be writable: stock writes `<input>-update-msa.json` beside it when it
searches or converts MSAs (multiple sequence alignments).

**Outputs.** They land where stock writes them (`--out_dir`). Per item, a `PHASE … fwd_s=…` and a `PEAK …` line print
on stdout.

**What a run prints** (on stderr).

- `check` prints `[protenix-opt] DRY-RUN mode=<m> … levers=…`, plus a `CARD LEVER SET` line and the checkpoint digest
  (the kit calls its individually switchable optimizations 'levers').
- A kit mode prints `[protenix-opt] ACTIVE mode=<m> n_gpu=… levers=… fallbacks=…` at start — the `ACTIVE` line, naming
  the mode and the optimizations engaged — then one `LEVER name=… state=…` line per optimization, and `FINAL …` and
  `EXIT …` at exit.
- `--mode off` prints `NOT ACTIVE: mode off: stock protenix …` and `ENV-CLEAN ok: …`, and returns stock's exit code.
- If a mode cannot engage, the command exits 3 with `NOT ACTIVE: <reason>` naming the optimization or gate that
  refused; it never falls back to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | ok |
| 1 | the command's own check failed — `pred`: incomplete outputs or unmet `--det 1` preconditions; `install --weights`: a file absent or off its pin |
| 2 | usage error |
| 3 | not active |
| other | stock's own exit code |

## Modes

- `off` — stock `protenix pred` as released, in a subprocess with the kit's switch variables and paths removed (proven
  before `protenix` is imported). Use it to reproduce upstream numbers.
- `exact` — the fused Pairformer block path; bit-exact triangle-multiplication, transition and diffusion-attention
  kernels; CUDA graphs for trunk and sampler. On H100 (pinned stack) the pair stacks' triangle attention runs on the
  shared core's own CUDA kernel (`common/opt_core`, the runtime all kits share), bit-identical to the stock operation,
  within the shapes its table vouches for; calls outside them, and every call on other cards, take the stock operation
  by name. Outputs identical to `off` under `--det 1`. Use it when outputs must not move.
- `fast` (default) — `exact` plus the `fast` variants of the shared core's FlashPairformer triangle kernels, and fused
  MSA-module and diffusion attention with fp16 tensor-core operands; bf16 / fp16 rounding-class differences, within
  stock's seed-to-seed variation. Use it for throughput.
- `big` — `fast`'s kernels without the CUDA-graph paths (and without the memory pool those paths keep reserved), the
  stock `n_token` guard lifted, chunked conditioning and pair bias, pair caches released early; numerics as `fast`.
  `--n_gpu P` row-shards the pair tensors over P GPUs. Use it when `fast` runs out of memory.

## Notes

- **Other cards** (`--config h200|a100|b200|b300`). Each mode's set of optimizations per compute capability is in
  `opt/protenix_opt/modes.py` and in the card's config comments; optimizations outside it print on the `CARD LEVER SET` line.
  - **H200** (compute capability 9.0) takes the H100's sets. Under `exact`, a shared-core kernel whose exactness table
    has no H200 entry runs the library operation instead (still bitwise vs stock).
  - **A100**: the stock fast-LayerNorm extension must carry code for compute capability 8.0. The image's build does; an
    own environment whose extension was built for 9.0 only stops with `no kernel image` in every mode, `off` included
    (rebuild it per STOCK.md's Stack section).
- **Switching optimizations off.** `MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]` runs a mode without the named optimizations
  (names as printed on the `LEVER` lines; a name outside the mode is refused, exit 3); the `ACTIVE` / `FINAL` lines then
  carry `ablated=<names>`.
- **`NAMED_FALLBACK:transition` lines under `exact`** are informational: outputs unchanged, exit code 0.
  - Inputs up to 256 tokens run the pair transition as the stock statement (the kernel table's entry there): the
    `LEVER` line reads `served=0` and one `NAMED_FALLBACK:transition` line prints.
  - The same line also names calls refused while a CUDA graph is being captured
    (`refused=…:init_during_capture served=caller:torch_swiglu note=raised`): a kernel's one-time set-up cannot run
    inside a capture, so those calls capture the stock statement.
- **Determinism.** `--det 1` selects the deterministic recipe (cuBLAS workspace setting, deterministic scatter) on any
  mode; `exact --det 1` equals `off --det 1` bit for bit.
- **`--dtype` fp32 / fp16** inputs run every mode: bf16-only kernels step aside per call to the stock computation,
  counted by name; `--dtype fp16` items whose sampler probes are non-finite run the eager sampler for that item, named on
  its route line.
- **Out of memory.** Under `fast` → use `big`; still out of memory → `big --n_gpu P`.
