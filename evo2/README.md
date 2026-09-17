# Evo 2 — optimization kit

Drop-in modes that make stock Evo 2 0.6.0 faster — `evo2` on `vtx` 1.1.0: `Evo2.score_sequences`, `model(ids)`,
`Evo2.generate`; models `evo2_7b`, `evo2_40b`. You call Evo 2 exactly as before; the kit adds a `--mode` to its `score`
command, or one environment variable, `EVO2_OPT`, for your own code:

- `off` — stock Evo 2, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster. **The default** of `run.sh score` and `check`; in your own process an unset
  `EVO2_OPT` means `off`.
- `fast` — small, documented numeric differences, faster still.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation.

## Setup

Pick ONE way to get the pinned stack — stock Evo 2 0.6.0 on vtx 1.1.0 and everything it needs (the 'Stack' section of
`STOCK.md` lists every pin; `--build-arg STACK=img_a100` builds it without Transformer Engine for cards without FP8):
**A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver for CUDA 12.8 (≥ 570) on the host;
- a weights directory, written `<your weights dir>` in the first block and mounted at `/weights`.

Route C additionally needs:

- the CUDA 12.8 toolkit and gcc (Transformer Engine is built from source);
- in every shell that runs Evo 2, the `LD_LIBRARY_PATH` export from STOCK.md's Stack section, which is the complete
  route-C recipe. An existing environment with the published `evo2` 0.6.0 and `vtx` 1.1.0 also works.

Type the first block from the directory that holds `evo2/` and `common/`:

```bash
# A — Docker (preferred): the whole pinned stack, stock and this kit in one image; the shell opens in /kit/evo2, so skip `cd evo2` below
docker build -f evo2/environment/Dockerfile -t evo2-kit:dev .
docker run --rm -it --gpus all -v <your weights dir>:/weights -e EVO2_OPT_WEIGHTS=/weights -v "$PWD/out":/kit/evo2/out evo2-kit:dev bash
# B — Apptainer / Singularity (the whole Setup for B): converts A's image from the local Docker daemon (building needs A; running does not).
apptainer build evo2.sif evo2/environment/apptainer.def    # from local evo2-kit:dev (A100: A's build + STACK=img_a100)
mkdir -p out jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --bind <your weights dir>:/weights --env EVO2_OPT_WEIGHTS=/weights evo2.sif "$@"; }   # kit = B's ./run.sh; --env covers the weights export hint
apptainer exec --bind <your weights dir>:/weights evo2.sif python -m evo2_opt.weights /weights --model_name evo2_7b   # /weights: evo2_7b.pt fetched or checked; read-only ok
kit check                                                  # each Run line alike: kit score --mode … (same paths)
#     B runs in your shell's directory, not /kit/evo2: out/, jit/ and upstream's ./activations_debug.log land there. jit/ stays so later runs and other nodes that see this directory reuse the compiled kernels; without the export (say, $PWD not writable) each node compiles under its /tmp.
# C — instead of A or B, on your own host (venv): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point: its `apptainer exec … python -m evo2_opt.weights` line is the same
weights step as `install --weights` below, and its Run lines are the same commands typed as `kit <command> …`.
Under **A** you are now in the container shell, which opens in `/kit/evo2`; the install is already in the image, so
start at the `--weights` line. Under **C** you are in your activated environment, in the directory holding `evo2/`
where STOCK.md's Stack section leaves you; type the whole block, the `install` line included. Run:

```bash
[ -f run.sh ] || cd evo2                                    # from the parent dir; a no-op once inside the kit dir
bash run.sh install                                            # kit + pin check (exit 3 on a mismatch) · C: type it
bash run.sh install --weights /weights --model_name evo2_7b    # 13.8 GB fetch if absent, else sha256 check (read-only ok)
export EVO2_OPT_WEIGHTS=/weights                            # C only: A's docker run line already set it with -e
bash run.sh check                                              # dry run, rc 0: `[evo2-opt] CHECK mode=exact … gpu=…`
```

What the blocks assume:

- **Weights directory.** `EVO2_OPT_WEIGHTS` is the directory holding `<model_name>.pt`. Without it, or when the file is
  not visible inside the container, upstream downloads the checkpoint itself (13.8 GB for `evo2_7b`).
- **Where route B writes.** B runs in your shell's directory, not `/kit/evo2`: `out/`, `jit/` and upstream's
  `./activations_debug.log` land there. `jit/` (`MODEL_OPT_JIT_ROOT`) stays so that later runs, and other nodes that see
  this directory, reuse the compiled kernels; without the export (say, `$PWD` not writable) each node compiles under its `/tmp`.
- **What `install` refuses.** `install` exits 3 when the installed `evo2` or `vtx` differs from the published files (a
  modified or editable checkout counts as different); torch, triton, flash-attn and Transformer Engine off the lock are
  only named.
- **What `check` reads.** Package metadata and torch's GPU query only — no model import, no weights.

## Run

```bash
printf '>w1\n%s\n' "$(printf 'ACGT%.0s' {1..2048})" > /tmp/w.fa   # an 8,192-bp example; B sees it (host /tmp is mounted)
bash run.sh score --mode off   --model_name evo2_7b --input /tmp/w.fa --out_dir "$PWD/out/off"     # stock, in a process with nothing of the kit imported
bash run.sh score --mode exact --model_name evo2_7b --input /tmp/w.fa --out_dir "$PWD/out/exact"   # the default when no mode is named
bash run.sh score --mode fast  --model_name evo2_7b --input /tmp/w.fa --out_dir "$PWD/out/fast"    # same scores as exact; fast differs only in generate()
EVO2_OPT=fast python -c "from evo2 import Evo2; m = Evo2('evo2_7b', local_path='/weights/evo2_7b.pt'); print(m.generate(['ACGT' * 512], n_tokens=64).sequences[0])"   # generation from your code; EVO2_OPT=exact = stock tokens
```

**What `score` does.** It calls `Evo2(model_name).score_sequences` once over every record and writes
`<out_dir>/scores.jsonl` (`{id, index, length, score, score_hex}` per record, in input order) and `run.json`.
`--batch_size N` and `--local_path CKPT` are upstream's arguments; `--mode` is the kit's.

**Your own code.** `EVO2_OPT=exact|fast python …` engages a mode at `import evo2`, as the last line shows; or call
`evo2_opt.enable()` / `enable("fast")` before the first model is built.

**What a run prints** (stderr):

- A kit mode prints `[evo2-opt] ACTIVE mode=<mode> evo2=… vtx=… stack=… transformer_engine=… gpu=… listed(…)`, then
  `[evo2-kit] APPLIED model=… route=… levers=N (…)` per model (the kit calls its optimizations 'levers') and
  `[evo2-kit] EXIT …` at exit; `score` ends with `[evo2-route] EXIT … status=ok` (rc 0).
- `off` imports nothing of the kit and prints `[evo2-route stock] ENV-CLEAN ok: …` instead.
- If a mode cannot engage — no CUDA device, `evo2` / `vtx` not the pinned version, or a model the kit cannot serve as
  constructed — the run prints `[evo2-opt] NOT ACTIVE: <reason>` or `[evo2-opt] KIT REFUSED …` and exits 3. It never falls
  back to stock silently.
- A GPU or library merely off the pinned stack is named on the `ACTIVE` line and served.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | ok |
| 1 | failed |
| 2 | usage error |
| 3 | not active |

**Comparing modes on the example.** On the example record `off`, `exact` and `fast` write the same `score_hex`. The
first `score` in a fresh environment spends minutes on the 13.8 GB checkpoint and kernel compiles (≈220 s, then
15–50 s a run), so compare timings from the second run.

## Modes

- `off` — stock: `EVO2_OPT` unset, nothing of the kit imported. `run.sh score --mode off` builds
  `Evo2(name, use_kernels=True)` for the one-device models and the constructor defaults for `evo2_40b` (STOCK.md, 'How
  stock is run').
- `exact` (default) — cached Hyena filters, spectra and FFT plans, fused Triton epilogues and RMSNorm, folded channel
  interleave, kept GEMM buffers, CUDA-graph replay of recurring scoring shapes and of the decode step. Outputs identical
  to `off` on the same stack. Use it when outputs must not move.
- `fast` — `exact` plus speculative sampling in `Evo2.generate` with the `evo2_1b_base` draft (k = 6): scoring identical,
  sampled tokens identical in distribution (a given seed's sequence differs). Use it for sampling throughput; it needs
  the draft checkpoint (Setup's weights line with `--model_name evo2_1b_base`).

## Notes

- **Where the gain is.** The kit's gain is in time per scoring forward. Each `score` process also pays ≈15–25 s of
  start-up and checkpoint load in every mode, so the one-record example takes about as long end to end in every mode;
  the gain grows with the records scored per process.
- **Per-shape warm-up.** The first forward of each (batch, length) in a process compiles the Triton kernels and fills
  that shape's caches; at a recurring shape the next few forwards select cuBLASLt configurations and record the CUDA
  graph, and later ones replay it.
- **Memory.** The caches are device memory beside the model, one shape resident at a time. An allocation that does not
  fit is torch's `OutOfMemoryError` from that call — a kit mode never drops to the stock path to avoid it.
- **Calls that take the stock path.** Cached-generation forwards, and forwards on a model carrying user hooks
  (`return_embeddings=True`), take the stock path for that call; this is named once on stderr and counted on the `EXIT` line.
- **A100 / cards without FP8.** Build with `--build-arg STACK=img_a100` (no Transformer Engine):
  - upstream runs `evo2_7b` on bf16 projections, and `exact` engages that route's own optimizations (its `APPLIED` line
    names the FP8, GEMM-buffer and graph-replay ones as not applicable);
  - `evo2_40b` fails in upstream's constructor;
  - `fast` prints `specdec=n/a` and generates as `exact`;
  - there is no card gate.
- **`evo2_40b`** needs two visible devices (vortex's layer split); set `CUDA_VISIBLE_DEVICES` accordingly.
