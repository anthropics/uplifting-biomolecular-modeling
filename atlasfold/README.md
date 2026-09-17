# AtlasFold — optimization kit

Drop-in modes that make stock AtlasFold v1.0.0 (`atlasfold monomer | multimer`) lighter on GPU memory, and faster above ≈400
residues. You call `atlasfold` exactly as before; the kit adds a `--mode`:

- `off` — stock AtlasFold, exactly as released ('stock' below always means this unmodified upstream release). **The default.**
- `exact` — identical outputs.
- `fast` — small, documented numeric differences; fastest on large inputs.
- `big` — lowest GPU memory, for large inputs.

Choose a kit mode for inputs above ≈400 residues, where the gains begin.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs · `fast` within stock's seed-to-seed variation · `big` lowest peak GPU memory — `big` / `fast` reach 4,000 tokens on one GPU (stock 2,000); faster than stock from mid-size inputs up, slower on small ones (Notes).

## Setup

One route: a Python venv on your own Linux x86_64 host; there is no container recipe. It needs:

- an NVIDIA driver ≥ 570 (CUDA 12.8);
- a C compiler (Triton compiles with it);
- `uv`, or any released CPython 3.11.

Step 1 — the pinned stack. The 'Stack' section of `STOCK.md` is this same block, annotated; skip it where AtlasFold v1.0.0
already runs:

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates build-essential   # bare Ubuntu host (root: no sudo); skip what you have
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user (skipped when present); the installer runs only when its sha256 matches
uv venv --seed --managed-python --python 3.11 ~/venv-atlasfold && . ~/venv-atlasfold/bin/activate
pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128
pip install "cuequivariance-torch==0.10.0" "cuequivariance-ops-torch-cu12==0.10.0" "numpy==1.26.4" "scipy==1.17.1" \
            "einops==0.8.0" "gemmi==0.7.5" "omegaconf==2.3.0" "numba==0.61.0" "huggingface_hub==0.36.2"
```

Step 2 — from the directory holding `atlasfold/` and `common/`, with that environment active:

```bash
[ -f run.sh ] || cd atlasfold                        # from the parent dir; a no-op once inside the kit dir
for r in atlaslm-3b-base atlasfold-260703 atlasfold-m-260725; do hf download "SeonghwanSeo/$r" --local-dir "/weights/atlasfold/$r"; done   # skip when the three repos are already on disk
export ATLASFOLD_WEIGHTS_DIR=/weights/atlasfold      # in every shell; the configs' default root is only a fallback
bash run.sh install --weights "$ATLASFOLD_WEIGHTS_DIR"   # --weights: sha256-checks DIR only (read-only ok), no fetch
bash run.sh check --config h100 --mode fast             # imports the stack, probes GPU, plans levers
```

What the blocks assume:

- **Weights.** The three Hugging Face repositories are downloaded into one root (`/weights/atlasfold/<repo>` above; skip
  the `hf download` line when they are already on disk). Export `ATLASFOLD_WEIGHTS_DIR` in every shell; the configs' default
  root is only a fallback.
- **What `install` does.** It adds the vendored `stock/src` and `opt/` editable (`--no-deps`) and runs the pin check (exit 1
  only if the vendored stock tree was modified). `--weights` reports the weight files' sha256 against the pins (read-only is
  fine; nothing is fetched).
- **What `check` does.** It reports package versions (`torch=… cueq=…`) without gating them. A shared core older than the
  kit's minimum, or absent, is refused (`NOT ACTIVE: reason=core_mismatch` / `core_missing`).
- **Shared core.** It is used in place: `run.sh` puts `../common/opt_core` (with `opt/` and `stock/src/src`) on `PYTHONPATH`.
  For the `ATLASFOLD_OPT=` route without `run.sh` (Run), `pip install -e ../common/opt_core` or export the same `PYTHONPATH` first.
- **GPU cards.** `--config h100|h200|a100` loads `configs/<card>.env`, which sets `ATLASFOLD_WEIGHTS_DIR`, `MODEL_OPT_JIT_ROOT`
  and `AFO_TARGET_GPU` unless already exported; every command but `install` takes it, before `--`. `configs/b300.env` exists,
  but its CUDA 13 stack is not shipped (STOCK.md, Stack section).
- **Compile cache.** The first run of a kit mode compiles its kernels once per machine (`fast` also captures one CUDA graph per
  input). `bash run.sh warm --mode M [--multimer]` does that ahead of time, and `MODEL_OPT_JIT_ROOT` (writable) keeps the cache
  between runs. The first run in a fresh cache directory also performs a one-time numerical self-check of the fused kernels on
  your GPU (seconds on H100; up to ~30 s on A100) before serving them; later runs skip it.

## Run

```bash
printf '>demo\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ\n' > demo.fasta      # any FASTA, one record per target
printf '>pair\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ:APILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ\n' > complex.fasta  # multimer: a record's chains joined with ':'
bash run.sh pred --config h100 --mode off   -- monomer  --input-fasta demo.fasta    --out-dir out/off      # stock, in a clean subprocess
bash run.sh pred --config h100 --mode exact -- monomer  --input-fasta demo.fasta    --out-dir out/exact    # a line re-run with its --out-dir: stock skips done targets
bash run.sh pred --config h100 --mode fast  -- monomer  --input-fasta demo.fasta    --out-dir out/fast
bash run.sh pred --config h100 --mode big   -- monomer  --input-fasta demo.fasta    --out-dir out/big
bash run.sh pred --config h100 --mode fast  -- multimer --input-fasta complex.fasta --out-dir out/pair --num-samples 1   # stock options pass through
```

**Options.** `pred` hands `atlasfold` everything after `--` verbatim. The kit's own flags, before `--`, are `--mode`, `--det`,
`--allow-partial`, `--weights` and `--config`.

Without `run.sh`, `ATLASFOLD_OPT=<mode> atlasfold monomer … --model-path … --lm-path …` engages a mode from the stock command
line (pass the weight files on that route; STOCK.md, 'Pin' section).

**Outputs.** They land in `--out-dir`, as stock writes them.

**What a run prints** (stderr).

- `check` prints `[atlasfold-opt] DRY-RUN mode=<m> …`, then `PLAN` / `KERNEL` / `WEIGHTS` lines, and exits 0.
- A kit mode prints `[atlasfold-opt] ACTIVE mode=<m> n_gpu=1 gpu=<name(smNN)> levers=<…> skipped=<…>` — the `ACTIVE` line,
  naming the mode and the optimizations engaged (the kit calls its individually switchable optimizations 'levers'). At exit
  it prints one `LEVER` line per optimization and `FINAL mode=<m> rc=0 exit=0 outputs=<n>/<n> gates=ok`.
- `off` prints `EXECUTION mode=off route=subprocess(clean env) …`, then `OUTPUTS` / `FINAL mode=off … exit=0`, and no
  `ACTIVE` line.
- If a mode cannot engage, the command prints `[atlasfold-opt] NOT ACTIVE: reason=<reason>` and exits 3; it never falls back
  to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | the prediction failed |
| 2 | usage error |
| 3 | not active |

## Modes

- `off` — stock `atlasfold` as released, in a clean subprocess with nothing of the kit importable.
- `exact` — identical to `off` under `--det 1`; use it when outputs must not move. It adds, each only where it reproduces
  the stock statement bit for bit:
  - fused projections around the triangle attention — itself the shared core's fused kernel, bit-identical to the stock op,
    on the cards and stacks the core holds a bitwise record for (the stock op elsewhere, named on the `LEVER` line);
  - fused LayerNorm / transition kernels and memory optimizations.
- `fast` — `exact` plus fused triangle-attention blocks and pair transitions, SDPA and bf16 / TF32 in the language model and
  the diffusion module, and the denoiser replayed as a CUDA graph; within stock's seed-to-seed variation. Use it for throughput.
- `big` — `fast` without the CUDA-graph optimizations (its one memory cost); numerics as `fast`. Use it when `fast` runs out
  of memory.

## Notes

- **Where the gain is.** Speed is prediction time per input (one H100, stock defaults): the kit modes are slower than `off`
  on small inputs, faster from a few hundred residues up, `big` with the lowest peak memory. Each run adds ≈11 s start-up and
  model load: ≈15 s per Run example.
- **Other cards: A100 and H200** (`--config a100`, `--config h200` load their `configs/<card>.env`). Every optimization
  installs on any card; the kernel is chosen per call by compute capability.
  - On compute capability 8.0, `triatt_block_exact` serves only inside the band in `hooks/card_rows.py`.
  - A call with no row for the card (`arch:smNN`) — or, under `exact`, no bitwise record for the card on the running
    stack — runs the stock or library op instead, named on its `LEVER` line.
- **A mode without some optimizations.** `MODEL_OPT_LEVERS_OFF=<lever>[,…]` runs a mode without the named ones
  (`ablated=<names>` on the `ACTIVE` line).
  - Size-gated optimizations run the stock statement below their floor (`below_min_tokens` on their line).
  - Under `exact` on the pinned stack, `dit_apb` and `exactln` have no bit-identity record and also serve stock
    (`vouch_not_recorded_on:<stack>`).
- **Deterministic runs.** `--det 1` selects the deterministic recipe (cuBLAS workspace pinned, TF32 off, deterministic
  algorithms; stock seeds per record); `exact --det 1` equals `off --det 1` bit for bit on the same card. `--allow-partial`
  proceeds when an optimization cannot install.
- **Multi-GPU.** `--n_gpu` accepts 1; P > 1 exits 3 (`NOT ACTIVE: reason=rowpair_tp_not_in_<version>`). Stock's `--gpu-ids`
  target parallelism passes through: each worker process activates the mode itself.
