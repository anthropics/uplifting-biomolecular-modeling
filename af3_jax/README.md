# AlphaFold 3 (JAX) — optimization kit

Drop-in modes that make stock AlphaFold 3 (JAX) faster and lighter on GPU memory — github.com/sokrypton/alphafold3 `v3.1.4`,
`run_alphafold.py` with the fork's ported `--of3_weights` checkpoint (variant `p2`). You call `run_alphafold.py` exactly as
before; the kit adds a `--mode`:

- `off` — stock plus one declared bug-fix patch ('stock' below always means this pinned upstream release; the patch:
  STOCK.md, 'Stock exceptions' section).
- `exact` — stock's arithmetic, faster; bit-identical to `off` when both compile into one `--cache_dir` (Notes).
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs; `--n_gpu P` splits one `big` prediction across P GPUs of one host.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` stock's arithmetic, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs, up to 6,000 tokens on one GPU · `--n_gpu P` splits `big` across P GPUs of one host.

## Setup

Pick ONE way to get the pinned stack (the 'Stack' section of `STOCK.md` lists every pin): **A — Docker**, **B — Apptainer**,
or **C — Python venvs on your own host** (two venvs and a from-source build of the stock wheel).

Every route needs:

- an NVIDIA GPU with driver ≥ 550.

Route C additionally needs:

- a C/C++ toolchain: gcc, make, zlib headers, git;
- network access for the stock wheel build.

Type the first block from the directory that holds `af3_jax/` and `common/`:

```bash
# A — Docker (preferred): stock and this kit in one image
docker build -f af3_jax/environment/Dockerfile -t af3_jax-kit:dev .
docker run --rm -it --gpus all -w /kit -v /weights:/weights -v $PWD/out:/kit/af3_jax/out af3_jax-kit:dev bash
# B — Apptainer / Singularity (the whole Setup for B): converts A's image (build A first; no Docker daemon needed at run time)
apptainer build af3_jax.sif af3_jax/environment/apptainer.def
mkdir -p out; kit() { apptainer run --nv --bind /weights:/weights --bind "$PWD/out":/kit/af3_jax/out af3_jax.sif "$@"; }   # kit = B's ./run.sh · cards: --config a100|h200
export AF3_JAX_PARAMS_ROOT=/weights/af3 && kit install --config h100 --weights /weights/af3   # /weights/af3/p2/*.bin.zst made or checked; read-only ok
kit check --config h100 --variant p2 --mode fast   # Run's X= is in-image; own inputs: absolute $HOME/$PWD path
# C — instead of A or B, on your own host (two venvs and a from-source stock wheel): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …` (their `out/…`
lands in `./out` on the host; the compile cache lives in `/tmp/model_opt_jit-uid<uid>`, the host's `/tmp` under Apptainer). Under
**A** you are now in the container shell, which opens in `/kit`. Under **C** you are in the kit environment that STOCK.md's
Stack section built and activated, inside `af3_jax/`. Run:

```bash
[ -f run.sh ] || cd af3_jax                              # from the parent dir; a no-op once inside the kit dir
bash run.sh install --config h100 --weights /weights/af3   # absent: fetch+convert 2.3GB; else hash-check, read-only ok
export AF3_JAX_PARAMS_ROOT=/weights/af3                  # required: holds p2/of3_ported_weights.bin.zst
bash run.sh check --config h100 --variant p2 --mode fast    # dry run: the mode's ACTIVE line, or NOT ACTIVE (exit 3)
```

What the blocks assume:

- **Weights.** `AF3_JAX_PARAMS_ROOT` is required and names the directory that holds `p2/of3_ported_weights.bin.zst`.
  `install --weights <dir>` fetches and converts the checkpoint when absent, otherwise hash-checks it (read-only is fine);
  `install` without `--weights` installs and checks only.
- **`--variant p2`** names the weights and is required on `pred`, `check` and `warm` (or set `AF3_JAX_VARIANT=p2`).
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env`: the target GPU and, where unset, `AF3_JAX_REPO`,
  `AF3_JAX_PY` and the pinned `XLA_*` values.
- **Pin check under C.** It accepts an existing install of the fork: a package release other than the lock's is reported
  (`PINS drift …`) and runs; only a missing package, checkout file or interpreter stops it (`PINS NOT MET`, exit 3).
- **Compile cache.** The first run of a kit mode compiles XLA executables once per GPU model and bucket length (≈½ min
  a bucket on H100; an image built with STOCK.md's optional compile-cache tar starts with the cache classes that tar
  holds). `bash run.sh warm --variant p2 --mode M --input_dir D`
  does it ahead of time.
  - A writable `AF3_JAX_CACHE_ROOT` keeps the cache across runs on that machine (`MODEL_OPT_JIT_ROOT` stands in for it when
    unset). Relocated to a root of your own, the cache is rebuilt once per mode on first use there, then persists.
  - The first input of every process also pays the script's one kernel-autotuning attempt (≈20 s in every kit mode), which
    no cache holds.

## Run

```bash
X=opt/forward/fast_inference/tests/inputs               # example input: barnase–barstar, 5 seeds, MSAs inline
bash run.sh pred --config h100 --variant p2 --mode off   --input_dir $X --output_dir out/off     # stock, in a clean subprocess
bash run.sh pred --config h100 --variant p2 --mode exact --input_dir $X --output_dir out/exact   # stock arithmetic, faster
bash run.sh pred --config h100 --variant p2 --mode fast  --input_dir $X --output_dir out/fast    # the default mode
bash run.sh pred --config h100 --variant p2 --mode big --input_dir $X --output_dir out/big
bash run.sh pred --config h100 --variant p2 --mode big --n_gpu 2 --input_dir $X --output_dir out/big_x2   # 2, 4 or 8 GPUs of one host
```

**Options.** `pred` passes `run_alphafold.py`'s own flags through unchanged — `--json_path <file>` for one input,
`--num_recycles`, `--buckets`, `--cache_dir`, …; a flag you give replaces the kit's default for it. The kit's own flags are:

- `--mode`, `--variant`, `--config`;
- `--n_gpu`;
- `--n_est` — your own token estimate (`big` only);
- `--model_dir` — a weights directory of your own.

`af3-jax-opt pred …` is the same command without `run.sh` (`source configs/h100.env` first). `AF3_JAX_OPT`, `AF3_JAX_VARIANT`
and `AF3_JAX_N_GPU` stand in for `--mode`, `--variant` and `--n_gpu` when the flags are absent.

**Inputs.** They carry their MSAs (multiple sequence alignments) and templates inline; `--norun_data_pipeline` is always on.

**Outputs.** They land where stock writes them:

- `<output_dir>/<job>/seed-<s>_sample-<k>/…` and `ranking_scores.csv` (`<job>_<timestamp>/` when that directory already exists);
- plus the model process's log, `<output_dir>/af3_jax_opt.log`.

Stock stamps each prediction with a notice that points at the AlphaFold 3 model-parameter and output terms of use; the terms
themselves are `stock/src/WEIGHTS_TERMS_OF_USE.md`, `WEIGHTS_PROHIBITED_USE_POLICY.md` and `OUTPUT_TERMS_OF_USE.md`, as upstream ships them.

**What a run prints** (stderr).

- A kit mode prints `[af3-jax-opt] ACTIVE mode=<m> … levers=<A+B+…> n_gpu=<P>` before the model starts — the `ACTIVE` line,
  naming the mode and the optimizations engaged (the kit calls its individually switchable optimizations 'levers'). It ends
  with one `LEVER name=… state=…` line per optimization and `DONE status=ok …`.
- `off` prints `ACTIVE mode=off … script=run_alphafold.py launcher=none …` and `STOCK … proof=ok`, with no `LEVER` lines.
- If a mode cannot engage, the command exits 3 with `NOT ACTIVE … reason=<reason>`; it never falls back to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | the prediction failed |
| 2 | usage error |
| 3 | not active |
| 5 | an attention implementation the mode relies on was absent or fell back (`KERNELS … REFUSED`) |

## Modes

- `off` — stock `run_alphafold.py` as released (the package carrying the one declared patch, STOCK.md 'Stock exceptions'),
  in a clean subprocess with none of the kit's variables; the kit passes `--norun_data_pipeline`, a multiple-of-64
  `--buckets` list and a fresh `--cache_dir` unless you state your own.
- `exact` — the kit script `run_alphafold_fast.py` (featurisation prefetch, writer thread, one autotune attempt) plus
  `GLUT` (fused triangle-multiplication GLU kernel) and `ATTNCFG` (pinned flash-attention tiles); outputs identical to `off`
  when both compile into one `--cache_dir` (Notes). Use it when outputs must not move.
- `fast` (default) — `exact`'s script plus the shared core's FlashPairformer triangle kernels, flash attention, the
  diffusion hoists and a bf16 sampler; bf16 re-association, within stock's seed-to-seed variation. Use it for throughput.
- `big` — above 1,408 padded tokens the pair stack's transitions, diffusion conditioning and triangle multiplication run in
  row shards with the hoist off (below that: `fast`'s program); numerics as `fast`. `--n_gpu 2|4|8` row-shards the pair
  representation over P GPUs of one host.

## Notes

- **A100** (`--config a100`): every optimization of every mode engages, except that `big`'s memory line leaves the fused
  triangle attention (`FPF_TRIATT`) and fused pair transition (`TTR`) off on that card class (a `BIG NOTE` line says so).
- **H200** (`--config h200`): the H100's compute capability, so the same optimizations, kernel tables and size gates as on
  H100, with a compile-cache class of its own (an H100-built cache does not serve it; its first run compiles).
- **Other cards** have no config; use the same generation's. A card without its own tile table gets safe tile rows
  (`tiles=safe:<gen>`); below compute capability 8.0, `fast` and `big` exit 3.
- **A mode without some optimizations.** `MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]` runs a mode without the named ones
  (names as printed on the `LEVER` lines); the `ACTIVE` line then carries `ablated=<names>`. A name outside the mode's
  composition, or any name under `off`, is rejected with a message naming it.
- **Seeing `exact` equal `off` bit for bit.** Two passes that compile separately — two `off` passes included — differ in
  trailing digits (XLA autotunes at each compilation, and diffusion samples then diverge). Compile both into one cache
  directory on one card: `pred --mode exact …` first, then `pred --mode off … --cache_dir <the cache= path on exact's ACTIVE line>`,
  with `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES` left unset.
- **Harmless messages.** tokamax's `Failed to parse autotuning cache file …` / `Tokamax autotune unavailable …` (all modes);
  the `os.fork() …` RuntimeWarning (kit modes).
- **Out of memory.** Under `fast` → use `big`; still out of memory → `big --n_gpu P`.
