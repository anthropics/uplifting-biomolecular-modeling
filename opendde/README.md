# OpenDDE — optimization kit

Drop-in modes that make stock OpenDDE 1.1.1 (`opendde pred`, model `opendde_v1`) faster and lighter on GPU memory.
You call OpenDDE exactly as before; the kit adds a `--mode`:

- `off` — stock OpenDDE as released ('stock' below always means this unmodified upstream release), with its two
  documented speed settings on, as under every mode (see Modes).
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs; `--n_gpu P` splits one `big` prediction across P GPUs of one host.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs, up to 4,000 tokens on one GPU · `--n_gpu P` splits `big` across P GPUs of one host.

## Setup

Pick ONE way to get the pinned stack (every pin: the 'Stack' section of `STOCK.md`): **A — Docker**, **B — Apptainer**,
or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver ≥ 560.

Route C additionally needs:

- the CUDA 12.6 toolkit (`nvcc`) and gcc;
- for `exact`, a libstdc++ from GCC 13 or newer (older: the one-line remedy in STOCK.md's Stack section, or
  `MODEL_OPT_LEVERS_OFF=dit_attn_exact`);
- `uv` and, on a bare host, the apt line — both in STOCK.md's Stack section. The route-C lines below are the whole
  recipe; STOCK.md's Stack section has the same lines annotated — run one or the other.

Type the first block from the directory that holds `opendde/` and `common/`:

```bash
# A — Docker (recommended: the whole pinned stack, stock and this kit in one image)
docker build -f opendde/environment/Dockerfile -t opendde-kit:dev .
docker run --rm -it --gpus all -v /weights/opendde:/weights/opendde -v $PWD/out:/kit/opendde/out opendde-kit:dev bash   # a shell in /kit/opendde
# B — Apptainer / Singularity (the whole Setup for B; `kit` below: a shell function = B's bash run.sh): converts A's image on any machine with Docker; copy the .sif to the cluster, which needs only Apptainer
apptainer build opendde-kit.sif opendde/environment/apptainer.def
mkdir -p out; kit() { apptainer run --nv --bind /weights/opendde:/weights/opendde --bind "$PWD/out":/kit/opendde/out opendde-kit.sif "$@"; }   # B's ./run.sh · bound dir: checkpoint/opendde.pt + common/
export OPENDDE_ROOT_DIR=/weights/opendde && kit install --weights /weights/opendde   # fetched if absent, else digest-checked; read-only ok
kit check --config h100 --mode fast                       # each Run line alike: kit <verb> …; cards: --config a100|h200
#     with --config <card>, B's compile cache lands in ~/.cache/opendde_opt/jit on the host (the config's MODEL_OPT_JIT_ROOT; Apptainer binds $HOME), seeded from the image on the first verb
# C — instead of A or B, on your own host (getting uv, the bare-host apt line: STOCK.md §Stack) — the whole route-C recipe; STOCK.md §Stack = these lines annotated (run one or the other)
uv venv --seed --managed-python --python 3.11 ~/odde && . ~/odde/bin/activate
grep -v -E '^(#|opendde==)' opendde/environment/requirements.lock > /tmp/stack.txt && pip install --no-deps -r /tmp/stack.txt   # ≈6 GB
pip install --no-deps opendde/stock/opendde-1.1.1-py3-none-any.whl
export PYTHONHASHSEED=0                                    # the images set it too
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …` on the
host (the `kit()` shell function defined above stands in for `bash run.sh`). Under **A** you are now in the container
shell, which opens in `/kit/opendde`, so the `cd` is a no-op. Under **C** you are in your activated environment: after
the route-C lines above, type the whole next block; if you used STOCK.md's Stack section instead, it already ran
`cd opendde` and the install line, so continue at `export`. Run:

```bash
[ -f run.sh ] || cd opendde                   # no-op once inside · C via §Stack: skip install too
bash run.sh install --weights /weights/opendde   # A: already installed, only --weights acts
export OPENDDE_ROOT_DIR=/weights/opendde      # required: checkpoint/opendde.pt and common/
bash run.sh check --config h100 --mode fast      # dry run, exit 0 when the mode resolves
```

What the blocks assume:

- **Weights directory.** `/weights/opendde` in the blocks (under B, the host directory bound there); it holds
  `checkpoint/opendde.pt` and `common/`, and `OPENDDE_ROOT_DIR` must name it.
  - `install --weights DIR` fetches what is absent with upstream's downloader and digest-checks what is present; omit
    `--weights` once downloaded. A complete directory may be read-only.
  - Under A the kit is already installed, so only `--weights` acts.
- **Outputs under A and B.** `out/…` is the mounted host directory `$PWD/out`.
- **Inputs under B.** Give your own inputs as absolute host paths; the shipped example's path exists inside the image.
- **Compile cache under B.** With `--config <card>`, it lands in `~/.cache/opendde_opt/jit` on the host (the config's
  `MODEL_OPT_JIT_ROOT`; Apptainer binds `$HOME`); the first command fills it — seeded from the image's cache when the
  image was built with STOCK.md's optional compile-cache tar, compiled otherwise.
- **What `install` checks.** It ends with `opendde 1.1.1 == pin 1.1.1; …/… package files are the pinned wheel's bytes`.
  The pin check, repeated at every command's start, accepts any install whose files are the pinned wheel's bytes (the
  bundled wheel, `pip install opendde==1.1.1`, a checkout at the pinned commit) and refuses another version or a
  patched, missing or extra file, naming it.
- **What `check` prints.** `[opendde-opt] DRY RUN mode=fast …`, `stack: STACK OK …` and
  `checkpoint: True at <root> match=pinned` (`False`: the weights root is not visible there). It compares versions and
  digests, loads no model and launches no kernel; exit 0 when the mode resolves.
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env` (`MODEL_OPT_JIT_ROOT`, `MODEL_OPT_TARGET_GPU`, the
  small-input floor; `h100.env` / `h200.env` also `big`'s two size gates); every command takes it.
- **First run compiles.** The first run of a kit mode compiles its kernels (Triton, cuEquivariance and LayerNorm caches)
  once per machine, about a minute; the ≈45 s LayerNorm build also happens on the first `--mode off`. The first run in a
  fresh cache directory also performs a one-time numerical self-check of the fused kernels on your GPU (seconds on H100;
  up to ~30 s on A100) before serving them; later runs skip it.
  - Compare modes on a second run.
  - `bash run.sh warm --config h100 [--mode M]` compiles ahead of time, and a writable `MODEL_OPT_JIT_ROOT` keeps the cache.

## Run

```bash
Q=stock/src/examples/example_without_msa.json; M="--use_msa false"         # upstream's example query (it carries no MSA files)
bash run.sh pred --config h100 --mode off   -i $Q -o out/off   $M             # stock, in a clean subprocess
bash run.sh pred --config h100 --mode exact -i $Q -o out/exact $M
bash run.sh pred --config h100 --mode fast  -i $Q -o out/fast  $M             # the default when no mode is named
bash run.sh pred --config h100 --mode big -i $Q -o out/big $M
bash run.sh pred --config h100 --mode big --n_gpu 2 -i $Q -o out/big_x2 $M   # P > 1 GPUs of one host, at most the GPUs visible
```

**Options.** `pred` hands `opendde pred` its own flags verbatim: `--seeds --cycle --step --sample --dtype --model_name`,
`--use_msa --use_template --use_rna_msa --need_atom_confidence`, `--triatt_kernel --trimul_kernel`; anything else after
`--`. The kit's own flags are `--mode`, `--n_gpu`, `--det`, `--allow-partial`, `--template_mmcif_dir` and `--root`.
Two other ways to make the same call:

- `opendde-opt pred --mode <mode> …` (or `python -m opendde_opt …`) — the same call without `run.sh`;
- `OPENDDE_OPT=<mode> opendde pred …` — engages a mode from the unchanged stock command line.

**Outputs.**

- They land where stock writes them: `<out>/<name>/seed_<S>/predictions/…`, plus `opt_manifest.json`. With no `--seeds`
  or `modelSeeds` given, each call draws a new seed, hence a new `seed_<S>/`.
- `--n_gpu P` ranks above 0 write under `<out>/.rowpair/`.

**What a run prints** (on stderr).

- A kit mode prints `[opendde-opt] ACTIVE mode=<mode> line=… levers=…` — the `ACTIVE` line, naming the mode and the
  optimizations engaged (the kit calls its individually switchable optimizations 'levers') — then a
  `LEVER name=… state=…` line per optimization and a closing `[opendde-opt] EXIT pid=… <counters>` tally.
- `--mode off` prints `[opendde-opt] NOT ACTIVE mode=off reason=mode off: stock opendde (…)` and runs stock (exit 0).
- If a mode cannot engage, the command prints `[opendde-opt] NOT ACTIVE [mode=<mode>] reason=<reason>` and exits 3; it
  never falls back to stock silently. An optimization that fell back mid-run is reported as `PARTIAL` (exit 3 unless
  `--allow-partial`).

**Exit codes.**

| code | meaning |
|---|---|
| 0 | ok |
| 1 | failed |
| 2 | usage error |
| 3 | not active (or `PARTIAL` without `--allow-partial`) |
| 5 | an accelerator the route expects was absent or fell back |

## Modes

- `off` — stock `opendde pred` as released, in a clean subprocess with nothing of the kit importable; upstream's two
  documented speed settings (`--dtype bf16`, `LAYERNORM_TYPE=fast_layernorm`) are on, as on every mode (STOCK.md).
- `exact` — bit-exact kernels from the shared core (`common/opt_core`, the runtime all kits share) for triangle
  multiplication, pair transition and sampler attention; triangle attention on a fused kernel bit-identical to the stock
  operation on H100 stacks the core vouches for, the stock operation elsewhere; conditioning hoisted, the denoiser step
  graph-replayed, per-step host syncs removed. Identical to `off` under `--det 1`. For outputs that must not move.
- `fast` (default) — `exact`'s scheduling optimizations plus a bf16 pair stack (the shared core's FlashPairformer
  triangle kernels, fused transition and LayerNorms) on trunk, refiner and confidence head, and fused sampler kernels;
  bf16 re-association, not bitwise. For throughput.
- `big` — `fast`'s kernels; from `MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS` tokens the pair tensors stream from pinned host RAM
  and samples run one at a time. `--n_gpu P` row-shards the pair tracks over P GPUs instead (exit 3 under another mode
  or with fewer GPUs visible).

## Notes

- **Where the gain is.** The kit's gain is in prediction time per input. Every call also pays ≈18 s of start-up,
  checkpoint load and featurisation in every mode, so a single ≈400-token input gains less end to end than its
  prediction time does — the gain grows with input size and with inputs per call.
- **Other cards:** `--config a100`, `--config h200` (the H100's settings; bit-exact records are per card). Kernel
  variants follow the running device (named on the `LEVER` lines).
  - On A100 `stepgraph` switches off above 1,024 tokens and `big`'s size gates drop under 64 GiB.
  - Under `exact`, a kernel with no bit-exact record for the running card and stack runs the stock operation and says
    so on its `LEVER … aside` / `NAMED_FALLBACK … served=<stock op>` line (A100: triangle multiplication, sampler
    attention; H200: triangle multiplication).
- **Switching optimizations off.** `MODEL_OPT_LEVERS_OFF=<lever>[,…]` runs a mode without the named optimizations
  (`ablated=<names>` on the `ACTIVE` line).
- **Small inputs.** Below `MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS` on every item, the trunk optimizations switch off (named
  on their `LEVER` lines). A stated `--triatt_kernel` / `--trimul_kernel` runs as stated.
- **Determinism.** `--det 1` selects upstream's deterministic recipe (`--deterministic true` + `CUBLAS_WORKSPACE_CONFIG`)
  on every mode, `off` included; `exact --det 1` equals `off --det 1` bit for bit.
- **MSAs** (multiple sequence alignments). Under upstream's default `--use_msa true`, every protein chain must carry its
  MSA files in the query (`pairedMsaPath` / `unpairedMsaPath` or `msa.precomputed_msa_dir`), else `pred` exits 3 naming
  the chain; nothing is searched or fetched, and `install --weights` brings no MSAs — hence `--use_msa false` in Run.
- **Templates.** `--use_template true`; each protein chain's `templatesPath` names its hits file, with the hit structures
  beside it as `<hits dir>/<pdb>.cif` (or under `--template_mmcif_dir`); `kalign` must be on `PATH`; nothing is searched
  or fetched.
- **Out of memory.** Under `fast` → use `big`; still out of memory → `big --n_gpu P`.
