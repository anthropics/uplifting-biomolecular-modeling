# ESMFold2 binder design — optimization kit

Drop-in modes that make the ESM cookbook's gradient-guided binder design (`cookbook/tutorials/binder_design.py`,
esm 3.4.0, on the Biohub `transformers` fork's ESMFold2 and ESM-C) faster per design step. You call the design recipe
exactly as before; the kit adds a `--mode`:

- `off` — stock (the unmodified upstream script) plus one declared dtype-cast fix the shipped script needs on this stack
  (STOCK.md, 'Stock exceptions').
- `exact` — stock's arithmetic, faster. Bit for bit the same as stock under `--det 1`; without `--det 1` neither stock
  nor `exact` repeats a design from run to run.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` stock's arithmetic, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs.

## Setup

Pick ONE way to get the pinned stack — esm 3.4.0 plus the Biohub `transformers` fork 4.57.6 and all they need (every
pin: the 'Stack' section of `STOCK.md`): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- one GPU per design, of compute capability 9.0 (H100) or 8.0 (A100, using the image built with
  `--build-arg STACK=img_ef2inv_a100`);
- an NVIDIA driver ≥ 570 on the host.

Route C additionally needs:

- `uv`, git, a C++ compiler and CUDA 12.8's `nvcc`; about 8.5 GB of disk. The route-C lines below are the whole recipe;
  STOCK.md's Stack section has the same lines annotated — run one or the other.

Type the first block from the directory that holds `ef2inv/` and `common/`:

```bash
# A — Docker (recommended: the whole pinned stack, stock and this kit in one image)
docker build -f ef2inv/environment/Dockerfile -t ef2inv-kit:dev .      # A100: add --build-arg STACK=img_ef2inv_a100
docker run --rm -it --gpus all -w /kit -v /weights/ef2inv/hf:/weights/ef2inv/hf -v "$PWD/out":/kit/ef2inv/out ef2inv-kit:dev bash
# B — Apptainer (the whole Setup for B): build the .sif once where Docker runs (converts A's image); the cluster then needs only Apptainer. In kit(), left of ":" = your host directory; the right side and HF_HOME are paths inside the image
apptainer build ef2inv.sif ef2inv/environment/apptainer.def            # from local ef2inv-kit:dev (A100: A + STACK=img_ef2inv_a100)
mkdir -p out jit; export MODEL_OPT_JIT_ROOT="$PWD/jit"; kit() { apptainer run --nv --env HF_HOME=/weights/ef2inv/hf --bind /weights/ef2inv/hf:/weights/ef2inv/hf --bind "$PWD/out":/kit/ef2inv/out ef2inv.sif "$@"; }   # kit = B's ./run.sh · other cards: --config a100|h200
kit install --weights /weights/ef2inv/hf         # /weights/ef2inv/hf: hub/… fetched or checked; read-only ok
kit check --config h100 --mode fast              # each Run line alike: kit <verb> … (out/… = ./out)
# C — instead of A or B, on your own host (venv; needs uv, git, a C++ compiler, CUDA 12.8's nvcc, ~8.5 GB): the whole recipe — STOCK.md §Stack has it annotated; run one or the other
uv venv --seed --managed-python --python 3.12.14 ~/ef2inv-env && . ~/ef2inv-env/bin/activate
grep -v -E '^(#|flash-attn==|transformer-engine==)' ef2inv/environment/requirements.lock > /tmp/stack.txt && pip install --no-deps -r /tmp/stack.txt
git clone https://github.com/Biohub/esm esm && git -C esm checkout d0207ea3c8cbf072679ece3bb332787d25e06852 && pip install --no-deps -e ./esm
bash ef2inv/environment/build_wheels.sh                               # builds flash-attn + transformer_engine, ≈16 min (24 jobs)
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 NVTE_FRAMEWORK=pytorch    # in every shell that runs the kit
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`.
Under **A** you are now in the container shell, which opens in `/kit`. Under **C** you are in your activated
environment: after the route-C lines above, type the whole next block; if you used STOCK.md's Stack section instead, it
already ran `cd ef2inv` and the install line, so continue at `export`. Run:

```bash
[ -f run.sh ] || cd ef2inv                      # no-op once inside · C via §Stack: skip install too
bash run.sh install --weights /weights/ef2inv/hf   # --weights: ≈30 GB if absent, else hash-check; read-only ok
export HF_HOME=/weights/ef2inv/hf               # holds hub/models--biohub--*/snapshots/… (STOCK §Pin)
bash run.sh check --config h100 --mode fast        # any mode; 1–2 min, no GPU work; exit 0 ACTIVE / 3 REFUSED
```

What the blocks assume:

- **Weights directory.** `/weights/ef2inv/hf` in the blocks; `HF_HOME` must name it. It holds the Hugging Face
  snapshots (`hub/models--biohub--*/snapshots/…`; STOCK.md, 'Pin').
  - `install --weights` fetches the snapshots if absent, else hash-checks what is there, so a complete directory may be read-only.
- **Paths under B.** In the `kit()` line, the left side of each `--bind …:…` is your host directory; the right side, and
  `HF_HOME`, are paths inside the image. The `out/…` of the Run examples is the host's `./out`.
- **Pin check on route C.** It stops with exit 3, naming the install, when esm is not the editable checkout at
  `d0207ea3`, the fork is not installed from commit `ef32577f`, or a cookbook / modelling file differs from the pin.
  Other flash-attn / transformer_engine versions are reported `NOT PINNED` and still run.
- **GPU cards.** `--config h100|a100|h200` loads `configs/<card>.env` (`EF2INV_GPU`, `EF2INV_GPU_MIB`, `EF2INV_REQUIRE_FAST_ENV`).
- **First run compiles.** A mode's first run compiles Inductor blocks and Triton kernels once; `bash run.sh warm
  --config h100 --mode fast --target-name pd-l1 --binder-len 80 --out out/warm` does that ahead of time. The first
  `design` launch on a machine also verifies the pinned weights by sha256 (about half a minute on H100, a minute on
  A100) and records the result under `~/.cache/ef2inv_opt`; later launches skip the check unless the files change. The
  digest memo lives in `$XDG_CACHE_HOME/ef2inv_opt/weights_sha256.json` (default `~/.cache/ef2inv_opt/`); persist or
  bind-mount that directory when running in a container, otherwise every launch re-hashes the ~28 GiB of weights
  (roughly 30–80 s).
- **Compile cache.** `MODEL_OPT_JIT_ROOT` keeps the compiled kernels; route B's block puts it in `./jit` on the host.
- **Harmless start-up lines.** The upstream `No checkpoint found … forward` lines printed while the stack imports are harmless.

## Run

```bash
T="--target-name pd-l1 --binder-len 80 --seed 0"                   # or your own: --target-name N --target-sequence <residues>
bash run.sh design --config h100 --mode off   $T --out out/off        # stock + the declared dtype-cast fix, in a clean subprocess
bash run.sh design --config h100 --mode off   $T --chunk-size none --kernel-backend cuequivariance --out out/stock   # upstream's two speed settings
bash run.sh design --config h100 --mode exact $T --out out/exact
bash run.sh design --config h100 --mode fast  $T --out out/fast       # the default when no mode is named
bash run.sh design --config h100 --mode big $T --out out/big
```

Every mode runs the unedited cookbook file's `ESMFold2Design().load()` → `.design()` recipe and applies the declared
dtype-cast fix after load; the kit modes attach their changes to the loaded models. One line of the block takes 2–3.5
min on an H100. Under `--mode off` the upstream model runs unchanged; its first run compiles upstream's own kernels
(roughly 20 s on H100) into the cache root and later runs reuse them.

**Options.** Besides the flags shown, the kit's own are `--det`, `--upstream-fix` and `--allow-partial` (run on with
`evidence=partial` when an optimization cannot engage, instead of exit 3). The cookbook's own parameters pass through
(`--batch-size` > 1: `off` only). Your own target: `--target-name N --target-sequence <residues>`.

**Outputs** (in `--out`).

- `design.pdb` / `.cif` / `.fasta`, and `critic_<name>.pdb` / `.cif`, `critics.json`;
- `trajectory.jsonl` (losses per step), `steps.jsonl` (time per fold call);
- `run.json`, `run.log`, `launch.log`, `activation.json`, `opt_manifest.json`;
- `stock_env_proof.json` in addition, under `off`.

**What a run prints** (on stderr).

- Each design prints `[ef2inv-opt] ACTIVE mode=<mode> …` once the mode is confirmed — the `ACTIVE` line, naming the
  mode and the optimizations engaged (`off`: `tier=stock kit_switch=none`).
- Then one `LEVER name=… state=…` line per optimization (the kit calls its individually switchable optimizations
  'levers'), and at the end `[ef2inv-opt] EXIT rc=<n> mode=<mode> … out=<dir>`.
- If a mode cannot engage, the command exits 3 with one `NOT ACTIVE … <reason>` line; it never falls back to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | complete |
| 1 | output missing |
| 2 | usage error |
| 3 | not active |
| 4 | the design loop failed |

## Modes

- `off` — stock plus the declared dtype-cast fix, in a clean subprocess (`stock_env_proof.json`); `--chunk-size` /
  `--kernel-backend` are upstream's own setters (the second `off` line in Run applies upstream's two speed settings).
- `exact` — less work and CUDA-graph replay of the same arithmetic: memory-planned checkpointing, tuned cuEquivariance
  tiles, unused passes skipped, ESM-C target states hoisted while a first-pass check holds. Bit-identical to
  `off --chunk-size none --kernel-backend cuequivariance` under `--det 1`.
- `fast` — `exact` plus fused Triton / CUDA C++ kernels for the pair trunk's triangle multiplication, transition and
  LayerNorm backward under grad, and a bf16 confidence head (bf16 operands, reordered fp32 accumulation: the documented
  small differences, not bitwise).
- `big` — `fast`'s kernels at the memory floor: every pair block checkpointed, no graph pools, the hoist and overlap
  optimizations off (`LEVER … state=off reason=mode:big`); numerics as `fast`. For when `fast` runs out of memory.

## Known upstream issues

| `--upstream-fix <ID>` | what it fixes | default |
|---|---|---|
| `EF2INV-0003` | ESM-C's rotary embedding runs outside autograd (flash-attn's raw Triton launcher), so the LM gradient term loses its q·k path; the fix runs the fork's torch rotary path under grad (`esmc_rope=torch`) | off |

- Fixes are opt-in on every mode, `off` included (`UPSTREAM-FIX <ID> applied …`).
- Always on: a dtype cast on `Transition._addmm_residual` (STOCK.md, 'Stock exceptions').

## Notes

- **A100** — `--config a100` on the `STACK=img_ef2inv_a100` image. There the ESM-C hoist and the transition kernel that
  exists only for compute capability 9.0 step aside at start (`LEVER … reason=cc_unproven:sm_80` and
  `… reason=not_sm_90:sm_80`), and under `fast` / `big` so does the no-save triangle-multiplication pass
  (`… reason=cc_no_gain:sm_80`).
- **H200** — `--config h200`: the H100's optimizations and compile-cache key. The hoist's first-pass check
  (hoisted == full, bitwise) fails there, so it steps aside (`reason=off:anchor`) and the full pass serves.
- **Any other card** prints one `NOTE hardware … proceeding` line and runs.
- **Switching optimizations off.** `MODEL_OPT_LEVERS_OFF=<lever>[,…]` runs a kit mode without the named optimizations
  (`ablated=` on the `ACTIVE` line).
- **`--no-compile`** is accepted for uniformity and changes nothing here: the only `torch.compile` is stock's own, kept
  in every mode (`compile=stock`).
