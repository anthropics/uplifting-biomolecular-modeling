# Enformer (official TensorFlow release) — optimization kit

A drop-in mode that makes the stock Enformer TF-Hub SavedModel `deepmind/enformer/1` faster on NVIDIA H100 / H200 and
A100 — `hub.load(handle).model.predict_on_batch(x)`, as `enformer-usage.ipynb` calls it. Your own script calls it exactly
as before; one environment variable, `ENFORMER_DEEPMIND_OPT`, switches the kit on per process (or an `enable()` call in code):

- `off` — the stock SavedModel, exactly as released ('stock' below always means this unmodified upstream release).
  This is what runs with the variable unset.
- `exact` — identical outputs, faster. The kit ships no other mode.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock.

## Setup

Pick ONE way to get the pinned stack — Python 3.11, TensorFlow 2.17.1, tensorflow-hub 0.16.1 and everything they need
(the 'Stack' section of `STOCK.md` lists every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your
own host**.

All three routes need only:

- an NVIDIA driver of release 545 or newer on the host.

Type the first block from the directory that holds `enformer_deepmind/`:

```bash
# A — Docker (recommended: the whole pinned stack and this kit in one image)
docker build -f enformer_deepmind/environment/Dockerfile -t enformer-deepmind-kit:dev .
docker run --rm -it --gpus all -w /kit -v /weights/tfhub:/weights/tfhub -v $PWD:/work enformer-deepmind-kit:dev bash   # then the next block; keep scripts in /work (= host $PWD)
# B — Apptainer / Singularity (the whole Setup for B; `kit` below: a shell function wrapping apptainer exec): converts the image built in A; needs A once, no Docker daemon at run time
apptainer build enformer-deepmind-kit.sif docker-daemon://enformer-deepmind-kit:dev
kit() { apptainer exec --nv --bind /weights/tfhub:/weights/tfhub --bind "$PWD":/work enformer-deepmind-kit.sif "$@"; }   # kit <cmd> runs it in the .sif; /work = $PWD, as under A
export TFHUB_CACHE_DIR=/weights/tfhub                    # hub's cache: one dir c444fdff3e18…, filled at first load
kit python -I /kit/enformer_deepmind/stock/check_pins.py --weights /weights/tfhub   # only for a pre-filled cache (read-only is fine): pin check
kit bash /kit/enformer_deepmind/run.sh check          # B's check; its Run lines are printed under Run
# C — instead of A or B, on your own host: a Python 3.11 venv — the whole route-C recipe; STOCK.md §Stack = these lines annotated (run one or the other)
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates   # bare Ubuntu host (root: no sudo); skip what you have
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; } # uv 0.12.15 itself, once per user; installer digest-checked
uv venv --seed --managed-python --python 3.11 enformer-env && . enformer-env/bin/activate
pip install --no-deps $(grep -E '^(pip|wheel|setuptools)==' enformer_deepmind/environment/requirements.lock)
pip install --no-deps -r enformer_deepmind/environment/requirements.lock       # CUDA 12.3 libraries arrive as wheels
```

Route B (Apptainer) is complete at this point; its Run lines are your own commands typed as `kit <command> …` (here
`kit` wraps `apptainer exec`). Under **A** you are now in the container shell, which opens in `/kit`; the image already
carries this install, so `install` there merely reinstalls it and re-checks the pins. Under **C** you are in your
activated environment, still in the directory holding `enformer_deepmind/` after the route-C lines above (or after
STOCK.md's Stack section, which is the same lines annotated), so type the whole block. Run:

```bash
[ -f run.sh ] || cd enformer_deepmind    # from the parent dir; a no-op once inside the kit dir
export TFHUB_CACHE_DIR=/weights/tfhub    # optional: hub's model cache (layout: STOCK.md §Pin)
bash run.sh install                         # kit + op library, then the pin check (STACK OK on a match)
bash run.sh check                           # dry run: says whether the mode engages here
```

What the blocks assume:

- **The model (weights).** The model itself is fetched by `tensorflow_hub` at the first `hub.load`, or read from
  `TFHUB_CACHE_DIR` when that cache holds it (fill it on a networked host for an offline one). In the blocks the host
  directory `/weights/tfhub` is that cache.
- **What `check` does.** It imports TensorFlow, finds the CUDA device and loads the op library, then prints
  `[enformer-deepmind-opt] DRY mode=exact gpu=… build=…` (exit 0); it reads no model.
- **If the mode cannot engage** — here or at run time — the kit prints
  `[enformer-deepmind-opt] NOT ACTIVE mode=exact reason=<reason>` and the process exits 3; it never falls back to stock silently.
- **Route C with an existing environment.** Any environment whose TensorFlow is 2.17.1 is accepted: `install` names
  other drift from the pin and carries on.

## Run

The kit adds no command: your own script runs unchanged, from any directory, with the switch in its environment.

- Route A: `cd /work` first — the mounted host directory — so the files outlive the container.
- Route B: on the host, put `kit` before `python`, as the last line of the block shows (the `off` line alike).

With upstream's example call as that script:

```bash
cat > predict.py <<'EOF'
import numpy as np, tensorflow as tf, tensorflow_hub as hub   # tensorflow before tensorflow_hub (Notes)
model = hub.load("https://tfhub.dev/deepmind/enformer/1").model
x = np.zeros((2, 393_216, 4), np.float32)        # one-hot ACGT windows, N = zeros
y = model.predict_on_batch(x)                     # shapes: human (2, 896, 5313), mouse (2, 896, 1643)
print({head: out.shape for head, out in y.items()})
EOF
python predict.py                                 # off: stock
ENFORMER_DEEPMIND_OPT=exact python predict.py     # exact
ENFORMER_DEEPMIND_OPT=exact kit python predict.py # exact under route B (kit from Setup)
```

**Outputs.** The call's return value, as stock; the script's exit status is its own. The kit prints no closing line.

**What a run prints** (stderr):

- Under `exact`: `[enformer-deepmind-opt] ACTIVE mode=exact gpu=<name>(smNN) kit=… levers=… build=… applied=deferred` as
  `import tensorflow` completes (`levers=` lists the optimizations engaged), and
  `[enformer-deepmind-opt] APPLIED model#1 levers=… cost=…s (…) weights=deepmind/enformer/1` at the model's first
  prediction. `weights=…` reads otherwise when the loaded SavedModel is not the released one by digest; it is served as is.
- Under `off`: nothing.

**From code instead of the variable.**

- `import enformer_deepmind_opt; enformer_deepmind_opt.enable()` before the model is loaded is the same switch. If the
  mode cannot engage it returns its report and leaves stock in place; `enable(strict=True)` raises `ActivationError` instead.
- `apply(model, batch_sizes=(2,))` rewrites at once instead of at the first prediction.
- `status()` reports; `disable()` restores stock.

**Exit codes** (`run.sh`).

| code | meaning |
|---|---|
| 0 | ok |
| 2 | usage error |
| 3 | `check` not active, or `install` found a cached model that differs from the pin |

## Modes

- `off` — `ENFORMER_DEEPMIND_OPT` unset or `off` and no `enable()`: the stock SavedModel function; the kit's working
  modules are never imported.
- `exact` — each loaded Enformer SavedModel's prediction graph is rewritten: elementwise, softmax, layer-norm and
  pooling node groups become fused GPU ops with the same float32 arithmetic, and input-independent subgraphs become
  constants (`CHANGES.md`). Outputs identical to `off` at the same batch size.

## Notes

- **One-time rewrite.** A model's first prediction pays the one-time graph rewrite (`cost=…s` on the `APPLIED` line);
  `enformer_deepmind_opt.apply(model, batch_sizes=(…))` pays it ahead of time, together with TensorFlow's own first-call
  work at each listed batch size.
- **Where the gain is.** The kit's gain is in time per `predict_on_batch` call from the second call on. A process also
  pays TensorFlow's start-up, model load and first-call work (≈10 s in either mode) and `exact` its ≈7 s rewrite, so the
  single 2-window call of the Run example ends sooner under `off`.
- **Calls the kit leaves to stock.** Under a `tf.GradientTape` or inside a `tf.function`, `predict_on_batch` runs the
  stock function (`STOCK-CALL model#k …`, printed once per model); a SavedModel whose prediction graph is not the released
  Enformer's runs untouched (`STOCK model#k <reason>`).
- **GPUs.** The op library carries native code for compute capability 8.0 (A100) and 9.0 (H100 / H200), and every
  optimization engages on both. An 8.6 / 8.9 device runs the 8.0 code and a newer one the library's PTX
  (`notes=untested device …` on the `ACTIVE` line); below 8.0 the mode cannot engage.
- **TensorFlow version.** TensorFlow must be 2.17.1, the version `opt/enformer_deepmind_opt/ops/edm_ops.so` is built
  against; `opt/enformer_deepmind_opt/ops/build.sh` rebuilds it for another.
- **Log lines.** The kit's lines start with `[enformer-deepmind-opt]` on stderr. TensorFlow's own start-up E/W/I lines
  (duplicate cuFFT/cuDNN/cuBLAS registration, TF-TRT absent, NUMA node) are expected; `TF_CPP_MIN_LOG_LEVEL=1`, set in
  the image, trims them.
- **Import order.** With the environment switch the kit engages as `import tensorflow` completes, so a script must import
  `tensorflow` before `tensorflow_hub`, as `enformer-usage.ipynb` does; one whose first TensorFlow import happens inside
  `import tensorflow_hub` gets `NOT ACTIVE`, exit 3. `enable()` in code has no such requirement.
- **Run-to-run agreement.** `exact` equals `off` within one process at TensorFlow's defaults. Two processes — even two
  stock runs — can differ in the convolutions, because TensorFlow's cuDNN autotuner picks algorithms per process;
  TensorFlow's `TF_CUDNN_USE_AUTOTUNE=0` in both makes them agree.
- **Batch limit.** At most 14 windows per call: from 15 up TensorFlow 2.17 aborts `predict_on_batch` (a 32-bit launch
  index in its GPU `BiasAdd`), kit or not.
