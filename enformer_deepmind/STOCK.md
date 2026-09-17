# Enformer (official TensorFlow release) — stock, as pinned

## Pin
Upstream: the TF-Hub SavedModel `https://tfhub.dev/deepmind/enformer/1` — one prediction function, `predict_on_batch`, taking a float32
one-hot tensor `(batch, 393216, 4)` (ACGT order, `N` = zeros) and returning `{'human': (batch, 896, 5313), 'mouse': (batch, 896, 1643)}`;
the network reads the central 196,608 bp. `stock/` holds the pin only (`stock/PINS.json`: the sha256 of the model's three files
`saved_model.pb`, `variables/variables.data-00000-of-00001`, `variables/variables.index`, and the documenting source
`google-deepmind/deepmind-research` commit `f5de0ede8430809180254ee957abf36ed62579ef`, directory `enformer/`). The model is fetched
by `tensorflow_hub` itself at the first `hub.load(handle)`, or read from `TFHUB_CACHE_DIR` when that cache holds it
(`<TFHUB_CACHE_DIR>/c444fdff3e183daf686869692c26e00391f6773c/`); `python -I stock/check_pins.py --weights <dir>` compares a cache or
model directory with the pinned digests, and `bash run.sh install` runs that comparison, `--quiet`, when `TFHUB_CACHE_DIR` is set to an existing directory: silent when the files match,
`WEIGHTS REFUSED: …` and exit 3 otherwise (run the script yourself without `--quiet` for a `WEIGHTS OK` line per file).
Licence: upstream's Apache-2.0 licence — the repository-root `LICENSE` of `google-deepmind/deepmind-research` at that commit, which governs
`enformer/` — is copied unmodified as `stock/LICENSE`. `stock/` is never edited.

## Stack
Debian 12 (`python:3.11.12-slim-bookworm`), Python 3.11, `tensorflow[and-cuda]==2.17.1` (CUDA 12.3 / cuDNN 8.9 libraries, ptxas and
libdevice from the `nvidia-*` wheels; no CUDA toolkit; host driver 545 or newer), `tensorflow-hub==0.16.1`, `numpy==1.26.4`,
`setuptools<81` (tensorflow-hub imports `pkg_resources`) — the full list is `environment/requirements.lock`; `environment/Dockerfile`
builds it. The same stack in an environment of your own (README.md route C), from the directory holding `enformer_deepmind/`:

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates   # bare Ubuntu host (root: no sudo); skip what you have
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; } # uv 0.12.15 itself, once per user; installer digest-checked
uv venv --seed --managed-python --python 3.11 enformer-env && . enformer-env/bin/activate   # uv's own released CPython 3.11 build; any released CPython 3.11 works instead (python.org, conda, or apt/deadsnakes: python3.11 + python3.11-venv + python3.11-dev) — not Ubuntu 22.04's python3.11 apt package (3.11.0rc1)
pip install --no-deps $(grep -E '^(pip|wheel|setuptools)==' enformer_deepmind/environment/requirements.lock)   # the pinned pip / setuptools / wheel first, as the Dockerfile does
pip install --no-deps -r enformer_deepmind/environment/requirements.lock         # the pinned stack; CUDA 12.3 libraries arrive as nvidia-* wheels (host driver 545 or newer)
export PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=1                                   # optional: the two settings the image carries (hash seed; TensorFlow start-up log trimmed to warnings)
```

Then README.md Setup's second block, typed once: `cd enformer_deepmind`, the optional export, `bash run.sh install` (the kit editable — it must stay editable: the op library and `stock/PINS.json` are found
from the package's location — then `stock/check_pins.py --stack`, which lists any installed `tensorflow` / `tensorflow-hub` / `numpy` that
differs from the pin — its own status 4 is informational: `install` carries on and finishes 0), `bash run.sh check`. An environment already at the pin works as
well: `install` names any drift. One stack serves H100 / H200 and A100.

## How stock is run
The user's own script without the switch: `tensorflow_hub.load("https://tfhub.dev/deepmind/enformer/1").model.predict_on_batch(x)`
(equivalently `tf.saved_model.load(<dir>).model.predict_on_batch(x)`) in a process where `ENFORMER_DEEPMIND_OPT` is unset or `off`
and `enable()` was not called, under TensorFlow's defaults: float32, TF32 tensor-core arithmetic where TensorFlow enables it (compute
capability 8.0 and newer), cuDNN / cuBLAS algorithm selection as TensorFlow ships it, no determinism or XLA option. The libraries
choose kernels by shape, so the stock call's bytes depend on the batch size: "identical outputs" always means identical to the stock
call at the same batch size on the same GPU model. TensorFlow's cuDNN autotuner picks convolution algorithms by timing, per process,
so two stock processes can return different bytes for one input; TensorFlow's `TF_CUDNN_USE_AUTOTUNE=0` makes the choice cuDNN's
heuristic, fixed per device and shape. The kit leaves the convolutions to TensorFlow and follows whatever its process chooses.

## Stock exceptions
- STACK: the SavedModel is run on TensorFlow 2.17.1 rather than the repository's `requirements.txt` pins (TensorFlow 2.5, dm-sonnet
  2.0.0), which have no build for CUDA 12 / compute capability 9.0. The SavedModel — the documented inference route — runs unmodified on
  this stack, with and without the kit; the repository's Sonnet-module route (`enformer.py` restored from the `sonnet_weights`
  checkpoint) needs the old pins and is outside this kit.

## Variables
| variable | required | default | effect |
|---|---|---|---|
| `ENFORMER_DEEPMIND_OPT` | no | unset (= `off`) | `exact` engages the kit at interpreter start for every Enformer SavedModel loaded afterwards; `off` does nothing; any other value is refused by name, exit 3 |
| `TFHUB_CACHE_DIR` | no | tensorflow-hub's default | tensorflow-hub's own cache location; `bash run.sh install` checks the cached model against the pin when it names an existing directory |
| `PYTHON` | no | `python` | the interpreter `run.sh` uses |

The kit's code reads no other variable and sets none, and changes no TensorFlow numerics or execution option. `environment/Dockerfile`
sets two in the container, for stock and kit alike: `PYTHONHASHSEED=0` and `TF_CPP_MIN_LOG_LEVEL=1` (TensorFlow's C++ log threshold:
INFO lines suppressed); neither affects what the model computes.
