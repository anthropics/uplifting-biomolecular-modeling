# Enformer — stock, as pinned

## Pin

Upstream: `enformer-pytorch` 0.8.12, https://github.com/lucidrains/enformer-pytorch at commit `29cd2529f31cc41456da7511e720d730f0bdd19d`,
shipped under `stock/` as the PyPI wheel and sdist (`enformer_pytorch-0.8.12-py3-none-any.whl`, `enformer_pytorch-0.8.12.tar.gz`; sources
unpacked under `stock/src/` for reading; file sizes and the commit in `stock/PINS.json`); weights: `EleutherAI/enformer-official-rough`
(`pytorch_model.bin`, snapshot and sha256 in `stock/PINS.json` `weights`), fetched by upstream's own `from_pretrained` through the Hugging
Face hub into `HF_HOME` at first use (their model card, carried as `stock/snapshot/README.md`, states the licence `cc-by-4.0`;
THIRD_PARTY_NOTICES.md) — the kit never loads, converts or caches weights; upstream's licence (MIT) as found under
`stock/src/LICENSE`. `stock/` is never edited and `run.sh install` installs the wheel as is.

## Stack

Debian 12 (`python:3.11.12-slim-bookworm`), CUDA 13.0 from the lock's NVIDIA wheels (the `cuda-toolkit` 13.0.3.0 meta-package and the
`nvidia-*` runtime, cuBLAS, cuDNN and NCCL wheels torch loads; no `nvcc`, nothing installed system-wide; host driver 580 or newer), Python 3.11.12, torch 2.13.0+cu130 (PyPI `torch==2.13.0`, cuDNN 9.20.0.48), triton 3.7.1, einops 0.8.2, numpy 2.4.6, transformers
4.56.2 (upstream's own pin: `Enformer` subclasses its `PreTrainedModel`), nvidia-cudnn-frontend 1.27.0 (the one distribution the lock adds
over upstream's dependency set) — the full list is `environment/requirements.lock`; `environment/Dockerfile` and `apptainer.def` build it.
Outside the images the pinned recipe is a fresh Python 3.11 environment and the lock, `--no-deps` (the same distributions the Dockerfile
installs; pip downloads torch 526.6 MB, triton 197.7 MB and the NVIDIA CUDA 13 wheels — several GB installed):

    sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential curl ca-certificates   # bare Ubuntu host (root: no sudo): build-essential is the one package the image adds, curl + ca-certificates fetch uv — skip what you have; the kit itself compiles nothing
    command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user; installer digest-checked
    uv venv --seed --managed-python --python 3.11.12 ~/venv-enformer && . ~/venv-enformer/bin/activate   # uv fetches its own released CPython 3.11.12; any released CPython 3.11 works instead (python.org, deadsnakes python3.11 + python3.11-venv, conda), not Ubuntu 22.04's python3.11 apt package (3.11.0rc1); 3.11 because the lock's nvidia-cudnn-frontend wheel is cp311-only
    cd enformer
    pip install --no-deps -r environment/requirements.lock                   # every pinned distribution; --no-deps because the lock is complete
    bash run.sh install                                                          # = README step 2's install line — type it once; step 2 continues at bash run.sh check

An existing environment that already runs `enformer-pytorch` 0.8.12 on this stack takes `bash run.sh install` alone (the stock wheel from
`stock/` and the kit from `opt/`, both `--no-deps`, the kit editable; then `stock/check_pins.py`, which confirms the installed
`enformer-pytorch` matches the pin file for file — exit 3 otherwise — and lists every distribution that differs from the lock by name,
exit 4, informational).
The wheel in `stock/` (like PyPI's) installs without a build step; an install from the sdist `stock/enformer_pytorch-0.8.12.tar.gz` or
from upstream's git URL makes pip fetch the build backend (hatchling) from the index at install time (network needed) — the lock pins
runtime packages only.
Other cards: nothing to configure — the kit selects its kernel build from the device's compute capability (README §Notes).

## How stock is run

Upstream has no command line; its product is the Python API, and this call is what the kit accelerates in place:

    model = enformer_pytorch.from_pretrained('EleutherAI/enformer-official-rough').cuda().eval()   # the loader sets use_tf_gamma=True for this id
    x = enformer_pytorch.str_to_one_hot(seq).cuda()     # one 196,608-bp window -> (196608, 4) float32; a list of windows -> (B, 196608, 4)
    with torch.no_grad():
        out = model(x)                                   # {'human': (896, 5313), 'mouse': (896, 1643)} float32; (B, 896, n) for a batch

`off` (`ENFORMER_OPT` unset or `off`, no `enable()`) is exactly this, in the caller's own process: the package's `.pth` installs no hook
and imports nothing of the levers. Stock settings, none of them set by the kit: fp32 weights and activations at PyTorch's default numerics
(matmul TF32 off, cuDNN TF32 on, cuDNN autotuner off, `float32_matmul_precision` highest); `eval()` and `no_grad()`; one-hot A/C/G/T with
`N` = zeros; 196,608 bp in (the pretrained route's `use_tf_gamma` positional table covers the trunk's 1,536 positions), 896 bins of 128 bp
out. Upstream documents no speed switch, needs no optional accelerator library, and batches natively (a list of windows -> one
`(B, L, 4)` call).

## Stock exceptions

None.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `ENFORMER_OPT` | no | unset | `exact` engages the kit at interpreter start (right after `enformer_pytorch` is first imported); unset or `off` = stock; any other value exits 3 by name |
| `PYTHON` | no | `python` | the interpreter `run.sh install` / `run.sh check` use |
| `HF_HOME` | no | Hugging Face default (`~/.cache/huggingface`) | upstream's weights cache root, read by `from_pretrained`: the checkpoint lives under `hub/models--EleutherAI--enformer-official-rough/` (snapshot commit and sha256: `stock/PINS.json` `weights`); the kit does not read it |
| `HF_HUB_OFFLINE` | no | unset | `1` = `from_pretrained` serves the snapshot already in `HF_HOME` without contacting the hub (Hugging Face's own switch; the kit does not read it) |

The kit declares no other variable, flag, file format or side file.
