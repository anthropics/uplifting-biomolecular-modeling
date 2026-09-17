# Evo 2 — stock, as pinned

## Pin

Upstream: PyPI `evo2` 0.6.0 (https://pypi.org/project/evo2/0.6.0/) on `vtx` 1.1.0 (import name `vortex`;
https://pypi.org/project/vtx/1.1.0/), shipped under `stock/` as the published wheels and sdists, byte-identical to the PyPI files (sizes and PyPI URLs in
`stock/PINS.json`), with the same modules unpacked, unmodified, under `stock/src/` for the file:line citations in CHANGES.md. Weights: `evo2_7b.pt`
(13.8 GB), `evo2_40b.pt` (82.3 GB, upstream's two parts concatenated in order as its loader does) and, for mode `fast`'s draft,
`evo2_1b_base.pt` (2.7 GB) from the `arcinstitute/<model_name>` repositories (revision and sha256 in PINS.json), fetched by
`bash run.sh install --weights DIR [--model_name M]` through `huggingface_hub`, the downloader upstream's loader uses; a file already
in DIR is kept and only digest-checked. Both packages are under the Apache License 2.0; their licence and notice files travel inside the
archives under `stock/`, with copies and the notices of the third-party code the vortex archives contain (FlashAttention, causal-conv1d,
NVIDIA CUTLASS) in `third_party_licenses/` and THIRD_PARTY_NOTICES.md. `stock/` is never edited, and the
kit never patches the installed packages on disk: `bash run.sh install` refuses (exit 3) unless the installed `evo2` and `vtx` are
these versions with every file byte-identical to the wheel.

## Stack

Ubuntu 22.04, CUDA 12.8 (driver ≥ 570), Python 3.11.5, torch 2.7.1+cu128 (cuDNN 9.7.1), triton 3.3.1, flash_attn 2.8.0.post2,
numpy 2.2.6 and Transformer Engine 2.3.0 (upstream's full install: FP8 input projections; `evo2_7b` on one device, `evo2_40b` on
two) — the full list is `environment/requirements-img_full.lock`; `environment/Dockerfile` and `apptainer.def` build it. Other
cards: `--build-arg STACK=img_a100` installs `environment/requirements-img_a100.lock`, the same list without Transformer Engine
(upstream's light install): upstream then runs `evo2_7b` on bf16 input projections with its warning `Transformer Engine not
installed. Falling back to bf16 projections`, and `evo2_40b` raises `ImportError` in upstream's constructor. Into an existing
environment that already runs Evo 2 at the pin: `bash run.sh install` (the kit, editable, then `stock/check_pins.py`, which confirms
the installed `evo2` / `vtx` match the wheels file for file and names any torch / triton / flash-attn / Transformer Engine version
off the lock). Without such an environment and without a container, the Dockerfile's steps are the recipe — from the directory holding
`evo2/`, with a released CPython 3.11 (the lock's wheels are cp311; the image unpacks python-build-standalone 3.11.5, `uv` below fetches
the same family), the CUDA 12.8 toolkit (`nvcc`) and gcc on PATH; for cards without FP8 read `requirements-img_a100.lock`
in both `grep` lines (it has no Transformer Engine line, so the build command is skipped):

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl   # bare Ubuntu host (root: no sudo); skip what you have
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; } # uv 0.12.15 itself, once per user; installer digest-checked
uv venv --seed --managed-python --python 3.11 ~/evo2-venv && . ~/evo2-venv/bin/activate      # uv's own released CPython 3.11, headers bundled; if you use apt/deadsnakes python instead of uv: python3.11 + python3.11-venv + python3.11-dev — not Ubuntu 22.04's python3.11 package (3.11.0rc1)
grep -v -E '^(#|evo2==|vtx==|transformer-engine-torch==)' evo2/environment/requirements-img_full.lock > /tmp/stack.txt
python -m pip install --no-deps -r /tmp/stack.txt                                              # every wheel at the lock's version, no resolver
grep -E '^transformer-engine-torch==' evo2/environment/requirements-img_full.lock > /tmp/te.txt
CUDNN_PATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cudnn" CC=gcc CXX=g++ NVTE_CUDA_ARCHS=90 MAX_JOBS=$(nproc) \
  python -m pip install --no-deps --no-build-isolation -r /tmp/te.txt                           # compiles Transformer Engine's torch extension for sm_90 (minutes)
python -m pip install --no-deps evo2/stock/vtx-1.1.0-py3-none-any.whl evo2/stock/evo2-0.6.0-py3-none-any.whl
export LD_LIBRARY_PATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cudnn/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The `export` belongs in every shell that runs Evo 2: Transformer Engine loads cuDNN 9 by name from its wheel, and without it `import
transformer_engine` fails with `libcudnn_adv.so.9: cannot open shared object file` (the images set it, together with `PYTHONHASHSEED=0` and
`CFLAGS=-g0`). Then `cd evo2 && bash run.sh install` as README §Setup continues. Two upstream messages appear on every run of this stack and are
harmless: Transformer Engine's `Supported flash-attn versions are … Found flash-attn 2.8.0.post2` warning and vortex's `Extra keys in
state_dict: {…}`. A GPU class not listed in `stock/PINS.json gpus` (H100 80GB, A100 80GB; matched by compute capability and memory) is served
and named `unlisted(…)` on the ACTIVE line (an H200, for example).

## How stock is run

`bash run.sh score --mode off` runs `route/evo2_route.py` in a process with `EVO2_OPT` unset and nothing of the kit imported (the driver
prints `[evo2-route stock] ENV-CLEAN ok` after confirming it): `Evo2('evo2_7b', use_kernels=True)` — upstream's documented
faster-inference switch, vortex's Triton kernels for the three Hyena convolutions — and for `evo2_40b` the constructor defaults
`Evo2('evo2_40b')`, because `use_kernels=True` over the two-device layer split scores NaN on every window, which makes the defaults
the fastest correct documented configuration. That is, `off` passes upstream's documented `use_kernels=True` switch ('for faster
inference', upstream's README) on the one-device models; upstream's bare constructor default is `use_kernels=False`. Scoring on every arm is one
`score_sequences(seqs, batch_size)` call at upstream's defaults (`batch_size=1`, forward strand, mean log-likelihood); no allocator
variable, precision flag or other setting is added. `exact` and `fast` use the `off` construction with `EVO2_OPT` set; any other
construction made under `EVO2_OPT` is served too, because the route is read from the object.

## Stock exceptions

None.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `EVO2_OPT` | no | unset (= `off`; `run.sh score` and `check` default to `exact`) | `exact` or `fast` engages that mode at `import evo2` in the process; `off` or unset imports nothing of the kit; any other value exits 2 |
| `EVO2_OPT_WEIGHTS` | no | unset | directory holding `<model_name>.pt`: `run.sh score` passes it as `local_path` when that file is present, and mode `fast` looks for `evo2_1b_base.pt` there when it is not beside the target's checkpoint; unset = upstream's own download. A set directory whose `<model_name>.pt` the process cannot see (missing, or not bound under Apptainer) is not refused: `score` then also takes upstream's own download (13.8 GB for `evo2_7b`) — make sure the file is visible inside the container first |
| `EVO2_OPT_HOME` | no | the tree `opt/` was installed from | the kit tree holding `stock/PINS.json`; `run.sh` exports it |
| `MODEL_OPT_JIT_ROOT` | no | unset: `run.sh` then supplies a private per-user root, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root) — the tools' own default locations this row names apply only without `run.sh` or when that root is refused | compile-cache root, read by `run.sh` only: when set, Triton's cache is `<root>/<stack key>/triton` (`img_full` or `img_a100`; a `TRITON_CACHE_DIR` you set yourself wins); unset = Triton's default under `~/.triton`. The image build takes an optional pre-filled compile cache `_jitcache/evo2-<stack key>-jit.tar` from the build context into `/opt/jit_cache` (the build is identical without one); in the image `run.sh` uses `/opt/jit_cache` in place when it is writable, else seeds the root (default `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`, i.e. `/tmp/model_opt_jit-uid<uid>` when `TMPDIR` is unset) from it once, printing the line below. A preset `MODEL_OPT_JIT_ROOT` the process cannot write is never compiled into: when the caller also exports `MODEL_OPT_STACK_KEY=img_full` or `img_a100` and that subtree holds at most `MODEL_OPT_JIT_SEED_MAX_FILES` files (default 5000), it is seeded once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` and used from there; otherwise the read-only root is left as is, no line is printed, and Triton keeps its default cache (the kit only points `TRITON_CACHE_DIR` at a writable directory). The printed `[evo2-kit] jit cache: <dir> (<how>)` line says which case applied |

The `evo2_opt` package reads only the three `EVO2_OPT*` variables; `run.sh` adds nothing else to the stock process beyond the cache root above.
