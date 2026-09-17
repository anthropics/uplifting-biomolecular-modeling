# Protenix v2 — stock, as pinned

## Pin

Upstream: https://github.com/bytedance/Protenix tag v2.0.0 (commit `2475421`), package `protenix` 2.0.0, shipped under `stock/` as
the wheel `protenix-2.0.0-py3-none-any.whl` (sha256 in `stock/PINS.json`) with the source snapshot `stock/src/`; weights:
`checkpoint/protenix-v2.pt` (1,859,785,497 bytes, sha256 `8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599`) plus the six `common/` data caches (Chemical Component
Dictionary, RDKit molecule cache, cluster and release-date tables; sha256 of each in PINS.json). `bash run.sh install --weights DIR` fetches them
through upstream's own downloader (`runner.inference.download_inference_cache`); upstream's download server does not currently serve the
checkpoint (HTTP 403 at the time of writing; the caches are served), so `install --weights` names the checkpoint absent (exit 1) until a
holder of the file places it at `DIR/checkpoint/protenix-v2.pt`, and digest-checks all seven files once it is there — a file off its pin (a `common/` cache stock regenerated at run time, say) is named with exit 1: delete it and re-run `install --weights DIR` to fetch the pinned copy; upstream's licence and notices as found under `stock/`. `stock/` is never edited.
`stock/PINS.json` also records the CLI defaults (`--seeds 101 --cycle 10 --step 200 --sample 5 --dtype bf16`,
`LAYERNORM_TYPE=fast_layernorm`) and the stack below.

## Stack

Ubuntu 24.04, CUDA 13.0 (driver 580 or newer), Python 3.11.5, torch 2.13.0+cu130, triton 3.7.1, cuequivariance-torch /
cuequivariance-ops-torch-cu13 0.11.1, deepspeed 0.17.5, numpy 2.4.6 — the full list is `environment/requirements.lock`;
`environment/Dockerfile` and `apptainer.def` build it (CUDA development image: stock compiles its fast-LayerNorm extension with nvcc at
first import; kalign 3.3.5 for template realignment). The wheel is installed with `--no-deps`: it declares torch 2.7.1, triton 3.3.1,
cuequivariance 0.8.0 (cu12), numpy 2.4.1, tqdm 4.67.1, torchvision 0.22.1 and torchaudio, where the lock carries the versions above,
tqdm 4.70.0, torchvision 0.28.0+cu130 and no torchaudio (imported nowhere in the inference code); every other declared pin is met.
From scratch (route C; `environment/Dockerfile` runs the same steps), from inside `protenix_v2/`, with the CUDA 13.0 toolkit (`nvcc`) and gcc on PATH
(ihm builds one C extension against the interpreter's own headers, which uv's Python carries):

    sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates wget git build-essential ninja-build kalign   # bare Ubuntu host (root: no sudo); skip what you have
    command -v uv >/dev/null || { t=$(mktemp) && wget -qO "$t" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t" && . "$HOME/.local/bin/env"; }   # uv itself, once per user (skipped when present); wget is in the apt line above
    uv venv --seed --managed-python --python 3.11.5 venv && . venv/bin/activate         # the image's CPython 3.11.5 (uv fetches that build); another 3.11 patch level also runs — `install` then prints `STACK not pinned: python …` once and proceeds (informational); with apt/deadsnakes python instead of uv: python3.11 + python3.11-venv + python3.11-dev, never Ubuntu 22.04's python3.11 apt package (3.11.0rc1)
    pip install $(grep -E '^pip==' environment/requirements.lock)                          # the stack's pip first
    grep -v -E '^(#|protenix==)' environment/requirements.lock > /tmp/stack.txt
    pip install --no-deps -r /tmp/stack.txt                                                # the pinned stack, no resolver (about 5.7 GB installed, mostly torch and the nvidia CUDA 13 wheels)
    pip install --no-deps stock/protenix-2.0.0-py3-none-any.whl                            # stock, as pinned
    python -c "import protenix.model.layer_norm.layer_norm"                                # nvcc compiles stock's fast-LayerNorm extension once, into the package directory
    bash run.sh install [--weights DIR]                                                       # = README step 2's install line — type it once; step 2 continues at its export line (kit + shared core editable, then stock/check_pins.py)

An environment that already runs Protenix 2.0.0 at the pin needs only the last two lines. The container image additionally sets
`TORCH_CUDA_ARCH_LIST=9.0+PTX` (needed only when the fast-LayerNorm extension is built on a machine without a GPU) and runs `kalign` with
stdin closed (Ubuntu's kalign otherwise waits on an open stdin under `--use_template true`).
`stock/check_pins.py` confirms the installed `protenix` matches the pinned wheel file for file (exit 3 otherwise); a different Python (any
3.11.x runs), torch, triton or cuequivariance is reported on one `STACK not pinned: …` line and the run proceeds. Other cards: `configs/h200.env`, `a100.env`,
`b200.env`, `b300.env` set the target card and cache key; the stock fast-LayerNorm extension must carry code for the card — an
environment whose prebuilt `fast_layer_norm_cuda_v2.so` was compiled for sm_90 only fails on A100 with `no kernel image` under
`--mode off` too; delete it and import `protenix.model.layer_norm.layer_norm` once to rebuild.
The image build takes an optional pre-filled compile cache `_jitcache/protenix_v2-*-jit.tar` from the build context and unpacks it under
`/opt/jit_cache/<stack key>/…` (the build is identical without one); in the image `run.sh`, with `MODEL_OPT_JIT_ROOT` unset, uses `/opt/jit_cache`
in place when it is writable and otherwise seeds `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` from it once; a set but empty `MODEL_OPT_JIT_ROOT` is seeded the
same way and a non-empty one is left as it is; it exports `MODEL_OPT_JIT_ROOT` accordingly before `configs/<card>.env` derives the cache directories
(`<root>/<stack key>/{triton,torch_extensions}`), and prints `[protenix_v2-kit] jit cache: <dir> (<how>)` (in-image | seeded from image | user). A preset `MODEL_OPT_JIT_ROOT` the process
cannot write is never compiled into: `run.sh` moves the root to `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`, seeded once from `<preset root>/<stack key>`
when `MODEL_OPT_STACK_KEY` is already set in the environment (up to `MODEL_OPT_JIT_SEED_MAX_FILES` files, default 5000; printed
'seeded from read-only root'), otherwise the read-only root is used as it is and no line is printed (nothing is compiled into it; caches already there are read;
`TRITON_CACHE_DIR` / `TORCH_EXTENSIONS_DIR` preset to writable directories are kept by the config) — the usual case here, since the
config derives the key only after this step.

## How stock is run

`--mode off` executes `protenix pred <your arguments>` in a subprocess whose environment has every name under `stock/PINS.json`
`must_be_absent_prefixes` removed (`PTX_*`, `FPF_*`, `INFOPT_*`, `PF_*`, `PROTENIX_OPT*`, `CUEQ_TRITON_CACHE_DIR`) and every `PYTHONPATH`
entry inside a kit directory removed; before `protenix` is imported the process proves it (no such name, no kit directory on `sys.path`,
no kit module loaded) and exits 3 otherwise. The deployment variables (`MODEL_OPT*`, the cache directories) stay in place; the stock
process reads `PROTENIX_ROOT_DIR`, `LAYERNORM_TYPE`, `TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR` as upstream does. `--det 1` is the one named
exception: the stock process then carries `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PTX_DET=1`, keeps one kit entry on `PYTHONPATH`
(`opt/forward/flashpairformer/src`, whose `sitecustomize.py` calls `torch.use_deterministic_algorithms(True, warn_only=True)` under
`PTX_DET=1` and does nothing else there), and runs with the deterministic scatter copy (`src/detref/scatter_utils.py`) placed over the
installed `protenix/utils/scatter_utils.py` for the run and restored at exit — the installed package directory must therefore be writable for that run (a read-only container image needs `apptainer run --writable-tmpfs` or a writable overlay; otherwise `--det 1` is refused by name, exit 1); the process states the exception on its `ENV-CLEAN` line
and exits 3 if its environment deviates from it. `off --det 1` is the reference `exact --det 1` equals. The checkpoint is not served by upstream at present ('Pin'): placed by hand under
`DIR/checkpoint/`, it is digest-checked by `install --weights`, which names it absent (exit 1) until then. At run time the stock CLI downloads a missing cache under the root; `PROTENIX_ROOT_FROZEN=1` makes that a refusal by name.
Stock writes `<input>-update-msa.json` beside `--input` whenever it searches or converts MSAs for an entry (and `<input>-final-updated.json`
when it adds templates or RNA MSAs), so the input's directory must be writable — under a read-only container image the input belongs in a
host directory; the shipped example carries precomputed single-sequence MSAs (nothing is searched) and the Run block copies it to `/tmp` first.

## Stock exceptions

None. Stock runs as released on every mode; the kit sets one stock argument itself, `--trimul_kernel torch`, and only under
`--mode big --n_gpu P` (CHANGES.md).

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `PROTENIX_ROOT_DIR` | yes | — | stock's weights root (`checkpoint/protenix-v2.pt` + `common/`); `configs/<card>.env` refuses by name when unset |
| `PROTENIX_OPT` | no | `fast` | the mode for the unchanged stock command line (`PROTENIX_OPT=exact protenix pred …`); `run.sh --mode` otherwise; the two disagreeing is a usage error |
| `MODEL_OPT_JIT_ROOT` | no | unset: `run.sh` then supplies a private per-user root, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root) — the tools' own default locations this row names apply only without `run.sh` or when that root is refused | cache root: `configs/<card>.env` then sets `TORCH_EXTENSIONS_DIR` / `TRITON_CACHE_DIR` = `<root>/<stack key>/{torch_extensions,triton}` and `PROTENIX_OPT_CACHE_DIR` = `<root>/weights` where unset (example `/weights/jit`); when set it must be writable — the compile caches are created under it per stack key (torch + CUDA + GPU architecture) on first run; left unset, none of the three is exported and torch, Triton and the kit each keep their own default location |
| `TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR` | no | under the cache root above (`configs/<card>.env`); each tool's own without a config | the JIT caches; a pre-set value is kept |
| `PROTENIX_OPT_CACHE_DIR` | no | `~/.cache/protenix_opt` | the kit's cache root (weights digest memo) |
| `MODEL_OPT_STACK_KEY` | no | derived | the JIT cache key `torch<ver>-cu<ver>-sm<cc>` (example `torch2.13.0-cu130-sm90`), computed by `python -m protenix_opt._stackkey` |
| `MODEL_OPT_TARGET_GPU` | no | per config (`H100` …) | the card the config targets; `check` / `pred` report a mismatch |
| `LAYERNORM_TYPE` | no | `fast_layernorm` | stock's LayerNorm switch, set by the config as stock's CLI default does |
| `INFOPT_FASTLN_PREBUILT` | no | chosen for the installed torch | directory of the prebuilt stream-correct fast-LayerNorm build (`opt/forward/flashpairformer/third_party/fastln_prebuilt*`) |
| `PROTENIX_OPT_OUTB`, `_OUTA`, `_ARM`, `_ENVSH`, `_KITSPEC` | no | set by the package | bookkeeping the package writes around its own sourcing of `env.sh` and reads back; not settings |
| `PROTENIX_OPT_FORCE` | no | unset | `1`: the stock-version gate (installed `protenix` ≠ the pin) and the no-CUDA gate warn instead of refusing; nothing else changes |
| `PROTENIX_ROOT_FROZEN` | no | unset | `1`: an absent checkpoint or cache is a refusal by name, never a download |
| `MODEL_OPT_LEVERS_OFF` | no | unset | comma-separated lever names removed from the mode for one run (CHANGES.md 'Switches') |
| `MODEL_OPT` | no | this directory | `run.sh` always exports it as its own directory (a caller's value is replaced); `configs/<card>.env` sets it only when unset; the package reads it — as `$MODEL_OPT/opt` — only to locate `opt/` when `protenix_opt` is not installed editable from this tree |
| `PROTENIX_OPT_HOME` | no | unset | the kit's `opt/` directory, read before `MODEL_OPT` for the same purpose; neither `run.sh` nor the configs set it |
| `PYTHONHASHSEED` | no | `0` for the ranks | `--n_gpu P`: an exported integer is handed to every rank; unset, empty or `random` become `0` |
| `PROTENIX_OPT_N_GPU` | no | unset (one GPU) | the GPU count for the unchanged stock command line; that route cannot launch ranks, so a value above 1 is refused by name (`NOT ACTIVE`, exit 3) — use `run.sh pred --mode big --n_gpu P`; a value that is not a positive integer is refused by name; the presence probes of `run.sh` and `configs/<card>.env` run with it removed |
| `PROTENIX_OPT_TP_ROUTE` | no | set by the launcher | `rowpair` inside every rank of `--n_gpu P`, exported by the kit's launcher (not a user setting); the presence probes run with it removed |

A `PROTENIX_OPT*` name the kit does not declare is refused by name at interpreter start (exit 3).
