# Chai-1 — stock, as pinned

## Pin

Upstream: `chai_lab` 0.6.1 (github.com/chaidiscovery/chai-lab tag `v0.6.1`, commit `8d5ac0f9`), shipped under `stock/` as the PyPI wheel
`chai_lab-0.6.1-py3-none-any.whl` plus the tag's source archive `stock/src/` (pins in `stock/PINS.json`; `stock/check_pins.py` refuses any
other installed `chai_lab`, exit 3; `pred`, `check` and `warm` run it first, `install` right after installing). Weights: the eight files of chai-lab's `downloads/` layout —
`models_v2/{feature_embedding,bond_loss_input_proj,token_embedder,trunk,diffusion_module,confidence_head}.pt`, `conformers_v1.apkl`,
`esm/` (ESM2-3B fp16, traced) — each pinned by sha256 in PINS.json, fetched and checked against the pin by `bash run.sh install --weights DIR`
(upstream's own downloader), and hashed once more at a mode's first activation (a file off its pin is named on a NOTE line). Upstream's licence and notices as found under
`stock/src/`; for the weights the upstream README states, as printed, "Chai-1 is released under an Apache 2.0 License (both code and model weights)"
(`stock/src/README.md` §Licence; the traced ESM-2 file's notice: THIRD_PARTY_NOTICES.md). `stock/` is never edited.

## Stack

Ubuntu 22.04, CUDA 13.0 (driver ≥ 580), Python 3.11.5, torch 2.13.0+cu130, cuDNN 9.20, triton 3.7.1, numpy 1.26.4, rdkit 2024.9.6, gemmi
0.6.7, gcc 11.4 — the full list is `environment/requirements.lock`; `environment/Dockerfile` and `apptainer.def` build it, one image for every
configured card. Triton builds its launchers at first use (gcc and the libc headers: `build-essential`; Python's own headers, which uv's
interpreter bundles — with apt / deadsnakes Python instead: `python3.11` + `python3.11-venv` + `python3.11-dev`); the compiled step links against the CUDA 13.0
development headers under `CUDA_HOME` (`cuda-cudart-dev-13-0`, `cuda-crt-13-0`). The pinned route into a fresh or existing Python 3.11
environment (about 5.7 GB once installed) — run from inside `chai1/`:

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl build-essential     # bare Ubuntu host (root: no sudo); skip what you have
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user (skipped when present); the installer runs only when its sha256 matches
uv venv --seed --managed-python --python 3.11.5 ~/venvs/chai1 && . ~/venvs/chai1/bin/activate     # uv's own CPython 3.11.5 build, Python headers included
grep -v -E '^(#|chai_lab==)' environment/requirements.lock > /tmp/stack.txt && python -m pip install --no-deps -r /tmp/stack.txt
python -m pip install --no-deps stock/chai_lab-0.6.1-py3-none-any.whl
export CUDA_HOME=/usr/local/cuda-13.0                                  # where the CUDA 13.0 development headers live (the image sets it)
bash run.sh install --weights /weights/chai1   # kit + shared core editable, pin check, weights fetched or checked (any writable dir) = README step 2's install line: type it once and skip it in step 2
```

The image build takes optional pre-filled compile caches `_jitcache/chai1-<stack key>-jit.tar` from the build context into `/opt/jit_cache`
(the build is identical without one); when that directory holds caches, `run.sh` uses it in place as the JIT root if no `MODEL_OPT_JIT_ROOT`
is set and it is writable, else seeds the JIT root (`MODEL_OPT_JIT_ROOT`, or `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` when unset; layout
`<root>/<stack key>/{triton,inductor,aoti}`) from it once — a root that already holds files is left as it is — and prints
`[chai1-kit] jit cache: <dir> (<how>)` — `<how>` is `in-image`, `seeded from image`, `seeded from read-only root`, `unseeded` (the copy failed:
cold) or `user` (your own writable root that already held files, used as is). A preset `MODEL_OPT_JIT_ROOT` the process cannot write is never compiled into: when `MODEL_OPT_STACK_KEY`
is set (`configs/h200.env` sets it) and the read-only root's `<stack key>/` subtree holds at most `MODEL_OPT_JIT_SEED_MAX_FILES` files (default
5000), `run.sh` seeds it once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` and says so on that line; otherwise the read-only root is used as is (caches already there are read, nothing is printed).

Fold settings are `run_inference`'s keyword defaults at the pin: 3 trunk recycles, 200 diffusion steps, 5 diffusion samples, 1 trunk sample,
ESM embeddings on, no MSA / template server, `low_memory=True`, no seed unless one is named; the crop is the smallest of chai-lab's model
sizes (256, 384, 512, 768, 1024, 1536, 2048 tokens) that holds the input. `pred` and `check` take `chai-lab fold`'s options for them and pass
them through unchanged on every mode: `--num-trunk-recycles`, `--num-diffn-timesteps`, `--num-diffn-samples`, `--num-trunk-samples`
(upstream's own loop, into `trunk_<i>/`), `--recycle-msa-subsample`, `--[no-]use-esm-embeddings`, `--[no-]low-memory`,
`--[no-]use-msa-server`, `--msa-server-url`, `--[no-]use-templates-server`, `--constraint-path`, `--template-hits-path`, `--device`
(`opt/chai1_opt/settings.py`).

## How stock is run

`--mode off` calls `run_inference` once per (input, seed) in a clean subprocess (`opt/chai1_opt/stock_fold.py`) whose environment carries
`CHAI_DOWNLOADS_DIR` and `CUDA_VISIBLE_DEVICES` and nothing of the kit: no `CHAI1_OPT*` variable, no kit directory on `sys.path`. Without a
seed it is upstream's unseeded call (outputs under `seed_none/`). `--det 1` under `off`: the same deterministic recipe as under the kit modes
(`opt/chai1_opt/det.py`: deterministic torch algorithms, `CUBLAS_WORKSPACE_CONFIG`, deterministic cuDNN); stock at default numerics is not
run-to-run bitwise on a GPU, so byte comparisons with `off` are meaningful under `--det 1` only.

## Stock exceptions

- torch pin relaxed: `chai_lab` 0.6.1 declares `torch<2.7,>=2.3.1` and is installed with `--no-deps` on torch 2.13.0+cu130, where it runs
  unmodified — applied identically on every mode, `off` included; changes no model arithmetic (`pip check` reports that one line).
- Stack note instead of refusal: on a torch / CUDA build other than the pinned one `check_pins.py` prints `STACK not pinned: torch <v>+<cuda> …`
  and passes; `--strict-stack` / `CHAI1_OPT_STRICT_STACK=1` refuses instead (exit 3). It compares torch and CUDA only and names the pinned
  stack by its key `torch2.13.0-cu130-sm90` (the card it was recorded on), so an A100 on the pinned build prints that same key.
- `exact` on compute capability 8.0: stock's TorchScript trunk runs one trace per crop, and the 1536 and 2048 traces order their chunked
  linear layers and attention differently from the smaller ones; the eager trunk's single statement reproduces those two traces bit for bit
  on sm90 but not on sm80, so on an A100 `exact --det 1` equals `off --det 1` through the 1024-token crop and is deterministic with
  `fast`-class differences above it.
- Under `exact` every run prints one `NAMED_FALLBACK:ln|… refused=exactln:aten_rowwise_path(misaligned_rows served=aten` line (`named_fallback=1 … alerts=1` on
  the `[opt_core] CELLS` line): one LayerNorm statement class the exact kernel does not take stays on stock's ATen op — expected, the Run example included; outputs unchanged, exit 0.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `CHAI_DOWNLOADS_DIR` | yes | — | upstream's weights root (the eight files above); unset or incomplete, every verb is `NOT ACTIVE` by name, exit 3 |
| `MODEL_OPT_TARGET_GPU` | no | `H100` / `A100` / `H200` from `configs/<card>.env` | the card the configuration targets; a GPU whose name lacks the word is noted on the ACTIVE line, never refused |
| `MODEL_OPT` | no | set by `configs/<card>.env` | this `chai1/` directory (the package locates the kit through it); `CHAI1_OPT_HOME` serves for an install without `--config`, `CHAI1_OPT_KIT` / `_EAGER` / `_DSTEP` / `_ERRATA02` per component |
| `MODEL_OPT_JIT_ROOT` | no | unset: `run.sh` then supplies a private per-user root, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root); without `run.sh`, or when that root is refused, the libraries' own cache locations and no packages kept | root of the Triton / Inductor caches (`<root>/<torch-cuda-arch key>/`) and of the compiled step's packages (`<root>/<key>/aoti/`); must be writable when set — the caches are created under it per GPU / stack key on first run |
| `CHAI1_OPT` | no | unset | the mode when `--mode` is absent (must agree with `--mode` when both are set); activates the package in any Python process of the environment through `chai1_opt_autoload.pth` |
| `MODEL_OPT_LEVERS_OFF` | no | unset | comma list of the mode's lever names to leave off for one run (CHANGES.md §Switches) |
| `CHAI1_OPT_ESM_MEMO_SCOPE` | no | `global` | `global` memoises ESM embeddings across every input of the process; `input` recomputes per input as stock does |
| `CHAI1_OPT_STRICT_STACK` | no | unset | `1` = `--strict-stack` |
| `CHAI1_OPT_ALLOW_PARTIAL` | no | unset | `1` = a partial activation proceeds, named, on the `CHAI1_OPT` route (`pred` takes `--allow-partial` instead) |
| `CHAI1_BIG_MSA_CHUNK_MIN_N`, `CHAI1_BIG_NOGRAPH_MIN_N`, `CHAI1_BIG_TRUNK_CHUNK_MIN_N`, `CHAI1_BIG_OPM_CHUNK_MIN_N`, `CHAI1_BIG_HOIST2_MAX_N` | no | the mode's gates (CHANGES.md) | the pair extent (crop) at which `msa_chunk`, `nograph`, `trunk_chunk`, `opm_chunk` engage and past which `hoist2` stands down under `big`; the only `CHAI1_BIG_*` names the kit reads — any other is refused by name |
| `CHAI1_OPT_AOTI_HOST_ISA` | no | read from torch / `/proc/cpuinfo` | the host CPU's vector-ISA word, for the compiled step's package check |

A `CHAI1_OPT_*` or `CHAI1_BIG_*` name the kit does not declare is refused by name (`NOT ACTIVE: undeclared variable(s) …`).
