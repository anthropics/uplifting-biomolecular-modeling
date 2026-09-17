# AtlasFold — stock, as pinned

## Pin
Upstream: github.com/SeonghwanSeo/atlasfold tag v1.0.0, commit `992067e67df29b665c501e0d2e9ead9dd4ba9b69`, shipped under
`stock/src` as the source tree (`docs/atlasfold.pdf` and `docs/images/` omitted; tree digest in `stock/PINS.json`, recomputed by
`stock/check_pins.py`). Weights: upstream's Hugging Face repositories `SeonghwanSeo/atlaslm-3b-base` (AtlasLM-3B),
`SeonghwanSeo/atlasfold-260703` (monomer head) and `SeonghwanSeo/atlasfold-m-260725` (AtlasFold-M head), revision and sha256 per
file in `PINS.json:weights`, fetched from Hugging Face (`hf download SeonghwanSeo/<repo> --local-dir <root>/<repo>` per repo) into
one root that must then contain `atlaslm-3b-base/weights/atlaslm_3b_base.pth`, `atlasfold-260703/weights/atlasfold-260703.pth` and
`atlasfold-m-260725/weights/atlasfold-m-260725.pth`; `bash run.sh install --weights <root>` only compares them with the pins and
reports (never downloads, never gates) — with weights already on disk `--weights` is optional. Weights terms: the MIT License,
as stated by the upstream author on the three model cards (`license: mit`) and in the upstream README §License ("The source
code, model weights, and released datasets are licensed under the MIT License"). Upstream's licence and notices as found under
`stock/src`. `stock/` is never edited.

## Stack
Linux x86_64, CUDA 12.8 user space, Python 3.11, torch 2.7.1+cu128, triton 3.3.1, cuequivariance-torch /
cuequivariance-ops-torch-cu12 0.10.0, numba 0.61.0, einops 0.8.0, gemmi 0.7.5, omegaconf 2.3.0, scipy 1.17.1;
`environment/requirements.lock` is the reference stack's full freeze (for comparison, not an install input).

A fresh environment at the pin = the block below (pinned top-level versions of upstream's `[fold,cuequiv]` dependency set; pip
resolves their own dependencies; stock AtlasFold itself is the vendored `stock/src`, added by `run.sh install`). Then, or into an
environment that already runs AtlasFold v1.0.0 on this stack, from inside `atlasfold/` (README step 2's `cd atlasfold`): `bash run.sh
install [--weights DIR]` (`pip install --no-deps -e stock/src` and `-e opt/`, then `stock/check_pins.py`, which confirms
`stock/src` matches the pin file for file); `run.sh` also puts `opt/`, `stock/src/src` and `../common/opt_core` on `PYTHONPATH`,
so the editable installs are a convenience, not a requirement. Other cards: `configs/a100.env` (compute capability 8.0) and
`configs/h200.env` (9.0, the H100 settings) beside `configs/h100.env`; `configs/b300.env` names a CUDA 13 stack that
`environment/` does not cover.

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates build-essential   # bare Ubuntu host (root: no sudo); skip what you have
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user (skipped when present); the installer runs only when its sha256 matches
# uv brings its own released CPython 3.11, headers included; apt/deadsnakes instead: python3.11 python3.11-venv python3.11-dev — never Ubuntu 22.04's python3.11 (3.11.0rc1)
uv venv --seed --managed-python --python 3.11 ~/venv-atlasfold && . ~/venv-atlasfold/bin/activate
pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128        # the +cu128 build; brings triton 3.3.1
pip install "cuequivariance-torch==0.10.0" "cuequivariance-ops-torch-cu12==0.10.0" "numpy==1.26.4" "scipy==1.17.1" \
            "einops==0.8.0" "gemmi==0.7.5" "omegaconf==2.3.0" "numba==0.61.0" "huggingface_hub==0.36.2"
```

## How stock is run
`--mode off` executes `python -c "from atlasfold.cli import main; main()" <everything after -->` in a clean subprocess: no
`ATLASFOLD_OPT*` / `AFO_*` / `FPF_*` / `OPT_CORE_*` variable in its environment and `opt/` off its `PYTHONPATH`
(`PINS.json:stock_environment`). `--det 1` under `off` adds `CUBLAS_WORKSPACE_CONFIG=:4096:8` and nothing else (stock seeds its
RNG per record itself). That process loads one kit file by path, `phase_timing.py` (standard library only): the per-item `PHASE` /
`PEAK` timing lines on stderr, no computed value touched. Stock defaults apply in every mode (`--kernel auto` = cuEquivariance
kernels on CUDA; monomer 4 recycles / 5 samples / 20–30–100 steps by length; multimer 10 recycles / 5 samples / 200 steps); the
kit changes no run setting and only adds `--model-path` / `--lm-path` from the weights root when the stock line lacks them.

## Stock exceptions
None. No file under `stock/src` is patched; `upstream_issues/` is empty.

## Variables
| variable | required | default | effect |
|---|---|---|---|
| `ATLASFOLD_WEIGHTS_DIR` | no | `configs/*.env`: `/weights/atlasfold` | weights root (`<root>/<repo>/weights/<file>.pth`); when set (or `--weights DIR`), `install` / `check` compare it with the pins and `pred` adds `--model-path` / `--lm-path` from it; unset and no `--weights`: `install` skips the comparison and the stock CLI resolves weights as upstream does |
| `ATLASFOLD_OPT` | no | unset = `off` | mode for the drop-in route; default of `--mode` |
| `ATLASFOLD_KIT_CONFIG` | no | `configs/h100.env` | config file `run.sh` sources when no `--config` is given |
| `MODEL_OPT_JIT_ROOT` | no | `configs/*.env`: `~/.cache/atlasfold_opt/jit` | root of the first-use compile caches; must be writable when set — the caches are created under it per GPU / stack key (`<root>/torch<version>-cu<cuda>-sm<cc>/…`) on first run, and a directory that cannot be created is named at start-up while that cache uses the library default; unset (drop-in route without a config file) = the libraries' own cache locations |
| `MODEL_OPT_STACK_KEY` | no | derived from the stack | name of that key; set by the kit when unset |
| `MODEL_OPT_LEVERS_OFF` | no | unset | levers removed from the mode's row (lever names, comma-separated; refused by name if not in the row) |
| `AFO_DET` · `AFO_ALLOW_PARTIAL` | no | `0` · unset | environment spellings of `--det` · `--allow-partial` (`1`) |
| `AFO_WARM_TOKENS` | no | `640` | length of the synthetic record `warm` folds |
| `AFO_TARGET_GPU` · `KMP_AFFINITY` | no | `configs/*.env`: `H100(sm90)` / `H200(sm90)` / `A100(sm80)` / `B300(sm103)` · `disabled` | the card a config targets · Intel OpenMP affinity, exported for the run |
| `AFO_TRANSITION_CHUNK_MIB` · `AFO_PAIR_TRANSITION_CHUNK` | no | `512` · `1` | row-block budget (MiB) of the chunked pair transitions · `0` = `pair_transition_chunk` off |
| `AFO_DENOISER_GRAPH_MAX_TOKENS` · `AFO_LM_SDPA_MIN_TOKENS` | no | `1024` · `1000` | token ceiling of `denoiser_graph` (`0` = eager denoiser) · token floor of `lm_sdpa` |
| `AFO_TRIATT_BLOCK_EXACT_MIN_TOKENS` · `_MAX_ROWS` | no | `512` · `1280²` | size band `triatt_block_exact` serves on compute capability 9.0 |
| `AFO_<LEVER>=0` for `ALLOC_EXPANDABLE`, `ATOM_BF16`, `ATOM_KDEDUP`, `ATOM_ROWS`, `ATOM_SDPA`, `ATOM_TF32`, `EXACTLN`, `GRAPH_REUSE`, `OUTPUT_OVERLAP`, `SAMPLER_HOIST`, `SAMPLER_HOSTSYNC` | no | on | the lever stays installed and every call takes the stock statement (`disabled` on its line); `AFO_ALLOC_EXPANDABLE=graphs` keeps it beside `denoiser_graph`; `AFO_SAMPLER_HOISTS=<list>` selects hoists |
| `AFO_{TRIMUL,TRIMUL_EXACT,TRIATTN,TRIATTN_EXACT,TRANSITION_EXACT,PAIR_TRANSITION,EXACTLN,DIT_APB}_WORD` | no | the mode's tier word | developer overrides naming one provider row or another tier word for that lever (`pinned=1` / the row on its line) |
