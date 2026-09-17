# Proteina-Complexa — stock, as pinned

## Pin

Upstream: `proteinfoundation` 1.1.0 from `https://github.com/NVIDIA-BioNeMo/Proteina-Complexa` (`stock/PINS.json` `repo`; the same
repository is served at `https://github.com/NVIDIA-Digital-Bio/proteina-complexa`, the URL `environment/requirements.lock` installs from)
at commit `916eaaedce5b07c205efb6ef32370c01d366591e` (branch `main`; the repository carries no tags), shipped under `stock/` as the source archive
`proteina-complexa-916eaaed.tar.gz` (`git archive` of the commit less the one file named under 'Stock exceptions'; the commit, version,
recipe and file checks in `stock/PINS.json`, byte copies of the cited configs, `pyproject.toml` and licences under `stock/src/`). Weights:
`complexa.ckpt` (denoiser, 2934289381 bytes) and `complexa_ae.ckpt` (autoencoder, 4100101779 bytes) from `hf:nvidia/NV-Proteina-Complexa-Protein-Target-160M-v1`
(revision and sha256 in PINS.json), fetched by `bash run.sh install --weights DIR` through upstream's own `env/download_startup.sh --complexa` and digest-checked;
both live under `$CKPT_PATH`. Code Apache-2.0, weights NVIDIA Open Model License, as found under `stock/` (loose copies of the licence texts that concern the kit's own files: `third_party_licenses/`; what those files restate: `THIRD_PARTY_NOTICES.md`). `stock/` is never edited.
`check` and `design` refuse by name (exit 3) when the installed upstream, its checkout under `$LOCAL_CODE_PATH` or the weights are not as
pinned; `install` exits 3 when the pin check refuses and 1 when the pip step fails or a fetched weight file is off its pin; `python stock/check_pins.py --weights "$CKPT_PATH"` recomputes the digests by hand.

## Stack

Debian 12, CUDA 12.6 runtime wheels (NVIDIA driver 560 or newer, or a data-centre driver from 525 on), Python 3.12.10, torch 2.7.0+cu126,
triton 3.3.0, torch_geometric 2.8.0 with the `pt27cu126` extension wheels, lightning 2.5.6, hydra-core 1.3.1, omegaconf 2.3.1,
atomworks 2.2.1, numpy 2.5.1 — upstream's `env/build_uv_env.sh` set. The full list is `environment/requirements.lock`;
`environment/Dockerfile` and `apptainer.def` build it, one image for H100 (sm90) and A100 (sm80). Into a fresh environment (line 3; an
existing Python 3.12 environment at the pin skips lines 2–4) on linux x86_64 with an NVIDIA driver (no CUDA toolkit: the CUDA 12.6 runtime comes as wheels) and
the host packages of line 1 — the image's own set plus `libxext6`, the two X client libraries being what the Open Babel wheel loads at import; the C and
Fortran toolchains matter at install time only (pip's source builds), nothing is compiled when the kit or stock runs. From inside `complexa/`:

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential git wget curl ca-certificates gfortran libxrender1 libxext6   # bare Ubuntu/Debian host (root: no sudo); skip what you have
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; } # uv 0.12.15 itself, once per user; installer digest-checked
uv venv --seed --managed-python --python 3.12.10 ~/complexa-venv && . ~/complexa-venv/bin/activate   # uv's own released CPython 3.12.10, headers included; any released CPython 3.12 works instead — apt/deadsnakes: python3.12 + python3.12-venv + python3.12-dev (not python3-dev), or python.org, or conda
grep -v '^-e ' environment/requirements.lock > /tmp/stack.txt && python -m pip install --no-deps -r /tmp/stack.txt   # the two -e lines are upstream itself
mkdir -p "$HOME/src" && tar -xzf stock/proteina-complexa-916eaaed.tar.gz -C "$HOME/src" && export LOCAL_CODE_PATH="$HOME/src/proteina-complexa-916eaaed"   # any writable directory (/opt/pc in the image)
python -m pip install --no-deps -e "$LOCAL_CODE_PATH" -e "$LOCAL_CODE_PATH/community_models/colabdesign"           # upstream's own two editable installs
bash run.sh install                         # the core and the kit, editable, then the pin check. README step 2's `install --weights "$CKPT_PATH"` is this SAME step plus the checkpoint fetch / hash-check into $CKPT_PATH — continuing with step 2, type only that one (repeating either is harmless)
export PYTHONHASHSEED=0 CFLAGS=-g0           # optional: the image's two other environment values (environment/Dockerfile ENV); LOCAL_CODE_PATH above is the required one
```

Line 4 downloads several GB of wheels (the finished environment is about 8 GB) and replaces the environment's own pip / setuptools / wheel with the
lock's versions; the editable installs (lines 6–7) print pip's legacy-editable deprecation notice at the pinned pip, which is harmless there, and write
their `*.egg-info` metadata beside the sources — the checkout, `opt/` and `../common/opt_core` must be writable while installing (read-only afterwards
is fine: a run writes only under `--out`, and Python skips `__pycache__` it cannot write). pip fetches the build backends from the package index at install time (network needed): hatchling for upstream's two editable installs,
setuptools and wheel for the kit's own; `environment/requirements.lock` pins runtime packages only (`environment/Dockerfile` additionally
holds those build tools to fixed versions through `PIP_CONSTRAINT`).

The generation path imports no flash-attn, cuequivariance-torch, xformers or deepspeed (its attention is `PairBiasAttention`, einsum +
softmax in fp32); upstream sets `torch.set_float32_matmul_precision('high')` itself; no `torch.compile`, CUDA graphs or autocast on this path,
and no kit lever compiles anything at run time (no C compiler needed to run).

## How stock is run

`--mode off` executes, with `--out` as working directory (upstream roots `./inference/…` and `./logs` there),

    complexa generate $LOCAL_CODE_PATH/configs/search_binder_local_pipeline.yaml --verbose \
      [++generation.task_name=<item> ++generation.target_dict_cfg={<item>:{…the --input entry…}}] \
      ++ckpt_path=$CKPT_PATH ++autoencoder_ckpt_path=$CKPT_PATH/complexa_ae.ckpt  <your overrides after `--`, verbatim, in order, last>

in a clean subprocess: your environment with the variables `stock/PINS.json` lists as must-be-absent (`COMPLEXA_OPT*`, `COMPLEXA_KIT*`,
`NVIDIA_TF32_OVERRIDE`, `CUBLAS_WORKSPACE_CONFIG`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`), `PYTHONSAFEPATH` and the kit's `PYTHONPATH` entries
removed (listed on the `ENV-CLEAN` line, recorded in `<out>/stock_env_proof.json`); of the kit only the inert `complexa_opt_autoload.pth` import runs at
interpreter start and, `COMPLEXA_OPT` absent, imports nothing else. The bracketed tokens appear only with `--input`; `--verbose` routes
upstream's log output to the terminal (output routing only). No override given = the shipped configuration (best-of-n search scored by
upstream's reward model, 4 designs, batch 16, seed 5; `opt/complexa_opt/settings.py` `SHIPPED` lists each value with its config file);
the generation stage alone is upstream's own `++generation.search.algorithm=single-pass ++generation.reward_model=null`. The kit modes
run this same command with `COMPLEXA_OPT=<exact|fast|big>` exported instead (CHANGES.md).

## Stock exceptions

No upstream file is patched and no numeric setting is changed on any mode, `off` included. One file of the commit is left out of the
shipped archive (`stock/PINS.json` `archive_omits`): `src/proteinfoundation/result_analysis/sc`, a pre-compiled shape-complementarity
executable — a third-party tool with no source in the repository, which upstream's `README.md` states is not distributed with the
repository, naming where the `sc` and `dssp` binaries are obtained. No file of the commit refers to that path: upstream's optional
bioinformatics reward (`rewards/bioinformatics_reward.py`) and binder evaluation (`evaluation/binder_eval.py`) read the tool's location
from `SC_EXEC`; the shipped generation pipeline enables neither (the `bioinformatics` reward block of `binder_generate.yaml` is commented
out) and `configs/*.env` stub `SC_EXEC` with `/bin/true` when unset ('Variables'). To score shape complementarity with upstream's reward or
evaluation code, obtain the tool as upstream's `README.md` describes and export `SC_EXEC=/path/to/sc` before the run.

## Variables

| variable | required | default (`configs/h100.env`; `a100.env` / `h200.env` = the same with `MODEL_OPT_TARGET_GPU=A100` / `H200`) | effect |
|---|---|---|---|
| `LOCAL_CODE_PATH` | yes | none (`/opt/pc` in the image) | upstream's checkout at the pin (upstream's own variable); `design` / `check` refuse by name without it |
| `CKPT_PATH` | yes | none | directory of `complexa.ckpt` + `complexa_ae.ckpt` (upstream's own variable) |
| `COMPLEXA_OPT` | no | unset (= `fast` when no `--mode`) | the mode, when `--mode` is not given; a `--mode` that disagrees is refused (exit 2) |
| `COMPLEXA_INIT` | no | `1` | upstream's CLI refuses to run before `complexa init` unless set |
| `COMMUNITY_MODELS_PATH`, `DATA_PATH`, `AF2_DIR` | no | `$LOCAL_CODE_PATH/community_models`, `$LOCAL_CODE_PATH/assets/target_data`, `$MODEL_OPT_STATE/af2_params` | upstream's variables, resolved eagerly by Hydra on every run (`AF2_DIR`: the reward model's parameters; a generation-stage run reads none of the three) |
| `ESM_DIR`, `RF3_CKPT_PATH`, `RF3_EXEC_PATH`, `FOLDSEEK_EXEC`, `MMSEQS_EXEC`, `DSSP_EXEC`, `SC_EXEC` | no | `/bin/true` when unset | upstream's optional reward / evaluation tools (resolved eagerly; name the real paths to use them) |
| `MODEL_OPT_TARGET_GPU` | no | `H100` / `A100` / `H200` | the GPU the config targets; `check` prints `target=` / `match=` — a report, never a gate |
| `MODEL_OPT_STATE` | no | `$HOME/.cache/complexa_opt` | out-of-tree state root |
| `MODEL_OPT_JIT_ROOT`, `MODEL_OPT_JIT_KEY` | no | unset | shared JIT-cache root: when set it must be writable — `TORCH_EXTENSIONS_DIR` / `TRITON_CACHE_DIR` default to `<root>/<key>/{torch_ext,triton}` and the compile caches are created there on first run, one directory per GPU/stack key = `torch<version>-cu<CUDA>-sm<cc>` of the running stack (`unknown`, named on stderr, when it cannot be established — on a host without a GPU, for one — never a refusal; preset `MODEL_OPT_JIT_KEY`, e.g. `torch2.7.0-cu126-sm90` for the H100, to key the caches without the probe); unset: nothing is exported and the tools' own cache locations apply. This kit's modes and stock's generation path compile nothing, so nothing is written there today (the optional AF2 reward stage's JAX compiles in memory only); the variables are honoured for a stack that does. |
| `MODEL_OPT_STACK_KEY` | no | derived: `torch<version>-cu<cuda>` of the installed torch, read from distribution metadata (no GPU needed), e.g. `torch2.7.0-cu126`; `unknown` when torch is absent | a label the manifest records, never a gate; preset it (e.g. `MODEL_OPT_STACK_KEY=torch2.7.0-cu126`) and the probe is skipped, e.g. when the config is sourced in an image build step or on a host without a GPU |

A `COMPLEXA_OPT*` or `COMPLEXA_KIT*` variable never reaches the stock subprocess; `COMPLEXA_OPT_RECORD` is set by the kit itself (the
directory of the run's `kit_records/`).
