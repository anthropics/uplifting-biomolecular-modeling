# Protenix v1 — stock, as pinned

## Pin

Upstream: https://github.com/bytedance/Protenix, PyPI package `protenix` 1.1.0, shipped under `stock/` as the wheel
`protenix-1.1.0-py3-none-any.whl` and the sdist `protenix-1.1.0.tar.gz` (the files PyPI serves; URL, size and upstream's own declared
requirements in `stock/PINS.json` "stock"); weights: the CLI's default checkpoint `checkpoint/protenix_base_default_v1.0.0.pt` plus the
`common/` data files upstream reads (sha256 and URLs in PINS.json), fetched by `bash run.sh install --weights DIR` through upstream's own
downloader (`download_inference_cache`); upstream's licence and notices as found inside the wheel — its README (the wheel's METADATA) states: "The
Protenix project including both code and model parameters is released under the Apache 2.0 License. It is free for both academic research and
commercial use." `stock/` is never edited.

## Stack

Ubuntu 22.04, CUDA 13.0 runtime wheels on the CUDA 12.9 toolkit image (driver ≥ 580), Python 3.11, torch 2.13.0+cu130, triton 3.7.1,
cuequivariance-torch / cuequivariance-ops-torch 0.11.1, deepspeed 0.17.5, numpy 2.4.1 — the full list is `environment/requirements.lock`;
`environment/Dockerfile` and `apptainer.def` build it. Into an environment of your own — a released CPython 3.11 (the images use 3.11.5; uv's
builds bundle the headers), nvcc (12.9 in the images) and a C++ compiler on PATH — from inside `protenix_v1/`:

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y git wget build-essential clang ninja-build kalign   # bare Ubuntu host (root: no sudo); skip what you have
command -v uv >/dev/null || { t=$(mktemp) && wget -qO "$t" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t" && . "$HOME/.local/bin/env"; } # uv itself, once per user (skipped when present; wget is in the apt line above)
uv venv --seed --managed-python --python 3.11.5 ~/protenix_v1-env && . ~/protenix_v1-env/bin/activate   # any released CPython 3.11; not 22.04's apt one (3.11.0rc1)
grep -v -E '^(#|protenix==|torch==|triton==|nvidia-[a-z-]+-cu13==)' environment/requirements.lock > /tmp/pass1.txt && pip install --no-deps -r /tmp/pass1.txt
grep -E '^(torch==|triton==|nvidia-[a-z-]+-cu13==)' environment/requirements.lock > /tmp/pass2.txt && pip install --no-deps -r /tmp/pass2.txt   # CUDA 13 wheels last
pip install --no-deps stock/protenix-1.1.0-py3-none-any.whl     # pass 1 leaves protenix out; this is the pinned wheel
bash run.sh install --weights /weights/protenix_v1                   # = README step 2's install line — type it once; step 2 continues at its export line
```

The two passes are the Dockerfile's: the CUDA 13 cuDNN / NCCL / cuSPARSELt wheels share directories with the CUDA 12 ones and must be laid down last (one pass
can leave torch unable to load). Stock's own fast-LayerNorm extension (`LAYERNORM_TYPE=fast_layernorm`) compiles with nvcc into the installed
`protenix/model/layer_norm/` at its first use — minutes, once per environment; the images do it at build time — so on this route site-packages must be
writable then; the kit's stream-correct build of it ships prebuilt (`opt/forward/v05_addon/lib/fastln_prebuilt/`, sm_70 … sm_100) and is rebuilt from
`lib/kit112_src` into `$TORCH_EXTENSIONS_DIR/fastln_stream` only where its load-time self-test rejects it. Stock also writes `<input>-update-msa.json` beside
the `--input` file, so an input inside a read-only tree (the image's copy of the example) needs a copy on a writable path. The image build takes an optional
pre-filled compile cache, `_jitcache/protenix_v1-*-jit.tar` (one per stack key) from the build context, into `/opt/jit_cache` (the build is identical without
one); in the image `run.sh`, before it sources the config, uses `/opt/jit_cache` in place when it is writable, else seeds the JIT root from it once
(`MODEL_OPT_JIT_ROOT` when set and empty, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` when unset; a non-empty root is left as it is; layout `<root>/<stack key>/…`),
exports `MODEL_OPT_JIT_ROOT` accordingly and prints `[protenix_v1-kit] jit cache: <dir> (<how>)` when `pred`, `check` or `warm` starts; `<how>` is
`in-image`, `seeded from image`, `user` (your non-empty root, used as it is), `unseeded` (the copy failed; that run compiles afresh) or
`seeded from read-only root` (below). A preset `MODEL_OPT_JIT_ROOT` the process cannot write is never compiled into: when `MODEL_OPT_STACK_KEY` is already
set and `<preset root>/<stack key>` holds at most `MODEL_OPT_JIT_SEED_MAX_FILES` files (default 5000), `run.sh` seeds that subtree once into
`${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` and moves the root there — `(seeded from read-only root)`; otherwise (the usual case, since the config derives the key after
this step) the read-only root is used as it is, silently: caches already there are read, nothing is written into it. Other cards: `configs/a100.env` and
`configs/h200.env` differ from `configs/h100.env` only in `MODEL_OPT_TARGET_GPU`; admission is by compute capability (`stock/PINS.json` "supported_gpus": H100
/ H200 9.0, B200 10.0, B300 10.3, A100 8.0) and every memory-dependent choice reads the running device. `stock/PINS.json` "pinned_stack" records the older
stack upstream 1.1.0 itself declares (its wheel's `Requires-Dist`: earlier torch, triton and cuequivariance releases on CUDA 12); the environment this kit
ships and runs on is the newer lock above, installed `--no-deps` over those declarations, and the pin check compares the installed `protenix` version and
wheel bytes only, not the stack.

## How stock is run

`--mode off` executes `protenix pred <your arguments>` in a clean subprocess with nothing of the kit importable and every
`PROTENIX_V1_OPT*` / `PTX_*` / `FPF_*` / `INFOPT_*` / `PROTENIX_DET_SCATTER` variable stripped (`opt/protenix_v1_opt/stock_pred.py`). The
1.1.0 CLI defaults already select every speed setting upstream exposes (cuEquivariance triangle kernels, bf16, TF32, fast LayerNorm,
caching and fusion on: `stock/PINS.json` "cli_defaults"), so stock is upstream exactly as shipped. `--det 1` under `off`: the same
deterministic recipe the kit modes use (`opt/protenix_v1_opt/det.py`), applied to the stock subprocess.

## Stock exceptions

None. The kit applies no fix or setting to the stock arm and has no `--upstream-fix` flag.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `PROTENIX_ROOT_DIR` | yes | — | the weights + data root upstream reads: `<root>/checkpoint/protenix_base_default_v1.0.0.pt`, `<root>/common/{components.cif, components.cif.rdkit_mol.pkl, clusters-by-entity-40.txt, obsolete_release_date.csv}` (+ `release_date_cache.json`, `obsolete_to_successor.json` under `--use_template true`); the `--config` verbs (`pred`, `check`, `warm`) refuse by name (rc 3) while it is unset or lacks a file — `install` and `probe` never read it (`install --weights DIR` fills DIR and needs no export) |
| `PROTENIX_V1_OPT` | no | unset | the mode on the environment route (`PROTENIX_V1_OPT=<mode> protenix pred …`); unset and no `--mode`: `fast` |
| `PROTENIX_V1_OPT_N_GPU` / `_DET` | no | unset | `--n_gpu P` / `--det 1` when the flag is not given — read by the verbs and by the environment route alike |
| `PROTENIX_V1_OPT_ALLOW_PARTIAL` | no | unset | `=1`: `--allow-partial` for the environment route (`PROTENIX_V1_OPT=<mode> protenix pred`), its only spelling there; the `run.sh` / `protenix-v1-opt` verbs take the flag and do not read this name |
| `PROTENIX_V1_TP_MSA_M` / `_CONF_FINISH` / `_MC_DROPOUT_FULL_MAX` | no | `token_sharded` / `exact` / `2048` | `--n_gpu P>1` statements: the MSA representation token-sharded or `replicated`; the confidence head's row-block finish `exact` or `rowsum`; the token count up to which MC dropout draws the stock keep-mask on every rank (`0` = rank-forked draws at every size) |
| `PROTENIX_V1_OPT_KIT` | no | unset | an explicit lever-code directory (default `opt/forward/v05_addon` beside the package) |
| `MODEL_OPT_TARGET_GPU` | no | `H100` (`A100` / `H200` in `a100.env` / `h200.env`) | the card the configuration is written for — informational, nothing in the kit reads it; the running card is what `check`'s `"gpu"` block and the ACTIVE line's `gpu='…' sm=…` print, and admission is by compute capability |
| `LAYERNORM_TYPE` | no | `fast_layernorm` | upstream's LayerNorm switch; the sampler-graph lever (`sg`) needs the fast path |
| `TORCH_EXTENSIONS_DIR` | no | `<jit root>/<stack key>/torch_extensions` | where torch builds extensions (the kit's LayerNorm rebuild, `fastln_stream/`, when needed); `<jit root>` is `MODEL_OPT_JIT_ROOT` — named by the caller or by `run.sh`, else the per-user `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made 0700, owned by this user alone, no group/other write bit, not a symbolic link; otherwise refused by name on stderr and a fresh private directory serves that run) |
| `TRITON_CACHE_DIR` | no | `<jit root>/<stack key>/triton` | the Triton JIT cache and the kit's cache root (kernels, the weights digest memo `weights_digests.json`; without a config the memo lives in `~/.cache/protenix_v1_opt`) |
| `PROTENIX_V1_OPT_WEIGHTS_MEMO` | no | unset | an explicit directory for the weights digest memo `weights_digests.json` (default: the cache root above); a run whose memo directory is not writable keeps its digest in a private directory and exports this variable to its rank processes |
| `MODEL_OPT_JIT_ROOT` | no | unset (the config then names the per-user root above) | persistent cache root; must be writable when set: the config exports the two directories above as `<root>/<stack key>/torch_extensions\|triton` unless pre-set, and the first run on a GPU / stack creates and fills them (key = `torch<version>-cu<cuda>-sm<cc>`, `python -m protenix_v1_opt._stackkey`; `MODEL_OPT_STACK_KEY` overrides); left unset, the caches stay at the two defaults above |
| `PTX_SAMPLER_GRAPH_MAXTOK` | no | unset | the sampler-graph token cap of `exact` / `fast` (unset = the mode's device-keyed cap, CHANGES.md; `0` = none) |
| `PROTENIX_V1_BIG_SIZE_N_TOKEN` | no | unset | `big`: the run's largest token count, stated instead of estimated from `--input` |
| `PROTENIX_V1_BIG_TRIMUL_TORCH_TOKENS` | no | `2048` | `big`: the token count above which upstream's torch triangle multiplication engages |
| `MODEL_OPT_LEVERS_OFF` | no | unset | ablation: the mode without the named levers (README Notes) |

A `PROTENIX_V1_OPT*` or `PROTENIX_V1_BIG_*` name the kit does not declare is refused by name. Values in `configs/<card>.env` are the
defaults above.
