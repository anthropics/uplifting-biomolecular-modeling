# ProteinMPNN — stock, as pinned

## Pin

Upstream: `https://github.com/dauparas/ProteinMPNN` at commit `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57` (no tag, no package), recorded in
`stock/PINS.json` with the entry points the kit drives (`protein_mpnn_run.py`, `helper_scripts/parse_multiple_chains.py`) carried under
`stock/src/` for the option table. `bash run.sh install --weights DIR` makes the clone at the pin with `git`; `DIR` is then `MPNN_DIR`. A clone, so
the `.fa` headers name the commit (`git rev-parse HEAD`); a checkout at another commit is refused by name (exit 3); one without `.git` runs with
a note (`git_hash=unknown` in the headers). Upstream's licence and notices as found in the clone. `stock/` is never edited.

## Weights

Upstream ships its weights in the same git repository as the code (`vanilla_model_weights/`, `soluble_model_weights/`, `ca_model_weights/`,
`training/exp_020/model_weights/`), so the clone is where they come from: the kit loads `vanilla_model_weights/v_48_020.pt` (variant `vanilla`,
upstream's default) or `soluble_model_weights/v_48_020.pt` (variant `soluble`) from `MPNN_DIR`, and `bash run.sh install --weights DIR` checks both
files' sha256 against `stock/PINS.json` after cloning. The container images carry no checkout and no weights: inside them `MPNN_DIR=/opt/ProteinMPNN`
names a directory you mount, filled once by `run.sh install --weights /opt/ProteinMPNN`. Weights kept outside the clone (a shared weights store) load with upstream's
`--path_to_model_weights DIR` on the design line, in both modes; `MPNN_DIR` still names the clone for the code and the pin check.

## Stack

Ubuntu 22.04, CUDA 12.4 runtime (driver ≥ 550), Python 3.11.5, torch 2.5.1+cu124, numpy 1.26.4, biopython 1.84, triton 3.1.0 (installed as torch's dependency, as
`stock/PINS.json` records it) — the full list is `environment/requirements-mpnn.lock`; `environment/Dockerfile` and `apptainer.def` build it (the images also carry
gcc and set `PYTHONHASHSEED=0`). Route C is the same lock in your own Python 3.11; prerequisites: a released CPython 3.11 with its headers (the images use 3.11.5; uv's managed build carries the headers — with apt/deadsnakes
python instead of uv: `python3.11 python3.11-venv python3.11-dev`), `git` and a C compiler (Ubuntu: `build-essential`) — Triton builds the `exact` draw kernel and its launcher with them at first use. From inside `proteinmpnn/`:

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl git build-essential   # bare Ubuntu host (root: no sudo); skip what you have
command -v uv >/dev/null || { t=$(mktemp) && curl -LsSf -o "$t" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t" && . "$HOME/.local/bin/env"; } # uv itself, once per user (skipped when present)
uv venv --seed --managed-python --python 3.11.5 ~/venv-mpnn && . ~/venv-mpnn/bin/activate     # uv's own released CPython build, headers included (apt's python3.11-dev + build-essential and a plain venv work instead of --managed-python)
python -m pip install --no-deps -r environment/requirements-mpnn.lock    # the whole pinned stack in one pass: torch's CUDA 12.4 wheel by URL, the nvidia-* wheels as lines
export PYTHONHASHSEED=0                                                  # the images' process environment; optional here
bash run.sh install --weights "$HOME/ProteinMPNN" && export MPNN_DIR="$HOME/ProteinMPNN"    # = README step 2's install + export lines, typed once here; README step 2 resumes at check
bash run.sh check --config h100 --mode exact
```

An environment that differs from the lock runs and is named on the `STACK` line of `check` / `design`. Every verb is re-runnable: `install` skips the pip step when both packages are already
installed from this tree, and `--weights DIR` keeps a clone already in `DIR` and only digest-checks it. One build serves H100, H200 and A100: nothing is compiled for a GPU at install time; the one run-time
compile is the `exact` line's Triton draw kernel (well under a second, a few hundred KB). Where its cache lands: with a JIT root `MODEL_OPT_JIT_ROOT` set (and `TRITON_CACHE_DIR` not) `run.sh` keys it as
`<root>/<stack key>/triton` (stack key such as `torch2.5.1-cu124-sm90`, from the installed torch and the device; `MODEL_OPT_STACK_KEY` overrides it); with neither set, `run.sh` supplies the private per-user root `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root, Triton then keeping `~/.triton/cache`) — under route B the host's `/tmp`.
The image build takes an optional pre-filled cache `_jitcache/proteinmpnn-<stack key>-jit.tar` from the build context into `/opt/jit_cache` (the build is identical
without one); only in an image that ships one does `run.sh` use `/opt/jit_cache` in place as the JIT root (writable, no root set) or seed the root from it once (an unset root
becomes `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`; a root that already holds files is left as it is) and print `[proteinmpnn-kit] jit cache: <dir> (<how>)`; an image without one prints no such line. Apptainer only (route B; skip on route C unless C itself runs inside Apptainer): a SIF built from `environment/apptainer.def` links `/opt/culib/libcuda.so` to the driver `--nv` binds and sets `TRITON_LIBCUDA_PATH` itself, so route B needs nothing extra;
the manual form below is only for an image converted straight from the Docker image (`apptainer build … docker-daemon://…` or `docker-archive://…`) or for route C run under Apptainer.
Under Apptainer
(`--nv`) on a host whose linker cache lists only `libcuda.so.1`, Triton's launcher link (`-lcuda`) fails and `exact` is refused (`fused_draw: PROBE FAIL … -lcuda`); the image carries no CUDA stubs, so point Triton at a directory holding a `libcuda.so` symlink to the bound driver library: `mkdir -p ~/libcuda && ln -sf /.singularity.d/libs/libcuda.so.1 ~/libcuda/libcuda.so`, then `apptainer run --nv --env TRITON_LIBCUDA_PATH=$HOME/libcuda …`.

## How stock is run

`--mode off` executes `python $MPNN_DIR/protein_mpnn_run.py <upstream's options, verbatim>` in a clean subprocess with nothing of the kit
importable, preceded by `helper_scripts/parse_multiple_chains.py` when the input is a directory of PDB files. Variables with prefixes
`PROTEINMPNN_`, `MPNN_` (except `MPNN_DIR`), `SELFTEST_`, and `PYTORCH_CUDA_ALLOC_CONF` / `CUBLAS_WORKSPACE_CONFIG` are kept out of that
subprocess. Every `design` argument other than the kit's own flags is an option of protein_mpnn_run.py, spelled and defaulted as upstream
defines it (an option upstream does not define exits 2, as upstream's argparse does); `--variant` selects the weight set as upstream's
default and `--use_soluble_model` do, and upstream's `--path_to_model_weights` / `--use_soluble_model` / `--model_name` take precedence.

## Stock exceptions

None. Nothing of upstream is patched or configured differently on any mode.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `MPNN_DIR` | yes | — (e.g. `/opt/ProteinMPNN`) | the ProteinMPNN clone: code, `.git`, both weight sets |
| `MODEL_OPT` | no | this directory (`configs/<card>.env`, run.sh) | where the package finds `opt/forward` and `stock/` |
| `MODEL_OPT_TARGET_GPU` | no | `H100` (`h100.env`) · `H200` (`h200.env`) · `A100` (`a100.env`) | the card the configuration targets; `check` / `design` print `target=` / `match=` |
| `MODEL_OPT_TARGET_GPU_MEM_MIB` | no | `81559` (`h100.env`) · `143771` (`h200.env`) · `81920` (`a100.env`) | the memory total `match=` asserts (nvidia-smi MiB); `40960` on an A100 40GB — a mismatch is reported, never refused |
| `PROTEINMPNN_OPT` | no | unset | the mode when `--mode` is not given; a flag that disagrees with it is refused |
| `PROTEINMPNN_VARIANT` | no | unset (= `vanilla`) | the weight set when `--variant` is not given |
| `PROTEINMPNN_OPT_ALLOW_PARTIAL` | no | unset | `1` = `--allow-partial` |
| `PROTEINMPNN_OPT_HOME` | no | unset | an alternative location of `opt/forward` (else found from the package or `MODEL_OPT`) |
