# AlphaFold 3 (JAX) — stock, as pinned

## Pin

Upstream: github.com/sokrypton/alphafold3 tag `v3.1.4`, commit `bc32b22ff5902e3daffd5d1f7203d7f2ab6cb997` (Apache-2.0), shipped under
`stock/` as a source archive (`stock/alphafold3-bc32b22f.tar.gz`; sha256 in `stock/PINS.json`) plus `stock/src/` for reading. No built wheel
ships (`stock/wheels/` holds only `.keep`): every route compiles `alphafold3_open-3.1.4-cp312-cp312-linux_x86_64.whl` from that source — the
image build does it, and §Stack's recipe does it in an environment of your own; a wheel you have built and placed in `stock/wheels/` is what
`--build-arg WHEELS_FROM=prebuilt` installs instead. `stock/unpack_src.sh DEST` lays the source out from the archive (or clones the pinned
commit when the archive is absent) and applies `stock/patches/`. Weights: variant `p2` = the public checkpoint `of3-p2-155k.pt` from
https://openfold.s3.amazonaws.com/staging/of3-p2-155k.pt (2.3 GB, Apache-2.0, sha256 in `stock/PINS.json`), fetched by
`bash run.sh install --weights DIR` and converted by the fork's own `convert_of3_weights.py` into `DIR/p2/of3_ported_weights.bin.zst`
(sha256 in `stock/PINS.json`); converted parameters already under `DIR/p2/` with that digest are kept (`CONVERT … status=PRESENT`, nothing
fetched), and `install` without `--weights` or `AF3_JAX_PARAMS_ROOT` installs and checks only. A `--model_dir` of your own is accepted and
its digest printed (`WEIGHTS … NOT PINNED`). No AlphaFold 3
model parameters are shipped, fetched or used. Upstream's licence and notices as found under `stock/src/`. No mode or verb edits `stock/`;
the archive is upstream's bytes, and the reading copy `stock/src/` is that source with the one declared patch of §Stock exceptions applied
(`src/alphafold3/model/network/template_modules.py`; every other file byte-identical to the archive member) and without the archive's
`src/alphafold3/test_data/` (THIRD_PARTY_NOTICES.md, 'Data files').

## Stack

Ubuntu 24.04, CUDA 12.6.3 base image (driver ≥ 550), Python 3.12.1; jax / jaxlib / jax-cuda12-plugin 0.10.2, dm-haiku 0.0.16, tokamax 0.0.12, numpy 2.4.1,
rdkit 2025.9.4, torch 2.7.1+cpu (the converter only) — the full list is `environment/requirements.lock`; `environment/Dockerfile` and `apptainer.def` build it
in upstream's own `docker/Dockerfile` layout (stock environment `/alphafold3_venv`, checkout `/app/alphafold`, the kit in `/kit/venv`; `--build-arg
WHEELS_FROM=prebuilt` installs the wheel from `stock/wheels/` instead of compiling it). The image build takes an optional pre-filled compile cache
`_jitcache/af3_jax-<stack>-jit.tar` from the build context into `/opt/jit_cache` (the build is identical without one; the tar carries cache classes as `warm`
lays them out under `AF3_JAX_CACHE_ROOT`: `<GPU model>__jax<v>_jaxlib<v>[__<mode>…]/` at its top level); in the image `run.sh` uses `/opt/jit_cache` in place
when it is writable, else seeds a root from it once (your `AF3_JAX_CACHE_ROOT` or `MODEL_OPT_JIT_ROOT` when set and empty, otherwise
`${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`), points `AF3_JAX_CACHE_ROOT` there and prints `[af3_jax-kit] jit cache: <dir> (<how>)`; a populated root of your own is used
as it is, and `MODEL_OPT_JIT_ROOT`, where set, stands in for an unset `AF3_JAX_CACHE_ROOT`; a preset `MODEL_OPT_JIT_ROOT` the process cannot write is used as
it is and no line is printed (nothing new is compiled into it; cache classes already there are read; the kit sets no stack-key subtree from which a writable
copy would be seeded). A cache class is valid at the path it was built at (the path is part of each compiled program's key); a copy elsewhere recompiles each
program once, then persists there. With neither `AF3_JAX_CACHE_ROOT` nor `MODEL_OPT_JIT_ROOT` set, the root is the per-user
`${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`, made with mode 0700 and used only when it is a directory this user owns, not a symbolic link and not writable by
group or others; anything else is named on a `CACHE REFUSED` line, nothing under it is read or written, and the process compiles into a private temporary
root for that run — export `AF3_JAX_CACHE_ROOT` (or `MODEL_OPT_JIT_ROOT`) to a directory of your own to keep a cache. Model-process environment: `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`,
`XLA_PYTHON_CLIENT_PREALLOCATE=true`, `XLA_CLIENT_MEM_FRACTION=0.95`, `PYTHONHASHSEED=0` (`configs/<card>.env` exports the three XLA variables where unset;
`PYTHONHASHSEED` is set only in the image — export it yourself on route C). The pinned recipe for environments of your own — Linux x86-64, CPython 3.12 for
stock (any Python ≥ 3.10 for the kit), an NVIDIA driver (no CUDA toolkit: jax's wheels bring the CUDA libraries), uv (the block below installs it when absent
and has it bring a released CPython 3.12.1 with headers; with apt or deadsnakes Python instead: python3.12 + python3.12-venv + python3.12-dev and `python3.12
-m venv`, any released 3.12 — another patch level is named on a NOTE line and runs), and for the wheel build gcc/g++ (11 and 13 both build it), make, zlib
headers, git and network access to PyPI, GitHub and the wwPDB; `/opt/af3/…` are example paths, any writable location works, and `AF3_JAX_REPO` stays writable
(the first `pred` of a kit mode copies `run_alphafold_fast.py` into it). From inside `af3_jax/`:

    sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y git wget curl ca-certificates gcc g++ make zlib1g-dev zstd patch   # bare Ubuntu host (root: no sudo); skip what you have
    command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user (skipped when present); the installer runs only when its sha256 matches
    uv venv --seed --managed-python --python 3.12.1 /opt/af3/venv && . /opt/af3/venv/bin/activate   # uv's own released CPython 3.12.1, headers included
    pip install --no-deps -r environment/requirements.lock
    bash stock/unpack_src.sh /opt/af3/alphafold                                            # the pinned source, patch applied
    (cp -a /opt/af3/alphafold /tmp/af3src && cd /tmp/af3src && /opt/af3/venv/bin/pip wheel --no-deps -w "$OLDPWD/stock/wheels" .)   # upstream's own build (always needed: no wheel ships); CMake fetches its dependencies, build_data the wwPDB components
    /opt/af3/venv/bin/pip install --no-deps stock/wheels/alphafold3_open-3.1.4-cp312-cp312-linux_x86_64.whl && /opt/af3/venv/bin/build_data
    export AF3_JAX_REPO=/opt/af3/alphafold AF3_JAX_PY=/opt/af3/venv/bin/python PYTHONHASHSEED=0
    uv venv --seed --managed-python --python 3.12 ~/.venvs/af3_jax_opt && . ~/.venvs/af3_jax_opt/bin/activate   # the kit's own environment (any released Python ≥ 3.10)

then `bash run.sh install [--config h100] [--weights DIR]` (kit + shared core editable, then `stock/check_pins.py`, which confirms the interpreter
`AF3_JAX_PY`, the checkout `AF3_JAX_REPO` and the pinned XLA variables against `stock/PINS.json`; a Python that differs from the pin in patch
level only is a `NOTE python … — proceeding` line, another package release or Python minor a `PINS drift …` line, and the verb runs either way;
exit 3 only when a kit file, a usable interpreter or the pinned stock install is absent). `pred`, `check` and `warm` repeat that check quietly:
they print the NOTE / drift lines only.
Other cards: `--config a100|h200` sources `configs/<card>.env`; one build serves every card, nothing is compiled per GPU at build time.

## How stock is run

`--mode off` executes `$AF3_JAX_PY $AF3_JAX_REPO/run_alphafold.py --norun_data_pipeline --cache_dir= --input_dir=<dir> --of3_weights
--model_dir=<AF3_JAX_PARAMS_ROOT>/p2 --buckets=<every multiple of 64 up to 5120> [your flags]` in a clean subprocess with nothing of the
kit importable and every kit variable removed (the `COMMAND argv=…` line prints it). The empty `--cache_dir=` is deliberate: the fork then reads
and writes no tokamax autotuning cache and keeps JAX's compilation cache at `./jax` under the pass's output directory, fresh per pass — the
ACTIVE line's `cache=<output_dir>/jax autotune=skipped(cache_dir empty)`; two such passes differ in trailing digits, because XLA's kernel
autotuning is decided anew for each compilation. A flag you state rides last and wins: `--cache_dir <exact's class>` shares that class, its
autotuning results included, with `exact` (the bit-for-bit comparison's route, README Notes); `--buckets=<upstream's list>
--cache_dir=/tmp/alphafold_cache` is the bare upstream command.

## Stock exceptions

- `04_of3_empty_template_restype_gap` (`stock/patches/04_of3_empty_template_restype_gap.diff`, 9 added lines in
  `alphafold3/model/network/template_modules.py`, gated on the fork's own `global_config.of3_weights`): the template embedder runs over 4
  padded template slots even in template-free inference, and the fork pads an absent slot's `template_aatype` with 0, which is alanine,
  while the ported weights were trained with empty slots presented as GAP; the patch presents the GAP one-hot for an all-padding slot.
  With AlphaFold 3's own weights and for real templates nothing changes. Applied identically on every mode, `off` included, because
  `stock/unpack_src.sh` applies it to the source every install route builds the package from; an environment assembled another way gets
  it with `patch -p1` at the source root and a reinstall (`stock/check_pins.py` checks the pinned files' presence, not their content).

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `AF3_JAX_PARAMS_ROOT` | yes | — | converted parameters root, `<root>/<variant>/of3_ported_weights.bin.zst`; unset, every verb refuses by name unless `--model_dir <dir>` names the weights |
| `AF3_JAX_CACHE_ROOT` | no | `MODEL_OPT_JIT_ROOT` when set, else `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (this user's own directory, mode 0700; one another account owns or can write is refused by name, above) | persistent compile-cache root; must be writable: one class per GPU model + jax build + mode (`<GPU model>__jax<v>_jaxlib<v>[__fast\|__big-…]`) is created under it on first run or by `warm`, and read and extended by `pred`. The ACTIVE / CACHE lines' `files=` counts the class's content (compiled programs under `<class>/jax/`, autotune results); `executables=` counts pre-serialized executables under `<class>/executables/`, which this stack does not produce, so it reads 0 |
| `AF3_JAX_REPO` | no | `/app/alphafold` | the fork checkout (`run_alphafold.py`, `convert_of3_weights.py`) |
| `AF3_JAX_PY` | no | `/alphafold3_venv/bin/python` | the fork's interpreter |
| `AF3_JAX_OPT`, `AF3_JAX_VARIANT`, `AF3_JAX_N_GPU` | no | `fast`, —, `1` | mode, variant and device count when `--mode` / `--variant` / `--n_gpu` are absent; a flag that disagrees with the set variable is refused (exit 2); `--n_gpu` wins over the variable |
| `MODEL_OPT_LEVERS_OFF` | no | unset | comma-separated lever ids removed from the mode's composition for one run (README Notes) |
| `MODEL_OPT_TARGET_GPU` | no | `H100` / `A100` per config | the GPU class the configuration targets; informational |
| `XLA_FLAGS`, `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_CLIENT_MEM_FRACTION` | no | the pinned values, filled by `configs/<card>.env` where unset | the model process's XLA settings; another value is named by the pin check, not refused |

Any other `AF3_JAX_*` name in the caller's environment is refused by name (`NOT ACTIVE: undeclared variable(s)`, exit 3): the lever switches
(`AF3_JAX_<LEVER>`, `AF3P_*`, `AF3_FLASHPAIRFORMER`, `AF3_DIFFUSION_HOIST`) are written by the mode table into the model process only.
