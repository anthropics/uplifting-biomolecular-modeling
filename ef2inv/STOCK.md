# ESMFold2 binder design — stock, as pinned

## Pin

Upstream: the ESM cookbook's `cookbook/tutorials/binder_design.py` from `https://github.com/Biohub/esm` at commit
`d0207ea3c8cbf072679ece3bb332787d25e06852` (esm 3.4.0; the file is 55007 bytes, sha256 in `stock/PINS.json`), run from an editable checkout of the
repository at that commit — the file ships in the checkout, not in a wheel. `stock/src/` holds the file, upstream's `pyproject.toml` and `LICENSE.md`
(MIT) for reading; `stock/esm-d0207ea3.tar.gz` is the `git archive` they come from. The models are the Biohub `transformers` fork 4.57.6 at commit
`ef32577f55da19a4989cd7b22e004dc43a4998cb` (`ESMFold2ExperimentalModel`, `ESMCForMaskedLM`); `stock/PINS.json` pins that commit and the sha256 of the
installed modelling files the levers touch (`modeling_esmfold2_experimental.py`, `modeling_esmfold2_common.py`, `modeling_esmfold2.py`, four
`kernels/*.py`, `esmc/modeling_esmc.py`, `esmc/tokenization_esmc.py`), and `stock/check_pins.py` compares them file by file. Weights: six Hugging Face
snapshots (table below), fetched by `bash run.sh install --weights DIR` through `huggingface_hub.snapshot_download` at their pinned revisions and
sha256-checked against `stock/PINS.json`. `stock/` is never edited; the one path written under it is `stock/wheels/`, where
`environment/build_wheels.sh` (route C) and a `WHEELS_FROM=prebuilt` image build keep the two compiled wheels.

**Weights.** Six Hugging Face snapshots under `HF_HOME` (`$HF_HOME/hub/models--biohub--<name>/snapshots/<commit>/`; no default, no other weights path), pinned by
snapshot commit and per-file sha256 in `stock/PINS.json` `weights`. At `design` / `check` / `warm` a repository with no snapshot, or a pinned file absent,
is the one refusal by name (`weights=… MISSING … — refused`, exit 3); another revision or a differing file is named `NOT PINNED` and the run proceeds.
`install --weights DIR` fetches, then sha256-checks every file: `WEIGHTS OK` and exit 0, or the differing / missing file named and exit 1 (nothing
deleted). Nothing is fetched at run time (`HF_HUB_OFFLINE=1` in the configs).

| repository | snapshot | role in the loop |
|---|---|---|
| `biohub/ESMFold2-Experimental-Fast` | `04dec820…` | inversion model 1 (the gradient flows through it) and hero critic 1 |
| `biohub/ESMFold2-Experimental-Fast-Cutoff2025` | `74b88548…` | inversion model 2 and hero critic 2 |
| `biohub/ESMFold2-Experimental` | `0515a117…` | hero critic 3 |
| `biohub/ESMFold2-Experimental-Cutoff2025` | `56f94f5c…` | hero critic 4 |
| `biohub/ESMC-6B` | `45b0fa5d…` | the language model: the ESM-C trunk every fold model shares (`REUSE_ESMC`) and the pseudo-perplexity head |
| `biohub/ESMFold2` | `1ebf0e34…` | the CCD pickle only (`ccd.pkl`) |

The 15 scaling-critic snapshots the shipped `__main__` can load are not among them (`--use-scaling-critics 1` reads them from `HF_HOME`).

## Stack

Ubuntu 22.04, CUDA 12.8.1 (driver ≥ 570), Python 3.12.14 via uv 0.12.5, torch 2.11.0+cu128, triton 3.6.0, xformers 0.0.35, cuequivariance 0.10.0, numpy
2.5.2, the `transformers` fork at `ef32577f`, esm 3.4.0 editable at `d0207ea3`, flash-attn 2.8.3 and transformer_engine 2.15.0+42b8400, with
`XFORMERS_IGNORE_FLASH_VERSION_CHECK=1` and `NVTE_FRAMEWORK=pytorch` in the environment — the full list is `environment/requirements.lock`;
`environment/Dockerfile` builds it and `environment/apptainer.def` converts that image for Apptainer. pip builds the lock's two VCS lines (the
`transformers` fork, `dockq`) and the editable esm checkout in build isolation and fetches their build backends from the index at install time —
setuptools, plus Cython and numpy for DockQ — so the install needs network access; the lock pins runtime packages only. flash-attn and
transformer_engine have no index build for this torch: the Dockerfile compiles both from their pinned sources (`--build-arg WHEELS_FROM=build`, the
default; `BUILD_JOBS=N` bounds the parallel compile jobs, each wanting a few GB of RAM; the two compiles are the build's longest steps) or installs two
wheel files placed in `stock/wheels/` after their sha256 match `stock/PINS.json` (`WHEELS_FROM=prebuilt`; no wheel ships in this tree). For the
cookbook's antibody-framework binders (`--binder-name <antibody>_framework_vhvl`, `--is-antibody 1`) the stack carries abnumber 0.4.4 + ANARCI
2026.2.13.2 (lock lines) and HMMER 3.4 in its own prefix `/opt/hmmer` (`environment/conda-hmmer.lock`); without them an antibody binder stops at its
first critic with `ModuleNotFoundError: abnumber` on every mode. Route C's block installs abnumber and ANARCI with the lock but not HMMER: the shipped
example and every non-antibody design never call it; an antibody binder on route C additionally needs HMMER 3.4's `hmmscan` on PATH (the
`conda-hmmer.lock` packages, or a distribution's hmmer).

The image build takes an optional pre-filled compile cache `_jitcache/ef2inv-*-jit.tar` from the build context (one per stack key, unpacking under
`/opt/jit_cache/<stack key>/…`; the build is identical without one). In the image `run.sh` uses `/opt/jit_cache` in place as `MODEL_OPT_JIT_ROOT` when
that variable is unset and the directory is writable, seeds `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` from it once when it is read-only (the Apptainer image),
seeds a set but empty `MODEL_OPT_JIT_ROOT` from it once and leaves a populated one untouched, and prints `[ef2inv-kit] jit cache: <dir> (in-image |
seeded from image | user)` before `configs/<card>.env` places the Triton / Inductor directories under `<root>/<stack key>/{triton,inductor}`; a preset
`MODEL_OPT_JIT_ROOT` the process cannot write is used as is — nothing is compiled into it, caches already there are read, no line is printed (the stack
key is derived only after this step, so no key subtree is seeded from it); without a cache in the image the line is absent and everything compiles on
first use as before.

Cards: the default build serves compute capability 9.0 (H100 and H200; `configs/h100.env`, `configs/h200.env` — the same settings). `--build-arg STACK=img_ef2inv_a100` is the A100 80 GB / 40 GB build
(`configs/a100.env` at run time; it also runs on the H100) and differs in exactly this: the two wheels are compiled inside the image for sm_80 + sm_90
(`FLASH_ATTN_CUDA_ARCHS` / `NVTE_CUDA_ARCHS=80;90`, `TORCH_CUDA_ARCH_LIST=8.0;9.0`) from the same sources, so `WHEELS_FROM=prebuilt` is refused for it and
no wheel sha256 is pinned for it (`image_recipe`'s digests are the sm_90 pair's; `stock/PINS.json` `pinned_stack.arch_variants.sm80` says so). Every
package, version string and environment variable of the stack is otherwise identical, so `stock/check_pins.py`'s version comparison holds unchanged.

Route C — an environment of your own (README §Setup C). The pinned recipe, literally (a released CPython 3.12 — the wheels are cp312; git (the esm clone, the lock's two git lines, the wheel
build), a C++ compiler and the CUDA 12.8 toolkit's nvcc for the compile step; network to github.com, download.pytorch.org and the Python index; about 8.5 GB of disk); an environment
that already holds the stack at these versions also works, and `bash run.sh install` names anything that drifts from the pin. Run it from the directory
holding `ef2inv/` and `common/` (README §Setup's fence 1 leaves you there; the block's last line does the `cd ef2inv`):

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl git build-essential   # bare Ubuntu host (root: no sudo); skip what you have
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user; installer digest-checked
uv venv --seed --managed-python --python 3.12.14 ~/ef2inv-env && . ~/ef2inv-env/bin/activate   # uv's released CPython 3.12.14 (= the image's; headers included). Instead of uv: python3.12 + python3.12-venv + python3.12-dev (deadsnakes on Ubuntu 22.04, stock packages on 24.04) with `python3.12 -m venv`, or conda's 3.12
grep -v -E '^(#|flash-attn==|transformer-engine==)' ef2inv/environment/requirements.lock > /tmp/stack.txt
pip install --no-deps -r /tmp/stack.txt                        # torch 2.11.0+cu128 and every other pinned line, the fork and DockQ from their git lines
git clone https://github.com/Biohub/esm esm && git -C esm checkout d0207ea3c8cbf072679ece3bb332787d25e06852 && pip install --no-deps -e ./esm
bash ef2inv/environment/build_wheels.sh                        # flash-attn 2.8.3 + transformer_engine 2.15.0+42b8400 compiled at the pins (≈16 min with 24 build jobs, longer on fewer cores), installed --no-index --no-deps; A100: add --stack img_ef2inv_a100; --jobs N caps the compile jobs
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 NVTE_FRAMEWORK=pytorch   # the image's two process variables, in every shell that runs the kit
cd ef2inv && bash run.sh install [--weights DIR]                    # = README step 2's install line — type it once; step 2 continues at its export line
```

Why two passes: `pip install -r environment/requirements.lock` as a whole stops at its flash-attn and transformer-engine lines — the flash-attn sdist
imports torch while it builds, and `transformer-engine==2.15.0+42b8400` is on no index. `environment/build_wheels.sh` is the Dockerfile's
`WHEELS_FROM=build` step for the current environment: it checks its prerequisites and refuses by name (exit 3) without them (torch 2.11.0+cu128
importable, nvcc 12.x / `CUDA_HOME`, git, a C++ compiler, the lock's nvidia-nccl / nvidia-cudnn wheels), downloads the flash-attn 2.8.3 sdist and checks
its sha256 against `stock/PINS.json`, clones TransformerEngine at `42b840051647` with its submodules (`core.abbrev 7`, so the version reads
`2.15.0+42b8400`), builds both with `--no-build-isolation --no-deps` for compute capability 9.0 (8.0 and 9.0 with `--stack img_ef2inv_a100`)
into `stock/wheels/`, installs them `--no-index --no-deps` and runs `pip check`. The two exports
are what the image sets: xformers 0.0.35 accepts this flash-attn build only with `XFORMERS_IGNORE_FLASH_VERSION_CHECK=1`, and `NVTE_FRAMEWORK=pytorch`
selects transformer_engine's torch binding. The fork's own start-up hints (`… run pip install flash-attn`, `pip install transformer-engine[pytorch]`) do
not apply at this pin — no index build matches torch 2.11.0+cu128; `build_wheels.sh` is the route. Without the two compiled packages every mode still
runs: `check` and each design print `stack flash_attn=… (pinned 2.8.3) NOT PINNED — the speed figures apply to the pinned stack only` (likewise
`transformer_engine`), the fork takes its SDPA / plain-torch paths (the `attention=` word on the ACTIVE line names them), and step times differ from the
README's At-a-glance line. `bash run.sh install` itself installs the kit — `pip install -e ../common/opt_core -e opt` (or `uv pip install` into that
interpreter when it carries no pip module) — and runs `stock/check_pins.py --checks software`, which refuses a cookbook file or fork modelling file that
is not the pinned bytes (exit 3).

## How stock is run

`--mode off` imports the pinned file from the installed esm checkout (`<esm package dir>/../cookbook/tutorials/binder_design.py`, sha256 checked
first; `EF2INV_COOKBOOK_STOCK` names another path) in a fresh subprocess with no kit variable set, no lever module importable and no kit directory on
`sys.path` — proven in that process before torch is imported and written as `stock_env_proof.json` — and drives the file's own
`ESMFold2Design().load(use_scaling_critics)` → `.design(…)` with the cookbook's parameters. The file's module-level `import modal` (its cloud SDK,
absent from this stack; the local route never uses it) resolves to a no-op stand-in, `opt/ef2inv_opt/_absent_sdk_stub.py`. Alone, `off` is the file
exactly as shipped: pair stack chunked at the fork's default 64 and the cookbook's own `set_kernel_backend("fused")`, which the fork engages in
no-grad folds only (the design steps then run the reference pair stack). `--mode off --chunk-size none --kernel-backend cuequivariance` adds
upstream's two documented speed settings — `set_chunk_size(None)` and `set_kernel_backend("cuequivariance")` on every loaded ESMFold2 model right
after load; the cuEquivariance kernels do reach the design steps under grad — and is the stock configuration `exact` reproduces bit for bit under
`--det 1`. `--det 1` on any mode: the fork's atomic `scatter_atom_to_token` replaced by an exact segment mean, the flash-attn callables called with
`deterministic=True`, `TORCHINDUCTOR_SHAPE_PADDING=0` + `TORCHINDUCTOR_DETERMINISTIC=1`, a private JIT cache (`opt/ef2inv_opt/det.py`).

What the `design` verb sets against the shipped `__main__`, each recorded as a named deviation in `run.json` / `opt_manifest.json`: `--binder-len L`
registers a fixed-length `PromptFactory` (absent: the cookbook's `--binder-name`, default `minibinder`, length sampled per seed);
`use_scaling_critics=False` unless `--use-scaling-critics 1` (the 15 scaling-critic snapshots are not among the six pinned ones); `--target-name NAME
[--target-sequence SEQ]` are the cookbook's own two parameters (a built-in `TARGET_SEQUENCES` name alone, or a label plus the sequence). The loop
constants (`STEPS`, `LEARNING_RATE`, `TEMPERATURE_MIN`, `LOSS_WEIGHTS`, `ESMC_MASK_FRACTION`, `LM_LOSS_BATCH_SIZE`, `LM_MASK_PASSES`, `COMPILE`,
`REUSE_ESMC`) are read back from the imported module and a changed one is refused (`opt/ef2inv_opt/settings.py`); everything else passes through
under the cookbook's own parameter names. The launcher writes the outputs (README §Run); the cookbook's local route holds its results in memory.

## Stock exceptions

One — `Transition._addmm_residual` dtypes. The fork's `Transition._addmm_residual` (`modeling_esmfold2_common.py`) hands `torch.addmm` a bf16 activation and the fp32 `w3` weight; on the pinned
stack the shipped script fails at its first design step with `RuntimeError: self and mat2 must have the same dtype, but got BFloat16 and Float`
(inside Inductor's `pad_mm` pass under `COMPILE=True`, directly under `COMPILE=False`). Every mode, `off` included, runs one run-time replacement of
that method (`opt/ef2inv_opt/patches.py` `FUNCTION_TEXT`, applied after the models load; no file of the fork is edited): the operands follow the
activation's dtype before `addmm` and the result is cast back to the input's — what autocast computes inside the design step's bf16 region; in the
fp32 confidence head no cast happens. Printed once per run as `PATCH upstream_bug_0002_transition_addmm_dtype applied=True sha=…` and recorded under
`run.json` `patches` / `opt_manifest.json` `deviations`. The change as it would read upstream:

```diff
-        out = torch.addmm(
-            x.contiguous().view(-1, x_shape[-1]),
-            hidden.view(-1, hidden.shape[-1]),
-            ffn.w3.weight.t(),
-        )
-        return out.view(x_shape)
+        h2 = hidden.view(-1, hidden.shape[-1])
+        w3_t = ffn.w3.weight.t()
+        x2 = x.contiguous().view(-1, x_shape[-1])
+        if w3_t.dtype != h2.dtype:
+            w3_t = w3_t.to(h2.dtype)
+        if x2.dtype != h2.dtype:
+            x2 = x2.to(h2.dtype)
+        out = torch.addmm(x2, h2, w3_t)
+        return out.to(x.dtype).view(x_shape)
```

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `HF_HOME` | yes (`design`, `check`, `warm`) | — | the Hugging Face cache root holding the six snapshots; unset or a snapshot missing: refused by name (exit 3) |
| `EF2INV_OPT` | no | `fast` | the mode when `--mode` is absent; a `--mode` that disagrees with it is a usage error (exit 2) |
| `EF2INV_COOKBOOK_STOCK` | no | the esm checkout's own file | path of the stock cookbook file when it is not the editable checkout's |
| `EF2INV_JIT_CACHE` | no | `shared` (the configs' default when unset) | `shared`: Triton / Inductor caches under `$MODEL_OPT_JIT_ROOT/<stack key>/{triton,inductor}` (nothing is exported while the root is unset — only without `run.sh`, which always supplies one; `EF2INV_JIT_CACHE=shared` set explicitly with no root stops by name: `configs/*.env` returns 2 before anything runs, and without a config `check` prints `REFUSED …` / `design` prints `NOT ACTIVE …` naming the variable, exit 3) · `per-box`: a private cache under `EF2INV_JIT_LOCAL_ROOT` (else a fresh temporary directory), what every `--det 1` run gets by itself |
| `MODEL_OPT_JIT_ROOT` | no | unset: `run.sh` then supplies a private per-user root, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root) — the tools' own default locations this row names apply only without `run.sh` or when that root is refused | the shared JIT-cache root, kept across runs; when set it must be writable — the Triton / Inductor caches are created under `<root>/<stack key>/{triton,inductor}` per GPU / stack key on a mode's first run; left unset, nothing is exported and Triton / Inductor use their defaults (`~/.triton/cache`, `/tmp/torchinductor_<user>`); on route C too, exporting it before a `--config` run keeps every compiled kernel under the one root between runs |
| `EF2INV_JIT_KEY` | no | `torch2.11.0-cu128-sm90` (`configs/h100.env`); derived from the running stack in `configs/a100.env` | the `<stack key>` of the shared cache; `check` refuses a shared directory keyed by another stack |
| `EF2INV_REQUIRE_FAST_ENV` | no | `1` from `configs/*.env`; `0` when no config is sourced (cli.py `require_fast_env`) | `1`: every run compares the attention / MLP / rotary paths the fork binds with the pinned stack's and names a difference on one info line (`fast environment NOT present (…) — proceeding, levers unchanged`) — never a refusal; `0`: the paths are reported without comparing |
| `EF2INV_GPU`, `EF2INV_GPU_MIB` | no | `NVIDIA H100 80GB HBM3`, `81559` (h100.env); `A100`, `81920\|40960` (a100.env) | the expected card (name containment + `nvidia-smi` memory.total); `MODEL_OPT_TARGET_GPU` is honoured as the name; any other reading: one `NOTE hardware …; proceeding` line |
| `MODEL_OPT` | no | this directory | the kit tree the lever modules are resolved from (`opt/forward/`) |
| `MODEL_OPT_LEVERS_OFF` | no | empty | CHANGES.md §Switches |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `PYTHONDONTWRITEBYTECODE` | no | `1` in the configs | offline loads; no `.pyc` in a mounted tree |

The arm processes run with every `EF2_*` / `EF2INV_*` / `ESMFOLD2_*` variable stripped and prove it before torch loads (`opt/ef2inv_opt/envproof.py`);
a kit arm carries exactly `EF2_FAST_KIT=<the mode's word>` (`exact` | `agk3` for `fast` | `big`), set by the launcher, never by the user.
