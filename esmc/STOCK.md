# ESM C — stock, as pinned

## Pin

Upstream: Biohub `esm` 3.4.0 @ `43ccece2ad485f27db46afdb67da2a9601e8f106` (github.com/Biohub/esm; MIT), shipped under `stock/` as
the source archive `esm-43ccece2.tar.gz` with its ESM C modules unpacked under `stock/src/esm/` for reading (next to them,
`stock/src/esm/cookbook/snippets/esmc.py` is a byte copy of upstream's file at the same commit, taken from the repository: the archive
carries no `cookbook/` directory), plus the Biohub
`transformers` fork 4.57.6 @ `ef32577f55da19a4989cd7b22e004dc43a4998cb` (Apache-2.0), installed from that commit
(`environment/requirements.lock`); its archive is not shipped — `PINS.json` names `stock/transformers-ef32577f.tar.gz` only as the
other install source the pin check accepts (matched by file name in pip's install record, never opened), and `archive_recipe`
reproduces it. Nothing outside this directory is needed to install or run the kit. Every pin on this page, with digests, is in
`stock/PINS.json`; upstream's licence and notices are as found in the archive. `stock/` is never edited, and the kit patches
nothing on disk.

Weights: the Hugging Face cache under `HF_HOME`, upstream's own layout (`hub/models--biohub--ESMC-<SIZE>/snapshots/<commit>/`,
`refs/main` naming the commit; the snapshot also carries `config.json` and the tokenizer files). `bash run.sh install --weights DIR
--variant V` fetches one size through upstream's own resolver (`esm.models.hub.resolve_model_dir`, the hub client underneath) into
`DIR/hf` — so `HF_HOME=DIR/hf` afterwards — and checks every weight file by sha256; a file off its pin is named and left in place
(exit 1). Run with `HF_HUB_OFFLINE=1` once the weights are in place: `ESMC.from_pretrained` otherwise asks the Hub for the repo's current
`main` and, when it is ahead of the snapshot on disk, downloads those newer files into a writable `HF_HOME` (a read-only one fails) — the
offline switch pins every load to the digest-checked snapshot. The container image sets it (`ENV HF_HUB_OFFLINE=1` in
`environment/Dockerfile`, inherited by the Apptainer image; `-e HF_HUB_OFFLINE=0` opts out), and `run.sh install --weights` clears it for
its own download command only. With no snapshot on disk and the switch unset, upstream's loader downloads `main` at the first load in
every mode.

| variant | SDK name | repo @ snapshot commit | weight files (sha256 per file in `PINS.json` `weights`) |
|---|---|---|---|
| `300m` | `esmc_300m` (d 960, 30 layers) | `biohub/ESMC-300M` @ `a59b831785f907e96e6a246b1d142bfb76df31ee` | `model.safetensors` |
| `600m` | `esmc_600m` (d 1152, 36 layers) | `biohub/ESMC-600M` @ `a7e82012c83126b9eedb055fea9fa84b6c02f094` | `model.safetensors` |
| `6b` | `esmc_6b` (d 2560, 80 layers) | `biohub/ESMC-6B` @ `45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a` | `model-0000{1..6}-of-00006.safetensors` |

Sizes on disk (`PINS.json` `total_weights_bytes`): 300m 1.3 GB, 600m 2.3 GB, 6b 25.4 GB. `HF_HOME` must contain
`hub/models--biohub--ESMC-<SIZE>/snapshots/<commit>/` with the files above (plus `config.json` and the tokenizer files the snapshot
carries); with the snapshot already in place `run.sh install --weights` downloads nothing and only digest-checks.
Terms of the weights, as their provider states them: the three model cards carry `license: mit, other` (`license_link` →
upstream's `THIRD_PARTY_NOTICE.md`) and upstream's README says "These models are available under the [MIT
license](https://github.com/Biohub/esm/blob/main/LICENSE.md)." and "Please follow our [Acceptable Use
Policy](https://biohub.org/acceptable-use-policy/) when using the model."; `stock/PINS.json` `licence.weights` records the same.

## Stack

Ubuntu 24.04, CUDA 13.0 (the Dockerfile's base image is `nvidia/cuda:13.0.1-runtime-ubuntu24.04`; host driver 580 or newer), Python 3.12.1 (the pin check does not compare the interpreter's patch release: a later 3.12 runs unremarked), torch 2.11.0+cu130 (cuDNN 9.19.0.56), triton 3.6.0,
transformers 4.57.6 (the fork), esm 3.4.0, huggingface_hub 0.36.2, safetensors 0.8.0, tokenizers 0.22.2, einops 0.8.2, and
upstream's own `accel` extra: flash-attn 2.7.4.post1 and transformer-engine 2.15.0 (upstream's prebuilt wheels for this Python /
torch / CUDA, by URL and digest in the lock), nvidia-cublas 13.6.1.10, numpy 2.5.2. `xformers` is absent and must stay absent
(upstream's `pyproject.toml` declares it; its wheel has no CUDA extension that loads on this torch). One stack (`PINS.json`
`stacks.accel`); every mode runs on it. The full list is `environment/requirements.lock`; `environment/Dockerfile` and `apptainer.def`
build it (no weights inside; `PINS.json` `stacks.accel.jit_key` names cache keys for H100 / H200 (sm90), A100 80 GB (sm80) and B200
(sm100), and the accelerator wheels are upstream's `py312-pt211-cu13-sm80-90` set). Under Apptainer `$HOME` is the invoking user's
bound home, so the image's prebuilt extension cache (`/root/.cache/esmc_sdkfused`) is not on the lookup path: the first kit-mode load
per user builds the extension once (nvcc and ninja are in the image) into `~/.cache/esmc_sdkfused` on the host and later runs import
it; keep the home bind (the default) or bind a persistent directory over `~/.cache`; root may pass `--no-home` to use the image's copy.
The image build takes an optional pre-filled compile cache `_jitcache/esmc-<stack key>-jit.tar` from the build context (any
`_jitcache/esmc-*-jit.tar`; the build is identical without one; stack keys are `PINS.json` `stacks.accel.jit_key`, e.g.
`torch2.11.0-cu130-sm90`); in an image that ships one, `run.sh` uses `/opt/jit_cache` in place when it is writable, else seeds the JIT root
(`MODEL_OPT_JIT_ROOT`, default `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`; layout `<root>/<stack key>/…`) from it once, and prints
`[esmc-kit] jit cache: <dir> (<how>)`; a preset `MODEL_OPT_JIT_ROOT` the process cannot write is never compiled into — `run.sh` moves
to a writable `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` seeded once from `<that root>/$MODEL_OPT_STACK_KEY` when that variable names the stack
key (up to `MODEL_OPT_JIT_SEED_MAX_FILES` files, default 5000) and says so on the printed line, otherwise the read-only root is used as it
is and nothing is printed (nothing is compiled into it; caches already there are read); a process that
loads a model reads the seeded Triton kernels when its `TRITON_CACHE_DIR` names `<dir>/<stack key>/triton`, while the extension is
found under `~/.cache/esmc_sdkfused` as above.
Into an existing Python 3.12 environment that already runs ESM C at
the pin (a released CPython 3.12 with its development headers — uv's managed builds carry them; with apt or deadsnakes python add
`python3.12-dev` — and, on PATH, git, a C/C++ compiler (`build-essential` on Debian-family hosts) and the CUDA 13.0 toolkit's `nvcc`;
OS floor: a libstdc++ that provides `CXXABI_1.3.15` — GCC 14's, i.e. Ubuntu 24.04; on 22.04 run
`sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y libstdc++6`,
or in a conda environment `conda install -y -c conda-forge 'libstdcxx-ng>=14'` with its `lib/` on `LD_LIBRARY_PATH`; the pinned flash-attn /
Transformer Engine wheels need it, the images carry it, and without it those two fail to import, so stock runs upstream's classes
without them while `exact` refuses by name): `pip install --no-deps -r environment/requirements.lock`,
then `bash run.sh install` — the kit editable (`pip install -e opt`: the `esmc_opt` package, the `esmc-opt` script,
`esmc_opt_autoload.pth`), then `stock/check_pins.py`, then the kit's CUDA extension built for the visible GPU (sm80 and sm90 when
none is visible). `pip install -e "opt[test]"` adds pytest and numpy for the package's CPU tests. pip builds the lock's three
git-pinned packages (`dockq`, `esm`, `transformers`) and the kit's own package in isolated build environments and fetches their build
backends from the index at install time — setuptools (the Dockerfile holds it to 84.0.0 for those builds; `setuptools>=64` for the
kit), plus Cython and numpy for DockQ's C extension — so installing needs network access; the lock pins the runtime packages plus
setuptools and ninja, which torch's extension builder needs in the kit's interpreter (an environment made with `python -m venv` has no
setuptools of its own). From inside `esmc/`: the whole of route C from a fresh interpreter, then the same steps through the images
(`esmc.sif` as built in the directory above):

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl git build-essential clang ninja-build   # bare Ubuntu host (root: no sudo); skip what you have; plus the CUDA 13.0 toolkit's nvcc (NVIDIA's cuda-nvcc-13-0 package or a full toolkit)
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; } # uv 0.12.15 itself, once per user; installer digest-checked
uv venv --seed --managed-python --python 3.12 ~/esmc-env && . ~/esmc-env/bin/activate   # uv's own released CPython 3.12, headers included; with apt/deadsnakes python instead: python3.12 + python3.12-venv + python3.12-dev
pip install --no-deps -r environment/requirements.lock           # the pinned stack, no resolver; setuptools and ninja included
bash run.sh install --weights /weights/esmc --variant 6b            # = README step 2's install line — type it once; step 2 continues at its export line (kit, pin check, CUDA extension; weights fetched or digest-checked)
export HF_HOME=/weights/esmc/hf HF_HUB_OFFLINE=1
bash run.sh check --variant 6b
# route A: a shell in the image, opened in /kit so the `cd esmc` line applies as written (HF_HUB_OFFLINE=1 is preset), then the install/export/check lines above:
docker run --rm -it --gpus all -w /kit -v /weights:/weights -v "$PWD":/work esmc-kit:dev bash
# route B (bind your weights directory to /weights; absolute paths; the runscript starts in /kit/esmc; the host's exported HF_HOME / ESMC_OPT pass into the container unless --cleanenv, HF_HUB_OFFLINE=1 is preset in the image):
apptainer run --nv --bind /weights:/weights ../esmc.sif install --weights /weights/esmc --variant 6b
apptainer exec --nv --bind /weights:/weights --env HF_HOME=/weights/esmc/hf --env ESMC_OPT=exact --bind "$PWD":/work ../esmc.sif python /work/your_script.py
```

`python -I stock/check_pins.py [--quiet] [--weights DIR --variant V]` (run by `run.sh` before every verb) checks the interpreter
against the pins — `esm` / `transformers` installed from those commits and nothing else, versions from distribution metadata, the
accelerators by import in a clean subprocess, optionally every weight file under `DIR` by sha256 — naming any stack drift it finds
(an accelerator off its pinned version or failing to import, `xformers` present; exit 0) and refusing (exit 3) only an upstream
package absent or installed from another source than its pinned commit / archive, torch that does not import, or a weight file
missing or off its digest.

## How stock is run

There is no stock command: `off` (or `ESMC_OPT` unset) leaves the process untouched, with nothing of the kit's levers imported, and
your code's `ESMC.from_pretrained(name, device=torch.device("cuda"))` resolves as upstream ships it — bf16 weights, the varlen
flash-attention class (`use_flash_attn=True`, upstream's default), Transformer Engine modules when importable; then `encode` +
`logits` per protein, or the padded batched forward under bf16 autocast. No TF32 override, no `torch.compile`, no numerics flag.
`use_flash_attn=False` is not treated as stock: its outputs equal the shipped class's only at short lengths. `exact` produces the same
output bytes as these calls for the same inputs and batch composition.

Every load, in every mode, prints two upstream `FutureWarning`s (torch.distributed's `reduce_op` deprecation and the ESM C `config.json`
field-name notice); they are upstream's and change nothing.

## Stock exceptions

None. Every mode runs `esm` 3.4.0 as shipped; no upstream issue is patched.

## Variables

The kit ships no configuration file; everything is read from the environment.

| variable | required | default | effect |
|---|---|---|---|
| `ESMC_OPT` | no | unset = nothing applied to the process; `run.sh check` and `python -m esmc_opt check` then resolve the package default, `exact` (`modes.DEFAULT_MODE`) | `exact` switches the kit on for the process at the first `import esm` (`esmc_opt.enable("exact")` from code does the same); `off` applies nothing; an unknown word is named on a `NOT ACTIVE` line and the process runs stock; `run.sh check` refuses a `--mode` that disagrees with it |
| `HF_HOME` | no | `~/.cache/huggingface` | upstream's weights cache, read in every mode |
| `TRITON_CACHE_DIR` | no | Triton's default (`~/.triton`) | Triton's JIT cache (upstream's rotary kernel; the kit's q/k kernel under `exact`); keep it persistent across processes |

`HF_HUB_OFFLINE` (1 for pinned loads, above; preset to 1 in the container image), `TORCH_EXTENSIONS_DIR`, `CUDA_VISIBLE_DEVICES` and `LD_LIBRARY_PATH` pass through as usual. The kit sets one marker
of its own, `ESMC_KIT`, when a lever applies; it is never read as a switch, and activation refuses to start with it already set.
Exit codes of `run.sh`: 0 ok · 1 the install step failed (pip, the extension build, a weight file off its pin) · 2 usage · 3 a kit mode
not active (unknown mode, `esm` absent or off its pinned commit, no GPU, the kit package not installed); `check --mode off` exits 0
once the pin check passes, GPU or not.
