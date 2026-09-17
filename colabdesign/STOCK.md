# ColabDesign — stock, as pinned

## Pin

Upstream: ColabDesign 1.1.3 @ `e31a56fe1d9b4de25c8697f3a28b75892941cc72` (https://github.com/sokrypton/ColabDesign), shipped under `stock/`
as the source archive `colabdesign-e31a56fe.tar.gz` plus reference copies of the three upstream files the kit reads (`stock/src/colabdesign/af/prep.py`,
`af/model.py`, `af/alphafold/model/modules.py`; sha256 in `stock/PINS.json`); and BindCraft @ `efb5bfeb8b4b1a5944256f979c34e0c8e6a82d9d`
(https://github.com/martinpacesa/BindCraft), shipped as `stock/bindcraft-efb5bfeb.tar.gz` with its design step vendored under
`stock/src/bindcraft/` (`functions/colabdesign_utils.py` `binder_hallucination` and the modules it imports, the settings
files, `example/PDL1.pdb`) and imported from there unmodified. BindCraft's two prebuilt third-party executables are not distributed here —
neither under `stock/src/` nor inside the archive (its recipe in `PINS.json` excludes `functions/dssp` and `functions/DAlphaBall.gcc`):
`functions/dssp`, which the design step runs (`optimise_beta`'s secondary-structure check), is fetched by `bash run.sh install` from the
BindCraft repository at the pinned commit (`https://raw.githubusercontent.com/martinpacesa/BindCraft/<commit>/functions/dssp`; network
needed once), accepted only when its sha256 and size equal the pin in `PINS.json` `upstream.bindcraft.fetched`, made executable (mode 755) and
placed at `stock/src/bindcraft/functions/dssp`, where BindCraft's default `dssp_path` expects it — other bytes are refused by name and the
install stops (exit 1); a file already in place is checked, never re-fetched (by hand: download that URL, compare `sha256sum` with the pin,
`chmod 755`, move it into place); `functions/DAlphaBall.gcc` serves only BindCraft's PyRosetta stage, which this kit does not run, and is
neither carried nor fetched — obtain it from the BindCraft repository if you enable that stage yourself. Their licences are their upstream
authors' (THIRD_PARTY_NOTICES.md). Weights: AlphaFold-Multimer v3, `params/params_model_{1..5}_multimer_v3.npz`
(sha256 in `PINS.json` `weights.files`), fetched by `bash run.sh install --weights DIR` from DeepMind's parameter archive
(`https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar`, the source BindCraft's own installer uses) into `DIR/params/`
(a file already there whose digest is not the pin is refused by name, exit 1, and left in place); `design` and `warm` digest the params
files they read, word each `pinned` or `NOT PINNED` by name and proceed, and refuse a root without `params/` (usage, exit 2); `check`
reports the same words when the root holds `params/` and prints `weights=not checked` otherwise. Upstream licences and notices as found under `stock/`:
ColabDesign's Beerware `stock/src/LICENSE.txt` (the `colabdesign.af.alphafold` model code inside it: Apache-2.0, DeepMind Technologies
Limited), BindCraft's MIT `stock/src/bindcraft/LICENSE`; the parameters are CC BY 4.0 and not redistributed (third-party notices and licence texts: THIRD_PARTY_NOTICES.md, `third_party_licenses/`). `stock/` is never edited;
BindCraft (beyond the vendored design step) and PyRosetta are not installed.

## Stack

Debian 12 (`mambaorg/micromamba:2.1.1`), CUDA 12.9 libraries from the conda environment (host driver of the CUDA 12 series), CPython
3.10.18, jax / jaxlib 0.6.0 with jax-cuda12-plugin / jax-cuda12-pjrt 0.6.0, cuDNN 9.10, dm-haiku 0.0.17, numpy 1.26.4, scipy 1.15.2,
colabdesign 1.1.3, freesasa, and on `PATH`: git (the VCS install), ffmpeg (BindCraft's design step writes its trajectory animation through
matplotlib's ffmpeg writer and fails at the end of the trajectory without it) and gcc with the Python headers (freesasa builds from source);
no `triton` distribution (the Pallas kernels lower through jaxlib's bundled Triton) — the full
list is `environment/conda-linux-64.lock` (conda) + `environment/requirements.lock` (pip); `environment/Dockerfile` and `apptainer.def`
build it. From inside `colabdesign/` (a fresh environment as below, or the pip lines into an existing one at these pins):

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential ca-certificates git wget bzip2   # bare Ubuntu host (root: no sudo); skip what you have
f=$(mktemp) && wget -qO "$f" https://github.com/mamba-org/micromamba-releases/releases/download/2.1.1-0/micromamba-linux-64 && echo "0d2dd49fb0171f27a63175e2c21941aafabdd6e46f521c498620b4b14ff604d8  $f" | sha256sum -c - && mkdir -p ~/.local/bin && install -m 755 "$f" ~/.local/bin/micromamba && rm -f "$f" && ~/.local/bin/micromamba shell init -s bash -r ~/micromamba   # micromamba itself (2.1.1, the image's), when the host has none: the release binary, put in place only when its sha256 matches (conda / mamba take the same lock file)
eval "$("$HOME/.local/bin/micromamba" shell hook -s bash)"              # hook micromamba into this shell (the installer only edits ~/.bashrc)
micromamba create -y -n colabdesign-kit -f environment/conda-linux-64.lock && micromamba activate colabdesign-kit   # activation-script `ERROR:` lines printed here (binutils, cuda-nvcc, gdk-pixbuf) are harmless when the line ends rc 0
python -m pip install --no-deps -r environment/requirements.lock        # stock ColabDesign at the pin (git URL) and freesasa, ≈5 min;
                                                                        # offline: pip install --no-deps stock/colabdesign-e31a56fe.tar.gz
bash run.sh install [--weights /weights/af2]                               # = README step 2's install line — type it once (it also fetches BindCraft's DSSP executable, see Pin); step 2 continues at its export line
```

pip builds ColabDesign (from the git URL), the kit and the shared core in isolated build environments and fetches their build backend
(setuptools, wheel) from the package index at install time (network needed); the locks pin runtime packages only.
The image build takes an optional pre-filled compile cache `_jitcache/colabdesign-<stack key>-jit.tar` from the build context into
`/opt/jit_cache` (the build is identical without one); in the image `run.sh` uses `/opt/jit_cache` in place when it is writable, else seeds the
JIT root `MODEL_OPT_JIT_ROOT` (default `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`) from it once, prints `[colabdesign-kit] jit cache: <dir> (<how>)`; a preset `MODEL_OPT_JIT_ROOT` the process cannot write is used as is — nothing is compiled into it (jax's cache
then stays in the lever's default directory) — unless `MODEL_OPT_STACK_KEY` names a subtree of it with at most
`MODEL_OPT_JIT_SEED_MAX_FILES` files (default 5000), which is seeded once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`, and,
whenever a writable JIT root is set, points jax's `JAX_COMPILATION_CACHE_DIR` at `<root>/jax0.6.0-jaxlib0.6.0-cuda12plugin0.6.0-<gpu>/xla` (the stack
key: the pinned jax stack and the card's product name) unless that variable is already set — the `compilecache` lever adopts it.
Accepted ColabDesign installs: one pip recorded from commit `e31a56fe` (the git URL; 1.1.3 is not on PyPI) or one reporting 1.1.3 without a VCS
record (a wheel built from that commit, your own checkout; an editable checkout is taken at its reported version); `af/prep.py`, `af/model.py` and
`af/alphafold/model/modules.py` are compared with the commit wherever pip's RECORD lists them; a missing package, another version or commit, or a differing file is refused by name (exit 3).
`stock/check_pins.py` confirms the installed colabdesign is the pinned commit, that jax, jaxlib, dm-haiku and numpy equal `PINS.json`
`pins` (the names in `pins_asserted`), and that the interpreter is the `PINS.json` `python` release series (another 3.10.x patch release
prints one `NOT PINNED` line; another series exits 3). At run time only a ColabDesign off its
pinned commit is refused (exit 3); a stack package off its pin is named on the `ACTIVE` line as `stack_drift=…` and the run proceeds.
Tested card: NVIDIA H100 80GB (`PINS.json` `gpu`, `configs/h100.env`); `configs/a100.env` and `configs/h200.env` differ only in their
`MODEL_OPT_TARGET_GPU` label; `gpu.cc_min` (8.0) is the kernels' compute-capability floor, not a gate — the card's own capability is on the
`ACTIVE` line (`gpu=<name>(sm<cc>,<MiB>MiB)`) and a card below the floor is noted in `check --json`, never refused. One image serves all three
cards: nothing is compiled for a GPU at build time.

## How stock is run

`--mode off` executes BindCraft's `binder_hallucination` on stock ColabDesign (`opt/colabdesign_opt/stock_design.py` driving the vendored
`stock/src/bindcraft/functions/colabdesign_utils.py`) in a clean subprocess with nothing of the kit importable and every kit variable
removed; the subprocess prints an `ENV-CLEAN` line stating the check. Settings are BindCraft's `--advanced` / `--filters` files (defaults
`settings_advanced/default_4stage_multimer.json`, `settings_filters/default_filters.json`), read at run time in every mode. The only
environment setter in either upstream tree is ColabDesign's import-time `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`, which fires in
every mode; the kit presets no `XLA_*` / `JAX_*` variable for stock, and `PYTHONHASHSEED=0` in every arm.

## Stock exceptions

None. No stock file is patched and no fix is applied to either upstream; every mode runs BindCraft's design step on ColabDesign as shipped.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `COLABDESIGN_PARAMS_DIR` | yes (or `--params-dir`) | BindCraft's folder `stock/src/bindcraft` | the params root (`<root>/params/params_model_{1..5}_multimer_v3.npz`); `design` / `warm` refuse a root without `params/` (exit 2) |
| `COLABDESIGN_OPT` | no | unset (= `--mode`, else `fast`) | the mode for an unchanged script (`COLABDESIGN_OPT=fast python …` engages at `import colabdesign`) and for run.sh when `--mode` is absent; must agree with `--mode` when both are set (exit 2) |
| `COLABDESIGN_OPT_HOME`, `MODEL_OPT` | no | the editable install's tree | this `colabdesign/` directory, for an interpreter whose package is not installed editable from this tree; `run.sh` and `configs/<card>.env` set `MODEL_OPT` |
| `MODEL_OPT_TARGET_GPU` | no | `H100` / `A100` from `configs/<card>.env` | a label recorded as `target_gpu` in `check --json`; a different visible card is recorded, not refused |
| `COLABDESIGN_OPT_LOWERCACHE` | no | unset | `relower`: lever `lowercache` lowers + compiles every design program it LOADS afresh as well and compares the first call's outputs of the loaded and the fresh executable bit for bit (`relower=same:<n>` on its LEVER line; the key's self-test — it costs the lowering the lever exists to remove). Read in-process (`route=api|env`); an arm launched by `run.sh` never carries `COLABDESIGN_OPT_*` (`must_be_absent_prefixes`), so the launcher names it under `env_dropped` and the arm runs without it. The executable store lives beside the compile cache: `<XDG_CACHE_HOME or ~/.cache>/colabdesign_opt/pcc/<stack key>/lowered/`, or `<dir>/../lowered/` beside an adopted `JAX_COMPILATION_CACHE_DIR` |
| `JAX_COMPILATION_CACHE_DIR` | no | `<XDG_CACHE_HOME or ~/.cache>/colabdesign_opt/pcc/<stack key>/` | jax's own variable; when set, the `compilecache` lever adopts it as the persistent cache directory |
