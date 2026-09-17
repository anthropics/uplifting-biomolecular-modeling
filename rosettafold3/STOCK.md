# RoseTTAFold3 — stock, as pinned

## Pin

Stock is `rf3 fold` from RosettaCommons/foundry at commit `4010e3e2e7350edada3e25a45c908c6bf407df4d` (`rc-foundry` 0.2.1.dev13,
not on PyPI at this version), vendored as `stock/foundry-4010e3e2e.tar.gz` with the five files the kit touches unpacked under
`stock/src/`, never edited. `stock/PINS.json` is the machine-readable pin (commit, install lines, checkpoint URL + sha256, CLI
defaults, stack, the two interpreters' file states); `stock/check_pins.py` makes every route refuse (exit 3) on an interpreter
that does not have upstream at the pin. The checkpoint is `rf3_foundry_01_24_latest_remapped.ckpt`, the file upstream's own
downloader fetches (`foundry install rf3 --checkpoint-dir <dir>`); `run.sh install --weights <dir>` calls that downloader when the
file is absent and checks its sha256 against the pin. Upstream exposes no speed setting beyond its defaults — cuEquivariance's
triangle kernels engage from the import, precision comes from the checkpoint — so `--mode off` is upstream exactly as shipped;
the kit applies no stock exception. CLI defaults: `n_recycles 10, diffusion_batch_size 5, num_steps 50,
early_stopping_plddt_threshold 0.5, seed null`.

## Stack

Pinned stack (`stock/PINS.json` "pinned_stack"): Python 3.12, torch 2.13.0+cu130, triton 3.7.1, cuequivariance 0.11.1 (cu13 ops),
lightning 2.6.5, atomworks 2.2.1. `environment/requirements.lock` is the full list, installed with `--no-deps`; stock is that
lock's `rc-foundry @ git+https://…@4010e3e2e…` line. Host: an NVIDIA driver that runs CUDA 13.0 (route C also: `git` and a C compiler on PATH). Three routes to the stack, from
the directory that holds `rosettafold3/` and `common/`:

    docker build -f rosettafold3/environment/Dockerfile -t rosettafold3-kit:dev .    # the add-on is applied at build; `run.sh install` inside only checks
    apptainer build rosettafold3.sif rosettafold3/environment/apptainer.def          # from that image; `apptainer run --nv --bind <weights dir>:/weights rosettafold3.sif <verb> …` = `run.sh <verb> …`
    sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates wget git build-essential   # route C on a bare Ubuntu host (root: no sudo); skip what you have
    command -v uv >/dev/null || { t=$(mktemp) && wget -qO "$t" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t" && . "$HOME/.local/bin/env"; }   # uv itself, once per user (skipped when present; wget is in the apt line above)
    uv venv --seed --managed-python --python 3.12.1 <dir> && . <dir>/bin/activate    # or your own environment; git + a C compiler on PATH (--managed-python = uv's own released CPython 3.12 build, headers included; python.org / conda 3.12 work too; if you use apt/deadsnakes python instead of uv: python3.12 + python3.12-venv + python3.12-dev, not python3-dev, then `python3.12 -m venv <dir>`)
    cd rosettafold3                                                                  # the lines below run inside the kit directory
    grep -v -E '^#' environment/requirements.lock > /tmp/stack.txt && python -m pip install --no-deps -r /tmp/stack.txt
    python -m pip install --no-deps --force-reinstall nvidia-cudnn-cu13==9.20.0.48 nvidia-nccl-cu13==2.29.7
    export PYTHONHASHSEED=0 CFLAGS=-g0                                               # the two variables the image sets (README C block's last line)
    bash run.sh install --make-venv --weights /weights/rosettafold3                     # = README step 2's install line; step 2 resumes at export

The image also sets `PYTHONHASHSEED=0` and `CFLAGS=-g0` (`environment/Dockerfile` `ENV`: fixed hash seed, no debug info in the launchers
Triton compiles) and `ROSETTAFOLD3_OPT_STOCK_PYTHON=/usr/local/bin/python`; in your own environment export the first two for the same process
environment and point the third at your pristine interpreter (README.md Setup). The image build takes optional pre-filled compile caches `_jitcache/rosettafold3-*-jit.tar` from the build context (one per stack
key, unpacking as `<stack key>/…` into `/opt/jit_cache`; the build is identical without one); in the image `run.sh` uses `/opt/jit_cache`
in place as the JIT root when `MODEL_OPT_JIT_ROOT` is unset and the directory is writable, else seeds `MODEL_OPT_JIT_ROOT` (when unset:
`${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`; when set and empty: that directory; layout `<root>/<stack key>/{triton,inductor}`) from it — the copy is skipped when
that directory already holds the seed, so it recurs wherever `/tmp` is fresh per run — a new node, a per-job `TMPDIR`, or `apptainer run --contain` — leaves a
non-empty `MODEL_OPT_JIT_ROOT` untouched, and prints `[rosettafold3-kit] jit cache: <dir> (<how>)` at every start (`<how>` names the provenance). A preset `MODEL_OPT_JIT_ROOT` the process cannot write is never compiled into: when `MODEL_OPT_STACK_KEY` is also preset and its
subtree there holds at most `MODEL_OPT_JIT_SEED_MAX_FILES` files (default 5000), that subtree is seeded once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`
(printed as above); otherwise the read-only root is used as is (nothing is compiled into it; caches already there are read) and nothing is printed.

pip builds `rc-foundry` from source with upstream's build backend, hatchling + hatch-vcs (foundry's `pyproject.toml` `[build-system]`),
which pip's build isolation fetches from the package index at install time (network needed; `environment/Dockerfile` constrains their
versions with `PIP_CONSTRAINT`); the lock pins runtime packages only. Without access to the upstream repository,
`pip install "stock/foundry-4010e3e2e.tar.gz[rf3]"` installs the same source (it reports version 0.0.0; the pin check accepts the archive
by its sha256); the build backend still comes from the index.

Two interpreters. The kit applies by writing five `rf3` files into site-packages — upstream's `inference_sampler.py`, `RF3_structure.py`,
`af3_diffusion_transformer.py` and `loss.py` modified, plus the new `graph_flags.py` (`opt/forward/rf3_xattempt_addon/install.sh`) —
and a tree with those bytes is not stock, so the pristine interpreter (`ROSETTAFOLD3_OPT_STOCK_PYTHON`) is only ever read and runs
`--mode off`, while `opt/venv` (`ROSETTAFOLD3_OPT_PYTHON`) receives the kit files. `run.sh install --make-venv` creates `opt/venv`
from the pristine interpreter (`python -m venv --system-site-packages` plus its own copy of the `rf3` and `foundry` packages; when
the stock interpreter is itself a venv its site-packages go into the new venv's `parent_venv.pth`, and when no `setuptools` is reachable the
lock's `setuptools` and `wheel` are pip-installed into it first, as the build backend),
installs `../common/opt_core` and `opt` editable into it, runs the pin check on both interpreters and applies the five files.
Every command classifies both interpreters' `rf3` files by sha256 against `stock/src/` and the kit's `patched/` files and refuses
on any other state. `--no-addon` does every step but the five files. `environment/Dockerfile` applies them at image build (into `opt/venv`
only; `/usr/local/bin/python` stays pristine), so `run.sh install` inside the container is a check that changes nothing, and
`environment/apptainer.def`'s `%post` runs the same check.
`--stock-python P` / `--opt-python P` name the interpreters explicitly. An environment patched by an earlier copy of the kit
takes this copy's files with `PATH=opt/venv/bin:$PATH bash opt/forward/rf3_xattempt_addon/install.sh --rebind` (it prints
`OVERLAY_PRESTATE: pristine|current|previous:<id>|foreign:<files>` and refuses a foreign edit without touching it).

Other cards. Kit modes run on NVIDIA GPUs of compute capability 8.0 and up from the one pinned stack (the cu130 wheels carry sm_80 and sm_90
code); Triton kernels compile per card into the cache keyed by `MODEL_OPT_STACK_KEY` (`torch2.13.0-sm90` / `torch2.13.0-sm80`). Kernel rows are
chosen by the running device, not by the configuration: `configs/a100.env` and `configs/h200.env` differ from `configs/h100.env` only in
`MODEL_OPT_TARGET_GPU` (a mismatch with the visible GPU is a note on `check`). On an H200 (compute capability 9.0, `--config h200`) the H100's
kernel rows and kernel cache apply; an exact tier serves its fused row where the provider lists that card's stack, stock's statements by name elsewhere. On an A100 80GB every mode runs with the same lever set; the differences, each named on
the run's `LEVER` / `XTR` / `SELECT` lines: the triangle-multiplication launch cells and the megakernel tiles are their 8.0 rows; `exact`'s
fused transitions serve the 128-wide pair transitions and run stock's statements by name (`route:<key>:<word>`) for the other widths; the fused
triangle attention's core is the shared core's Triton kernel (its CUDA kernels are sm_90-only); the MSA module binds the pair-weighted-averaging
cell alone, the outer-product mean staying stock's; `fast` captures no pairformer CUDA graph (`levers_budget_skipped=fpf_tg(…)`, exit 0). A lever
that cannot engage at all on the running GPU refuses the run by name (exit 1, or `NOT ACTIVE` exit 3); size gates are census, exit 0. `exact`
equals `--mode off` run on the same card; nothing in this tree compares one card with another. On the H100, `exact`'s triangle attention is served by the shared core's kernel where its table lists the running library stack and the call's shape (bf16, head width 32, pair lengths 101–4,160; `LEVER name=fpf_xatt … strategy=F1.triattn_exact`, and `[fpf_rf3 triattn] CENSUS component=xatt … rows=…` counts the calls per row) — bit-identical to `cuequivariance_torch.triangle_attention`, checked against the library on the process's first served call — and by the library op by name for every other call, stack or card; `LEVER name=fpf_xmul …` and `fpf_xln` likewise name the shared core's exact tiers, which serve a call only where their tables list it bitwise for the card and stack and run the library op / stock's statements by name otherwise (CHANGES.md `exact`). Under `exact` (either card) `pred.log` also carries `NAMED_FALLBACK:ln|…` lines — `…|c16|… refused=cells:no_family:c16 served=aten_autocast` and `…|fp32|c833|N=…|fwd.eager` / `…|fwd.graph … refused=cells:no_family:c833 served=aten` —, counted in `named_fallback=` and `alerts=` on the `[opt_core] CELLS` line: informational — LayerNorms of those widths have no fused row and run the stock op by name; the outputs are `exact`'s.

Settings: the kit has none of its own. `run.sh pred … [key=value …]` (and `rosettafold3-opt pred`) appends `rf3 fold`'s hydra
overrides verbatim, in order, on every mode alike; the kit itself writes only `inputs=`, `out_dir=`, `ckpt_path=` (refused as
overrides by name) and, with `--seeds S[,S…]`, one `seed=<S>` per fold (`opt/rosettafold3_opt/settings.py`).

## How stock is run

`--mode off` executes `$ROSETTAFOLD3_OPT_STOCK_PYTHON -m rf3.cli fold inputs=<input> out_dir=<out_dir> ckpt_path=<ckpt> [key=value …]`
(`seed=<S>` and `out_dir=<out_dir>/seed-<S>` per fold with `--seeds`) in a clean subprocess on the pristine interpreter with nothing of
the kit importable and no kit variable in its environment (`opt/rosettafold3_opt/stock_fold.py`); it prints
`[rosettafold3-opt stock] ENV-CLEAN ok: …` with the interpreter's file state first.

## Stock exceptions

None. `--mode off` is upstream exactly as shipped; the kit applies no fix or setting to stock.

## Variables

`configs/h100.env` (`configs/a100.env` / `configs/h200.env` = the same file with the A100 / H200 label) fills the unset ones with the defaults shown and keeps
a pre-set value — except `ROSETTAFOLD3_OPT_CKPT`, which has no default, and the three cache directories, which it exports only when
`MODEL_OPT_JIT_ROOT` names a cache root.

| variable | meaning | default |
|---|---|---|
| `ROSETTAFOLD3_OPT_CKPT` | the checkpoint every mode passes as `ckpt_path=` (a per-command `--ckpt <file>` overrides it); required — `pred` / `warm` refuse by name without it, `check` reports it absent | none |
| `ROSETTAFOLD3_OPT_STOCK_PYTHON` | the pristine interpreter (`--mode off`, and the source of `--make-venv`) | `$MODEL_OPT/stock/venv/bin/python` |
| `ROSETTAFOLD3_OPT_PYTHON` | the patched interpreter every kit mode runs on | `opt/venv/bin/python` when it exists, else `python` on PATH |
| `MODEL_OPT_STACK_KEY` | the JIT cache key `torch<version>-sm<cc>`, probed from the patched interpreter when unset | derived |
| `MODEL_OPT_JIT_ROOT` | optional persistent cache root: set, the config exports `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` = `<root>/$MODEL_OPT_STACK_KEY/{triton,inductor}` and `ROSETTAFOLD3_OPT_DIGEST_DIR` = `<root>/weights` (each unless pre-set); unset, none of the three is exported; when set it must be writable: the compile caches are created under it per GPU / stack key (`$MODEL_OPT_STACK_KEY`) on first run | unset: `run.sh` then supplies a private per-user root, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root) — the tools' own default locations this row names apply only without `run.sh` or when that root is refused |
| `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` | the Triton / Inductor JIT caches; a pre-set value is kept as given | `$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/{triton,inductor}` when `MODEL_OPT_JIT_ROOT` is set, else the libraries' own defaults |
| `ROSETTAFOLD3_OPT_DIGEST_DIR` | the weights sha256 memo directory (`weights_digests.json`; `check` rewrites it afresh) | `$MODEL_OPT_JIT_ROOT/weights` when `MODEL_OPT_JIT_ROOT` is set; otherwise `<TRITON_CACHE_DIR>/../../weights`, else `~/.cache/rosettafold3_opt/weights` |
| `MODEL_OPT_TARGET_GPU` | the GPU class the configuration targets (`check` notes a mismatch with the visible GPU) | `H100` (`configs/a100.env`: `A100`; `configs/h200.env`: `H200`) |
| `ROSETTAFOLD3_OPT` | the mode on the environment route; a `--mode` that disagrees with it is refused | unset |
| `MODEL_OPT_LEVERS_OFF` | comma list of lever names withheld from the mode; `compile` is accepted as a no-op (`--no-compile`) | unset |
| `ROSETTAFOLD3_BIG_ALLOW_PARTIAL` | `1` = `--allow-partial` for an `rf3 fold` process activated by `ROSETTAFOLD3_OPT=big` (`pred` sets it from the flag) | unset |
