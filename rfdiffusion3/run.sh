#!/bin/bash
# RFdiffusion3 optimized — single entry point: a thin wrapper over `python -m rfdiffusion3_opt` (the installed rfdiffusion3_opt package on the shared core:
# `run.sh install` puts them in, once per interpreter; every other verb runs the package's core pin gate first and stops, rc 3, on an absent core or one that is not the pinned one).
#   run.sh install [--weights DIR]                              the install step, once per interpreter (its python first on PATH): the shared core and this kit installed editable
#                 (`pip install -e ../common/opt_core -e opt`; a tree already installed from here is named and the pip step skipped), then the pin check
#                 (`stock/check_pins.py --expect any`: stock RFdiffusion3 at the pin in this interpreter; exit 3 when it is not); --weights DIR also fetches upstream's
#                 checkpoint into DIR with upstream's own installer and checks it against stock/PINS.json (sha256) — DIR/rfd3_latest.ckpt is then your RFD3_CKPT
#   run.sh design [--config h100] [--mode exact|fast|off] [--ckpt <file>] [--det 0|1] [--allow-partial] [--] inputs=<spec> out_dir=<dir> [key=value ...]
#                 one `rfd3 design` run: every key=value token is upstream's own hydra override, passed through verbatim (its names and defaults:
#                 inputs= out_dir= seed=101 diffusion_batch_size=16 n_batches=2 inference_sampler.step_scale=<x> compile_model=true …; no token = upstream's
#                 configuration untouched); `--ckpt` (default $RFD3_CKPT, configs/<cfg>.env) becomes upstream's `ckpt_path=` unless the line names one.
#                 opt_manifest.json beside the outputs (off: the stock caller in the pristine interpreter; its environment proof is the manifest's `stock_env_proof`)
#   run.sh check  [--config h100] [--mode exact|fast|off] [--json] [--allow-partial]   dry run: gates the mode on this box, prints the DRY-RUN line and the pins report; nothing is exported
#   run.sh warm   [--config h100] [--mode exact|fast|off] [--ckpt <file>] [--json] [--allow-partial]   the checkpoint load under the mode (imports, checkpoint, model on the GPU)
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (checkpoint, JIT cache dirs, target GPU). Mode = --mode when given, else RFDIFFUSION3_OPT
# from the environment, else the package default (fast) — the core's mode argument: --mode wins over the variable. `--mode off` runs in the interpreter
# RFDIFFUSION3_STOCK_PYTHON names (the pristine one; default: the `python` on PATH, which then must be pristine): STOCK.md.
# Pins (stock/PINS.json: the upstream commit / archive, foundry/utils/torch.py, the pinned stack) are REPORTED — `check` prints stock/check_pins.py's report,
# `design` / `warm` print one `PINS …` line per finding (prefixed `[rfdiffusion3-opt]` on the kit routes, `[rfdiffusion3-opt stock]` in the stock child) — and never refuse a run: a checkout upstream itself runs is stock (`install` alone stops on them, 3: it answers whether this interpreter holds the pinned upstream). What refuses is
# deployment: the package or its core not installed (3), the kit's ten files not installed in this interpreter for a kit mode (3), the stock interpreter's rfd3
# tree not pristine for `off` (NOT STOCK, 3), no GPU (3), no checkpoint (2).
# Exit codes: 0 ok; 1 the run failed (or the install's pip step, or a weight file off its pin); 2 usage; 3 NOT ACTIVE (a refused deployment gate, the install's pin check, a stock process not proven stock, upstream's compile_model=true under
# exact | fast — no such kit route, `--mode off` runs it —, or a PARTIAL activation — the lever without evidence of application — unless --allow-partial /
# RFDIFFUSION3_OPT_ALLOW_PARTIAL=1 records the allowance in opt_manifest.json and lets the run proceed with its own exit); 5 KERNELS refused: an accelerator a KIT
# route's lever needs is absent or fell back, or the dense atom-attention pre-flight bound (the stock routes never stop: they name the path upstream picks). A design line
# that asks upstream for a computation the levers do not drive — classifier-free guidance (`inference_sampler.use_classifier_free_guidance=true`) or the
# low-memory tokenization (`low_memory_mode=true`), or selects the symmetry sampler (`inference_sampler.kind=symmetry`) — is refused by name too under exact | fast
# (NOT ACTIVE, 3, nothing runs): `--mode off` serves it.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
usage() { sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d' >&2; exit 2; }   # the whole header: line 2 up to (not including) the `set -euo pipefail` line
CMD=""; CFG=""; MODE=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config)   [ $# -ge 2 ] || usage; CFG=$2; shift 2 ;;
    --config=*) CFG=${1#--config=}; shift ;;
    --mode)     [ $# -ge 2 ] || usage; MODE=$2; shift 2 ;;
    --mode=*)   MODE=${1#--mode=}; shift ;;
    *) if [ -z "$CMD" ]; then CMD=$1; else ARGS+=("$1"); fi; shift ;;
  esac
done
case "$CMD" in design|check|warm) ;; install) ;; *) usage ;; esac
if [ "$CMD" = install ]; then                                       # the install step: everything below it presupposes the installed package
  WEIGHTS=""
  [ -z "$CFG" ] && [ -z "$MODE" ] || { echo "run.sh: install takes no --config / --mode (usage: run.sh install [--weights DIR])" >&2; exit 2; }
  set -- ${ARGS[@]+"${ARGS[@]}"}
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
      --weights=*) WEIGHTS=${1#*=}; [ -n "$WEIGHTS" ] || { echo "run.sh: install --weights= takes a directory" >&2; exit 2; }; shift ;;
      *) echo "run.sh: install takes no argument '$1' (usage: run.sh install [--weights DIR])" >&2; exit 2 ;;
    esac
  done
  command -v python >/dev/null || { echo "run.sh: no python on PATH — put the interpreter this kit installs into first on PATH (README.md Install: once per interpreter)" >&2; exit 3; }
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('rfdiffusion3_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: rfdiffusion3_opt and opt_core are installed from this tree already in $(command -v python) ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" --expect any || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the lines above — stock RFdiffusion3 is not installed as pinned in $(command -v python))" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then python -m rfdiffusion3_opt.weights "$WEIGHTS" || exit $?; fi   # upstream's installer into DIR, then the sha256 check against stock/PINS.json (opt/rfdiffusion3_opt/weights.py)
  exit 0
fi
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[rfdiffusion3-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
}
if [ -n "$J" ] && ! { mkdir -p "$J" && [ -w "$J" ]; } 2>/dev/null; then   # a preset root this process cannot write: this stack's key dir (≤ MODEL_OPT_JIT_SEED_MAX_FILES files, default 5000) seeds a writable copy; larger / no key: the root is left as is
  R="$J"; J="$T"; K="${MODEL_OPT_STACK_KEY:-}"; W=""
  if ! seedroot; then :
  elif [ -e "$J/.seeded" ]; then W="seeded from read-only root"
  elif [ -n "$K" ] && [ -d "$R/$K" ] && [ "$(find "$R/$K" -type f 2>/dev/null | head -n $((N+1)) | wc -l)" -le "$N" ]; then
    { mkdir -p "$J/$K" && cp -a "$R/$K/." "$J/$K/" && chmod -R u+w "$J" && find "$J" -name '__grp__*.json' -exec sed -i "s#$R/#$J/#g" {} + && touch "$J/.seeded" && W="seeded from read-only root"; } || rm -rf "$J/$K"
  fi; if [ -n "$W" ]; then mkdir -p "$J"; else J="$R"; fi   # nothing seeded: the root stays where the caller put it (readers still hit it; the writers step aside by name)
fi
if [ -d "$I" ] && [ -n "$(ls -A "$I" 2>/dev/null)" ]; then
  if [ -z "$J" ]; then if [ -w "$I" ]; then J="$I"; W="in-image"; elif seedroot; then J="$T"; W="seeded from image"; fi
  elif [ -z "$(ls -A "$J" 2>/dev/null)" ]; then W="seeded from image"; elif [ -z "$W" ]; then W="user"; fi
  if [ "$W" = "seeded from image" ]; then [ -e "$J/.seeded" ] || { mkdir -p "$J" && cp -a "$I/." "$J/" && chmod -R u+w "$J" && { [ "$J" != "$T" ] || chmod 700 "$T"; } && find "$J" -name '__grp__*.json' -exec sed -i "s#$I/#$J/#g" {} + && touch "$J/.seeded"; } || W="unseeded"; fi   # cp -a gives the copy the image directory's mode: the per-user root keeps 0700
elif [ -z "$J" ] && seedroot; then J="$T"; export MODEL_OPT_JIT_ROOT="$J"; fi   # no preset root and no image cache: the private per-user root seedroot made or checked is this run's cache root, exported without a printed line; a refused one is named by seedroot and the run has no cache root
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[rfdiffusion3-kit] jit cache: $J ($W)"; }
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses when the package is not importable or the core is not the pinned one (rc 3) or the JIT cache key cannot be derived (rc 2)
fi
MODE=${MODE:-${RFDIFFUSION3_OPT:-}}                              # --mode, else the variable; not validated here: the mode table (opt/rfdiffusion3_opt/modes.py) judges it and names the reason
env -i PATH="$PATH" PYTHONPATH="${PYTHONPATH:-}" python -c "import rfdiffusion3_opt" 2>/dev/null || { echo "run.sh: rfdiffusion3_opt is not importable on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/../common/opt_core -e $HERE/opt" >&2; exit 3; }   # importability only, in a clean environment: under RFDIFFUSION3_OPT[_*] the installed .pth refuses at interpreter start, which is the next probe's line to print, not this one's to swallow
python -c "from rfdiffusion3_opt._autoload import core_gate; core_gate()" >/dev/null || exit 3   # the core pin gate: absent, or older than the pinned floor -> its NOT ACTIVE line, rc 3, before any verb
if [ "$CMD" = check ]; then                                       # the pins REPORT (never a gate): the stock interpreter's for `off` (RFDIFFUSION3_STOCK_PYTHON, else this python), this interpreter's otherwise
  if [ "$MODE" = off ]; then "${RFDIFFUSION3_STOCK_PYTHON:-python}" -I "$HERE/stock/check_pins.py" --expect pristine >&2 || true; else python -I "$HERE/stock/check_pins.py" --expect any >&2 || true; fi
fi
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m rfdiffusion3_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
