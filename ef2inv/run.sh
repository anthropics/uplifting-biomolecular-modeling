#!/bin/bash
# ef2inv optimized — single entry point: a thin wrapper over `python -m ef2inv_opt` (the installed ef2inv_opt package and the shared core it pins in
# opt/pyproject.toml [tool.opt_core]: `run.sh install`).
#   run.sh design [--config h100] [--mode off|exact|fast|big] --target-name NAME [--target-sequence SEQ] [--binder-len L | --binder-name minibinder] --seed S --out <dir> [--target-hotspot-ids 56 57] [--det 0|1] [--chunk-size none|N] [--kernel-backend fused|cuequivariance|None] [--batch-size 1] [--is-antibody 0|1] [--epitope-contact-distance 12.0] [--binder-sequence SEQ] [--use-scaling-critics 0|1] [--upstream-fix ID[,ID...]] [--no-compile] [--allow-partial]
#   run.sh check  [--config h100] [--mode M] [--json]                                   the mode row, the pins, the kit bytes, the device: ACTIVE | REFUSED
#   run.sh warm   [--config h100] [--mode M] --target-name NAME [--target-sequence SEQ] --binder-len L --out <dir> [--chunk-size none|N] [--kernel-backend …] [--upstream-fix ID[,ID...]] [--no-compile]  the one-time costs through the mode's line (the JIT caches filled: first folds + one design-step-shaped grad pass)
#   run.sh install [--weights DIR]                                                      the install step: the shared core and this kit installed editable into the python on PATH (pip, or uv when
#                                                                                       that python carries no pip module), then the stock software pins checked (stock/check_pins.py --checks software);
#                                                                                       --weights DIR also fetches the six pinned weight snapshots into DIR (a Hugging Face cache root: HF_HOME=DIR at run
#                                                                                       time) with huggingface_hub's own downloader and checks every file against stock/PINS.json (sha256) — files already
#                                                                                       in DIR are kept and checked, never re-fetched
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (the tree, the data paths, the stack key). Mode = --mode when given,
# else EF2INV_OPT from the environment, else the package default (fast); a --mode that disagrees with a set EF2INV_OPT is a usage error (exit 2).
# Exit codes: 0 complete · 1 the install failed / a weight file off its pin · 2 usage (no/unknown command, missing config, --mode vs EF2INV_OPT disagreement, argument errors) · 3 refused by name / NOT ACTIVE / pins · 4 the loop failed (a CUDA out-of-memory arm by name).
# `--mode off` = the stock route: the upstream cookbook file in a subprocess proven clean of all kit variables, modules and directories, exactly as shipped;
# `--chunk-size none --kernel-backend cuequivariance` add upstream's two documented speed settings (the stock arm the rows time against).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="${1:-}"; [ -n "$CMD" ] || { sed -n '2,13p' "$0" >&2; exit 2; }; shift
CFG=""; MODE=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CFG="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    *) ARGS+=("$1"); shift;;
  esac
done
case "$CMD" in design|check|warm|install) ;; *) echo "run.sh: unknown command $CMD (design|check|warm|install)" >&2; exit 2;; esac
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
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md Install)" >&2; exit 3; }
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('ef2inv_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: ef2inv_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the install step is skipped"   # the container image ships them installed; a read-only image cannot re-run the installer
  elif python -m pip --version >/dev/null 2>&1; then
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  elif command -v uv >/dev/null; then
    echo "run.sh: $(command -v python) carries no pip module — installing with uv pip into that interpreter"   # the pinned stack's environment is made by uv and has no pip (environment/Dockerfile)
    uv pip install --python "$(command -v python)" -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (uv's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  else
    echo "run.sh: $(command -v python) carries no pip module and uv is not on PATH — add pip to the environment (python -m ensurepip) or put uv on PATH, then re-run" >&2; exit 3
  fi
  python -I "$HERE/stock/check_pins.py" --checks software || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py --checks software: the lines above — the pinned stock stack is not installed as pinned in this environment)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then python -m ef2inv_opt.weights "$WEIGHTS" || exit $?; fi   # huggingface_hub's downloader into DIR, then the sha256 check against stock/PINS.json (opt/ef2inv_opt/weights.py)
  exit 0
fi
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[ef2inv-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[ef2inv-kit] jit cache: $J ($W)"; }
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no configs/$CFG.env" >&2; exit 2; }
  set -a; . "$HERE/configs/$CFG.env"; set +a
fi
export MODEL_OPT="${MODEL_OPT:-$HERE}"
if [ -n "$MODE" ] && [ -n "${EF2INV_OPT:-}" ] && [ "$MODE" != "$EF2INV_OPT" ]; then
  echo "run.sh: --mode $MODE disagrees with EF2INV_OPT=$EF2INV_OPT" >&2; exit 2   # a usage error (2), not a refusal (3)
fi
# the card comparison lives in the package (opt/ef2inv_opt/hardware.py; EF2INV_GPU / EF2INV_GPU_MIB from the config set the expected card): design | warm |
# check word any other card on ONE `NOTE hardware ...; proceeding` line and run unchanged; the design verb refuses a box with no CUDA device by name (3).
# Installed = the shared core this kit pins (opt/pyproject.toml [tool.opt_core]) and the package, both editable (`run.sh install`):
python -c "import opt_core" 2>/dev/null || { echo "run.sh: opt_core is not installed on $(command -v python || echo 'python (not on PATH)'): bash $HERE/run.sh install (pip install -e $HERE/../common/opt_core -e $HERE/opt)" >&2; exit 3; }
python -c "import ef2inv_opt" 2>/dev/null || { echo "run.sh: ef2inv_opt is not installed on $(command -v python || echo 'python (not on PATH)'): bash $HERE/run.sh install (pip install -e $HERE/../common/opt_core -e $HERE/opt)" >&2; exit 3; }
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m ef2inv_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
