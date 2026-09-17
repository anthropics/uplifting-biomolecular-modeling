#!/bin/bash
# flashzoi optimized — single entry point: a thin wrapper over `python -m flashzoi_opt` (the installed flashzoi_opt package: `pip install -e opt`).
#   run.sh pred   [--config h100|h200|a100] [--mode M] --input <dir> --out <dir> [--items a,b] [--tracks SPEC] [--jobs FILE] [--det] [--allow-partial]   one process: every <item>.npy one-hot window (uint8 or float32,
#                 (4, 524288), rows A,C,G,T) in <dir> through the documented call; <out>/<item>.npy + rows.jsonl (item, shape, dtype, wall_s) + opt_manifest.json
#   run.sh check  [--config h100|h200|a100] [--mode M]    dry run: resolves the mode against the kit's own tables on this GPU; nothing is applied
#   run.sh warm   [--config h100|h200|a100] [--mode M] [--allow-partial]    imports + the kit apply + one forward: fills the Triton cache
#   run.sh install [--weights DIR]    the install step: this kit installed editable into the python on PATH (pip install -e opt), then the pin check (stock/check_pins.py);
#                 --weights DIR also fetches the four pinned replicates into the hub cache DIR with huggingface_hub and checks their digests (then export FLASHZOI_WEIGHTS=DIR)
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (weights cache, JIT cache, target GPU). Mode = --mode when given, else
# FLASHZOI_OPT from the environment, else the package default (exact: the kit's one route, bitwise to stock); a --mode that disagrees with a set FLASHZOI_OPT is refused. Modes
# (off|exact) are resolved by the package from the kit's own tables (opt/forward/kits_v1_25/engines/flashzoi/kits/v1_25: LEVERS, PINS);
# this script validates none of them. `--mode off` is the stock route: the tree's one stock caller (opt/flashzoi_opt/stock_pred.py — upstream's API, nothing
# from the kit on the path) in a clean subprocess that proves its environment; its one command is pred (the stock arm has no dry run and no
# warm-up: check / warm under --mode off exit 2). Every route refuses (rc 2, with the install line) when the package is not importable and (rc 3) unless the stock is
# installed at its pin (stock/check_pins.py --package-only; the rest of the stack is named when it differs, never refused). Exit codes: 0 ok, 1 failed (a run, the install, a weight file off its pin), 2 usage / package not installed, 3 not active /
# pins not met / partial (a lever of the mode without the kit's evidence after the run; `--allow-partial` records and proceeds).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
stack_note() {   # the rest of stock/PINS.json (torch, triton, flash-attn, transformers, python, CUDA): a difference is NAMED and the run goes on — the kit says it again on its ACTIVE line (drift=[...])
  local d; d="$(python "$@" "$HERE/stock/check_pins.py" --quiet 2>&1 >/dev/null)" && return 0
  echo "run.sh: NOTE the installed stack differs from stock/PINS.json ($(printf %s "$d" | sed 's/^check_pins: //' | paste -sd ';' - | sed 's/;/; /g')) — running; outputs on this stack are not covered by the kit's pins" >&2
}
usage() { sed -n '2,16p' "$0" >&2; exit 2; }
CFG=""; CMD=""; MODE=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config)    [ $# -ge 2 ] || usage; CFG=$2; shift 2 ;;
    --config=*)  CFG=${1#--config=}; shift ;;
    --mode)      [ $# -ge 2 ] || usage; MODE=$2; shift 2 ;;
    --mode=*)    MODE=${1#--mode=}; shift ;;
    *) if [ -z "$CMD" ]; then CMD=$1; else ARGS+=("$1"); fi; shift ;;
  esac
done
case "$CMD" in pred|check|warm|install) ;; *) usage ;; esac
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
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md Install)" >&2; exit 2; }
  if python -I -c "import os,sys,importlib.util as u; d=os.path.realpath(sys.argv[1]); x=u.find_spec('flashzoi_opt'); sys.exit(0 if x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) else 1)" "$HERE/opt" 2>/dev/null; then
    echo "run.sh: flashzoi_opt is installed from this tree already ($HERE/opt) — the pip step is skipped"   # the container image ships it installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the kit package is expected at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" --package-only || { echo "run.sh: installed, but the stock is not at its pin (stock/check_pins.py: the line above) — every mode, off included, is defined against the pinned upstream release" >&2; exit 3; }
  stack_note -I
  if [ -n "$WEIGHTS" ]; then python -m flashzoi_opt.weights "$WEIGHTS" || exit $?; fi   # huggingface_hub's downloader into the hub cache DIR, then the sha256 check against stock/PINS.json (opt/flashzoi_opt/weights.py)
  exit 0
fi
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 2) when flashzoi_opt is not importable
fi
KIT=flashzoi; J0="${MODEL_OPT_JIT_ROOT:-}"   # [install]-jitcache mapping (before the block): the log tag it prints under; the root as preset, to recognise below a TRITON_CACHE_DIR that configs/<card>.env derived from it
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[${KIT:-kit}-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[${KIT:-kit}-kit] jit cache: $J ($W)"; }
if [ -n "${MODEL_OPT_JIT_ROOT:-}" ] && [ -n "${MODEL_OPT_STACK_KEY:-}" ] && { [ -z "${TRITON_CACHE_DIR:-}" ] || { [ "$MODEL_OPT_JIT_ROOT" != "$J0" ] && [ "${TRITON_CACHE_DIR:-}" = "$J0/$MODEL_OPT_STACK_KEY/triton" ]; }; }; then
  export TRITON_CACHE_DIR="$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/triton"   # [install]-jitcache mapping (after the block): the variable this kit's Triton kernels read, derived as configs/<card>.env derives it — when unset, or when the config derived it from a preset root the block has just stepped off
fi
FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
ENVMODE=${FLASHZOI_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with FLASHZOI_OPT=$ENVMODE from $FROM; one run has one mode — drop one of them" >&2; exit 2
fi
MODE=${MODE:-$ENVMODE}
python -c "import flashzoi_opt" 2>/dev/null || { echo "run.sh: flashzoi_opt is not installed on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/opt" >&2; exit 2; }
python "$HERE/stock/check_pins.py" --package-only >/dev/null || { echo "run.sh: the stock is not installed at its pin (stock/PINS.json 'package') — every mode, off included, is defined against that upstream release" >&2; exit 3; }   # the same interpreter and path the package runs on (the image puts packages on PYTHONPATH: no -I); the ONE environment refusal
stack_note
if [ "$MODE" = off ]; then                                        # stock route: pred = the tree's stock caller in a clean subprocess — the arm's one command
  case "$CMD" in pred) ;; *) echo "run.sh: --mode off has one command, pred (the stock arm has no check and no warm-up)" >&2; exit 2 ;; esac
fi
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m flashzoi_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
