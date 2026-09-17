#!/usr/bin/env bash
# atlasfold kit — run.sh install|pred|check|warm
#   ./run.sh install [--weights DIR]        pip install --no-deps -e stock/src + pip install --no-deps -e opt/ ; prints the stock/src digest and, when a weights root is given, WEIGHTS present/missing as information (weights are never pinned or gated)
#   ./run.sh check  [--config h100|h200|a100|b300] [--mode M]   DRY-RUN: core/stock/GPU gates + lever plan
#   ./run.sh warm   [--config …] [--mode M] [--multimer] one synthetic 640-token fold (AFO_WARM_TOKENS) to compile every first-use kernel of the mode
#   ./run.sh pred   [--config …] --mode off|exact|fast|big [--det 0|1] [--n_gpu P] -- monomer|multimer <stock atlasfold flags>
# --config <cfg> (any verb but install; before the stock separator `--`) sources configs/<cfg>.env — deployment parameters only (weights root, JIT-cache
# root, target GPU), never the mode; without it ATLASFOLD_KIT_CONFIG=<path> names the file (default configs/h100.env). An unknown <cfg> exits 2 by name.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_FILE="${ATLASFOLD_KIT_CONFIG:-$HERE/configs/h100.env}"; CFG_NAME=""; ARGS=()
while [ $# -gt 0 ]; do                                   # --config is the kit's; everything from `--` on is the stock CLI's, untouched
  case "$1" in
    --config)   [ $# -ge 2 ] && [ -n "$2" ] || { echo "run.sh: --config takes a name (see $HERE/configs/)" >&2; exit 2; }; CFG_NAME=$2; shift 2 ;;
    --config=*) CFG_NAME=${1#--config=}; [ -n "$CFG_NAME" ] || { echo "run.sh: --config= takes a name (see $HERE/configs/)" >&2; exit 2; }; shift ;;
    --)         ARGS+=("$@"); break ;;
    *)          ARGS+=("$1"); shift ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}
verb="${1:-}"; shift || true
if [ -n "$CFG_NAME" ]; then
  [ "$verb" != install ] || { echo "run.sh: install takes no --config (usage: run.sh install [--weights DIR])" >&2; exit 2; }
  CFG_FILE="$HERE/configs/$CFG_NAME.env"
  [ -f "$CFG_FILE" ] || { echo "run.sh: no such config: $CFG_NAME (see $HERE/configs/)" >&2; exit 2; }
fi
[ -f "$CFG_FILE" ] && source "$CFG_FILE"
export ATLASFOLD_KIT_CONFIG="$CFG_FILE"                   # the file in force, for the report (check prints config=<name>)
export PYTHONPATH="$HERE/opt:$HERE/stock/src/src:$HERE/../common/opt_core${PYTHONPATH:+:$PYTHONPATH}"
case "$verb" in
  install)
    python3 -m pip install --no-deps -e "$HERE/stock/src" >/dev/null 2>&1 || echo "[atlasfold-opt] install: editable install of stock/src skipped (read-only tree?) — PYTHONPATH route in use"
    python3 -m pip install --no-deps --no-build-isolation -e "$HERE/opt" >/dev/null 2>&1 || echo "[atlasfold-opt] install: editable install of opt/ skipped — PYTHONPATH route in use (the .pth drop-in route needs the install)"
    python3 "$HERE/stock/check_pins.py" ${ATLASFOLD_WEIGHTS_DIR:+--weights "$ATLASFOLD_WEIGHTS_DIR"} "$@" || exit 1     # exits non-zero only on a stock/src TREE digest mismatch; weights lines are information
    if [ -n "${ATLASFOLD_WEIGHTS_DIR:-}" ]; then python3 -m atlasfold_opt install --weights "$ATLASFOLD_WEIGHTS_DIR" || true; fi ;;
  check|warm|pred) exec python3 -m atlasfold_opt "$verb" "$@" ;;
  *) echo "usage: run.sh install|pred|check|warm ..." >&2; exit 2 ;;
esac
