#!/usr/bin/env bash
# pxdesign/run.sh — the one launcher:
#   run.sh install [--weights DIR]                                       the install step: the shared core and this kit installed editable into the python on
#                                                                        PATH, then the pin check of the three stock packages (stock/check_pins.py); with --weights,
#                                                                        the seven weight and CCD-cache files fetched into DIR/checkpoint and DIR/ccd_cache by
#                                                                        upstream's downloader and checked against stock/PINS.json (README.md Setup)
#   run.sh design|check|warm [--config h100] [--mode exact|fast|big|off] ...   the design verb, the activation dry run, the cache warm
# Every verb but install gates the core pin first (`pxdesign_opt.core_gate()`: rc 3 with one `[pxdesign-opt] NOT ACTIVE: reason=...` line when the importable
# opt_core is not the one opt/pyproject.toml pins, on every route), sources configs/<config>.env (deployment parameters only), then runs `python -m pxdesign_opt <verb> ...`.
# The stock pins (stock/PINS.json: the installed upstream packages against the tree's copy) are checked once by `install` and reported by `check`
# (its PINS line); this launcher never runs that check on the run routes — the package refuses a kit mode at activation when the installed
# upstream is not the pinned stock ([pxdesign-opt] NOT ACTIVE: ..., rc 3), and `off` runs whatever checkout upstream itself would run.
# The mode comes from --mode or PXDESIGN_OPT (they must agree) and is forwarded to every verb; the default is the package's (modes.DEFAULT_MODE:
# fast, the tier-2 line; exact and big by name). Mode off = the stock caller in a clean subprocess (design, warm) or the stock dry run
# (check; plus the stock pins report). Exit codes are the package's (0 ok, 1 failed / outputs incomplete / the install or a weight file off its pin,
# 2 usage, 3 not active / stock environment not clean / the pinned upstream not installed as pinned).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
usage() { echo "usage: $0 install [--weights DIR] | $0 design|check|warm [--config h100] [--mode exact|fast|big|off] [args...]" >&2; exit 2; }
[ $# -ge 1 ] || usage
VERB=$1; shift
case "$VERB" in design|check|warm|install) ;; *) usage ;; esac
if [ "$VERB" = install ]; then                                       # the install step: everything below it presupposes the installed package
  WEIGHTS=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
      --weights=*) WEIGHTS=${1#*=}; [ -n "$WEIGHTS" ] || { echo "run.sh: install --weights= takes a directory" >&2; exit 2; }; shift ;;
      *) echo "run.sh: install takes no argument '$1' (usage: run.sh install [--weights DIR])" >&2; exit 2 ;;
    esac
  done
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md Install)" >&2; exit 3; }
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('pxdesign_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: pxdesign_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the lines above — a pinned upstream package is not installed as pinned in this environment; STOCK.md)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then python -m pxdesign_opt.weights "$WEIGHTS" || exit $?; fi   # upstream's downloader into DIR/checkpoint and DIR/ccd_cache, then the sha256 check against stock/PINS.json (opt/pxdesign_opt/weights.py)
  exit 0
fi
CONFIG=h100; MODE=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG=$2; shift 2 ;;
    --config=*) CONFIG=${1#--config=}; shift ;;
    --mode) MODE=$2; ARGS+=("--mode" "$2"); shift 2 ;;
    --mode=*) MODE=${1#--mode=}; ARGS+=("--mode" "$MODE"); shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [ -n "$MODE" ]; then
  case "$MODE" in off|exact|fast|big) ;; *) echo "[pxdesign-opt] unknown --mode $MODE (modes: off|exact|fast|big)" >&2; exit 2 ;; esac
  if [ -n "${PXDESIGN_OPT:-}" ] && [ "${PXDESIGN_OPT}" != "$MODE" ]; then echo "[pxdesign-opt] --mode $MODE disagrees with PXDESIGN_OPT=$PXDESIGN_OPT" >&2; exit 2; fi
fi
PXDESIGN_OPT= python -c "import pxdesign_opt" 2>/dev/null || { echo "run.sh: pxdesign_opt is not importable on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/../common/opt_core -e $HERE/opt" >&2; exit 3; }   # importability only: the variable is neutralised here so the installed .pth stays inert on this silenced probe (its gate speaks on the next line)
python -c "import pxdesign_opt as k; k.core_gate()" >/dev/null || exit 3          # the core pin gate, the first gate of every route (mode off included): [pxdesign-opt] NOT ACTIVE: reason=... on an absent or mismatched opt_core
CFG="$HERE/configs/$CONFIG.env"
[ -f "$CFG" ] || { echo "[pxdesign-opt] no configuration $CFG (configs/: $(ls "$HERE/configs" | tr '\n' ' '))" >&2; exit 2; }
# shellcheck disable=SC1090
source "$CFG"
exec python -m pxdesign_opt "$VERB" "${ARGS[@]}"
