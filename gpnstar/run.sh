#!/bin/bash
# GPN-Star exact kit — entry point. Two verbs; everything else is upstream's own command line or your own code with the kit switched on
# (GPNSTAR_OPT=exact gpn star vep …, or `import gpnstar_opt; gpnstar_opt.enable()` — README.md):
#   run.sh install [--weights DIR [--model KEY|REPO]]
#                         pip-install the shared core (../common/opt_core) and this kit (opt/, editable) into the python on PATH, then check the
#                         install against stock/PINS.json (stock/check_pins.py: gpn at the pinned commit, transformers at upstream's pin; exit 3 = a pin
#                         does not hold, named). --weights DIR also stages the pinned checkpoint snapshot (default: the primary model) into DIR (an
#                         HF_HOME) with the hub library's own downloader and checks every file against stock/PINS.json (sha256) — DIR is then your HF_HOME
#   run.sh check          the dry run: resolve the kit on this machine (CUDA device, the gpn pin, transformers' pin, the primary checkpoint's digests when
#                         staged) and print the one line — `[gpnstar-opt] DRY-RUN … would_refuse=none` exit 0, or `… would_refuse=<reason>` exit 3; nothing is applied
# PYTHON=<interpreter> selects the python (default: python). Exit codes: 0 ok · 1 the install failed · 2 usage · 3 NOT ACTIVE / a pin refused
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; }
install_verb() {   # run.sh install [--weights DIR [--model KEY|REPO]]
  local WEIGHTS="" WMODEL=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "[gpnstar-opt] run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR [--model KEY|REPO]])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
      --weights=*) WEIGHTS=${1#*=}; [ -n "$WEIGHTS" ] || { echo "[gpnstar-opt] run.sh: install --weights= takes a directory" >&2; exit 2; }; shift ;;
      --model) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "[gpnstar-opt] run.sh: install --model takes a model key or repository id (stock/PINS.json weights)" >&2; exit 2; }; WMODEL=$2; shift 2 ;;
      --model=*) WMODEL=${1#*=}; [ -n "$WMODEL" ] || { echo "[gpnstar-opt] run.sh: install --model= takes a model key or repository id" >&2; exit 2; }; shift ;;
      *) echo "[gpnstar-opt] run.sh: install takes no argument '$1' (usage: run.sh install [--weights DIR [--model KEY|REPO]])" >&2; exit 2 ;;
    esac
  done
  [ -z "$WMODEL" ] || [ -n "$WEIGHTS" ] || { echo "[gpnstar-opt] run.sh: install --model names what --weights DIR stages: give --weights DIR too" >&2; exit 2; }
  command -v "$PY" >/dev/null || { echo "[gpnstar-opt] run.sh: no $PY on PATH — activate the environment this kit installs into (README.md Setup)" >&2; exit 3; }
  local CORE="$HERE/../common/opt_core"
  if "$PY" -I -c "import os,sys,importlib.util as u; d=os.path.realpath(sys.argv[1]); x=u.find_spec('gpnstar_opt'); sys.exit(0 if x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) else 1)" "$HERE/opt" 2>/dev/null; then
    echo "[gpnstar-opt] run.sh: gpnstar_opt is installed from this tree already ($HERE/opt) — the pip step is skipped"   # the container image ships it installed; a read-only image cannot re-run pip
  else
    [ -f "$CORE/pyproject.toml" ] || { echo "[gpnstar-opt] run.sh: the core package is expected at $CORE (the release tree's common/opt_core, installed beside this kit) — not found" >&2; exit 1; }
    "$PY" -m pip install -e "$CORE" -e "$HERE/opt" || { echo "[gpnstar-opt] run.sh: the install failed (pip's words above): the core is expected at $CORE and the kit package at $HERE/opt" >&2; exit 1; }
  fi
  "$PY" -I "$HERE/stock/check_pins.py" || { echo "[gpnstar-opt] run.sh: installed, but refused by the pin check (stock/check_pins.py: the lines above — the installed gpn is not the pinned stock, or transformers is off upstream's pin)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then local WM=(); [ -n "$WMODEL" ] && WM=(--model "$WMODEL"); "$PY" -m gpnstar_opt.weights "$WEIGHTS" ${WM[@]+"${WM[@]}"} || exit $?; fi   # the hub downloader into DIR, then the sha256 check against stock/PINS.json (opt/gpnstar_opt/weights.py)
  "$PY" -c "import gpnstar_opt; print('[gpnstar-opt] installed gpnstar_opt', gpnstar_opt.__version__, 'from', gpnstar_opt.__file__)"
}
[ $# -ge 1 ] || { usage >&2; exit 2; }
CMD="$1"; shift
case "$CMD" in
  install) install_verb "$@" ;;
  check)
    [ $# -eq 0 ] || { echo "[gpnstar-opt] run.sh: check takes no arguments" >&2; usage >&2; exit 2; }
    "$PY" -c "import gpnstar_opt" 2>/dev/null || { echo "[gpnstar-opt] NOT ACTIVE: gpnstar_opt is not installed on $(command -v "$PY" || echo "$PY (not on PATH)"): bash $HERE/run.sh install" >&2; exit 3; }
    exec "$PY" -m gpnstar_opt check
    ;;
  -h|--help|help) usage ;;
  *) echo "[gpnstar-opt] run.sh: unknown command '$CMD' (install | check)" >&2; usage >&2; exit 2 ;;
esac
