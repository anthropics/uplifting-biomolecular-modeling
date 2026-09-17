#!/bin/bash
# Borzoi (TensorFlow, calico/borzoi) optimized — single entry point: a thin wrapper over `python -m borzoi_opt` (the installed borzoi_opt package: `pip install -e opt`).
#   run.sh sad    [--config h100|a100|h200] [--mode M] [--det 0|1] [--allow-partial] <the stock borzoi_sad.py arguments>   the documented command (VCF -> <out_dir>/sad.h5) in mode M; stock's arguments pass through unread
#   run.sh check  [--config h100|a100|h200] [--mode M] [--det 0|1]                        dry run: resolves and gates M on this machine; nothing is applied
#   run.sh install [--weights DIR]                                                        the install step: this kit installed editable into the python on PATH (it needs no other package of the bundle), then the pin check (stock/check_pins.py: borzoi and baskerville at their pins); --weights DIR also fetches the fold-0 weights to DIR/f0/model0_best.h5 from the URL in stock/PINS.json and checks its sha256 — that path is then sad's <model_file> argument
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (the target GPU; no path — README.md "Variables"). Mode = --mode
# when given, else BORZOI_OPT from the environment, else the package default (exact); a --mode that disagrees with a set BORZOI_OPT is refused.
# Modes (exact|off) are resolved by the package against the kit's own files (opt/datapath/pipeline_tf/v17/); this
# script validates none of them. `--mode off` is the stock route: the package's one stock caller (opt/borzoi_opt/stock_sad.py — the stock
# script, nothing from the kit on the path) in a clean subprocess that proves its environment; `sad` and `check` are its commands.
# `--det 1` (sad, check) exports the reproducibility recipe into the job. Every route refuses (rc 3) unless the package is
# installed on the `python` on PATH. Exit codes: 0 ok (the job's own), 1 failed, 2 usage, 3 not active or partial (a lever of the mode could not engage — options outside the chunked post's set run the stock post per call, named ROUTED, exit unaffected —, by the kit's stamp; `--allow-partial` records and proceeds).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,12p' "$0" >&2; exit 2; }
CFG=""; CMD=""; MODE=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config)    [ $# -ge 2 ] || usage; CFG=$2; shift 2 ;;
    --config=*)  CFG=${1#--config=}; shift ;;
    --mode)      [ -z "$CMD" ] || [ ${#ARGS[@]} -eq 0 ] || usage; [ $# -ge 2 ] || usage; MODE=$2; shift 2 ;;
    --mode=*)    MODE=${1#--mode=}; shift ;;
    *) if [ -z "$CMD" ]; then CMD=$1; else ARGS+=("$1"); fi; shift ;;
  esac
done
case "$CMD" in sad|check|install) ;; *) usage ;; esac
if [ "$CMD" = install ]; then   # the install step: nothing of the run-time routes below applies (no config, no mode)
  [ -z "$CFG" ] && [ -z "$MODE" ] || { echo "run.sh: install takes no --config / --mode (it installs into the python on PATH; a mode is chosen per run)" >&2; exit 2; }
  WDIR=""; set -- ${ARGS[@]+"${ARGS[@]}"}
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights)   [ $# -ge 2 ] && [ -n "$2" ] || usage; WDIR=$2; shift 2 ;;
      --weights=*) WDIR=${1#--weights=}; [ -n "$WDIR" ] || usage; shift ;;
      *) echo "run.sh: install: unexpected argument $1 (usage: run.sh install [--weights DIR])" >&2; exit 2 ;;
    esac
  done
  command -v python > /dev/null || { echo "run.sh: no python on PATH — activate the environment that holds the stack of stock/PINS.json (README.md 'Install')" >&2; exit 3; }
  if python -I -c "import os,sys,borzoi_opt; sys.exit(0 if os.path.realpath(borzoi_opt.__file__).startswith(os.path.realpath('$HERE/opt')+os.sep) else 1)" 2>/dev/null; then
    echo "[borzoi-opt install] this kit ($HERE/opt) is installed from this tree already — pip step skipped"
  else
    echo "[borzoi-opt install] python = $(command -v python); installing this kit ($HERE/opt), editable — it depends on no other package of the bundle"
    python -m pip install --no-deps --no-build-isolation -e "$HERE/opt" || { echo "run.sh: pip install failed (exit $?)" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || exit $?           # 3 = borzoi / baskerville / the stock entry are not the pinned ones — refused by name above
  if [ -n "$WDIR" ]; then python -m borzoi_opt.weights "$WDIR" || exit $?; fi
  echo "[borzoi-opt install] done: bash run.sh check --config h100 shows what engages on this machine${WDIR:+; the <model_file> argument of sad is then $WDIR/f0/model0_best.h5}"
  exit 0
fi
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 2) when borzoi_opt is not importable
fi
FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
ENVMODE=${BORZOI_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with BORZOI_OPT=$ENVMODE from $FROM; one run has one mode — drop one of them" >&2; exit 2
fi
MODE=${MODE:-$ENVMODE}
python -c "import borzoi_opt" 2>/dev/null || { echo "run.sh: borzoi_opt is not installed on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/opt" >&2; exit 3; }
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m borzoi_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
