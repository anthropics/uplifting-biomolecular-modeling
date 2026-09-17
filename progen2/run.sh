#!/usr/bin/env bash
# progen2/run.sh — the one entry point: a thin wrapper over `python -m progen2_opt <command> ...` (== `progen2-opt`; the installed package: `pip install -e opt`).
# The flags of `sample` / `score` are sample.py's / likelihood.py's own (names, types, defaults) plus the mode word; --model takes the stock's
# checkpoint names progen2-small|medium|oas|base|large|BFD90|xlarge (progen2-bfd90 accepted for progen2-BFD90).
#   run.sh install [--weights DIR [--model M]]                       the install step: this kit (package progen2_opt, no other dependencies) installed editable into
#                                                                     $PROGEN2_PYTHON (default `python` on PATH), then the pin check (stock/check_pins.py); with --weights
#                                                                     also fetches the checkpoints into DIR (all seven sizes, or the one --model names) and checks them
#                                                                     against the pins (README.md 'Setup')
#   run.sh check  [--mode exact|off] [--model M]                     dry run: pins + the activation resolved and gated (the card read from the device); nothing applied
#   run.sh sample [--mode exact|off] [--model progen2-large] [--device cuda:0] [--rng-seed 42] [--rng-deterministic true] [--p 0.95] [--t 0.2]
#                 [--max-length 256] [--num-samples 1] [--fp16 true] [--context 1] [--sanity true]
#                                                                     sample.py: mode off = the stock script in ONE clean process (its own output and exit code); exact =
#                                                                     in-process, the model loaded once with the generation kit's levers — stdout = the block the stock prints
#   run.sh score  [--mode exact|off] [--model progen2-base] [--device cuda:0] [--rng-seed 42] [--rng-deterministic true] [--fp16 true]
#                 [--context <likelihood.py's default>] [--sanity false]
#                                                                     likelihood.py: mode off = the stock script in ONE clean process; exact = the scoring kit in-process —
#                                                                     stdout = ll_sum= / ll_mean= (after likelihood.py's own sanity lines when --sanity true)
#     the one flag upstream lacks, on both: --input FILE --out_dir DIR   a multi-item job through ONE loaded model (the load paid once; exact holds the static K/V slots of the job's
#                                                                     batch sizes from the load): FILE = JSON lines keyed by the per-item flags' dests (sample: context,
#                                                                     max_length, num_samples, t, p, rng_seed; score: context; + item_id); DIR/items/<item_id>/block.txt
#                                                                     per item, nothing else
# The weights root ($PROGEN2_WEIGHTS), the interpreter ($PROGEN2_PYTHON, default `python`), the stock checkout ($PROGEN2_STOCK_DIR, default this tree's
# stock/src/progen2) and the run directory ($PROGEN2_RUN_DIR) come from the environment (STOCK.md 'Variables'); no environment variable selects a mode or
# a size, and the card is read from the device (nothing is configured per card class). Modes are exact (the default) | off; one composition per mode
# and a mode is all of its levers; every stock flag reaches the route as the stock types it under both (exact names a setting outside the tested
# defaults on its stack line and engages; --device cpu / no CUDA device, or a lever that cannot run on the box: refused by name, exit 3 — --mode off
# runs the stock). Past `install` this script validates nothing itself: every flag is the package's (`progen2-opt --help` prints the full surface and
# the exit codes); when the package is not importable on the chosen python it prints `NOT ACTIVE: kit package not installed` with the install line
# and exits 3. Exit: 0 ok · 1 failed (a run, an item, the install, a weight file off its pin) · 2 usage · 3 not active / refused by name / partial.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=("$@")
CMD="${ARGS[0]:-}"
case "$CMD" in
  install)                                                           # the install step: everything below it presupposes the installed package
    PY="${PROGEN2_PYTHON:-python}"; WEIGHTS=""; MODEL=()
    set -- "${ARGS[@]:1}"
    while [ $# -gt 0 ]; do
      case "$1" in
        --weights) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR [--model M]])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
        --weights=*) WEIGHTS=${1#*=}; [ -n "$WEIGHTS" ] || { echo "run.sh: install --weights= takes a directory" >&2; exit 2; }; shift ;;
        --model) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --model takes a checkpoint name (usage: run.sh install --weights DIR --model progen2-<size>)" >&2; exit 2; }; MODEL=(--model "$2"); shift 2 ;;
        --model=*) MODEL=(--model "${1#*=}"); shift ;;
        *) echo "run.sh: install takes no argument '$1' (usage: run.sh install [--weights DIR [--model M]])" >&2; exit 2 ;;
      esac
    done
    [ -z "${MODEL[*]:-}" ] || [ -n "$WEIGHTS" ] || { echo "run.sh: install --model goes with --weights DIR (the size whose checkpoint to fetch)" >&2; exit 2; }
    command -v "$PY" >/dev/null || { echo "run.sh: no such interpreter: PROGEN2_PYTHON=$PY — set PROGEN2_PYTHON to the environment this kit installs into, or put its python on PATH (README.md Install)" >&2; exit 3; }
    if "$PY" -I -c "import os,sys,importlib.util as u; d=os.path.realpath(sys.argv[1]); x=u.find_spec('progen2_opt'); sys.exit(0 if x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) else 1)" "$HERE/opt" 2>/dev/null; then
      echo "run.sh: progen2_opt is installed from this tree already ($HERE/opt) — the pip step is skipped"   # the container image ships it installed; a read-only image cannot re-run pip
    else
      "$PY" -m pip install -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the kit package is expected at $HERE/opt" >&2; exit 1; }
    fi
    "$PY" -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the line above — the pinned stack or the stock files are not as pinned in this environment)" >&2; exit 3; }
    if [ -n "$WEIGHTS" ]; then "$PY" -m progen2_opt.weights "$WEIGHTS" ${MODEL[@]+"${MODEL[@]}"} || exit $?; fi   # the checkpoints into DIR the way upstream's README fetches them, then the sha256 check against stock/PINS.json (opt/progen2_opt/weights.py)
    exit 0 ;;
esac
export PYTHONDONTWRITEBYTECODE=1                                    # no bytecode caches written into the tree or the stock checkout
PY="${PROGEN2_PYTHON:-python}"
if ! "$PY" -c "import progen2_opt" 2>/dev/null; then   # nothing can activate: NOT ACTIVE by name (rc 3); rc 2 stays the usage code
  echo "[progen2-opt] NOT ACTIVE: kit package not installed on this python ($PY) — install it with: $PY -m pip install -e $HERE/opt" >&2
  exit 3
fi
exec "$PY" -m progen2_opt "${ARGS[@]}"
