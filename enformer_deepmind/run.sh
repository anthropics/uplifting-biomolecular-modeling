#!/bin/bash
# Enformer (official TensorFlow implementation, TF-Hub SavedModel) exact kit — entry point. Two verbs; everything else is your own code with the kit
# switched on (ENFORMER_DEEPMIND_OPT=exact python your_script.py, or `import enformer_deepmind_opt; enformer_deepmind_opt.enable()` — README.md):
#   run.sh install        pip-install this kit (editable, opt/) into the python on PATH, then list any difference between the installed stack and
#                         the pinned one (stock/PINS.json; named, informational) — with TFHUB_CACHE_DIR set, also check the cached SavedModel's digests
#   run.sh check [--json] the dry run: resolve the kit on this box (CUDA device by TensorFlow, kit files, the levers' builds for the device) and print
#                         the one line — `[enformer-deepmind-opt] DRY …` exit 0, or `… NOT ACTIVE … reason=…` exit 3; nothing is applied
# PYTHON=<interpreter> selects the python (default: python). Exit codes: 0 ok · 2 usage · 3 NOT ACTIVE / a pin refused · 4 the stack differs from the pinned one (install; named, informational)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; }
[ $# -ge 1 ] || { usage; exit 2; }
CMD="$1"; shift
case "$CMD" in
  install)
    [ $# -eq 0 ] || { echo "[enformer-deepmind-opt] run.sh: install takes no arguments" >&2; usage >&2; exit 2; }
    "$PY" -m pip install --no-deps --no-build-isolation -e "$HERE/opt"                        # the kit, editable (its files are resolved from the package's own location)
    rc=0; "$PY" -I "$HERE/stock/check_pins.py" --stack || rc=$?                                # the pinned stack: differences are listed by name (exit 4) — informational, the kit names a differing tensorflow on its own line
    [ "$rc" -eq 0 ] || [ "$rc" -eq 4 ] || exit "$rc"
    if [ -n "${TFHUB_CACHE_DIR:-}" ] && [ -d "$TFHUB_CACHE_DIR" ]; then "$PY" -I "$HERE/stock/check_pins.py" --quiet --weights "$TFHUB_CACHE_DIR" || exit $?; fi   # the cached model is the released one, else exit 3 by name
    "$PY" -c "import enformer_deepmind_opt as k; print('[enformer-deepmind-opt] installed enformer_deepmind_opt', k.__version__, 'from', k.__file__)"
    ;;
  check)
    exec "$PY" -m enformer_deepmind_opt check "$@"
    ;;
  -h|--help|help) usage ;;
  *) echo "[enformer-deepmind-opt] run.sh: unknown command '$CMD' (install | check)" >&2; usage >&2; exit 2 ;;
esac
