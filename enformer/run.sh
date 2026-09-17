#!/bin/bash
# Enformer exact kit — entry point. Two verbs; everything else is your own enformer_pytorch code with the kit switched on
# (ENFORMER_OPT=exact python your_script.py, or `import enformer_opt; enformer_opt.enable()` — README.md):
#   run.sh install        pip-install the pinned stock (stock/enformer_pytorch-0.8.12 wheel) and this kit (editable, opt/) into the python on PATH,
#                         then check the install against stock/PINS.json (exit 3/4 = a pin does not hold, named)
#   run.sh check [--json] the dry run: resolve the kit on this box (CUDA device, kernel build for its architecture, kit files, upstream pin) and
#                         print the one line — `[enformer-opt] DRY …` exit 0, or `[enformer-opt] NOT ACTIVE … reason=…` exit 3; nothing is applied
# PYTHON=<interpreter> selects the python (default: python). Exit codes: 0 ok · 2 usage · 3 NOT ACTIVE / a pin refused · 4 the stack differs from the pinned one (install; named, informational)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; }
[ $# -ge 1 ] || { usage; exit 2; }
CMD="$1"; shift
case "$CMD" in
  install)
    [ $# -eq 0 ] || { echo "[enformer-opt] run.sh: install takes no arguments" >&2; usage >&2; exit 2; }
    "$PY" -m pip install --no-deps "$HERE/stock/enformer_pytorch-0.8.12-py3-none-any.whl"      # the stock, from the tree's own copy of the PyPI wheel (its dependencies are the pinned stack's: environment/requirements.lock)
    "$PY" -m pip install --no-deps --no-build-isolation -e "$HERE/opt"                        # the kit, editable (the vendored kernels are resolved from the package's own location)
    "$PY" -I "$HERE/stock/check_pins.py" || exit $?                                           # the stock pin holds (version + files), else exit 3 by name
    rc=0; "$PY" -I "$HERE/stock/check_pins.py" --quiet --stack kit || rc=$?                   # the pinned stack: differences are listed by name (exit 4) — informational, the kit names a differing torch on its own line
    [ "$rc" -eq 0 ] || [ "$rc" -eq 4 ] || exit "$rc"
    "$PY" -c "import enformer_opt; print('[enformer-opt] installed enformer_opt', enformer_opt.__version__, 'from', enformer_opt.__file__)"
    ;;
  check)
    exec "$PY" -m enformer_opt check "$@"
    ;;
  -h|--help|help) usage ;;
  *) echo "[enformer-opt] run.sh: unknown command '$CMD' (install | check)" >&2; usage >&2; exit 2 ;;
esac
