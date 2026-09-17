#!/bin/bash
# ColabFold optimized — single entry point: a thin wrapper over `python -m colabfold_opt` (the installed colabfold_opt package: `pip install -e ../common/opt_core -e opt`).
#   run.sh pred   [--config h100] [--mode M] [--n_gpu P] <input> <results> [colabfold_batch options] [--det 0|1]
#                                                          colabfold_batch's own command line (input: a3m|fasta|csv|dir; its options verbatim); colabfold's own file set in <results>/
#   run.sh check  [--config h100] [--mode M] [--n_gpu P] [--json]                 dry run: resolves and gates the mode (and the GPU count) on this machine; nothing is applied
#   run.sh warm   [--config h100] [--mode M] [--out DIR]                            one public-input prediction through pred: the route end to end
#   run.sh install [--weights DIR]                                                  the install step: the shared core and this kit installed editable into the python on PATH, then the pin check (stock/check_pins.py --checks packages,files);
#                                                                                   --weights DIR also fetches the AlphaFold2-Multimer v3 parameters into DIR with upstream's own downloader and checks the five files against stock/PINS.json (sha256) — DIR is then your COLABFOLD_OPT_DATA_DIR
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (the target GPU, the compile-cache root; it requires COLABFOLD_OPT_DATA_DIR, the parameters root — STOCK.md 'Variables'). Mode = --mode when given, else
# COLABFOLD_OPT from the environment, else the package default (opt/colabfold_opt/modes.py DEFAULT_MODE, read from the package here: `fast`);
# --mode and a set COLABFOLD_OPT must agree. Modes: off | exact | fast | big [--n_gpu P] (opt/colabfold_opt/modes.py TABLE). Exit: 0 ok · 1 failed (or outputs
# short of the request: missing_outputs; the install, a parameter file off its pin) · 2 usage · 3 not active or partial (the stock pin, the kit's installation, no GPU, a refused
# switch or --n_gpu, a lever of the mode that cannot run on this GPU or engaged no call — a mode is all of its levers; --mode off runs stock).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
usage() { sed -n '2,13p' "$0" >&2; exit 2; }
CMD=""; CFG=""; MODE=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CFG=$2; shift 2 ;;
    --config=*) CFG=${1#*=}; shift ;;
    --mode) MODE=$2; shift 2 ;;
    --mode=*) MODE=${1#*=}; shift ;;
    -h|--help) usage ;;
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
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md Install)" >&2; exit 3; }
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('colabfold_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: colabfold_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" --checks packages,files || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the lines above — the pinned upstream is not installed as pinned in this environment)" >&2; exit 3; }   # the software pins; the parameters are --weights' and every route's check
  if [ -n "$WEIGHTS" ]; then python -m colabfold_opt.weights "$WEIGHTS" || exit $?; fi   # upstream's downloader into DIR, then the sha256 check against stock/PINS.json (opt/colabfold_opt/weights.py)
  exit 0
fi
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 2) when colabfold_opt is not importable
fi
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[colabfold-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[colabfold-kit] jit cache: $J ($W)"; }
[ -z "${MODEL_OPT_JIT_ROOT:-}" ] || export COLABFOLD_OPT_JIT_ROOT="${COLABFOLD_OPT_JIT_ROOT:-$MODEL_OPT_JIT_ROOT}"                # after the block: the kit's own cache-root variable follows the seeded root when nothing set it
# a shipped compile cache (image routes): jax reads and extends the tree of THIS configuration's stack key under the root the block chose —
# /opt/jit_cache in place when writable (docker), the seeded writable copy when not (read-only SIF); the image never links one card's tree at build
# time. Another JAX_COMPILATION_CACHE_DIR (yours), no shipped cache, or no config: nothing changes.
if [ "${JAX_COMPILATION_CACHE_DIR:-}" = /root/.cache/jax ] && [ -n "${MODEL_OPT_STACK_KEY:-}" ] && [ -n "${MODEL_OPT_JIT_ROOT:-}" ]; then
  case "$W" in in-image|"seeded from image"|"seeded from read-only root"|user) export JAX_COMPILATION_CACHE_DIR="$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/default/jax";; esac
fi
ENVMODE=${COLABFOLD_OPT:-}; FROM="the environment"
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with COLABFOLD_OPT=$ENVMODE from $FROM; a run has one mode — drop one of them" >&2; exit 2
fi
CLIMODE=$MODE; MODE=${MODE:-$ENVMODE}
PROBE_ERR=$(python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('colabfold_opt') else 4)" 2>&1 >/dev/null); PROBE_RC=$?; case $PROBE_RC in 0) ;; 4|126|127) echo "run.sh: colabfold_opt is not installed on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/../common/opt_core -e $HERE/opt" >&2; exit 3 ;; *) [ -n "$PROBE_ERR" ] && printf '%s\n' "$PROBE_ERR" >&2; exit "$PROBE_RC" ;; esac   # located, not imported: 'not installed' only when find_spec finds no colabfold_opt (or no python); any other rc is the interpreter's own start-up refusal (the kit's autoload .pth under COLABFOLD_OPT=<mode>: core pin gate / hook not importable) — its words and its rc, verbatim
python -c "import sys, colabfold_opt._autoload as a
a.gate_core()                                   # the core pin gate (the package's copy of the shared core's kit template): an absent or another opt_core is NOT ACTIVE (reason=core_missing:opt_core / core_mismatch), rc 3
try:
    from colabfold_opt import stack             # a module of the pinned core this package imports missing: NOT ACTIVE core_missing:<module>, rc 3 — never a traceback
except ImportError as e:
    sys.exit(a.core_missing(e))" || exit $?
if [ -z "$CLIMODE" ] && [ -n "$ENVMODE" ] && [ "$ENVMODE" != off ]; then   # the ENVIRONMENT ROUTE (COLABFOLD_OPT names a kit mode, no --mode): the .pth's hook must be LIVE in a fresh interpreter
  python -c "import sys; from colabfold_opt import report, stack; r, d = stack.autoload_pth_check(); r and (sys.stderr.write(report.NOT_ACTIVE_FMT.format(prefix=report.PREFIX, reason=r[0]) + chr(10)), sys.exit(report.EXIT_NOT_ACTIVE))" || exit $?
fi                                                                 # (--mode <kit mode> is un-gated: the launcher sets the mode for the model process itself, and the manifest verdict refuses a launch without its activation report; --mode off / COLABFOLD_OPT=off need no hook)
if [ -z "$MODE" ]; then                                           # no --mode, no COLABFOLD_OPT: the package default, read from its one mode table
  MODE=$(python -c "from colabfold_opt import modes; print(modes.DEFAULT_MODE)") || { echo "run.sh: the package default mode could not be read" >&2; exit 3; }
  echo "run.sh: no --mode and no COLABFOLD_OPT: the package default mode, $MODE (opt/colabfold_opt/modes.py DEFAULT_MODE)" >&2
fi
case "$MODE" in
  off|exact|fast|big) ;;
  *) echo "run.sh: '$MODE' is not a mode (off|exact|fast|big, opt/colabfold_opt/modes.py)" >&2; exit 2 ;;
esac
python -I "$HERE/stock/check_pins.py" --quiet || { echo "run.sh: the pinned upstream is not installed as pinned (stock/PINS.json, STOCK.md)" >&2; exit 3; }
exec python -m colabfold_opt "$CMD" --mode "$MODE" ${ARGS[@]+"${ARGS[@]}"}
