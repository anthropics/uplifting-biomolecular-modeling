#!/bin/bash
# mosaic optimized — single entry point: a thin wrapper over `python -m mosaic_opt` (the installed mosaic_opt package on the shared core; `run.sh install` installs both).
#   run.sh install [--weights DIR]                        the install step: the shared core and this kit installed editable into the python on PATH, the pin check (stock/check_pins.py),
#                                                         then the kit's lever files placed inside the installed mosaic package; --weights DIR also fetches the Boltz-2 checkpoint + CCD
#                                                         molecules into DIR with boltz's own downloader and checks the checkpoint against stock/PINS.json (DIR = MOSAIC_CACHE_DIR at run time)
#   run.sh design [--config h100] [--mode M] --out <dir> [--seed S] [--binder-length L] [--target-copies N] [--target-fasta F [--first-record] [--msa A]] [--epitope 12,15,40-48] [--tag T] [--steps1 N] [--steps2 N] [--det 0|1]
#                                                         one binder design through the kit driver in a fresh process; the driver's file set (inputs: F = ONE record, refused rc 2 unless --first-record takes record 1; --epitope = 1-based target residue positions the contact loss keeps, every mode)
#   run.sh check  [--config h100] [--mode M] [--binder-length L] [--target-copies N] [--json]   dry run: the row, the levers, the shape's cache state, the GPU; nothing is applied
#   run.sh warm   [--config h100] [--mode exact] [--binder-length L] [--target-copies N] [--seed S]   stage the weights, freeze the shape's features, populate its P1 files
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (weights cache, P1 files root, target GPU). Mode = --mode when given, else
# MOSAIC_OPT from the environment, else the package default (fast); a --mode that disagrees with a set MOSAIC_OPT is refused. Modes
# (fast|exact|big|off) are resolved by the package from the kit's activation rows (opt/mosaic_opt/modes.py ROWS); this script validates none
# of them. `--mode off` is the stock route: the package's one stock caller (opt/mosaic_opt/stock_design.py — the kit driver with every lever off,
# nothing else from the kit on the path) in a clean subprocess that proves its environment; `design` and `check` are its only commands (the
# stock arm has no warm of its own). Every route refuses (rc 3) unless the package is importable, its core pin gate passes (`mosaic_opt.core_gate`,
# first on every route: an absent or older shared core is one `[mosaic-opt] NOT ACTIVE: reason=...` line, a newer core passes) and the three pinned upstream packages are
# installed at their pinned commits (stock/check_pins.py). Exit codes: 0 ok, 1 failed (a design, the install, a weights file off its pin), 2 usage, 3 not active / pins not met, or partial (`--allow-partial` records and proceeds).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,17p' "$0" >&2; exit 2; }
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
case "$CMD" in design|check|warm|install) ;; *) usage ;; esac
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
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('mosaic_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: mosaic_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the line above — mosaic, joltz or boltz is not installed at its pinned commit in this environment)" >&2; exit 3; }
  python -m mosaic_opt.leverfiles || exit $?      # the kit's lever files into the installed mosaic package (site-packages/mosaic/fast/), add-only: fast and big run from that copy (opt/mosaic_opt/leverfiles.py)
  if [ -n "$WEIGHTS" ]; then python -m mosaic_opt.weights "$WEIGHTS" || exit $?; fi   # boltz's downloader into DIR, then the sha256 check against stock/PINS.json (opt/mosaic_opt/weights.py)
  exit 0
fi
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 3) when mosaic_opt is not importable, its core pin gate refuses, or the P1 key cannot be derived
fi
JIT_ROOT_PRESET="${MODEL_OPT_JIT_ROOT:-}"   # before the block: the root configs/<card>.env may already have derived MOSAIC_OPT_CACHE_ROOT from; the block can move a read-only one
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[mosaic-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[mosaic-kit] jit cache: $J ($W)"; }
[ -z "$CFG" ] || [ -z "${MODEL_OPT_JIT_ROOT:-}" ] || [ -z "${MODEL_OPT_STACK_KEY:-}" ] || { [ -n "${MOSAIC_OPT_CACHE_ROOT:-}" ] && [ "${MOSAIC_OPT_CACHE_ROOT}" != "$JIT_ROOT_PRESET/$MODEL_OPT_STACK_KEY/mosaic" ]; } || [ "${MOSAIC_OPT_CACHE_ROOT:-}" = "$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/mosaic" ] || export MOSAIC_OPT_CACHE_ROOT="$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/mosaic"   # after the block: the kit reads MOSAIC_OPT_CACHE_ROOT — derived from the root the block chose (or moved) exactly as configs/<card>.env derives it; a value you set yourself is kept
FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
ENVMODE=${MOSAIC_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with MOSAIC_OPT=$ENVMODE from $FROM; a run has one mode — drop one of them" >&2; exit 2
fi
MODE=${MODE:-$ENVMODE}
MOSAIC_OPT= python -c "import mosaic_opt" 2>/dev/null || { echo "run.sh: mosaic_opt is not importable on $(command -v python || echo 'python (not on PATH)'): bash $HERE/run.sh install (README.md Install)" >&2; exit 3; }   # importability alone: the variable is neutralised for this probe (an installed .pth gates at interpreter start under MOSAIC_OPT; the gate speaks once, on the next line)
python -c "from mosaic_opt import core_gate; core_gate()" >/dev/null || exit 3     # the core pin gate, first on every route (the stock route included): an absent / older / newer / edited shared core is its one NOT ACTIVE line, rc 3
python -I "$HERE/stock/check_pins.py" --quiet || { echo "run.sh: the pinned upstream packages are not installed at their pinned commits (stock/PINS.json, STOCK.md)" >&2; exit 3; }
if [ "$MODE" = off ]; then                                        # stock route: `design --mode off` = the package's stock caller in a clean subprocess
  [ "$CMD" = design ] || [ "$CMD" = check ] || { echo "run.sh: --mode off has two commands, design and check (the stock arm has no warm of its own)" >&2; exit 2; }
fi
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m mosaic_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
