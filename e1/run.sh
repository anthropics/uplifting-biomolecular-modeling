#!/bin/bash
# E1 optimized — single entry point: a thin wrapper over `python -m e1_opt` (the installed e1_opt package: `run.sh install`).
#   run.sh score  [--config h100] [--mode off|exact] [--variant V | --model-name Profluent-Bio/E1-V] [--det 0|1]
#                 --parent-path P --mutants-path M --output-path O            one assay, upstream's own flags; the tool's scores.csv where named
#                 [--max-batch-tokens N] [--scoring-method S]                 every other E1.tools.score option passes through verbatim
#                 [--context-path F] [--context-reduction R]
#   run.sh check  [--config h100] --variant V [--mode M]    dry run: resolves and gates the mode on this GPU; nothing is applied
#   run.sh warm   [--config h100] --variant V [--mode M]    one scoring of the package's fixture to fill the JIT caches
#   run.sh install [--weights DIR]                          the install step: this kit's package installed editable into the python on PATH, then the software
#                 pin check (stock/check_pins.py); --weights DIR also stages the three checkpoints and the hub RMSNorm kernel into DIR (an HF_HOME) with
#                 upstream's own downloaders and checks them against stock/PINS.json (sha256) — DIR is then your HF_HOME for every run
# --variant 150m|300m|600m (the kit's pins module, opt/forward/engines/e1/kits/pins.py WEIGHTS): one variant per process, never two.
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (offline switches, target GPU class; cache locations: STOCK.md "Variables").
# Mode = --mode when given, else E1_OPT from the environment, else the package default (exact); a --mode that disagrees with a set E1_OPT is
# refused as a usage error (rc 2, a NOT ACTIVE line); the same rule binds --variant and E1_VARIANT. Modes (off|exact) are resolved by the
# package's one mode table (opt/e1_opt/modes.py); this script validates none of them. `--mode off` is the stock route: the package's one
# stock runner (opt/e1_opt/stock_score.py — the upstream CLI, nothing from the kit on the path) in a clean subprocess that proves its
# environment. `--mode exact` runs the same upstream CLI with the kit armed: its one lever set applies at the tool's model construction.
# A route refuses (rc 3) only when it cannot be what was asked: the package not installed or the kit's files missing, the installed E1 not
# the pinned stock (commit / version), the weights file absent, no GPU, `--det 1` without the recipe's environment, or a lever of the set
# that cannot run in the tool's process (the kit's refusal by name — the mode is all of its levers, never a subset under its name). The
# host's environment — a dependency off its pin, an accelerator that does not import, a GPU outside the tested classes — is NAMED on the
# ACTIVE line (`notes="…"`) and the run proceeds (`stock/check_pins.py` is the user's own check of an environment against the pins;
# `install` runs its software checks). Exit codes: 0 ok, 1 failed (the tool, the install, a staged file off its pin), 2 usage (incl. a
# --mode / --variant disagreeing with E1_OPT / E1_VARIANT), 3 not active.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,13p' "$0" >&2; exit 2; }
CFG=""; CMD=""; MODE=""; VARIANT=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config)    [ $# -ge 2 ] || usage; CFG=$2; shift 2 ;;
    --config=*)  CFG=${1#--config=}; shift ;;
    --mode)      [ $# -ge 2 ] || usage; MODE=$2; shift 2 ;;
    --mode=*)    MODE=${1#--mode=}; shift ;;
    --variant)   [ $# -ge 2 ] || usage; VARIANT=$2; shift 2 ;;
    --variant=*) VARIANT=${1#--variant=}; shift ;;
    *) if [ -z "$CMD" ]; then CMD=$1; else ARGS+=("$1"); fi; shift ;;
  esac
done
case "$CMD" in score|check|warm|install) ;; *) usage ;; esac
echo "[e1-kit] Built with Profluent-E1 (Profluent Bio Inc.), used under the Profluent-E1 Clickthrough License Agreement: $HERE/stock/src/LICENSE (NOTICE and ATTRIBUTION beside it)" >&2   # attribution notice, printed once at the start of every verb
if [ "$CMD" = install ]; then                                       # the install step: everything below it presupposes the installed package
  WEIGHTS=""
  [ -z "$CFG" ] && [ -z "$MODE" ] && [ -z "$VARIANT" ] || { echo "run.sh: install takes no --config / --mode / --variant (usage: run.sh install [--weights DIR])" >&2; exit 2; }
  set -- ${ARGS[@]+"${ARGS[@]}"}
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
      --weights=*) WEIGHTS=${1#*=}; [ -n "$WEIGHTS" ] || { echo "run.sh: install --weights= takes a directory" >&2; exit 2; }; shift ;;
      *) echo "run.sh: install takes no argument '$1' (usage: run.sh install [--weights DIR])" >&2; exit 2 ;;
    esac
  done
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md Install)" >&2; exit 3; }
  if python -I -c "import os,sys,importlib.util as u; d=os.path.realpath(sys.argv[1]); x=u.find_spec('e1_opt'); sys.exit(0 if x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) else 1)" "$HERE/opt" 2>/dev/null; then
    echo "run.sh: e1_opt is installed from this tree already ($HERE/opt) — the pip step is skipped"   # the container image ships it installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the kit package is expected at $HERE/opt" >&2; exit 1; }   # the package alone: e1_opt imports nothing from the shared core (opt/pyproject.toml)
  fi
  python -I "$HERE/stock/check_pins.py" --checks package,stack || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the lines above — stock E1 or the stack is not installed as pinned in this environment)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then python -m e1_opt.weights "$WEIGHTS" || exit $?; fi   # upstream's downloaders into DIR, then the sha256 check against stock/PINS.json (opt/e1_opt/weights.py)
  exit 0
fi
PRE_E1_OPT=${E1_OPT:-}
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
# (placed before the config is sourced: configs/<card>.env derive TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR from MODEL_OPT_JIT_ROOT at that point)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[e1-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[e1-kit] jit cache: $J ($W)"; }
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 2) when e1_opt is not importable
fi
FROM="the environment"; { [ -n "$CFG" ] && [ "${E1_OPT:-}" != "$PRE_E1_OPT" ]; } && FROM="configs/$CFG.env"   # where the set E1_OPT came from
ENVMODE=${E1_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "[e1-opt] NOT ACTIVE: mode $MODE disagrees with E1_OPT=$ENVMODE from $FROM; a run has one mode — drop one of them"; exit 2
fi
MODE=${MODE:-$ENVMODE}
ENVVARIANT=${E1_VARIANT:-}
if [ -n "$VARIANT" ] && [ -n "$ENVVARIANT" ] && [ "$VARIANT" != "$ENVVARIANT" ]; then
  echo "[e1-opt] NOT ACTIVE: variant $VARIANT disagrees with E1_VARIANT=$ENVVARIANT from $FROM; one variant per process — drop one of them"; exit 2
fi
VARIANT=${VARIANT:-$ENVVARIANT}
case "$VARIANT" in ""|150m|300m|600m) ;; *) echo "run.sh: '$VARIANT' is not a variant (150m|300m|600m, the kit's pins module WEIGHTS)" >&2; exit 2 ;; esac
python -c "import e1_opt" 2>/dev/null || { echo "[e1-opt] NOT ACTIVE: e1_opt is not installed on $(command -v python || echo 'python (not on PATH)'): bash $HERE/run.sh install"; exit 3; }
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
VARARG=();  [ -n "$VARIANT" ] && VARARG=(--variant "$VARIANT")
exec python -m e1_opt "$CMD" ${VARARG[@]+"${VARARG[@]}"} ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
