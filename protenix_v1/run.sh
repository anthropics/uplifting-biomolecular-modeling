#!/bin/bash
# Protenix v1 optimized — single entry point: a thin wrapper over `python -m protenix_v1_opt` (the installed protenix_v1_opt package: `run.sh install`).
#   run.sh install [--weights DIR] [pip options]                                              the install step: pip install -e ../common/opt_core -e opt (the shared core, then this package) into the `python` on PATH (further options go to pip), then the pin check (stock/check_pins.py); --weights DIR then fetches the checkpoint + data files into DIR (= PROTENIX_ROOT_DIR) with upstream's downloader and checks them (opt/protenix_v1_opt/weights.py)
#   run.sh probe                                                                               the package probe (configs/*.env run it): rc 0 importable; rc 2 `not installed` (absent only); else python's own refusal and code
#   run.sh pred   [--config h100] [--mode M] [--det 0|1] [--allow-partial] <stock `protenix pred` arguments>   one process: the stock CLI with the mode's levers on its runner
#   run.sh check  [--config h100] [--mode M] [--det 0|1] [--allow-partial]                    dry run: the mode's arm and the gates on this machine; nothing from the kit is imported
#   run.sh warm   [--config h100] [--mode M] [--det 0|1] [--allow-partial] --out_dir <dir>    one prediction on the kit's p995_1brs input: weights load, graph capture
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (weights root, JIT caches, target GPU). Mode = --mode when given,
# else PROTENIX_V1_OPT from the environment, else the package default (fast); a --mode that disagrees with a set PROTENIX_V1_OPT is refused.
# Modes (exact|fast|big|off) resolve to the kit's arm strings in opt/protenix_v1_opt/modes.py; this script validates none of them. `--mode off`
# is the stock route: the stock `protenix pred` in a clean subprocess that proves its environment (opt/protenix_v1_opt/stock_pred.py);
# `pred` and `warm` are its commands. `--det 1` = the kit's deterministic recipe on either route (opt/protenix_v1_opt/det.py).
# Exit codes (opt/protenix_v1_opt/report.py): 0 ok, 1 failed, 2 usage, 3 not active — a gate, the kit's refusal of the arm, or a PARTIAL
# activation (a lever of the mode not applied, applied with no served call, or fallen back from at run time: the kit's own counters);
# `--allow-partial` names the partial levers on the PARTIAL line and the exit is the run's own.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,12p' "$0" >&2; exit 2; }
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
probe_package () {   # the ONE package probe (configs/*.env call `run.sh probe`): python's own words and exit code pass through; 'not installed' names an ABSENT package only
  python -c "pass" || return $?                  # the interpreter start under the caller's environment as given: an installed .pth's refusal (an undeclared / mistyped PROTENIX_V1_OPT* name, an unknown mode) prints its own NOT ACTIVE line and its exit code is the probe's
  python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('protenix_v1_opt') else 1)" \
    || { echo "[protenix-v1-opt] NOT ACTIVE: protenix_v1_opt is not installed on $(command -v python || echo 'python (not on PATH)'): $HERE/run.sh install" >&2; return 2; }
  python -c "import protenix_v1_opt" || return $?   # a refusal raised while importing the package prints its own text (stderr as is) and its exit code is the probe's — never 'not installed'
}
case "$CMD" in install|probe) ;; pred|check|warm) ;; *) usage ;; esac   # install and probe are this script's own steps (below); pred|check|warm are the package's verbs
if [ "$CMD" = install ]; then                                       # the install step: everything below it presupposes the installed package
  WEIGHTS=""; PIPARGS=()
  [ -z "$CFG" ] && [ -z "$MODE" ] || { echo "run.sh: install takes no --config / --mode (usage: run.sh install [--weights DIR] [pip options])" >&2; exit 2; }
  set -- ${ARGS[@]+"${ARGS[@]}"}
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#-}" = "$2" ] || { echo "run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR] [pip options])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
      --weights=*) WEIGHTS=${1#*=}; [ -n "$WEIGHTS" ] || { echo "run.sh: install --weights= takes a directory" >&2; exit 2; }; shift ;;
      *) PIPARGS+=("$1"); shift ;;                                  # every other argument is pip install's, as given (e.g. --no-build-isolation, -q)
    esac
  done
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md 'Install')" >&2; exit 3; }
  if [ ${#PIPARGS[@]} -eq 0 ] && python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('protenix_v1_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: protenix_v1_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip; pip options given = pip runs
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" ${PIPARGS[@]+"${PIPARGS[@]}"} || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the line above — the pinned protenix is not installed as pinned in this environment)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then python -m protenix_v1_opt.weights "$WEIGHTS" || exit $?; fi   # upstream's downloader into DIR, then the weights gate against stock/PINS.json (opt/protenix_v1_opt/weights.py)
  exit 0
fi
if [ "$CMD" = probe ]; then probe_package; exit $?; fi
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[protenix_v1-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[protenix_v1-kit] jit cache: $J ($W)"; }
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 2) when protenix_v1_opt is not importable, (rc 3) when PROTENIX_ROOT_DIR is unset or lacks the pinned weights
fi
CLIMODE=$MODE                                                     # the command-line mode, kept apart from the environment: the environment-route hook guard below reads it before any merge
MODE=${MODE:-${PROTENIX_V1_OPT:-}}                               # --mode, else the environment — only to tell the stock route apart below; the mode rule (the default, a --mode that disagrees with a set PROTENIX_V1_OPT: rc 2) is the package's (opt/protenix_v1_opt/cli.py)
if [ "$MODE" = off ]; then                                        # stock route: `pred --mode off` (and `warm`, one stock prediction) = the stock CLI in a clean subprocess
  case "$CMD" in pred|warm) ;; *) echo "run.sh: --mode off has two commands, pred and warm (the stock arm has no check)" >&2; exit 2 ;; esac
fi
probe_package || exit $?                         # the package probe above: a start-time or import-time refusal exits with python's own code and words; an absent package is `not installed`, rc 2
python -m protenix_v1_opt._producers || exit 3   # the producers gate: the shared core present with every module this package imports, else NOT ACTIVE reason=core_missing|producer_missing rc 3 (before the hook probe below imports the package)
if [ -z "$CLIMODE" ] && [ -n "${PROTENIX_V1_OPT:-}" ] && [ "$PROTENIX_V1_OPT" != off ]; then   # the environment route only (PROTENIX_V1_OPT names a kit mode, no --mode): the
  python -c "from protenix_v1_opt import kit; kit.autoload_hook_main()" || exit $?   # autoload hook must be LIVE in a fresh interpreter (the .pth's effect), else NOT ACTIVE rc 3 with the diagnostic
fi
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m protenix_v1_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
