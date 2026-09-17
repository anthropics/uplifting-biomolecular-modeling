#!/bin/bash
# Caliby optimized — single entry point: a thin wrapper over `python -m caliby_opt` (the installed caliby_opt package and the shared core it pins).
#   run.sh install [--weights DIR]                              the install step: the shared core and this kit installed editable into the python on PATH (pip install -e ../common/opt_core -e opt), then the pin check (stock/check_pins.py);
#                                                               --weights DIR also fetches the weight files into DIR with upstream's own downloader and checks them against stock/PINS.json (sha256) — DIR is then your MODEL_PARAMS_DIR
#   run.sh design [--config h100] [--mode M] [--variant V] --input <structures...> --out_dir <dir> [--seed S] [--det 0|1] [upstream's keywords, forwarded only when given: --model_name --num_seqs_per_pdb --batch_size --omit_aas A,B --temperature --num_workers --verbose --sampling_overrides K=V.. --pos_constraint_csv --clean_workers --device --sampling_cfg_path; ensemble32: --num_samples_per_pdb --pp_batch_size --sampling_yaml_path --max_num_conformers --include_primary_conformer --use_primary_res_type]
#   run.sh check  [--config h100] [--mode M] [--variant V]      dry run: resolves the mode to its kit row on this box; nothing is installed or applied
#   run.sh warm   [--config h100] [--mode M] [--variant V]      one design of the smallest public example (Triton JIT + CUDA-graph capture)
# --variant single|ensemble32 (the package's modes.VARIANTS; stock/PINS.json "variants"): one variant per process, default single. --config <cfg>
# sources configs/<cfg>.env: deployment parameters only (weights, JIT cache, target GPU, state directory). Mode = --mode when given, else
# CALIBY_OPT from the environment, else fast on both variants (opt/caliby_opt/modes.py DEFAULT_MODE; never exact); a --mode that disagrees with a set
# CALIBY_OPT is refused; the same rule binds --variant and CALIBY_VARIANT. Modes off|exact|fast (single: off|fast; ensemble32: off|exact|fast — fast runs the exact row there) resolve in the package to the kits'
# own activation lines (modes.py), which refuses an unknown mode and exact on single by name (exit 2); this script validates the variant name only. `--mode off` is the stock route: the package's one stock
# caller (opt/caliby_opt/stock_design.py — the upstream API on the pinned tree, every kit switch stripped and proven absent, the tree digest
# asserted, no import hook) in a clean subprocess; `design`, `warm` and `check` are its commands. Every design process prints the [caliby-opt] report lines of
# opt/caliby_opt/report.py CENSUS_LINES (STACK, OUTPUTS_WRITTEN, PEAK; the parent: ENV-CLEAN, DESIGNS); nothing inside upstream's loops is stamped. Every upstream keyword is forwarded under its own name only when given, so
# upstream's default applies otherwise (any --model_name name or path, any --seed or none, --det 0|1 = upstream's scripts' cuDNN-deterministic recipe); the design child runs to its end (no deadline). A kit mode (`--mode fast` on single, `--mode exact` on ensemble32)
# loads the kits' files by import hook inside its one design process; site-packages is never written (CHANGES.md). Every route refuses (rc 3) unless the package is importable with the shared core it pins
# — the core pin gate first (opt/caliby_opt/_core_gate.py: an absent, older/newer or edited core is one NOT ACTIVE line before anything of the core is imported) — and the pinned upstream packages
# are installed at their pinned commits (stock/check_pins.py). Exit codes: 0 ok, 1 failed (a design, the install, a weight file off its pin; or outputs short of the request: `incomplete`), 2 usage, 3 not active (also a run in which a lever of the mode could not run: a mode is all of its levers) / pins not met.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,19p' "$0" >&2; exit 2; }
if [ "${1:-}" = install ]; then                                     # the install step: everything below it presupposes the installed package
  shift; WEIGHTS=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
      --weights=*) WEIGHTS=${1#*=}; [ -n "$WEIGHTS" ] || { echo "run.sh: install --weights= takes a directory (usage: run.sh install [--weights DIR])" >&2; exit 2; }; shift ;;
      *) echo "run.sh: install takes no argument '$1' (usage: run.sh install [--weights DIR])" >&2; exit 2 ;;
    esac
  done
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md Install)" >&2; exit 3; }
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('caliby_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: caliby_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the lines above — the pinned upstream packages are not installed at their pinned commits in this environment; STOCK.md)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then python -m caliby_opt.weights "$WEIGHTS" || exit $?; fi   # upstream's downloader into DIR, then the sha256 check against stock/PINS.json (opt/caliby_opt/weights.py)
  exit 0
fi
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
case "$CMD" in design|check|warm) ;; *) usage ;; esac
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 3, NOT ACTIVE) when caliby_opt is not importable or the core pin gate refuses the shared core
fi
_J0="${MODEL_OPT_JIT_ROOT:-}"   # the root configs/<card>.env derived cache dirs from, if any (the block below may move it)
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[caliby-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[caliby-kit] jit cache: $J ($W)"; }
if [ -n "${MODEL_OPT_JIT_ROOT:-}" ] && [ -n "${MODEL_OPT_STACK_KEY:-}" ]; then   # key the compile caches under the root the block chose; a dir configs/<card>.env derived from a root the block moved follows it, a dir set elsewhere is kept
  if [ -z "${TRITON_CACHE_DIR:-}" ] || [ "${TRITON_CACHE_DIR:-}" = "$_J0/$MODEL_OPT_STACK_KEY/triton" ]; then export TRITON_CACHE_DIR="$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/triton"; fi
  if [ -z "${TORCHINDUCTOR_CACHE_DIR:-}" ] || [ "${TORCHINDUCTOR_CACHE_DIR:-}" = "$_J0/$MODEL_OPT_STACK_KEY/inductor" ]; then export TORCHINDUCTOR_CACHE_DIR="$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/inductor"; fi
fi
FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
ENVMODE=${CALIBY_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with CALIBY_OPT=$ENVMODE from $FROM; each run has exactly one mode; pass one of them" >&2; exit 2
fi
MODE=${MODE:-$ENVMODE}
ENVVARIANT=${CALIBY_VARIANT:-}
if [ -n "$VARIANT" ] && [ -n "$ENVVARIANT" ] && [ "$VARIANT" != "$ENVVARIANT" ]; then
  echo "run.sh: --variant $VARIANT disagrees with CALIBY_VARIANT=$ENVVARIANT from $FROM; one variant per process — drop one of them" >&2; exit 2
fi
VARIANT=${VARIANT:-$ENVVARIANT}
case "$VARIANT" in ""|single|ensemble32) ;; *) echo "run.sh: '$VARIANT' is not a variant (single|ensemble32, stock/PINS.json)" >&2; exit 2 ;; esac
python -c "import caliby_opt" 2>/dev/null || { echo "run.sh: caliby_opt is not importable on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/../common/opt_core -e $HERE/opt" >&2; exit 3; }
python -c "from caliby_opt.stack import core_gate; core_gate()" >/dev/null || exit 3     # the core pin gate, first on every route (--mode off included): an absent / older / newer / edited shared core is its one NOT ACTIVE line, rc 3
python -I "$HERE/stock/check_pins.py" --quiet || { echo "run.sh: the pinned upstream packages are not installed at their pinned commits (stock/PINS.json, STOCK.md)" >&2; exit 3; }
if [ "$MODE" = off ]; then                                        # stock route: the package's stock caller in a clean subprocess
  case "$CMD" in design|warm|check) ;; *) echo "run.sh: --mode off has three commands: design, warm and check" >&2; exit 2 ;; esac
fi
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
VARARG=();  [ -n "$VARIANT" ] && VARARG=(--variant "$VARIANT")
exec python -m caliby_opt "$CMD" ${VARARG[@]+"${VARARG[@]}"} ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
