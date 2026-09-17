#!/bin/bash
# ESM C optimized — the kit's install and dry-run commands: a thin wrapper over `python -m esmc_opt` (the installed esmc_opt package:
# `pip install -e opt`). There is no inference command: ESM C is a Python API and the kit is a drop-in beside it — set ESMC_OPT=exact and
# run your own ESM C code (README.md Run).
#   run.sh install [--weights DIR --variant V]              the install step: this kit's package installed editable into the python on PATH (it imports nothing else of the bundle), the pin check (stock/check_pins.py),
#                                                          then the fused residual+LayerNorm CUDA extension built once for the visible GPU (esmc_opt.kits.residual_ln.build: needs the CUDA toolkit's nvcc + ninja; cached under ~/.cache/esmc_sdkfused);
#                                                          --weights DIR --variant V also fetches the variant's weights snapshot into DIR with upstream's own downloader and checks it against
#                                                          stock/PINS.json (sha256) — DIR is the weights root of STOCK.md 'Weights' (the HF cache is DIR/hf: export HF_HOME=DIR/hf afterwards)
#   run.sh check  --variant V [--mode M]                     dry run: resolves the mode against the package's mode table on this GPU; nothing is applied, no model is loaded
# --variant 300m|600m|6b, or the SDK's model names esmc_300m|esmc_600m|esmc_6b (modes.VARIANTS; stock/PINS.json "variants"). Mode = --mode when given,
# else ESMC_OPT from the environment, else the package default (exact); a --mode that disagrees with a set ESMC_OPT is refused. Modes (exact|off;
# there is no fast mode for this model) are resolved by the package (opt/esmc_opt/modes.py); this script validates none of them.
# Weights: the HF cache under HF_HOME (upstream's own location). Caches: the extension's build under ~/.cache/esmc_sdkfused; set TRITON_CACHE_DIR to a
# persistent path to keep the Triton JIT across processes (Triton's own variable; unset = its own default, ~/.triton).
# `check` refuses (rc 3) unless the pinned upstream packages are installed at their pinned COMMITS (stock/check_pins.py: a different commit changes
# what "stock" means) and the package is installed. Exit codes: 0 ok, 1 the install failed (pip, the extension build, a weight file off its
# pin), 2 usage, 3 not active (unknown mode, stock absent or off its commit, no GPU).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
usage() { sed -n '2,17p' "$0" >&2; exit 2; }
CMD=""; MODE=""; VARIANT=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --mode)      [ $# -ge 2 ] || usage; MODE=$2; shift 2 ;;
    --mode=*)    MODE=${1#--mode=}; shift ;;
    --variant)   [ $# -ge 2 ] || usage; VARIANT=$2; shift 2 ;;
    --variant=*) VARIANT=${1#--variant=}; shift ;;
    *) if [ -z "$CMD" ]; then CMD=$1; else ARGS+=("$1"); fi; shift ;;
  esac
done
case "$CMD" in check|install) ;; *) usage ;; esac
case "${VARIANT,,}" in esmc_300m) VARIANT=300m ;; esmc_600m) VARIANT=600m ;; esmc_6b) VARIANT=6b ;; esac   # the SDK's model names (stock/PINS.json variants[].model_name) name the same sizes
KIT=esmc   # the log tag of the [install]-jitcache seed block below
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[${KIT:-kit}-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[${KIT:-kit}-kit] jit cache: $J ($W)"; }
if [ "$CMD" = install ]; then                                       # the install step: everything below it presupposes the installed package
  WEIGHTS=""
  [ -z "$MODE" ] || { echo "run.sh: install takes no --mode (usage: run.sh install [--weights DIR --variant V])" >&2; exit 2; }
  set -- ${ARGS[@]+"${ARGS[@]}"}
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR --variant V])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
      --weights=*) WEIGHTS=${1#*=}; [ -n "$WEIGHTS" ] || { echo "run.sh: install --weights= takes a directory" >&2; exit 2; }; shift ;;
      *) echo "run.sh: install takes no argument '$1' (usage: run.sh install [--weights DIR --variant V])" >&2; exit 2 ;;
    esac
  done
  if [ -n "$WEIGHTS" ]; then
    [ -n "$VARIANT" ] || { echo "run.sh: install --weights DIR takes --variant (300m|600m|6b): the variant whose weights are fetched into DIR" >&2; exit 2; }
    case "$VARIANT" in 300m|600m|6b) ;; *) echo "run.sh: '$VARIANT' is not a variant (300m|600m|6b, stock/PINS.json)" >&2; exit 2 ;; esac
  else
    [ -z "$VARIANT" ] || { echo "run.sh: install takes --variant only together with --weights DIR (usage: run.sh install [--weights DIR --variant V])" >&2; exit 2; }
  fi
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md Setup)" >&2; exit 3; }
  if python -I -c "import os,sys,importlib.util as u; d=os.path.realpath(sys.argv[1]); x=u.find_spec('esmc_opt'); sys.exit(0 if x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) else 1)" "$HERE/opt" 2>/dev/null; then
    echo "run.sh: esmc_opt is installed from this tree already ($HERE/opt) — the pip step is skipped"   # the container image ships it installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the kit package is expected at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the lines above — the pinned upstream packages are not installed at their pinned commits in this environment, or torch does not import)" >&2; exit 3; }
  python -m esmc_opt.kits.residual_ln.build || { echo "run.sh: installed, but the fused residual+LayerNorm extension could not be built (the lines above: it needs setuptools and ninja in this environment and the CUDA toolkit's nvcc); without it the kit mode cannot engage — stock (\`--mode off\`) is unaffected" >&2; exit 1; }   # kernels.cu compiled once into ~/.cache/esmc_sdkfused for this GPU (no GPU visible: for sm80 and sm90)
  if [ -n "$WEIGHTS" ]; then HF_HUB_OFFLINE=0 python -m esmc_opt.weights "$WEIGHTS" "$VARIANT" || exit $?; fi   # upstream's downloader into DIR (online for this one command, whatever HF_HUB_OFFLINE says), then the sha256 check against stock/PINS.json (opt/esmc_opt/weights.py)
  exit 0
fi
ENVMODE=${ESMC_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with ESMC_OPT=$ENVMODE in the environment; one mode per process — drop one of them" >&2; exit 2
fi
MODE=${MODE:-$ENVMODE}
[ -n "$VARIANT" ] || { echo "run.sh: --variant is required (300m|600m|6b)" >&2; exit 2; }
case "$VARIANT" in 300m|600m|6b) ;; *) echo "run.sh: '$VARIANT' is not a variant (300m|600m|6b, stock/PINS.json)" >&2; exit 2 ;; esac
python -I "$HERE/stock/check_pins.py" --quiet || { echo "run.sh: the pinned upstream packages are not installed at their pinned commits (stock/PINS.json), or torch does not import" >&2; exit 3; }
python -c "import esmc_opt" 2>/dev/null || { echo "run.sh: esmc_opt is not installed on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/opt" >&2; exit 3; }
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m esmc_opt "$CMD" --variant "$VARIANT" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
