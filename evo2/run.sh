#!/usr/bin/env bash
# Evo 2 exact kit — one entry point over the installed `evo2_opt` package and the tree's route driver.
#   run.sh install [--weights DIR [--model_name evo2_7b|evo2_40b|evo2_1b_base]]   pip install -e opt (the kit + its autoload hook) into the python on PATH, then the
#                                                                   pin check (stock/check_pins.py: evo2 / vtx / the stack against stock/PINS.json);
#                                                                   --weights DIR also fetches the model's checkpoint from the pinned Hugging Face
#                                                                   revision into DIR and checks its sha256 — DIR is then your EVO2_OPT_WEIGHTS
#   run.sh check                                                    the dry run: the [evo2-opt] CHECK line (what the kit reads here), nothing applied
#   run.sh score [--mode exact|fast|off] --model_name M --input <fasta> --out_dir <dir> [--batch_size N] [--local_path CKPT]
#                                                                   route/evo2_route.py: one score_sequences call; exact (default) sets EVO2_OPT=exact,
#                                                                   off runs the stock in a clean process
# In your own code: `export EVO2_OPT=exact` (or `import evo2_opt; evo2_opt.enable()`) before constructing evo2.Evo2 — README.md.
# Exit codes: 0 ok · 1 a failure · 2 usage · 3 the kit refused by name ([evo2-opt] NOT ACTIVE / KIT REFUSED).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export EVO2_OPT_HOME=$HERE
KIT=evo2
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
# [install]-jitcache — evo2 mapping: Triton is this kit's only compile cache; under a root it lives in <root>/<stack key>/triton
# (stack key = img_full when Transformer Engine is installed, else img_a100 — the stacks of stock/PINS.json) when that directory is writable;
# a read-only root the block left in place is never handed to Triton (its cache stays at the default). Nothing changes when no root is set.
if [ -n "${MODEL_OPT_JIT_ROOT:-}" ] && [ -z "${TRITON_CACHE_DIR:-}" ]; then
  T="$MODEL_OPT_JIT_ROOT/${MODEL_OPT_STACK_KEY:-$(python -c 'import importlib.util as u; print("img_full" if u.find_spec("transformer_engine") else "img_a100")' 2>/dev/null || echo img_full)}/triton"
  if { mkdir -p "$T" && [ -w "$T" ]; } 2>/dev/null; then export TRITON_CACHE_DIR="$T"; fi
fi
usage() { sed -n '2,12p' "$0" >&2; exit 2; }
[ $# -ge 1 ] || usage
CMD=$1; shift
case "$CMD" in
  install)
    WEIGHTS=""; MODEL="evo2_7b"
    while [ $# -gt 0 ]; do
      case "$1" in
        --weights) [ $# -ge 2 ] || usage; WEIGHTS=$2; shift 2 ;; --weights=*) WEIGHTS=${1#*=}; shift ;;
        --model_name) [ $# -ge 2 ] || usage; MODEL=$2; shift 2 ;; --model_name=*) MODEL=${1#*=}; shift ;;
        *) echo "run.sh: install: unknown argument $1" >&2; usage ;;
      esac
    done
    python -m pip install -e "$HERE/opt" || { echo "run.sh: pip install -e $HERE/opt failed (pip's words above)" >&2; exit 1; }
    python "$HERE/stock/check_pins.py" || exit $?
    if [ -n "$WEIGHTS" ]; then python -m evo2_opt.weights "$WEIGHTS" --model_name "$MODEL" || exit $?; fi
    ;;
  check)
    [ $# -eq 0 ] || usage
    python -c "import evo2_opt" 2>/dev/null || { echo "run.sh: evo2_opt is not installed on $(command -v python): bash run.sh install" >&2; exit 3; }
    exec python -m evo2_opt check
    ;;
  score)
    MODE=exact; ARGS=()
    while [ $# -gt 0 ]; do
      case "$1" in
        --mode) [ $# -ge 2 ] || usage; MODE=$2; shift 2 ;; --mode=*) MODE=${1#*=}; shift ;;
        *) ARGS+=("$1"); shift ;;
      esac
    done
    case "$MODE" in
      exact) export EVO2_OPT=exact ;;
      fast) export EVO2_OPT=fast ;;                         # scoring under fast is exact's (the same levers); fast adds speculative sampling to generate()
      off) unset EVO2_OPT ;;
      *) echo "run.sh: --mode $MODE is not a mode (exact | fast | off)" >&2; exit 2 ;;
    esac
    exec python "$HERE/route/evo2_route.py" --mode "$MODE" ${ARGS[@]+"${ARGS[@]}"}
    ;;
  *) usage ;;
esac
