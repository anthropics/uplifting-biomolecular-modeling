#!/bin/bash
# AF2 initial-guess optimized — single entry point: a thin wrapper over `python -m af2ig_opt` (the installed af2ig_opt package and the tree's core: `pip install -e ../common/opt_core -e opt`).
#   run.sh pred   [--config h100] [--mode M] --pdbdir <dir> --out <dir> [--det 1] [--precompile [N]] [--allow-partial] [-- <driver settings switches>]
#                 exact|fast|big: the kit driver's process with the mode's flags appended; off: the same driver with no lever flag, through the stock caller in a clean subprocess
#   run.sh check  [--config h100] [--mode M] [--json]                        dry run: resolves the mode and every gate on this box; nothing runs
#   run.sh warm   [--config h100] [--out DIR] [--det 1] [--mode M --lengths 400,800,…]   the kit's warm-up process: one public complex through the stock line — or, with --lengths, the cache warm-up (ccache + program store) for those lengths through mode M; outputs discarded
#   run.sh install [--weights DIR]                                           the install step: the shared core and this kit installed editable into the python on PATH, the pinned upstream tree
#                 unpacked from stock/ and patched with the kit's series at ./dl_binder_design (AF2IG_DIR is then ./dl_binder_design/af2_initial_guess), then the pin check (stock/check_pins.py);
#                 --weights DIR also fetches the AlphaFold-2 parameter file into DIR/params/ (upstream's documented archive) and checks it against stock/PINS.json (sha256) — DIR is then your AF2_PARAMS
# --config <cfg> sources configs/<cfg>.env (h100 | a100 | h200): deployment parameters only (target GPU, bytecode).
# Mode = --mode when given, else AF2IG_OPT from the environment, else the package default (opt/af2ig_opt/modes.py DEFAULT_MODE); a --mode that
# disagrees with a set AF2IG_OPT is refused by the package (exit 2). Modes (off|exact|fast|big) are resolved and composed by the package from the kit's
# own statement (opt/forward/af2ig_kit/MANIFEST.json `levers`); this script validates none of them. Every route needs AF2IG_DIR (the patched checkout's af2_initial_guess directory) and AF2_PARAMS (the directory
# holding params/params_model_1_ptm.npz) in the environment and refuses (rc 3) unless the package is installed; the pinned stack (stock/PINS.json) is the
# package's own gate: a required distribution that is absent refuses by name, one installed at another version is named on the printed line (pins=drift) and runs; an absent or mismatched core is `[af2ig-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: …` (rc 3; af2ig_opt/_core_gate.py); a package whose import itself fails or refuses ends the script with that python's own stderr and exit status (no word added). Exit codes: 0 ok, 1 failed or incomplete (fewer outputs than inputs), 2 usage, 3 not active or partial (`--allow-partial` records and proceeds) / pins not met.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,15p' "$0" >&2; exit 2; }
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
case "$CMD" in pred|check|warm|install) ;; *) usage ;; esac
if [ "$CMD" = install ]; then                                       # the install step: everything below it presupposes the installed package and the patched tree
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
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md Setup)" >&2; exit 3; }
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('af2ig_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: af2ig_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  CHECKOUT=$HERE/dl_binder_design                                   # the pinned upstream tree, patched: AF2IG_DIR is $CHECKOUT/af2_initial_guess (STOCK.md)
  if [ -f "$CHECKOUT/af2_initial_guess/predict_pdb.py" ]; then
    echo "run.sh: the patched upstream tree is at $CHECKOUT already — kept, and checked against the pin below"   # the container image ships it; a second run only checks
  else
    [ ! -e "$CHECKOUT" ] || { echo "run.sh: $CHECKOUT exists but holds no patched af2_initial_guess/predict_pdb.py — remove it and re-run this step" >&2; exit 1; }
    ARCHIVE=("$HERE"/stock/dl_binder_design-*.tar.gz)
    [ ${#ARCHIVE[@]} -eq 1 ] && [ -f "${ARCHIVE[0]}" ] || { echo "run.sh: expected exactly one stock/dl_binder_design-*.tar.gz (the pinned upstream archive, STOCK.md), found: ${ARCHIVE[*]}" >&2; exit 1; }
    command -v tar >/dev/null && command -v patch >/dev/null || { echo "run.sh: tar and GNU patch are needed on PATH to unpack ${ARCHIVE[0]##*/} and apply the kit's patches (opt/forward/af2ig_kit/patches)" >&2; exit 3; }
    tar -xzf "${ARCHIVE[0]}" -C "$HERE" || { echo "run.sh: unpacking ${ARCHIVE[0]} into $HERE failed (tar's words above)" >&2; exit 1; }
    for p in "$HERE"/opt/forward/af2ig_kit/patches/[0-9][0-9]_*.diff; do
      patch -d "$CHECKOUT" -p1 -s < "$p" || { echo "run.sh: $p did not apply to $CHECKOUT (patch's words above) — remove $CHECKOUT and re-run this step" >&2; exit 1; }
    done
    echo "run.sh: unpacked ${ARCHIVE[0]##*/} and applied the kit's patch series (opt/forward/af2ig_kit/patches/[0-9][0-9]_*.diff) at $CHECKOUT"
  fi
  python -I "$HERE/stock/check_pins.py" --checkout "$CHECKOUT" || { echo "run.sh: $CHECKOUT is not the pinned, patched upstream tree (stock/check_pins.py --checkout: the lines above) — remove it and re-run this step" >&2; exit 1; }
  python -I "$HERE/stock/check_pins.py" --gpu || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the lines above — the pinned stack, its CUDA wheels included, is not installed as pinned in this environment)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then python -m af2ig_opt.weights "$WEIGHTS" || exit $?; fi   # the AlphaFold-2 parameter file into DIR/params/, then the sha256 check against stock/PINS.json (opt/af2ig_opt/weights.py)
  echo "run.sh: installed — before pred, check and warm (README.md Setup): export AF2_PARAMS=${WEIGHTS:-<the directory holding params/>}; on a source checkout also export AF2IG_DIR=$CHECKOUT/af2_initial_guess (the container image presets it)"
  exit 0
fi
JIT_ROOT_PRESET="${MODEL_OPT_JIT_ROOT:-}${AF2IG_OPT_JIT_ROOT:-}"   # a JIT root named before the config is read (either variable) wins over a preset JAX_COMPILATION_CACHE_DIR below; the config's default root does not
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 3, NOT ACTIVE) when af2ig_opt is not importable
fi
[ -n "${MODEL_OPT_JIT_ROOT:-}" ] || [ -z "${AF2IG_OPT_JIT_ROOT:-}" ] || export MODEL_OPT_JIT_ROOT="$AF2IG_OPT_JIT_ROOT"   # before the block
JIT_ROOT_BEFORE="${MODEL_OPT_JIT_ROOT:-}"   # the root the block starts from; the kit's own root word follows it below if the block moves it
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[af2ig-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[af2ig-kit] jit cache: $J ($W)"; }
[ -z "${MODEL_OPT_JIT_ROOT:-}" ] || { [ -n "${AF2IG_OPT_JIT_ROOT:-}" ] && [ "$AF2IG_OPT_JIT_ROOT" != "$JIT_ROOT_BEFORE" ]; } || export AF2IG_OPT_JIT_ROOT="$MODEL_OPT_JIT_ROOT"   # after the block: the kit's root word follows the block's root (moved or not); one you set to something else yourself is kept
[ -z "${JAX_COMPILATION_CACHE_DIR:-}" ] || { [ -z "$JIT_ROOT_PRESET" ] && [ -z "$W" ]; } || { echo "[af2ig-kit] jit cache: JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR not used — a JIT root is named (MODEL_OPT_JIT_ROOT=$MODEL_OPT_JIT_ROOT), so the kit places <root>/<stack key>/<recipe>/jax and its program store beside it"; unset JAX_COMPILATION_CACHE_DIR; }   # a root you exported, or an image's shipped cache, wins over a preset jax cache directory; with only the config's default root a preset directory is kept as yours
python -c "import importlib.util, sys; sys.exit(4) if importlib.util.find_spec('af2ig_opt') is None else None; import af2ig_opt, af2ig_opt._core_gate as g; g.gate(list(af2ig_opt.__path__)[0], tag=af2ig_opt.TAG)" || { rc=$?; [ $rc != 4 ] || { echo "[af2ig-opt] NOT ACTIVE: package_missing:af2ig_opt on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/../common/opt_core -e $HERE/opt" >&2; rc=3; }; exit $rc; }   # the package present (a genuine absence, find_spec None, is the one `package_missing` word, rc 3) AND its core pin gate (af2ig_opt/_core_gate.py: NOT ACTIVE reason=core_missing|core_mismatch|core_pin_unreadable, rc 3); any other failure of the import is that python's own words and exit status, passed through — before anything else
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m af2ig_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}     # the entry runs the same gate as its first statement, then names any core sub-module the pinned core lacks (`NOT ACTIVE: core_missing:<module>`, rc 3) — never a traceback
