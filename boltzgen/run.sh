#!/bin/bash
# BoltzGen optimized — single entry point: a thin wrapper over `python -m boltzgen_opt` (the installed boltzgen_opt package + its core: `run.sh install`).
#   run.sh install [--weights DIR]                         the install step: the shared core (the one opt/pyproject.toml [tool.opt_core] pins) and this package installed editable into the
#                                                          python on PATH (pip install -e ../common/opt_core -e opt), then the pin check (stock/check_pins.py); --weights DIR also fetches the
#                                                          six stock weight / data files into DIR with upstream's own downloader and checks them against stock/PINS.json (sha256) — DIR is then your BOLTZGEN_CACHE
#   run.sh design [--config h100] [--mode M] <spec.yaml> --output <dir> [--seed N] [<any `boltzgen run` option>] [-- <more of them>]
#                                                          one design job, spelled like `boltzgen run`: upstream's `configure` with the caller's own options as given, then the mode's launch
#                                                          (a kit mode: the kit's runner; off: upstream alone in a clean environment, one process per pipeline step)
#   run.sh check  [--config h100] [--mode M]                dry run: resolves the mode against the kit's files and this GPU; nothing is applied
#   run.sh warm   [--config h100] [--mode M]                one design on the bundled example spec: Triton JIT + graph capture into the caches
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (weights cache, JIT caches, target GPU). `--config` and
# `--mode` are this script's only options and only BEFORE the first `--`: everything after it is upstream's own `boltzgen configure` argv,
# passed through as given (`-- --diffusion_batch_size 16 --use_kernels false --config design compile_pairformer=true` is upstream's, `--config`
# included, not this script's). Mode = --mode when given, else BOLTZGEN_OPT from the environment, else the package default (fast: the tolerance
# tier); a --mode that disagrees with a set BOLTZGEN_OPT is refused. Modes (exact|fast|big|off) are the package's one table
# (opt/boltzgen_opt/modes.py, read from the kit's own files); this script validates none of them. `--mode off` is the stock route: the
# package's one stock caller (opt/boltzgen_opt/stock_design.py — upstream alone, nothing of the kit on the path) one clean subprocess per
# pipeline step, each proving its environment; it serves every request upstream's `boltzgen run` serves. `--seed N`: step i runs with N + i on
# every arm; without it the stock route is upstream's unseeded run and a kit mode is not active (rc 3, the line names --seed).
# Exit codes: 0 ok, 1 failed (a design, the install, a weight file off its pin), 2 usage, 3 not active — the mode cannot activate on this box (a kit
# directory, pin, GPU or seed missing; the line names the escape, `--mode off`), a lever of the mode could not run on the box (partial: a mode is all of its
# levers and refuses by name), or the pin check refuses. An untested card is named (NOTE) and never refused.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,21p' "$0" >&2; exit 2; }
CFG=""; CMD=""; MODE=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --)          [ -n "$CMD" ] || usage; ARGS+=("$@"); break ;;      # end of this script's options: `--` and every token after it are upstream's argv (configure args), passed through untouched
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
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('boltzgen_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: boltzgen_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the line above — the pinned upstream is not installed as pinned in this environment)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then python -m boltzgen_opt.weights "$WEIGHTS" || exit $?; fi   # upstream's downloader into DIR, then the sha256 check against stock/PINS.json (opt/boltzgen_opt/weights.py)
  exit 0
fi
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses by name (rc 3) when boltzgen_opt is not importable or the core pin gate refuses
fi
_JIT_PRESET="${MODEL_OPT_JIT_ROOT:-}"   # [install]-jitcache mapping (1/2): the root configs/<card>.env derived the cache variables from, before the block below may replace it
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[boltzgen-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[boltzgen-kit] jit cache: $J ($W)"; }
if [ -n "${MODEL_OPT_JIT_ROOT:-}" ] && [ -n "${MODEL_OPT_STACK_KEY:-}" ]; then for _v in TRITON_CACHE_DIR:triton TORCH_EXTENSIONS_DIR:torch_extensions TORCHINDUCTOR_CACHE_DIR:inductor NUMBA_CACHE_DIR:numba; do _n=${_v%%:*}; eval "_c=\${$_n:-}"; if [ -z "$_c" ] || { [ -n "$_JIT_PRESET" ] && [ "$MODEL_OPT_JIT_ROOT" != "$_JIT_PRESET" ] && [ "${_c#"$_JIT_PRESET"/}" != "$_c" ]; }; then export "$_n=$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/${_v#*:}"; fi; done; unset _v _n _c; fi; unset _JIT_PRESET   # [install]-jitcache mapping (2/2): a cache variable that is unset, or that the config derived from a root the block replaced, follows the root the block chose (configs/h100.env's layout); anything set elsewhere stays
FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
ENVMODE=${BOLTZGEN_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with BOLTZGEN_OPT=$ENVMODE from $FROM; one run has one mode — drop one of them" >&2; exit 2
fi
MODE=${MODE:-$ENVMODE}
env -u BOLTZGEN_OPT python -c "import boltzgen_opt" 2>/dev/null || { echo "[boltzgen-opt] NOT ACTIVE: boltzgen_opt is not importable on $(command -v python || echo 'python (not on PATH)'): bash $HERE/run.sh install (= pip install -e $HERE/../common/opt_core -e $HERE/opt)" >&2; exit 3; }
python -c "import boltzgen_opt as k; k.core_gate()" >/dev/null || exit 3               # the core pin gate, first on every route (off included): an absent / older / newer / edited shared core is its one NOT ACTIVE line, rc 3
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m boltzgen_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
