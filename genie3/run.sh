#!/bin/bash
# Genie 3 optimized — single entry point: a thin wrapper over `python -m genie3_opt` (the installed genie3_opt package: `pip install -e ../common/opt_core -e opt`).
#   run.sh design [--config h100] [--mode M] --input|-c <experiment.yaml> [--out_dir <dir>] [--n N] [--selections k1,k2] [--seed S] [--batch_size B] [--det 0|1] [--tag T]
#                 [upstream's generate flags, passed through: --verbose --log-dir <dir> --num-devices N --shard-id K --num-shards M]
#   run.sh design [--config h100] [--mode M] [kit flags] -- <upstream's own generate argv verbatim: -c|--config <experiment.yaml> --log-dir <dir> …>
#   run.sh check  [--config h100] [--mode M] [--json]                        dry run: resolves the mode against the package's mode table on this GPU, reports the stock pins and the weights digests; nothing runs
#   run.sh warm   [--config h100] [--mode M] --out_dir <dir> [--input <experiment.yaml>] [--n N]     one public request through the mode's line
#   run.sh install [--weights DIR]                       this kit and the shared core installed editable into the python on PATH, then the pin check (stock/check_pins.py:
#                 exit 3 names a checkout that is not the pin; torch / lightning / numpy differences are reported, not refused); --weights DIR also fetches upstream's
#                 pretrained/ tree into DIR with upstream's own downloader (scripts/setup/download.sh's
#                 `hf download`) and checks the two files the kit reads against stock/PINS.json — GENIE3_WEIGHTS is then DIR/pretrained/v1 (opt/genie3_opt/weights.py)
# --config <card> (before `--`) sources configs/<card>.env: deployment parameters only (checkout, weights, target GPU); upstream's own
# `-c/--config <experiment.yaml>` is `--input|-c <file>` here or rides verbatim after `--`. Mode = --mode when given, else GENIE3_OPT from the
# environment, else the package default (modes.DEFAULT_MODE — fast; `--mode exact` selects the bit-exact line); a --mode that disagrees with a set
# GENIE3_OPT is refused. Modes (off|exact|fast) are resolved by the package from its one mode table (opt/genie3_opt/modes.py); this script
# validates none of them. `--mode off` is the stock route: the package's one stock caller (opt/genie3_opt/stock_cli.py — upstream's `genie3 generate`
# with the caller's own generate flags, in a clean subprocess that proves its environment); its commands are `design` and `check` (there is no stock warm).
# The request is upstream's experiment YAML exactly. Dataset shards (--num-shards M --shard-id K) and the sequence stage (predict_sequence) run on every
# mode; a request naming a computation a kit line does not run — beam search, the sidechain stage, more than one device — is refused by name before
# anything runs (exit 3; `--mode off` runs it, as it runs every key and flag upstream accepts). The stock pins
# (stock/check_pins.py: the checkout's bytes at the pin, the torch / lightning / numpy freeze) are REPORTED on every route and gate nothing; `check`
# prints the full report; `install` refuses one thing, a checkout that is not the pin (exit 3) — a different torch / lightning / numpy is reported and the kit
# runs on it. Uncertainty about the environment (GPU, triton / torch patch level, a card without a cell row) is named on the activation lines and every lever
# engages — never a refusal. A mode is ALL of its levers: a planned lever that cannot run on this box (a kernel that will not compile or launch, an unsupported
# shape) makes the mode refuse BY NAME after the pass (`NOT ACTIVE: mode=<m> refused — lever(s) <ids> could not run …`, exit 3), never a subset. The shared
# core's pin is the first line of every route (an absent or mismatched core is its NOT ACTIVE line, rc 3).
# Exit codes: 0 ok, 1 failed (or: outputs short of the request — `incomplete`; a forbidden driver-log line; the install; a weight file off its pin), 2 usage, 3 not active
# (a deployment fact named — no CUDA device, no checkout, a weight file missing, the shared core; a batch refused by the driver's memory model, the batch that fits
# named; or a planned lever that could not run — the mode refuses by name).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,28p' "$0" >&2; exit 2; }
CFG=""; CMD=""; MODE=""; ARGS=()
take_config() {   # the kit's card: configs/<card>.env (deployment parameters). Upstream's own -c/--config <experiment.yaml> is --input/-c here, or verbatim after `--`.
  [ -f "$HERE/configs/$1.env" ] || { echo "run.sh: no such card: $1 (see $HERE/configs/); upstream's own -c/--config <experiment.yaml> is --input <file> here, or goes verbatim after \`--\`" >&2; exit 2; }
  CFG=$1
}
while [ $# -gt 0 ]; do
  case "$1" in
    --)          shift; ARGS+=("$@"); break ;;               # everything after `--` is upstream's own argv, verbatim (its -c/--config <experiment.yaml>, --verbose, --log-dir, --num-devices, --shard-id, --num-shards)
    --config)    [ $# -ge 2 ] || usage; take_config "$2"; shift 2 ;;
    --config=*)  take_config "${1#--config=}"; shift ;;
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
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('genie3_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: genie3_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the lines above — the checkout at \$GENIE3_ROOT is not the pinned commit byte for byte; stock/PINS.json). A torch / lightning / numpy off the pinned stack is reported above, never refused" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then python -m genie3_opt.weights "$WEIGHTS" || exit $?; fi   # upstream's downloader into DIR, then the sha256 check against stock/PINS.json (opt/genie3_opt/weights.py)
  exit 0
fi
if [ -n "$CFG" ]; then
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 3) when genie3_opt is not importable or its core pin gate refuses
fi
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[genie3-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[genie3-kit] jit cache: $J ($W)"; }
# [install]-jitcache (this kit): with a JIT root, Triton's and inductor's caches are keyed under <root>/<stack key>/ (the key of
# configs/<card>.env, probed here when no config was sourced); a directory already named in TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR
# is kept, and without a root nothing is exported (Triton's ~/.triton, the driver's own inductor default).
if [ -n "${MODEL_OPT_JIT_ROOT:-}" ]; then
  _G3_JIT_KEY="${MODEL_OPT_STACK_KEY:-$(command -v python >/dev/null && python -c 'from genie3_opt.stack import jit_cache_key; print(jit_cache_key())' 2>/dev/null || true)}"
  case "$_G3_JIT_KEY" in ""|*unknown:*|*[!A-Za-z0-9._+-]*) ;; *)
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$MODEL_OPT_JIT_ROOT/$_G3_JIT_KEY/triton}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$MODEL_OPT_JIT_ROOT/$_G3_JIT_KEY/inductor}" ;;
  esac
fi
FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
ENVMODE=${GENIE3_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with GENIE3_OPT=$ENVMODE from $FROM; a pass runs one mode — drop one of them" >&2; exit 2
fi
MODE=${MODE:-$ENVMODE}
python -c "import genie3_opt" 2>/dev/null || { echo "run.sh: genie3_opt is not importable on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/../common/opt_core -e $HERE/opt" >&2; exit 3; }
python -c "from genie3_opt._core import core_gate; core_gate()" >/dev/null || exit 3     # the core pin gate, first on every route (off included): an absent / older / newer / edited shared core is its one NOT ACTIVE line, rc 3
STACKARG=(); [ "$MODE" = off ] && STACKARG=(--no-stack)     # the stock route on another stack is a labelled run (the package records the stack)
python -I "$HERE/stock/check_pins.py" --quiet ${STACKARG[@]+"${STACKARG[@]}"} || echo "run.sh: the stock pins REPORT above names a difference (checkout bytes at the pin, or torch/lightning/numpy vs the pinned stack; stock/PINS.json) — reported, not gated: the pass runs on the checkout as installed" >&2
if [ "$MODE" = off ] && [ "$CMD" = warm ]; then                   # stock route: `design --mode off` = the package's stock caller in a clean subprocess; `check --mode off` reports it; there is no stock warm
  echo "run.sh: --mode off has no warm (warm runs one design through a kit line; the stock route's commands are design and check)" >&2; exit 2
fi
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m genie3_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
