#!/bin/bash
# ProteinMPNN optimized — single entry point: a thin wrapper over `python -m proteinmpnn_opt` (the installed proteinmpnn_opt package and the core it pins).
#   run.sh install [--weights DIR]                              the install step: the shared core and this kit installed editable into the python on PATH (skipped when they
#                                                             already import from this tree), then the pin check of what is installed (stock/check_pins.py --checks package);
#                                                             --weights DIR first fetches upstream's own files into DIR — the ProteinMPNN clone at the pinned commit, code and both
#                                                             repository weight sets — and checks them against stock/PINS.json (sha256); DIR is then your MPNN_DIR
#   run.sh design [--config h100] --mode off|exact [--variant V] [--bb_batch K] [--det 0|1] [--hybrid_gemm 0|1] [--allow-partial] <input> <output folder> [<stock options>]
#                                                             one design pass: the stock layout of outputs (seqs/ scores/ probs/), opt_manifest.json beside them;
#                                                             input  = --jsonl_path <parsed.jsonl> | --pdb_path <file.pdb> (upstream's own) | --input <dir of PDBs | parsed.jsonl | file.pdb>
#                                                             output = --out_folder <dir> (upstream's own) | --out <dir>
#   run.sh check  [--config h100] --mode exact [--variant V] [--bb_batch K] [--det 0|1] [--hybrid_gemm 0|1] [--json]   dry run: resolves and gates the mode on this box, prints the DRY-RUN line and the checkout / weights report; nothing is applied
#   run.sh warm   [--config h100] --mode exact [--variant V] [--bb_batch K] [--det 0|1] [--hybrid_gemm 0|1] [--keep] [--allow-partial] [--json]   one public-input design pass in the mode, its lines relayed
# V: the weight set, vanilla (default, upstream's) | soluble (modes.VARIANTS); upstream's own --use_soluble_model / --path_to_model_weights select the weights too, as protein_mpnn_run.py reads them.
# K: backbones per worker batch in a kit mode (default 16): speed and GPU memory only, the outputs do not depend on it.
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (the checkout MPNN_DIR, the target GPU and its memory total). Mode = --mode when given, else PROTEINMPNN_OPT from the environment; neither is a usage error (rc 2) whose one line names off|exact — the family
# default, fast, is a tier this engine does not ship, so it has no default mode; a --mode that disagrees with a set PROTEINMPNN_OPT is refused. `--det 0|1` is
# accepted and inert (both routes are deterministic by construction at the pass's own --seed). `--mode off` is the stock route and exists for design only
# (check/warm refuse it, rc 2). Names are validated by the package
# (modes.py, the one table; unknown or refused names exit 2 / 3). Every route runs on the installed package and the shared core it pins: the core pin
# gate is the first line of every route (a core absent or older than the kit's pin is its NOT ACTIVE line, rc 3). Exit codes are the package's (0 ok, 1 failed or
# incomplete, 2 usage, 3 the mode refused by name: what a pass cannot run without is absent, the checkout is at another commit than the pin, or a lever of the
# mode's line cannot engage here — no CUDA device (refused at activation), or the worker's start-up probe fails on this GPU (`hybrid_gemm: REFUSED`, nothing
# designed; `--hybrid_gemm 0` runs the line without that lever by name). A mode is all of its levers: it never runs under its name with a subset; `--allow-partial`
# is the one recorded override for levers that cannot run). A kit mode serves the design pass with every stock option (seed, sequence counts, temperatures, residue
# dictionaries, PSSM, any input form) and refuses by name, exit 3, nothing launched, the few passes its worker does not re-state (`--ca_only`, `--score_only`, the
# probability-only passes, `--tied_positions_jsonl`, `--backbone_noise` above 0): `--mode off` runs those.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,26p' "$0" >&2; exit 2; }
if [ "${1:-}" = install ]; then                                   # the install step: everything below it presupposes the installed package
  shift; WEIGHTS=""; IVAR=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights)   [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
      --weights=*) WEIGHTS=${1#*=}; [ -n "$WEIGHTS" ] || { echo "run.sh: install --weights= takes a directory" >&2; exit 2; }; shift ;;
      --variant)   [ $# -ge 2 ] && [ -n "$2" ] || { echo "run.sh: install --variant takes a name (soluble | vanilla)" >&2; exit 2; }; IVAR=$2; shift 2 ;;
      --variant=*) IVAR=${1#*=}; shift ;;
      *) echo "run.sh: install takes no argument '$1' (usage: run.sh install [--weights DIR])" >&2; exit 2 ;;
    esac
  done
  case "${IVAR:-soluble}" in soluble|vanilla) ;; *) echo "run.sh: install --variant $IVAR: the variants are soluble | vanilla (one checkout serves both weight sets)" >&2; exit 2 ;; esac
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md Install)" >&2; exit 3; }
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('proteinmpnn_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: proteinmpnn_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -c "from proteinmpnn_opt import core_gate; core_gate()" >/dev/null || exit 3     # the core pin gate: a shared core absent or older than the kit's pin is its one NOT ACTIVE line, rc 3; any core at or above the pin activates
  if [ -n "$WEIGHTS" ]; then                                         # upstream's own fetch into DIR, then the sha256 check against stock/PINS.json (opt/proteinmpnn_opt/weights.py)
    python -m proteinmpnn_opt.weights "$WEIGHTS" || exit $?
    export MPNN_DIR=$(cd "$WEIGHTS" && pwd)                          # the pin check below reads the directory just filled
  fi
  python -I "$HERE/stock/check_pins.py" --checks package || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the line above — MPNN_DIR must name the ProteinMPNN checkout: run.sh install --weights DIR makes one)" >&2; exit 3; }
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
    --) ARGS+=("$1"); shift; while [ $# -gt 0 ]; do ARGS+=("$1"); shift; done ;;
    *) if [ -z "$CMD" ]; then CMD=$1; else ARGS+=("$1"); fi; shift ;;
  esac
done
case "$CMD" in design|check|warm) ;; *) usage ;; esac
PREMODE=${PROTEINMPNN_OPT:-}; PREVAR=${PROTEINMPNN_VARIANT:-}          # what the environment held before the config was sourced
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config runs the same two probes first (the package importable, the core pin gate: NOT ACTIVE, rc 3)
fi
ENVMODE=${PROTEINMPNN_OPT:-}; ENVVAR=${PROTEINMPNN_VARIANT:-}
FROMMODE="the environment"; [ -n "$CFG" ] && [ "$ENVMODE" != "$PREMODE" ] && FROMMODE="configs/$CFG.env"     # the config is the source only when it set the variable
FROMVAR="the environment"; [ -n "$CFG" ] && [ "$ENVVAR" != "$PREVAR" ] && FROMVAR="configs/$CFG.env"
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with PROTEINMPNN_OPT=$ENVMODE from $FROMMODE; a run has one mode — drop one of them" >&2; exit 2
fi
MODE=${MODE:-$ENVMODE}
if [ "$MODE" = off ] && [ "$CMD" != design ]; then echo "run.sh: $CMD --mode off: the stock route is a design pass only (run.sh design --mode off ...)" >&2; exit 2; fi
if [ -n "$VARIANT" ] && [ -n "$ENVVAR" ] && [ "$VARIANT" != "$ENVVAR" ]; then
  echo "run.sh: --variant $VARIANT disagrees with PROTEINMPNN_VARIANT=$ENVVAR from $FROMVAR — drop one of them" >&2; exit 2
fi
VARIANT=${VARIANT:-$ENVVAR}
python -c "import proteinmpnn_opt" 2>/dev/null || { echo "[proteinmpnn-opt] NOT ACTIVE: proteinmpnn_opt is not installed on $(command -v python || echo 'python (not on PATH)'): $MODEL_OPT/run.sh install (= pip install -e $MODEL_OPT/../common/opt_core -e $MODEL_OPT/opt)" >&2; exit 3; }
python -c "from proteinmpnn_opt import core_gate; core_gate()" >/dev/null || exit 3     # the core pin gate, first on every route (the stock route included): a shared core absent or older than the kit's pin is its one NOT ACTIVE line, rc 3; any core at or above the pin activates
# [install]-jitcache — seed the compile caches shipped in the image; with no preset root and no image cache, a private per-user root (KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[proteinmpnn-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
}
if [ -d "$I" ] && [ -n "$(ls -A "$I" 2>/dev/null)" ]; then
  if [ -z "$J" ]; then if [ -w "$I" ]; then J="$I"; W="in-image"; elif seedroot; then J="$T"; W="seeded from image"; fi
  elif [ -z "$(ls -A "$J" 2>/dev/null)" ]; then W="seeded from image"; else W="user"; fi
  if [ "$W" = "seeded from image" ]; then [ -e "$J/.seeded" ] || { mkdir -p "$J" && cp -a "$I/." "$J/" && chmod -R u+w "$J" && { [ "$J" != "$T" ] || chmod 700 "$T"; } && find "$J" -name '__grp__*.json' -exec sed -i "s#$I/#$J/#g" {} + && touch "$J/.seeded"; } || W="unseeded"; fi   # cp -a gives the copy the image directory's mode: the per-user root keeps 0700
elif [ -z "$J" ] && seedroot; then J="$T"; export MODEL_OPT_JIT_ROOT="$J"; fi   # no preset root and no image cache: the private per-user root seedroot made or checked is this run's cache root, exported without a printed line; a refused one is named by seedroot and the run has no cache root
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[proteinmpnn-kit] jit cache: $J ($W)"; }
# [install]-jitcache — this kit compiles one thing at run time, the exact line's Triton draw kernel: under a JIT root its cache is keyed
# <root>/<stack key>/triton (opt_core.jit_cache names the key); with no root (the per-user one refused above) Triton keeps its own default location.
if [ -n "${MODEL_OPT_JIT_ROOT:-}" ] && [ -z "${TRITON_CACHE_DIR:-}" ]; then
  JK=$(python -I -c "from opt_core import jit_cache as j; print(j.key(strict=False))" 2>/dev/null) || JK=""
  export TRITON_CACHE_DIR="$MODEL_OPT_JIT_ROOT/${MODEL_OPT_STACK_KEY:-${JK:-unknown}}/triton"
fi
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
VARARG=(); [ -n "$VARIANT" ] && VARARG=(--variant "$VARIANT")
exec python -m proteinmpnn_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${VARARG[@]+"${VARARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
