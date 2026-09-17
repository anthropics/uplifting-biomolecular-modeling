#!/bin/bash
# colabdesign optimized — single entry point: a thin wrapper over `python -m colabdesign_opt` (the installed colabdesign_opt package and the shared core it pins: `pip install -e ../common/opt_core -e opt`).
#   run.sh design    [--config h100] [--mode M] --starting-pdb <pdb> --chains A[,B,C] --binder-len L [--target-hotspot-residues SPEC] [--seed S] --out <dir> [--params-dir D] [--binder-name NAME] [--advanced FILE] [--filters FILE]
#                                                          one binder design (BindCraft's binder_hallucination) in a subprocess; design.pdb, design.fasta, trajectory.jsonl and the log run.log under --out
#   run.sh check     [--config h100] [--mode M] [--json] [--params-dir D]                        dry run: the mode resolved on this machine (pins, core pin, GPU) and the weights' digests; nothing is designed
#   run.sh warm      [--config h100] [--mode M] --out <dir> [--params-dir D] [--starting-pdb <pdb> --chains C --binder-len L] [--seed S]
#                                                          BindCraft's bundled example (or the case named) designed once under <out>/warm/: pays the first-run compiles and reports the one-time costs
#   run.sh install   [--weights DIR]                          the install step: the shared core and this kit installed editable into the python on PATH, BindCraft's DSSP executable fetched from its repository at the pinned commit and sha256-checked (stock/check_pins.py --fetch), then the pin check (stock/check_pins.py --no-gpu);
#                                                          --weights DIR also fetches the five AlphaFold-Multimer v3 parameter files into DIR/params/ and checks them against stock/PINS.json (sha256) — DIR is then your COLABDESIGN_PARAMS_DIR
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (the tree, the params root, the target GPU). Mode = --mode
# when given, else COLABDESIGN_OPT from the environment, else the package default (fast); a --mode that disagrees with a set COLABDESIGN_OPT is
# refused. Modes (off|exact|fast, named by guarantee, plus the subtractive word <mode>-no-<lever>; any other word is unknown → rc 3) are the package's table
# (opt/colabdesign_opt/modes.py, lever sets from registry.py); this script validates none of them. `--mode off` is the stock route: the design script in a subprocess
# proven clean of all the kit's variables and modules. Every route refuses (rc 3) when colabdesign is not installed at its pinned commit (stock/check_pins.py
# --no-stack: another commit changes what "stock" means) or the package and the core are not installed — configuration. A LEVER that cannot engage on this host
# steps aside by name (`LEVER name=<id> state=skipped reason=…`; `skipped=` on the ACTIVE line) and the mode runs the rest of its set.
# The rest of the environment is named on the ACTIVE line, never refused: any other card, or one below the kernel's
# compute-capability floor, is named (`gpu=`; the levers engage), a stack distribution off its pin is `stack_drift=`.
# Exit codes: 0 ok, 1 failed (or the file set short of the request: `incomplete`; the install; a weights file off its pin), 2 usage, 3 not active
# (the pinned upstream commit or the kit's install not met, an unknown mode, a lever of the mode that cannot run here) or partial (a lever of
# the mode did not engage in the arm: refused by name, no opt-out).
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
    --)          ARGS+=("$1"); shift; while [ $# -gt 0 ]; do ARGS+=("$1"); shift; done ;;
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
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('colabdesign_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: colabdesign_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" --fetch || { echo "run.sh: the install stops: BindCraft's DSSP executable is not in place as pinned (the lines above) — this tree does not carry it; the install fetches it from the BindCraft repository at the pinned commit into stock/src/bindcraft/functions/ (stock/PINS.json upstream.bindcraft.fetched; STOCK.md 'Pin' has the by-hand route)" >&2; exit 1; }   # sha256-checked before it is placed or kept; network needed once
  python -I "$HERE/stock/check_pins.py" --no-gpu || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the line above — the pinned upstream is not installed as pinned in this environment)" >&2; exit 3; }   # the installed software, not the box: the card is the routes' check
  if [ -n "$WEIGHTS" ]; then python -m colabdesign_opt.weights "$WEIGHTS" || exit $?; fi   # the five parameter files into DIR/params/, then the sha256 check against stock/PINS.json (opt/colabdesign_opt/weights.py)
  exit 0
fi
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 3) when colabdesign_opt is not importable
fi
KIT=colabdesign   # [install]-jitcache: the tag on the seed block's one line
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
# [install]-jitcache mapping: the compilecache lever adopts jax's own JAX_COMPILATION_CACHE_DIR (the stock arm strips it, so `off` still
# compiles cold); derived from the root only when a root is set and the variable is not: <root>/<stack key>/xla, the key naming the pinned
# jax stack and the visible card; a root that cannot be created or written (probed by writing a file: a read-only volume mount can pass
# `test -w` for root and still refuse the write) leaves the variable unset — the lever's own default directory
if [ -n "${MODEL_OPT_JIT_ROOT:-}" ] && [ -z "${JAX_COMPILATION_CACHE_DIR:-}" ] && mkdir -p "$MODEL_OPT_JIT_ROOT" 2>/dev/null && { : > "$MODEL_OPT_JIT_ROOT/.w.$$"; } 2>/dev/null; then
  rm -f "$MODEL_OPT_JIT_ROOT/.w.$$"
  G=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) || G=""          # the card's product name, empty without a working nvidia-smi
  G=$(printf '%s\n' "$G" | head -n1 | tr 'A-Z' 'a-z' | tr -cs 'a-z0-9' '_' | sed 's/^_*//; s/_*$//')
  export JAX_COMPILATION_CACHE_DIR="$MODEL_OPT_JIT_ROOT/jax0.6.0-jaxlib0.6.0-cuda12plugin0.6.0-${G:-nogpu}/xla"
fi
FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
ENVMODE=${COLABDESIGN_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with COLABDESIGN_OPT=$ENVMODE from $FROM; one run has one mode — drop one of them" >&2; exit 2
fi
MODE=${MODE:-$ENVMODE}
python -I "$HERE/stock/check_pins.py" --quiet --no-gpu --no-stack || { echo "run.sh: colabdesign is not installed at its pinned commit (stock/PINS.json upstream.colabdesign) — another commit changes what stock means" >&2; exit 3; }   # the stack and the card are named by the package on its ACTIVE line (stack_drift=, gpu=), never refused
python -c "import importlib.util, sys; sys.exit(4) if importlib.util.find_spec('colabdesign_opt') is None else None; import colabdesign_opt, colabdesign_opt._core_gate as g; g.gate(list(colabdesign_opt.__path__)[0], tag=colabdesign_opt.TAG)" || { [ $? = 3 ] || echo "[colabdesign-opt] NOT ACTIVE: package_missing:colabdesign_opt on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/../common/opt_core -e $HERE/opt" >&2; exit 3; }   # the package AND its core pin gate (colabdesign_opt/_core_gate.py: NOT ACTIVE reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …, rc 3)
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m colabdesign_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}     # the entry runs the same gate as its first statement, then names any core sub-module the pinned core lacks (`NOT ACTIVE: core_missing:<module>`, rc 3) — never a traceback
