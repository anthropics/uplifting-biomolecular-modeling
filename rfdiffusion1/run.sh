#!/bin/bash
# RFdiffusion-1 optimized — single entry point: a thin wrapper over `python -m rfdiffusion1_opt` (the installed rfdiffusion1_opt package on the shared core: `pip install -e ../common/opt_core -e opt`).
#   run.sh design [--config h100] [--mode off|exact|fast] KEY=VALUE ... [--pack K] [--det 0|1] [--dry-run]      one target per call: upstream's own hydra overrides (scripts/run_inference.py's tokens), nothing else
#   run.sh check  [--config h100] [--mode M] [KEY=VALUE ...] [--pack K] [--json]      dry run: resolves and gates the mode on this GPU, reports the upstream pin and the installed stack (stock/check_pins.py); nothing runs
#   run.sh warm   [--config h100] [--mode M] [KEY=VALUE ...] --out_dir <dir>   the first design of each bundled example target (the cold start, off the results, into a scratch dir; the design pass's exit code)
#   run.sh install [--weights DIR]      the shared core and this kit installed editable into the python on PATH (pip), then the pin check (stock/check_pins.py); --weights DIR also fetches the two
#                                       pinned checkpoints into DIR and checks their digests against stock/PINS.json (files already there are kept and checked) — DIR is then your WEIGHTS
# KEY=VALUE: upstream's hydra overrides verbatim — the target itself, exactly as scripts/run_inference.py takes it (`inference.input_pdb=t.pdb 'contigmap.contigs=[A1-115/0 80-80]'
# 'ppi.hotspot_res=[A59,A83,A91]' inference.output_prefix=out/des inference.num_designs=4 inference.design_startnum=0 ...`; outputs at the typed prefix as upstream writes them,
# the run record — opt_manifest.json, run.log, run_timings.json — beside them), and any other key.
# --mode off passes them to scripts/run_inference.py unchanged (the package's stock caller, opt/rfdiffusion1_opt/stock_cli.py: a clean subprocess that proves its environment);
# a kit mode composes them on its resident driver's configuration (potentials, noise schedules, partial diffusion, ... ride with the request) and refuses by name,
# exit 3, what its line cannot serve (symmetry, cyclic peptides, fold conditioning, sequence inpainting, ...: README.md "Notes"; `--mode off` runs those).
# --pack K: K resident workers of the kit line (exact or fast) on one GPU under CUDA MPS (common/mps_packing/mps_workers.sh); K is an axis, not a mode; refused by name on --mode off. --det 1: the deterministic
# recipe (upstream's per-design seed) on both arms; default 0. --config <cfg> sources configs/<cfg>.env: deployment parameters only (target GPU, JIT cache key and
# directory; the checkout RFD_ROOT and the weights WEIGHTS come from the environment — STOCK.md "Variables"). Mode = --mode when given, else RFDIFFUSION1_OPT from
# the environment, else `fast` (the default; `--mode exact` for byte-equal equality with off);
# a --mode that disagrees with a set RFDIFFUSION1_OPT is refused (rc 2). Modes are resolved by the package from its one mode table (opt/rfdiffusion1_opt/modes.py); this
# script validates none of them; an unknown name is refused by the package, by name (exit 3).
# Every route refuses (rc 3) unless the package is importable and its core pin gate passes (opt/rfdiffusion1_opt/_core_gate.py, the
# first probe below). Exit codes: 0 ok, 1 failed (outputs short of the request, whatever else happened; on --mode off the stock command line's own code), 2 usage, 3 not
# active / `check` would refuse / a lever on the stock path after a pass (`partial`: `[rfdiffusion1-opt] NOT ACTIVE: partial activation — <levers>: <reason>; exit 3`,
# the outputs kept and the levers named in opt_manifest.json).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,23p' "$0" >&2; exit 2; }
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
case "$CMD" in design|check|warm|install) ;; *) usage ;; esac
if [ "$CMD" != install ]; then                        # the routes — each ends in exec or exit; `install`, the one step that presupposes no installed package, is the block after them
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 3) when rfdiffusion1_opt is not importable or the core pin gate refuses (its probes are the two below, first)
fi
_jit_root_preset="${MODEL_OPT_JIT_ROOT:-}"   # the root configs/<cfg>.env derived TRITON_CACHE_DIR from, before the block below may move it off a read-only preset
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[rfdiffusion1-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[rfdiffusion1-kit] jit cache: $J ($W)"; }
if [ -n "${MODEL_OPT_JIT_ROOT:-}" ] && [ -n "${MODEL_OPT_STACK_KEY:-}" ] && { [ -z "${TRITON_CACHE_DIR:-}" ] || [ "${TRITON_CACHE_DIR:-}" = "${_jit_root_preset:-}/$MODEL_OPT_STACK_KEY/triton" ]; }; then export TRITON_CACHE_DIR=$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/triton; fi   # configs/h100.env's Triton cache line again, for the root the block chose: derived when unset, re-derived when it still points under the preset root the block moved away from; a value set elsewhere is kept
ENVMODE=${RFDIFFUSION1_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with RFDIFFUSION1_OPT=$ENVMODE from the environment; one invocation runs one mode — drop one of them" >&2; exit 2
fi
MODE=${MODE:-$ENVMODE}
env -u RFDIFFUSION1_OPT python -c "import rfdiffusion1_opt" 2>/dev/null || { echo "run.sh: rfdiffusion1_opt is not importable on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/../common/opt_core -e $HERE/opt" >&2; exit 3; }   # the variable unset on this probe alone: under a kit mode the installed .pth would run the core gate here, its line lost to /dev/null
python -c "import rfdiffusion1_opt as k; from rfdiffusion1_opt.report import TAG; from rfdiffusion1_opt._core_gate import gate; gate(k.__file__, tag=TAG)" >/dev/null || exit 3   # the core pin gate, the first gate of every route: an absent, older / newer or edited core prints its own `[rfdiffusion1-opt] NOT ACTIVE: reason=core_…` line; rc 3
if [ "$MODE" = off ] && [ "$CMD" = warm ]; then echo "run.sh: warm has nothing to do on --mode off (the stock command line has no resident state to warm)" >&2; exit 2; fi
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m rfdiffusion1_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
fi
# run.sh install [--weights DIR]: the shared core (../common/opt_core) and this kit (opt/) installed editable into the python on PATH, then the pin check —
# stock/check_pins.py: the RFdiffusion checkout pip resolves (or RFD_ROOT) byte for byte against the pinned archive, and torch / dgl against the pinned
# stack; a tree that is installed from these two directories already (the container image's /kit) says so and skips pip. --weights DIR then fetches the
# pinned checkpoints into DIR with their digests checked (opt/rfdiffusion1_opt/weights.py); nothing else reads or writes weights here.
[ -z "$CFG" ] && [ -z "$MODE" ] || { echo "run.sh: install takes no --config / --mode (usage: run.sh install [--weights DIR])" >&2; exit 2; }
WEIGHTS_DIR=""
set -- ${ARGS[@]+"${ARGS[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
    --weights)   [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR])" >&2; exit 2; }; WEIGHTS_DIR=$2; shift 2 ;;
    --weights=*) WEIGHTS_DIR=${1#--weights=}; [ -n "$WEIGHTS_DIR" ] || { echo "run.sh: install --weights= takes a directory (usage: run.sh install [--weights DIR])" >&2; exit 2; }; shift ;;
    *) echo "run.sh: install takes no argument '$1' (usage: run.sh install [--weights DIR])" >&2; exit 2 ;;
  esac
done
command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the Python 3.11 environment this kit is to be installed into (README.md 'Install')" >&2; exit 3; }
if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('rfdiffusion1_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
  echo "run.sh: rfdiffusion1_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
else
  python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's report above) — the shared core is expected at $HERE/../common/opt_core and the kit package at $HERE/opt" >&2; exit 1; }
fi
python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but the pin check refuses this environment (stock/check_pins.py, the lines above: the pinned RFdiffusion checkout, torch or dgl is not the pinned one — README.md 'Install')" >&2; exit 3; }
if [ -n "$WEIGHTS_DIR" ]; then python -m rfdiffusion1_opt.weights "$WEIGHTS_DIR" || exit $?; fi
exit 0
