#!/bin/bash
# Proteina-Complexa — single entry point: a thin wrapper over `python -m complexa_opt` (the installed complexa_opt package and the core it pins).
#   run.sh install [--weights DIR]                             the install step: the shared core and this kit installed editable (pip install -e ../common/opt_core -e opt, the core first: opt/pyproject.toml [tool.opt_core]) into the python on PATH —
#                                                              the interpreter that runs upstream's `complexa` — then the pin check (stock/check_pins.py); --weights DIR also fetches complexa.ckpt + complexa_ae.ckpt into DIR with upstream's own
#                                                              downloader and checks both against stock/PINS.json (sha256) — DIR is then your CKPT_PATH
#   run.sh design --out <dir> [--mode M] [--config h100] [--input <entry.json>] [-- <upstream's `complexa generate` arguments: Hydra overrides + options, verbatim>]
#                                                              one generation run: <out>/inference/search_binder_local_pipeline_<item>_<run>/job_*/…pdb (the designs, upstream's own layout), <out>/logs (upstream's),
#                                                              plus opt_manifest.json, design.log and stock_env_proof.json (off) or kit_records/ (exact | fast | big)
#   run.sh check  [--config h100] [--mode M]                   dry run: pins, checkout, weights, GPU vs target, the mode's route and levers, the hook — one DRY-RUN line; nothing runs
# M: the package's mode names (modes.MODES: off | exact | fast | big; `python -m complexa_opt design --help` lists them and each mode's levers). `off` = stock: upstream's
# `complexa generate` on the shipped pipeline configuration in a proven-clean child, nothing added. `exact` / `fast` / `big` = the SAME command with COMPLEXA_OPT=<mode> in
# the child's environment: the install's autoload hook (complexa_opt_autoload.pth) then installs the mode's levers inside upstream's generation process (CHANGES.md: exact =
# byte-identical designs, faster, lower peak memory; big = byte-identical at the lowest peak memory; fast = tolerance class, the fastest); a lever that cannot be
# installed refuses the whole mode by name (exit 3). No --mode (and no COMPLEXA_OPT) asks for the default mode `fast`. What a run computes is upstream's own
# to say: everything after `--` is passed to `complexa generate` verbatim and last — any Hydra key and upstream's `--job-id N` — e.g. the generation stage alone (one pass of the
# model, no reward model scored), 32 designs at seed 5, dataloader batch 32, run name r1 (none given: the shipped configuration — best-of-n search scored by upstream's reward model,
# 4 designs, seed 5, batch 16):  -- ++run_name=r1 ++generation.search.algorithm=single-pass ++generation.reward_model=null ++generation.dataloader.dataset.nres.nsamples=32 ++seed=5 ++generation.dataloader.batch_size=32
# --input names ONE target entry in upstream's targets_dict form (optional: without it the target is the overrides' `++generation.task_name=…` or the configuration's own); a run
# is reproducible as shipped (upstream seeds every RNG itself from `++seed`). --config <cfg> sources configs/<cfg>.env: deployment parameters only
# (upstream's own variables, the JIT caches, the target GPU; LOCAL_CODE_PATH and CKPT_PATH come from your environment — STOCK.md 'Variables'). Mode = --mode when given, else
# COMPLEXA_OPT from the environment, else fast; a --mode that disagrees with a set COMPLEXA_OPT is refused. Every route runs on the installed package and the shared
# core it pins: the core pin gate is the first line of every route (an absent or mismatched core is its NOT ACTIVE line, rc 3). Exit codes are the package's (0 ok, 1 failed
# or incomplete, 2 usage — one named `USAGE refused:` line for the kit's OWN words; upstream's arguments after `--` are never judged here, 3 not active: the core gate,
# pins/checkout/weights not as pinned, a stock process that is NOT STOCK, the hook absent from upstream's interpreter, a mode whose levers could not all be installed or did not
# all engage; install: 1 = the pip step failed or a weight file is off its pin, 3 = the pin check refused).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,25p' "$0" >&2; exit 2; }
if [ "${1:-}" = install ]; then                                      # the install step: everything below it presupposes the installed package
  shift; WEIGHTS=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
      --weights=*) WEIGHTS=${1#*=}; [ -n "$WEIGHTS" ] || { echo "run.sh: install --weights= takes a directory (usage: run.sh install [--weights DIR])" >&2; exit 2; }; shift ;;
      *) echo "run.sh: install takes no argument '$1' (usage: run.sh install [--weights DIR])" >&2; exit 2 ;;
    esac
  done
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md 'Install')" >&2; exit 3; }
  if python -I -c "import os,sys,site,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('complexa_opt','opt_core')]; pth=any(os.path.isfile(os.path.join(d,'complexa_opt_autoload.pth')) for d in site.getsitepackages()+[site.getusersitepackages()]); sys.exit(0 if pth and all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: complexa_opt and opt_core are installed from this tree already, the autoload hook laid ($HERE/opt, $HERE/../common/opt_core, complexa_opt_autoload.pth) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the line above — the pinned upstream is not installed as pinned in this environment; LOCAL_CODE_PATH names its checkout)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then python -m complexa_opt.weights "$WEIGHTS" || exit $?; fi   # upstream's downloader into DIR, then the sha256 check against stock/PINS.json (opt/complexa_opt/weights.py)
  exit 0
fi
CFG=""; CMD=""; MODE=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config)    [ $# -ge 2 ] || usage; CFG=$2; shift 2 ;;
    --config=*)  CFG=${1#--config=}; shift ;;
    --mode)      [ $# -ge 2 ] || usage; MODE=$2; shift 2 ;;
    --mode=*)    MODE=${1#--mode=}; shift ;;
    --) ARGS+=("$1"); shift; while [ $# -gt 0 ]; do ARGS+=("$1"); shift; done ;;
    *) if [ -z "$CMD" ]; then CMD=$1; else ARGS+=("$1"); fi; shift ;;
  esac
done
case "$CMD" in design|check) ;; *) usage ;; esac
probe_err=$(python -c "import complexa_opt" 2>&1 >/dev/null) || { case "$probe_err" in *"[complexa-opt] NOT ACTIVE"*) echo "$probe_err" >&2 ;; *) echo "[complexa-opt] NOT ACTIVE: complexa_opt is not installed on $(command -v python || echo 'python (not on PATH)'): $HERE/run.sh install (= pip install -e $HERE/../common/opt_core -e $HERE/opt)" >&2 ;; esac; exit 3; }   # the install's autoload hook refuses a stray COMPLEXA_OPT* name at interpreter start: its own line, not "not installed"
python -c "from complexa_opt import core_gate; core_gate()" >/dev/null || exit 3        # the core pin gate: its one NOT ACTIVE line on stderr, rc 3, before any config probe
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no configs/$CFG.env (configs: $(ls "$HERE/configs" | sed 's/\.env$//' | tr '\n' ' '))" >&2; exit 2; }
  # shellcheck disable=SC1090
  source "$HERE/configs/$CFG.env"
fi
MODE_ARGS=(); [ -n "$MODE" ] && MODE_ARGS=(--mode "$MODE")
exec python -m complexa_opt "$CMD" "${MODE_ARGS[@]}" "${ARGS[@]}"
