#!/bin/bash
# ESM-IF1 — single entry point: a thin wrapper over `python -m esm_if1_opt` (the installed esm_if1_opt package and the shared core it pins).
#   run.sh install [--weights DIR]              the install step: the shared core and this kit installed editable into the `python` on PATH (pip install -e
#                                               ../common/opt_core -e opt, the core first: opt/pyproject.toml [tool.opt_core]) — a tree already installed from here says so and
#                                               skips pip — then the pin check (python -I stock/check_pins.py: fair-esm is the pinned commit); --weights DIR also fetches the one
#                                               checkpoint into DIR with the transfer upstream's loader uses (torch.hub) and checks it against stock/PINS.json (bytes, sha256):
#                                               DIR/esm_if1_gvp4_t16_142M_UR50.pt is then your ESM_IF1_WEIGHTS
#   run.sh design [--config h100|a100|h200] [--mode off|fast] (PDBFILE | --input <dir|file>) [--out <dir> | --outpath FILE] [--chain C] [--temperature T]
#                 [--num-samples N] [--multichain-backbone] [--nogpu] [--seed N] [--batch_size B] [--det [0|1]]
#   run.sh check  [--config h100|a100|h200] [--mode M]   what the tree carries and what this box has (upstream pin, fair-esm, torch, the weights file, the GPU); nothing runs
# M (modes.MODES = off|fast): `off` is stock — upstream's sample_sequences.py as shipped, one interpreter per structure, unseeded, in a
# proven-clean child; `fast` (the default when --mode and ESM_IF1_OPT are both absent) is the kit's one tier — batched sampling: the unmodified
# upstream modules over --batch_size (backbone, sample) rows per forward, the model loaded once, torch.manual_seed(--seed) once (README, Modes);
# any other mode word is a usage error: exit 2 with one line naming the modes (never a silent fallback). --seed (default 37)
# and --batch_size (default 64; --batch is accepted too) are fast's inputs and are named NOT APPLIED under off; --det never applies here
# (named NOT APPLIED). The other arguments are upstream sample_sequences.py's own, names and defaults unchanged (pdbfile, --chain, --temperature 1.0,
# --outpath, --num-samples 1, --multichain-backbone, --nogpu); --input / --out are the directory forms. --config <cfg> sources configs/<cfg>.env:
# deployment parameters only (state directory, TORCH_HOME, target GPU).
# Exit: 0 ok · 1 failed / incomplete (a pass, the pip step, a weights file off its pin) · 2 usage (incl. an unknown mode word) · 3 NOT STOCK (a stock child's
# environment did not prove clean), the mode cannot run here (esm_if1_opt / opt_core / fair-esm / torch not installed, the tree's stock/ or opt/ absent), or
# install's pin check refused the installed fair-esm (not the pinned commit).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MODEL_OPT="$HERE"
usage() { sed -n '2,21p' "$0" >&2; exit 2; }
if [ "${1:-}" = install ]; then
  shift; WEIGHTS_DIR=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights) [ $# -ge 2 ] && [ -n "$2" ] || usage; WEIGHTS_DIR="$2"; shift 2 ;;
      --weights=*) WEIGHTS_DIR="${1#--weights=}"; [ -n "$WEIGHTS_DIR" ] || usage; shift ;;
      *) usage ;;
    esac
  done
  # A tree already installed editable from HERE (the container image's /kit) names the fact and skips pip: a read-only image cannot rerun it, and need not.
  if python -I -c "import importlib.util as u, os, sys
want = {'esm_if1_opt': sys.argv[1], 'opt_core': sys.argv[2]}
for name, root in want.items():
    s = u.find_spec(name)
    if s is None or not s.origin or not os.path.realpath(s.origin).startswith(os.path.realpath(root) + os.sep):
        sys.exit(1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "[esm_if1-opt] install: esm_if1_opt and opt_core are already installed from this tree ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "[esm_if1-opt] install FAILED: pip install -e ../common/opt_core -e opt (exit $?)" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || exit $?
  if [ -n "$WEIGHTS_DIR" ]; then python -m esm_if1_opt.weights "$WEIGHTS_DIR" || exit $?; fi
  exit 0
fi
CMD=""; CFG=""; MODE=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config) [ $# -ge 2 ] || usage; CFG="$2"; shift 2 ;;
    --config=*) CFG="${1#--config=}"; shift ;;
    --mode) [ $# -ge 2 ] || usage; MODE="$2"; shift 2 ;;
    --mode=*) MODE="${1#--mode=}"; shift ;;
    -h|--help) usage ;;
    --) shift; while [ $# -gt 0 ]; do ARGS+=("$1"); shift; done ;;
    *) if [ -z "$CMD" ] && [[ "$1" != -* ]]; then CMD="$1"; else ARGS+=("$1"); fi; shift ;;
  esac
done
case "$CMD" in design|check) ;; *) usage ;; esac
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no configs/$CFG.env (have: $(ls "$HERE/configs" | sed 's/\.env$//' | tr '\n' ' '))" >&2; exit 2; }
  # shellcheck disable=SC1090
  source "$HERE/configs/$CFG.env" || exit 2
fi
python -c 'import esm_if1_opt' 2>/dev/null || { echo "[esm_if1-opt] NOT ACTIVE: esm_if1_opt is not installed on $(command -v python || echo 'python (not on PATH)'): bash $HERE/run.sh install" >&2; exit 3; }
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m esm_if1_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
