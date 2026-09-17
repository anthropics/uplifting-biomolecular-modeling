#!/bin/bash
# PyTorch AlphaFold 3 port (xfold) + OpenFold3 weights, optimized — single entry point: a thin wrapper over `python -m af3_torch_opt` (the installed af3_torch_opt package: `pip install -e ../common/opt_core -e opt` — the tree's shared core plus this package).
#   run.sh pred    [--config h100] [--mode M] --output_dir <out> (--json_path <json> ... | --input_dir <dir>) [the stock CLI's flags: --model_dir --[no]run_data_pipeline --[no]run_inference --db_dir … --num_diffusion_samples N --fastnn|--nofastnn] [--num_recycles N] [--diffusion_steps N]   one process chain: [data pipeline +] featurise -> forward -> postprocess (inputs/1BRS.json: a ready example)
#   run.sh check   [--config h100] [--mode M] [--json] [--no-compile]                   dry run: resolves + gates the mode on this box; nothing is launched (--no-compile, here and on pred: the mode without the kit's torch.compile lever — an alias of MODEL_OPT_LEVERS_OFF=compile; the ACTIVE line says compile=on|off:user|off:mode)
#   run.sh warm    [--config h100] [--mode M|all] [<fold input json> ...]                 the one-time kernel and cache setup, ahead of the first pred: one short prediction per mode (all = every mode in turn) over synthetic single-chain inputs of 448 / 832 / 1216 tokens, or over the named inputs — fills the Triton / Inductor / JAX caches under AF3_TORCH_CACHE_ROOT; one WARM line per mode (cache files before / after, seconds); a second run finds them warm
#   run.sh stock   [--config h100] --output_dir <out> (--json_path <json> ... | --input_dir <dir>)   the stock route: xfold's own CLI (run_alphafold.py, the pinned archive's bytes) at its shipped defaults on the composed stock venv ($AF3_TORCH_STOCK_PY, opt/af3_torch_opt/stock_venv.py); no mode, no lever
#   run.sh install [--weights DIR [--fetch]]                                             the install step: the shared core and this kit installed editable into the python on PATH (pip install -e ../common/opt_core -e opt), xfold's pinned archive made present under stock/ (stock/fetch_upstream.py: fetched from GitHub at the pinned commit when the tree arrived without it), then the pin check (stock/check_pins.py: the two interpreters AF3_TORCH_PY / AF3_TORCH_JAX_PY at their pins); --weights DIR checks the converted checkpoint under DIR against stock/PINS.json — with --fetch, a DIR without one gets it: the public OpenFold3-preview2 checkpoint is downloaded and converted with the reference fork's converter (README.md 'Install') — DIR is then your AF3_TORCH_PARAMS_DIR
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (the two interpreters, the parameters dir, the cache root, the target GPU).
# Modes: off | exact | fast | big (the memory line). Mode = --mode when given, else AF3_TORCH_OPT from the environment, else the package default (opt/af3_torch_opt/modes.py, the one mode
# table); a --mode that disagrees with a set AF3_TORCH_OPT is refused. Modes are validated by the package alone (modes.py resolves a mode
# to one of the kit's own lever sets, opt/forward/af3t/af3_torch/af3_torch_api.py LEVER_SETS); this script carries no table and validates
# none. `--mode off` is stock on this route (xfold as shipped: the kit's eager set with xfold's fastnn kernels on, no padding, no DTK, the same 5-sample form, in a model process
# whose environment carries no AF3_TORCH_OPT* variable — the STOCK line proves it); the stock route is `run.sh stock` (xfold's own CLI). One variant (the OpenFold3-preview2 parameters at AF3_TORCH_PARAMS_DIR): no --variant switch.
# Every route refuses (rc 3) unless the pinned stack is installed (stock/check_pins.py: the two interpreters' pinned packages — its
# diagnostic line is printed above the refusal) and the package is installed. Exit codes: 0 ok, 1 failed, 2 usage, 3 not active / pins not met / partial (a degraded lever set; --allow-partial accepts it, said on the DONE line).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE                          # the tree root for the package (stack.home())
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
case "$CMD" in pred|check|stock|install|warm) ;; *) usage ;; esac
if [ "$CMD" = install ]; then                                       # the install step: everything below it presupposes the installed package
  WEIGHTS=""; FETCH=()
  [ -z "$CFG" ] && [ -z "$MODE" ] || { echo "run.sh: install takes no --config / --mode (usage: run.sh install [--weights DIR [--fetch]])" >&2; exit 2; }
  set -- ${ARGS[@]+"${ARGS[@]}"}
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR [--fetch]])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
      --weights=*) WEIGHTS=${1#*=}; [ -n "$WEIGHTS" ] || { echo "run.sh: install --weights= takes a directory" >&2; exit 2; }; shift ;;
      --fetch) FETCH=(--fetch); shift ;;
      *) echo "run.sh: install takes no argument '$1' (usage: run.sh install [--weights DIR [--fetch]])" >&2; exit 2 ;;
    esac
  done
  [ ${#FETCH[@]} -eq 0 ] || [ -n "$WEIGHTS" ] || { echo "run.sh: install --fetch goes with --weights DIR (the directory the checkpoint is fetched and converted into)" >&2; exit 2; }
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md Install)" >&2; exit 3; }
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('af3_torch_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: af3_torch_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/fetch_upstream.py" || { echo "run.sh: the install stopped at xfold's pinned archive (stock/fetch_upstream.py: the line above names the remedy)" >&2; exit 1; }   # present in the tree, or fetched from GitHub at the pinned commit
  env -u AF3_TORCH_OPT python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the line above — AF3_TORCH_PY / AF3_TORCH_JAX_PY must name the two interpreters of the pinned stack, README.md Variables)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then env -u AF3_TORCH_OPT python -m af3_torch_opt.weights "$WEIGHTS" ${FETCH[@]+"${FETCH[@]}"} || exit $?; fi   # the converted checkpoint under DIR checked against stock/PINS.json (opt/af3_torch_opt/weights.py); --fetch downloads + converts it when DIR has none
  exit 0
fi
# The seed block runs before the config is sourced: configs/<card>.env derives AF3_TORCH_CACHE_ROOT from the MODEL_OPT_JIT_ROOT it exports.
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[af3_torch-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[af3_torch-kit] jit cache: $J ($W)"; }
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 3) through the package entry route: not installed / core_missing / producer_missing / pins
fi
FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
ENVMODE=${AF3_TORCH_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ] && ! { [ "$CMD" = warm ] && [ "$MODE" = all ]; }; then   # warm --mode all names no single mode: every mode in turn, nothing to disagree with
  echo "run.sh: --mode $MODE disagrees with AF3_TORCH_OPT=$ENVMODE from $FROM; one run has one mode — drop one of them" >&2; exit 2
fi
CLIMODE=$MODE; MODE=${MODE:-$ENVMODE}
# the two helper probes run without AF3_TORCH_OPT: the mode variable is judged by the wrapper below (an unknown selection = its NOT ACTIVE line, exit 3), never by a probe
# the install probe: `not installed` ONLY for a genuine ModuleNotFoundError of the package; any other failure of the import (a broken tree, a
# refusal at interpreter start) prints the interpreter's own words verbatim and exits with ITS rc — the probe never renames a refusal
PROBE_RC=0; PROBE_ERR=$(env -u AF3_TORCH_OPT python -c "import af3_torch_opt" 2>&1 >/dev/null) || PROBE_RC=$?
if [ "$PROBE_RC" -ne 0 ]; then
  if printf '%s' "$PROBE_ERR" | grep -q "ModuleNotFoundError: No module named 'af3_torch_opt'"; then
    echo "run.sh: af3_torch_opt is not installed on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/../common/opt_core -e $HERE/opt" >&2; exit 3
  fi
  printf '%s\n' "$PROBE_ERR" >&2; echo "run.sh: \`import af3_torch_opt\` failed on $(command -v python || echo python) (rc $PROBE_RC; its words above)" >&2; exit "$PROBE_RC"
fi
# the env route (the mode from AF3_TORCH_OPT, none on the command line) needs the fail-loud hook LIVE in this interpreter: a fresh `python`
# (the route's own form — no -I: the exec line below runs `python` without it, so a user-site install counts here as it does there)
# must hold af3_torch_opt._autoload in sys.modules at start (the site-processed af3_torch_opt_autoload.pth's effect — a direct import of the
# kit's api under the variable then refuses instead of running stock); a package merely importable (a path .pth, PYTHONPATH) has no live hook
# -> refuse by name, the diagnostic in the line (present but not processed / a copy beside a PYTHONPATH entry / absent from the searched sites).
# The --mode route is not gated: the wrapper activates the named mode in-process (its ACTIVE line is the evidence); off and stock need no hook.
if [ -n "$ENVMODE" ] && [ "$ENVMODE" != off ] && [ -z "$CLIMODE" ] && [ "$CMD" != stock ]; then
  DIAG=$(env -u AF3_TORCH_OPT python - "$HERE/opt/af3_torch_opt_autoload.pth" <<'PYH'
import os, site, sys
if "af3_torch_opt._autoload" in sys.modules: sys.exit(0)
F = "af3_torch_opt_autoload.pth"; want = open(sys.argv[1], "rb").read().strip()
sites = list(dict.fromkeys(site.getsitepackages() + [site.getusersitepackages()]))
present = [d for d in sites if os.path.isfile(os.path.join(d, F))]
beside = [d for d in (os.environ.get("PYTHONPATH") or "").split(os.pathsep) if d and os.path.isfile(os.path.join(d, F))]
stale = [d for d in present + beside if open(os.path.join(d, F), "rb").read().strip() != want]
if stale: print(F + " is a stale copy (its line differs from the kit's " + sys.argv[1] + "): " + ", ".join(stale))
elif present: print(F + " present but not processed (in a site dir site.py did not process at start): " + ", ".join(present))
elif beside: print(F + " present but not processed (a copy beside a PYTHONPATH entry, not a site dir): " + ", ".join(beside))
else: print(F + " absent from the searched sites: " + ", ".join(sites))
sys.exit(1)
PYH
) || { echo "run.sh: the env route (AF3_TORCH_OPT=$ENVMODE) refused: the fail-loud hook af3_torch_opt._autoload is not live in $(command -v python) — $DIAG; the package is importable but not installed there: pip install -e $HERE/../common/opt_core -e $HERE/opt — or name the mode on the command line (--mode $ENVMODE)" >&2; exit 3; }
fi
env -u AF3_TORCH_OPT python -I "$HERE/stock/check_pins.py" --quiet >&2 || { echo "run.sh: pins not met (the line above; stock/check_pins.py)" >&2; exit 3; }
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
[ "$CMD" = stock ] && MODEARG=()                                    # stock is xfold's own CLI: it takes no mode
exec python -m af3_torch_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
