#!/bin/bash
# Protenix v2 optimized — single entry point: a thin wrapper over `python -m protenix_opt` (the installed protenix_opt package: `pip install -e opt`).
#   run.sh pred   [--config h100] [--mode exact|fast|big|off] [--n_gpu P] [--det 0|1] <the stock `protenix pred` arguments>   drop-in: same inputs and outputs (stock's files, nothing else);
#                 --n_gpu P: the GPU count, explicit, default 1 (P > 1 under exact/fast/off is refused by name; fewer than P visible GPUs is refused by name); --mode big --n_gpu P (P > 1) = the multi-GPU LINE of big: P torchrun ranks, tensor-parallel over the pair rows (opt/forward/PTX_TP), every rank on the fast levers;
#                 the world size, the ranks' completion and the row-shard map are printed (EXECUTION line) and gate the exit (a run short of P ranks exits 1 by name); with --n_gpu 1 (or no --n_gpu) big is its single-GPU line; ACTIVE/FINAL/EXIT lines carry n_gpu=P sharding=rowpair|none;
#                 --det 1 = the kit's deterministic recipe on both arms (CUBLAS_WORKSPACE_CONFIG=:4096:8 PTX_DET=1 + the detref scatter copy, restored at the end): off and exact then produce byte-identical outputs;
#                 a partial activation (a lever of the mode fallen back, at activation or in the kit's end-of-run records) exits 3 by name — a mode is all of its levers on the card, never a subset under its name; an untested environment is named on the activation line, not refused;
#                 fewer output files than requested (entries x seeds x samples) is `incomplete` on the OUTPUTS line and exits 1
#   run.sh warm   [--config h100] [--mode exact|fast|big|off] [--det 0|1] [--out_dir DIR]   the JIT warm-up: one small pred on the shipped example input (opt/protenix_opt/warm_input.json; cycle 1, step 2, sample 1, no MSA) so the mode's kernels are compiled before the first real call; exit codes are pred's
#   run.sh install [--weights DIR] [pip arguments]   the install step: the shared core (../common/opt_core) and this kit (opt) installed editable into the `python` on PATH (extra arguments go to pip; already installed from this tree and no pip
#                 arguments: the pip step is skipped), then the pin check (stock/check_pins.py); --weights DIR then fetches the checkpoint and the six stock data caches into DIR with upstream's downloader and checks them against stock/PINS.json — DIR is the PROTENIX_ROOT_DIR of every route
#   run.sh check  [--config h100] [--mode exact|fast|big|off] [--n_gpu P] [--det 0|1]   the pin / version / GPU gates and the activation line, with --det 1 the recipe's preconditions (DET lines), + dry run: resolves and gates the mode on this GPU, prints the DRY-RUN line; nothing is applied; a partial dry run exits 3 by name
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (weights root, JIT cache dirs, target GPU, LayerNorm kernel). Mode = --mode when given, else
# PROTENIX_OPT from the environment, else the package default (fast); a --mode that disagrees with a set PROTENIX_OPT is refused. Exit codes are the
# package's, one per condition across verbs: 0 ok | 1 the verb's own check failed (incomplete outputs, DET, console script) | 2 usage |
# 3 the mode did not activate, or only part of its lever set could run on this card — refused by name | pred otherwise passes the stock exit code through.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,13p' "$0" >&2; exit 2; }
CFG=""; CMD=""; MODE=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config)   [ $# -ge 2 ] || usage; CFG=$2; shift 2 ;;
    --config=*) CFG=${1#--config=}; shift ;;
    --mode)     [ $# -ge 2 ] || usage; MODE=$2; CLI_MODE=$2; shift 2 ;;
    --mode=*)   MODE=${1#--mode=}; CLI_MODE=$MODE; shift ;;
    *) if [ -z "$CMD" ]; then CMD=$1; else ARGS+=("$1"); fi; shift ;;
  esac
done
case "$CMD" in pred|check|warm|install) ;; *) usage ;; esac
if [ "$CMD" = install ]; then                                       # the install step: everything below it presupposes the installed package
  WEIGHTS=""; PIPARGS=()
  [ -z "$CFG" ] && [ -z "$MODE" ] || { echo "run.sh: install takes no --config / --mode (usage: run.sh install [--weights DIR] [pip arguments])" >&2; exit 2; }
  set -- ${ARGS[@]+"${ARGS[@]}"}
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR] [pip arguments])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
      --weights=*) WEIGHTS=${1#*=}; [ -n "$WEIGHTS" ] || { echo "run.sh: install --weights= takes a directory" >&2; exit 2; }; shift ;;
      *) PIPARGS+=("$1"); shift ;;                                  # anything else is pip's (an index, a proxy, -v …), as `pip install` would take it
    esac
  done
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md Install)" >&2; exit 3; }
  if [ ${#PIPARGS[@]} -eq 0 ] && python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('protenix_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: protenix_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" ${PIPARGS[@]+"${PIPARGS[@]}"} || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the line above — the pinned upstream is not installed as pinned in this environment)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then python -m protenix_opt.weights "$WEIGHTS" || exit $?; fi   # upstream's downloader into DIR, then the sha256 check against stock/PINS.json (opt/protenix_opt/weights.py)
  exit 0
fi
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[protenix_v2-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[protenix_v2-kit] jit cache: $J ($W)"; }
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses by name: rc 2 when protenix_opt is not installed, the interpreter's own code when it refuses to start (an undeclared PROTENIX_OPT* name: 3)
fi
ENVMODE=${PROTENIX_OPT:-}; FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with PROTENIX_OPT=$ENVMODE from $FROM; a run has one mode — drop one of them" >&2; exit 2
fi
MODE=${MODE:-$ENVMODE}
case "$MODE" in ""|exact|fast|big|off) ;; *) echo "run.sh: PROTENIX_OPT=$MODE from $FROM is not a mode (exact|fast|big|off)" >&2; exit 2 ;; esac
# Presence probe (find_spec executes nothing of the package): exit 0 = importable; 41 = genuinely absent -> the 'not installed' word, rc 2; 127 = no python on PATH,
# rc 2. Any OTHER exit is the interpreter's own refusal at start-up (the kit's hook: an undeclared PROTENIX_OPT* name, exit 3) - its words are printed
# verbatim and run.sh exits with ITS code, never masked as 'not installed'. An absent or older shared core is refused by name by the entry itself
# (`python -m protenix_opt`, exit 3), not here.
_probe_rc=0; _probe_err=$(env -u PROTENIX_OPT -u PROTENIX_OPT_N_GPU -u PROTENIX_OPT_TP_ROUTE python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('protenix_opt') else 41)" 2>&1 >/dev/null) || _probe_rc=$?
if [ "$_probe_rc" -eq 41 ] || [ "$_probe_rc" -eq 127 ] || [ "$_probe_rc" -eq 126 ]; then
  echo "run.sh: protenix_opt is not installed on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/opt" >&2; exit 2
elif [ "$_probe_rc" -ne 0 ]; then
  [ -n "$_probe_err" ] && printf '%s\n' "$_probe_err" >&2
  echo "run.sh: $(command -v python) exited $_probe_rc before the protenix_opt probe could run (its words above)" >&2; exit "$_probe_rc"
fi
unset _probe_rc _probe_err
# Start-up hook gate (env route only): with no --mode on the command line and PROTENIX_OPT naming a non-off mode, the run and every child of
# `pred` activate through the start-up hook `protenix_opt._autoload` (protenix_opt_autoload.pth in the interpreter's site) — a package merely
# importable runs STOCK silently under the variable. The check is the hook's EFFECT: the module in sys.modules of a FRESH interpreter (the same
# `python` the route runs, no -I: the route's interpreter processes the user site when it is enabled, so a `pip install --user` form passes; the
# variable unset for the probe); refused by name (exit 3) otherwise, the line naming the diagnostic (the .pth absent from the searched sites |
# present in a site but not processed | a copy on PYTHONPATH, never processed — site.py reads .pth files from site directories only). The --mode
# route is not gated (it activates in-process; its ACTIVE line is the evidence); --mode off / PROTENIX_OPT=off exempt (the stock route needs no hook).
if [ -z "${CLI_MODE:-}" ] && [ -n "$ENVMODE" ] && [ "$ENVMODE" != off ]; then   # the command-line mode (kept apart from the merge above) and the environment's own value
  DIAG=$(env -u PROTENIX_OPT RF6_PYTHONPATH="${PYTHONPATH:-}" python - <<'PY'
import os, site, sys, sysconfig
if "protenix_opt._autoload" in sys.modules: print("LIVE"); raise SystemExit
name = "protenix_opt_autoload.pth"; dirs = []
for d in [sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib")] + (site.getsitepackages() if hasattr(site, "getsitepackages") else []) + ([site.getusersitepackages()] if site.ENABLE_USER_SITE else []):
    if d and d not in dirs: dirs.append(d)
in_site = [os.path.join(d, name) for d in dirs if os.path.isfile(os.path.join(d, name))]
on_path = [os.path.join(p, name) for p in os.environ.get("RF6_PYTHONPATH", "").split(os.pathsep) if p and os.path.isfile(os.path.join(p, name))]
if in_site: print(f"present but not processed: {in_site[0]} is in a site directory yet the hook did not load at interpreter start (a stale or broken copy: its line must read `import protenix_opt._autoload`)")
elif on_path: print(f"a copy on PYTHONPATH at {on_path[0]} is never processed (site.py reads .pth files from site directories only); absent from the searched sites {':'.join(dirs)}")
else: print(f"absent from the searched sites {':'.join(dirs)}")
PY
)
  [ "$DIAG" = LIVE ] || { echo "[protenix-opt] NOT ACTIVE: the env route (PROTENIX_OPT=$ENVMODE, no --mode) needs the start-up hook protenix_opt._autoload live in a fresh interpreter ($(command -v python)) and it is not — $DIAG; protenix_opt is importable but PROTENIX_OPT=<mode> + the stock CLI would run STOCK silently; install the kit into this interpreter: pip install -e $HERE/opt" >&2; exit 3; }
fi
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m protenix_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
