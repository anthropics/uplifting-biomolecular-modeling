#!/bin/bash
# RoseTTAFold3 optimized — single entry point: a thin wrapper over `python -m rosettafold3_opt` (the installed rosettafold3_opt package:
# `run.sh install` puts it, editable, into the PATCHED interpreter only).
#   run.sh pred    [--config h100] [--mode M] --input <json|cif|dir|list> --out_dir <dir> [--seeds S] [key=value ...]   rf3 fold on the input as given (key=value: rf3 fold's own overrides, verbatim; --seeds: one fold per seed)
#   run.sh check   [--config h100] [--mode M]      dry run: the tree state of both interpreters, the row, the pins, the GPU, cuEquivariance importable; nothing is applied
#   run.sh warm    [--config h100] [--mode M]      one public-input prediction to fill the caches
#   --no-compile (pred|check|warm, any mode): the release tree's sanctioned word for "no torch.compile" = MODEL_OPT_LEVERS_OFF=compile; this kit has no compile
#                  lever and upstream rf3 compiles nothing, so it is accepted as a no-op and reported (`compile=none` on the ACTIVE / DRY-RUN / EXIT lines)
#   run.sh install [--make-venv] [--weights DIR] [--stock-python P] [--opt-python P] [--no-addon] [--config h100]   the install step: the patched interpreter
#                  opt/venv (derived from the stock interpreter by --make-venv when absent), the shared core and this kit installed editable into it, the pin check on
#                  both interpreters (stock/check_pins.py), the add-on applied once with both tree states asserted (--no-addon: every step but the add-on — the patched
#                  interpreter's rf3 files stay as found until a later `run.sh install`; a container image is built so); --weights DIR also fetches the pinned checkpoint
#                  into DIR with upstream's own downloader and checks its sha256 (stock/PINS.json "weights") — DIR/rf3_foundry_01_24_latest_remapped.ckpt is then
#                  your ROSETTAFOLD3_OPT_CKPT; the stock interpreter = --stock-python, else ROSETTAFOLD3_OPT_STOCK_PYTHON, else stock/venv/bin/python, else python on PATH
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (the two interpreters, the checkpoint, the JIT cache).
# Mode = --mode when given, else ROSETTAFOLD3_OPT from the environment, else the package default (fast — the standing rule; exact
# reproduces stock's outputs; README.md names the cold one-item trade-off); a --mode that disagrees with a set ROSETTAFOLD3_OPT is refused. Modes (fast|exact|big|off) are the rows of opt/rosettafold3_opt/modes.py;
# this script validates none of them.
# `--mode off` is the stock route: the package's one stock caller (opt/rosettafold3_opt/stock_fold.py — the upstream CLI on the pristine
# interpreter, nothing from this tree on its path) in a clean subprocess that proves its environment and its tree by sha; `pred` is its
# only command (the stock arm has no warm). Every route runs on the patched interpreter's package and refuses
# (rc 3) unless the pinned upstream is installed at the pin (stock/check_pins.py) there. Exit codes: 0 ok, 1 failed, 2 usage, 3 not active / pins not met.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,20p' "$0" >&2; exit 2; }
CFG=""; CMD=""; MODE=""; CLIMODE=""; NOCOMPILE=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config)    [ $# -ge 2 ] || usage; CFG=$2; shift 2 ;;
    --config=*)  CFG=${1#--config=}; shift ;;
    --mode)      [ $# -ge 2 ] || usage; MODE=$2; CLIMODE=1; shift 2 ;;
    --mode=*)    MODE=${1#--mode=}; CLIMODE=1; shift ;;
    --no-compile) NOCOMPILE=1; shift ;;
    *) if [ -z "$CMD" ]; then CMD=$1; else ARGS+=("$1"); fi; shift ;;
  esac
done
case "$CMD" in pred|check|warm|install) ;; *) usage ;; esac
[ -n "$NOCOMPILE" ] && export MODEL_OPT_LEVERS_OFF="${MODEL_OPT_LEVERS_OFF:+${MODEL_OPT_LEVERS_OFF},}compile"   # --no-compile: the levers-off word `compile` (leversoff.ABSENT: accepted, a no-op here, reported compile=none)
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[rosettafold3-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[rosettafold3-kit] jit cache: $J ($W)"; }
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 3, the package's NOT ACTIVE line) when rosettafold3_opt or its pinned core is not importable
fi
FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
ENVMODE=${ROSETTAFOLD3_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with ROSETTAFOLD3_OPT=$ENVMODE from $FROM; one run has one mode — drop one of them" >&2; exit 2
fi
MODE=${MODE:-$ENVMODE}
if [ "$CMD" = install ]; then                                                                     # THE INSTALL STEP (README.md 'Install'): every piece of the kit goes into the PATCHED
  CORE=$HERE/../common/opt_core                                                                   # interpreter; the stock interpreter is only ever read. The shared core, pinned by
  WEIGHTS=""; MAKEVENV=""; STOCKPY=""; OPTPY=""; PASS=()                                          # opt/pyproject.toml [tool.opt_core]. --weights DIR is this step's own flag; --make-venv,
  set -- ${ARGS[@]+"${ARGS[@]}"}                                                                  # --stock-python, --opt-python and anything else are the package's install command's
  while [ $# -gt 0 ]; do                                                                          # (python -m rosettafold3_opt install), passed on to it
    case "$1" in
      --weights)        [ $# -ge 2 ] || usage; WEIGHTS=$2; shift 2 ;;
      --weights=*)      WEIGHTS=${1#--weights=}; shift ;;
      --make-venv)      MAKEVENV=1; PASS+=("$1"); shift ;;
      --stock-python)   [ $# -ge 2 ] || usage; STOCKPY=$2; shift 2 ;;
      --stock-python=*) STOCKPY=${1#--stock-python=}; shift ;;
      --opt-python)     [ $# -ge 2 ] || usage; OPTPY=$2; shift 2 ;;
      --opt-python=*)   OPTPY=${1#--opt-python=}; shift ;;
      *)                PASS+=("$1"); shift ;;
    esac
  done
  STOCK=${STOCKPY:-${ROSETTAFOLD3_OPT_STOCK_PYTHON:-}}                                             # the pristine interpreter: --stock-python, else ROSETTAFOLD3_OPT_STOCK_PYTHON, else
  if [ -z "$STOCK" ]; then if [ -x "$HERE/stock/venv/bin/python" ]; then STOCK=$HERE/stock/venv/bin/python; else STOCK=$(command -v python || true); fi; fi   # stock/venv/bin/python when it exists, else `python` on PATH
  OPT=${OPTPY:-${ROSETTAFOLD3_OPT_PYTHON:-$HERE/opt/venv/bin/python}}                             # the patched interpreter: --opt-python, else ROSETTAFOLD3_OPT_PYTHON, else opt/venv/bin/python
  [ -f "$CORE/pyproject.toml" ] || { echo "run.sh: the shared core is not at $CORE — unpack or clone the release so that common/ sits beside rosettafold3/ (README.md 'Install')" >&2; exit 3; }
  [ -n "$STOCK" ] && [ -x "$STOCK" ] || { echo "run.sh: no stock interpreter (--stock-python, ROSETTAFOLD3_OPT_STOCK_PYTHON, stock/venv/bin/python or python on PATH): the interpreter that has the pinned stack and stock rc-foundry installed (README.md 'Install')" >&2; exit 3; }
  if [ ! -x "$OPT" ]; then                                                                        # no patched interpreter yet: --make-venv derives it from the stock one — the package's own
    [ -n "$MAKEVENV" ] || { echo "run.sh: no patched interpreter at $OPT: run.sh install --make-venv derives it from $STOCK (or create it with the pinned stack installed and name it: ROSETTAFOLD3_OPT_PYTHON / --opt-python)" >&2; exit 3; }
    PYTHONPATH=$HERE/opt${PYTHONPATH:+:$PYTHONPATH} "$STOCK" -m rosettafold3_opt install --stock-python "$STOCK" --opt-python "$OPT" ${PASS[@]+"${PASS[@]}"} || exit $?   # install command, run from this tree by the stock interpreter
  fi                                                                                              # (nothing of the kit is installed anywhere yet): the venv, the core, the add-on, the states
  "$OPT" -m pip install -q --no-build-isolation --no-deps -e "$CORE" || { echo "run.sh: pip install -e $CORE into $OPT failed" >&2; exit 1; }           # the shared core, editable
  "$OPT" -m pip install -q --no-build-isolation --no-deps -e "$HERE/opt" || { echo "run.sh: pip install -e $HERE/opt into $OPT failed" >&2; exit 1; }   # this kit, editable (its .pth hook with it)
  for _p in "$OPT" "$STOCK"; do                                                                   # the pin check: stock rc-foundry at the pinned commit, on both interpreters
    "$_p" -I "$HERE/stock/check_pins.py" || { echo "run.sh: the pinned upstream is not installed at the pin on $_p (stock/PINS.json)" >&2; exit 3; }
  done
  "$OPT" -m rosettafold3_opt install --stock-python "$STOCK" --opt-python "$OPT" ${PASS[@]+"${PASS[@]}"} || exit $?                                    # the add-on applied once (kept when already applied; left out under --no-addon); both tree states asserted by sha
  if [ -n "$WEIGHTS" ]; then "$OPT" -m rosettafold3_opt.weights "$WEIGHTS" || exit 1; fi          # --weights DIR: the pinned checkpoint, fetched with upstream's downloader when absent, digest-checked
  _w="${WEIGHTS:-<dir>}/rf3_foundry_01_24_latest_remapped.ckpt"; [ -n "$WEIGHTS" ] || _w="$_w (run.sh install --weights <dir> fetches it)"
  echo "[rosettafold3-opt install] DONE stock=$STOCK opt=$OPT — next: export ROSETTAFOLD3_OPT_STOCK_PYTHON=$STOCK ROSETTAFOLD3_OPT_CKPT=$_w; then: bash run.sh check --config h100"
  exit 0
fi
PY=${ROSETTAFOLD3_OPT_PYTHON:-}
if [ -z "$PY" ]; then if [ -x "$HERE/opt/venv/bin/python" ]; then PY=$HERE/opt/venv/bin/python; else PY=$(command -v python || true); fi; fi
[ -n "$PY" ] && [ -x "$PY" ] || { echo "run.sh: no interpreter: set ROSETTAFOLD3_OPT_PYTHON or create opt/venv (run.sh install --make-venv)" >&2; exit 3; }
export ROSETTAFOLD3_OPT_PYTHON=$PY
if [ "$MODE" = off ]; then                                        # stock route: `pred --mode off` = the package's stock caller in a clean subprocess
  case "$CMD" in pred|check) ;; *) echo "run.sh: --mode off has one command, pred (plus check; the stock arm has no warm)" >&2; exit 2 ;; esac
fi
_rc=0; _imp=$("$PY" -m rosettafold3_opt._require 2>&1) || _rc=$?                                                 # the package's producer check (opt_core and every module it loads, _core.REQUIRED): whatever it prints is surfaced verbatim and its
if [ "$_rc" -ne 0 ]; then                                                                                            # exit code is run.sh's (NOT ACTIVE reason=core_missing|producer_missing → 3; any other import-time failure → that rc);
  [ -n "$_imp" ] && echo "$_imp" >&2                                                                                 # 'not installed' is said ONLY when a stripped interpreter answers the absence sentinel (41), exit 3
  _abs=0; [ "$_rc" -eq 3 ] || { ( for _v in $(compgen -e | grep '^ROSETTAFOLD3_OPT' || true); do unset "$_v"; done; exec "$PY" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("rosettafold3_opt") is not None else 41)' ) 2>/dev/null || _abs=$?; }   # 3 = refused by name: nothing to ask; else ONE question,
  if [ "$_abs" -eq 41 ]; then echo "run.sh: rosettafold3_opt is not installed on $PY: $PY -m pip install -e $HERE/opt" >&2; exit 3; fi            # every ROSETTAFOLD3_OPT* name stripped: is the
  exit "$_rc"                                                                                                        # package ABSENT (find_spec None → sentinel 41)? any other answer (a start-up hook refusing, no python) is not absence: the original code
fi; unset _imp _rc
if [ -z "$CLIMODE" ] && [ -n "$ENVMODE" ] && [ "$ENVMODE" != off ]; then case "$CMD" in pred|check|warm)     # the ENV ROUTE (ROSETTAFOLD3_OPT=<mode> names the mode, no --mode: the variable's route, whose activation is the .pth hook's) needs the
  env -u ROSETTAFOLD3_OPT "$PY" - "$HERE/opt" <<'PY' || exit 3                                                                          # hook LIVE in this interpreter: a fresh interpreter (the route's own `python`: no -I, so a user-site install counts when the user site
                                                                                                                           # is enabled) must carry the hook module at start-up (the site-processed .pth's effect;
import os, site, sys                                                                                                       # a copy on PYTHONPATH or an unprocessed file proves nothing) — refused by name, exit 3, the diagnostic in the line; the --mode route
                                                                                                                           # is not gated (the package's CLI activates in-process; its ACTIVE line is the evidence); --mode off / ROSETTAFOLD3_OPT=off need no hook
MOD, NAME = "rosettafold3_opt._autoload", "rosettafold3_opt_autoload.pth"
LINE = open(os.path.join(sys.argv[1], NAME)).read()                                                                       # the kit's own .pth text (generated by opt/_build_backend.py): a site copy with other text is stale
if MOD in sys.modules:
    sys.exit(0)
searched = []
for d in list(site.getsitepackages() if hasattr(site, "getsitepackages") else []) + [site.getusersitepackages()] + [p for p in sys.path if p.rstrip("/").endswith(("site-packages", "dist-packages"))] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]:
    if d and d not in searched:
        searched.append(d)
hits = [os.path.join(d, NAME) for d in searched if os.path.isfile(os.path.join(d, NAME))]
stale = [h for h in hits if open(h).read() != LINE]
if stale:
    why = f"{stale[0]} is a stale copy (its text is not the kit's opt/{NAME})"
elif hits:
    why = f"{hits[0]} is present but not processed (site.py did not execute it: a copy beside a PYTHONPATH entry, a user site that is disabled — ENABLE_USER_SITE={site.ENABLE_USER_SITE} — or an import that failed at start-up)"
else:
    why = f"{NAME} is absent from the searched sites ({', '.join(searched)})"
print(f"[rosettafold3-opt] NOT ACTIVE: the .pth hook is not live on {sys.executable} ({MOD} is not imported at interpreter start): {why} — the env route ROSETTAFOLD3_OPT=<mode> would run stock silently on it; install the kit into this interpreter: {sys.executable} -m pip install -e {os.environ['MODEL_OPT']}/opt", file=sys.stderr)
sys.exit(3)
PY
;; esac; fi
"$PY" -I "$HERE/stock/check_pins.py" >/dev/null || { echo "run.sh: the pinned upstream is not installed at the pin on $PY (stock/PINS.json)" >&2; exit 3; }
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec "$PY" -m rosettafold3_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
