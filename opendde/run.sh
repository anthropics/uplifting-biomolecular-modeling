#!/bin/bash
# OpenDDE optimized — single entry point: a thin wrapper over `python -m opendde_opt` (the installed opendde_opt package: `pip install -e opt`).
#   run.sh pred   [--config h100] [--mode M] -i <query.json> -o <dir> [upstream flags: --seeds --cycle --step --sample --dtype --use_msa --use_template …] [--det 0|1] [--template_mmcif_dir D] [-- <extra opendde pred args>]   upstream's action; its own flags pass through as stated (README "Stock")
#   run.sh check  [--config h100] [--mode M] [--json]                                      dry run: resolves and gates the mode on this box; nothing is applied
#   run.sh warm   [--config h100] [--mode M] [-o <dir>]                                   one prediction of upstream's smallest documented example (JIT + caches)
#   run.sh install [--weights DIR]                                                          the install step: the shared core and this kit installed editable into the python on PATH, then the pin check (stock/check_pins.py);
#                                                                                          --weights DIR also fetches the checkpoint and the four common/ files into DIR with upstream's own downloader and checks them against stock/PINS.json (sha256) — DIR is then your OPENDDE_ROOT_DIR
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (weights root, JIT caches, target GPU). Mode = --mode when given, else
# OPENDDE_OPT from the environment, else the package default (modes.DEFAULT_MODE); a --mode that disagrees with a set OPENDDE_OPT is
# refused.
# Modes (off|exact|fast|big) are resolved by the package's one mode table; this script validates none of them (exact = the kit's S1,
# fast = LSTAR2A). `--mode off` is the stock route: the package's one stock caller (opt/opendde_opt/stock_pred.py — the upstream CLI, nothing from
# the kits on the path) in a clean subprocess that proves its environment; `pred` is its only command (the stock arm has no warm). Every route refuses (rc 3) unless the pinned upstream wheel is installed (stock/check_pins.py) and the package is
# installed. Exit codes: 0 ok, 1 failed, 2 usage, 3 not active / PARTIAL without --allow-partial / pin not met, 5 expected accelerator absent / fell back.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,14p' "$0" >&2; exit 2; }
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
case "$CMD" in pred|check|warm|install) ;; *) usage ;; esac
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
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('opendde_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: opendde_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the line above — the pinned upstream wheel is not installed as pinned in this environment)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then python -m opendde_opt.weights "$WEIGHTS" || exit $?; fi   # upstream's downloader into DIR, then the sha256 check against stock/PINS.json (opt/opendde_opt/weights.py)
  exit 0
fi
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 2) when opendde_opt is not importable
fi
_J0="${MODEL_OPT_JIT_ROOT:-}"   # the root the config named, before the seed block may step off it
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[opendde-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[opendde-kit] jit cache: $J ($W)"; }
if [ -n "$_J0" ] && [ "${MODEL_OPT_JIT_ROOT:-}" != "$_J0" ]; then   # the block stepped off a root it cannot write: re-point the cache dirs the config derived from that root
  for _v in TRITON_CACHE_DIR TORCH_EXTENSIONS_DIR MODEL_OPT_WEIGHTS_DIGEST_DIR; do
    case "${!_v:-}" in "$_J0"/*) export "$_v=$MODEL_OPT_JIT_ROOT${!_v#"$_J0"}" ;; esac
  done
fi
FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
ENVMODE=${OPENDDE_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with OPENDDE_OPT=$ENVMODE from $FROM; a row has one mode — drop one of them" >&2; exit 2
fi
CLI_MODE=$MODE                                                    # the mode as given on the command line (empty: the environment names it)
MODE=${MODE:-$ENVMODE}
case "$MODE" in ""|off|exact|fast|big) ;; *) echo "run.sh: '$MODE' is not a mode (off|exact|fast|big)" >&2; exit 2 ;; esac
if [ "$MODE" = off ]; then                                        # stock route: `pred --mode off` = the package's stock caller in a clean subprocess
  [ "$CMD" = pred ] || { echo "run.sh: --mode off has one command, pred (the stock arm has no warm)" >&2; exit 2; }
fi
_PROBE_ERR=$(python -m opendde_opt help 2>&1 >/dev/null) || {                   # the real entry, before any other probe: the package's own words and exit code pass through
  _rc=$?; [ -n "$_PROBE_ERR" ] && printf '%s\n' "$_PROBE_ERR" >&2
  if printf '%s' "$_PROBE_ERR" | grep -qE "No module named '?opendde_opt"; then           # the one condition this script names itself: the package is not installed for this interpreter
    echo "run.sh: opendde_opt is not installed on $(command -v python || echo 'python (not on PATH)') (pip install -e $HERE/../common/opt_core -e $HERE/opt)" >&2; exit 3
  fi
  echo "run.sh: opendde_opt refused at start on $(command -v python || echo python) — its line above is the reason (rc $_rc)" >&2; exit "$_rc"
}
unset _PROBE_ERR
python -I "$HERE/stock/check_pins.py" --quiet || { _rc=$?; echo "run.sh: the installed upstream is not the pinned wheel (stock/check_pins.py's line above; stock/PINS.json) (rc $_rc)" >&2; exit "$_rc"; }
# Env route only: a mode named by OPENDDE_OPT and not on the command line is the env route — it exists only through
# opendde_opt_autoload.pth processed by site (the stock CLI under the variable); a package that is merely importable (PYTHONPATH) would run
# STOCK silently there. The proof is the hook itself: a FRESH `python` (the route's own interpreter form — user site enabled, the variable
# unset so the hook only imports) has opendde_opt._autoload in sys.modules before any user code — a copy of the .pth on PYTHONPATH (opt/
# carries one) is not processed by site and proves nothing; a `pip install --user` copy in an enabled user site is processed, and passes. The --mode route activates in-process (python -m opendde_opt below) and is not
# gated; --mode off is the stock route. CLI_MODE is the command-line mode alone, read before the merge with the environment.
if [ -z "$CLI_MODE" ] && [ -n "$ENVMODE" ] && [ "$ENVMODE" != off ]; then
  LIVE_FROM=$(env -u OPENDDE_OPT python -c 'import os, sys; m = sys.modules.get("opendde_opt._autoload"); print(os.path.realpath(os.path.dirname(os.path.dirname(m.__file__))) if m else "")' 2>/dev/null)
  if [ -n "$LIVE_FROM" ] && [ "$LIVE_FROM" != "$(python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$HERE/opt")" ]; then
    echo "run.sh: the env route OPENDDE_OPT=$ENVMODE would run a stale copy — the live autoload hook of $(command -v python) imports opendde_opt from $LIVE_FROM, not this tree ($HERE/opt); install this tree into the interpreter (pip install -e $HERE/../common/opt_core -e $HERE/opt) or name the mode on the command line (--mode $ENVMODE)" >&2; exit 3
  fi
  [ -n "$LIVE_FROM" ] || {
    DIAG=$(env -u OPENDDE_OPT HERE="$HERE" python - <<'PY'
import os, site
sites = [d for d in list(site.getsitepackages()) + ([site.getusersitepackages()] if site.ENABLE_USER_SITE else []) if os.path.isdir(d)]   # the user site counts when this interpreter enables it
insite = [d for d in sites if os.path.isfile(os.path.join(d, "opendde_opt_autoload.pth"))]
onpath = [d for d in os.environ.get("PYTHONPATH", "").split(os.pathsep) if d and d not in sites and os.path.isfile(os.path.join(d, "opendde_opt_autoload.pth"))]
usersite = site.getusersitepackages()
if not insite and not site.ENABLE_USER_SITE and os.path.isfile(os.path.join(usersite, "opendde_opt_autoload.pth")):
    print(f"opendde_opt_autoload.pth present in the user site {usersite} but the user site is disabled for this interpreter (a venv without system site packages, PYTHONNOUSERSITE, or python -s/-I): not processed"); raise SystemExit(0)
if insite:
    print(f"opendde_opt_autoload.pth present in the site {insite[0]} but not processed at start (site.py skipped it: python -S, or the site dir is not on this interpreter's path)")
elif onpath:
    print(f"opendde_opt_autoload.pth present at {onpath[0]} but not processed (a PYTHONPATH entry is not a site dir)")
else:
    print(f"opendde_opt_autoload.pth absent from the searched sites {sites}")
PY
    )
    echo "run.sh: the env route OPENDDE_OPT=$ENVMODE needs the autoload hook, which is not live in $(command -v python) ($(python -c 'import sys; print(sys.prefix)')): $DIAG; the stock CLI under the variable would run stock silently — install the kit into this interpreter (pip install -e $HERE/../common/opt_core -e $HERE/opt) or name the mode on the command line (--mode $ENVMODE)" >&2; exit 3; }
fi
SEL=(); [ -n "$MODE" ] && SEL=(--mode "$MODE")
exec python -m opendde_opt "$CMD" ${SEL[@]+"${SEL[@]}"} ${ARGS[@]+"${ARGS[@]}"}
