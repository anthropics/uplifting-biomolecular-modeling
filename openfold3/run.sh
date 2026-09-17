#!/bin/bash
# OpenFold3 optimized — single entry point: a thin wrapper over `python -m openfold3_opt` (the installed openfold3_opt package: `pip install -e ../common/opt_core -e opt`).
#   run.sh pred   [--config h100] [--mode M] [--n_gpu P] --query-json <q.json> --output-dir <dir> [--num-model-seeds N] [--num-diffusion-samples S] [--use-msa-server true|false] [--use-templates true|false] [--runner-yaml Y]… [--det 0|1] [--no-compile]
#                 one process, one `run_openfold predict` call on one query JSON
#   run.sh check  [--config h100] [--mode M] [--n_gpu P] [--det D] [--no-compile]   dry run: resolves and gates the mode on this GPU with the pred's route refusals; nothing is applied
#   run.sh warm   [--config h100] [--mode M] --out <dir> [--no-compile]     one public-input prediction: imports, JIT builds, first captures
#   run.sh install [--weights DIR]                                   the install step: the shared core and this kit installed editable into the python on PATH, then the pin check (stock/check_pins.py);
#                                                                    --weights DIR also fetches of3-p2-155k.pt into DIR from the URL stock/PINS.json pins (upstream's S3 client) and checks its sha256 against the pin — DIR/of3-p2-155k.pt is then your OPENFOLD3_CKPT
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (weights, cache roots, JIT caches, target GPU). Mode = --mode when
# given, else OPENFOLD3_OPT from the environment, else the package default (fast); a --mode that disagrees with a set OPENFOLD3_OPT is
# refused. Modes (exact|fast|big|off; `big --n_gpu 2|4|8` the multi-GPU form) are resolved by the package from its one mode
# table (opt/openfold3_opt/modes.py); this script validates none of them. `--mode off` is the stock route: the package's one stock caller
# (opt/openfold3_opt/stock_pred.py — the upstream CLI with the stock configuration, nothing from the add-ons on the path) in a clean subprocess
# that proves its environment; `pred` is its only command (the stock arm has no warm; its check is the stock caller's proof).
# Every route refuses (rc 3) unless openfold3 is installed at its pin, file for file (stock/check_pins.py), and the package is installed.
# Exit codes: 0 ok, 1 failed (incl. `incomplete`: fewer structures than queries x seeds x samples; the install; a weights file off its pin), 2 usage, 3 not active / not stock / pins not
# met / a partial line (a requested lever without the kit's record: a mode is all of its levers — named on the exit lines, no opt-out).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,17p' "$0" >&2; exit 2; }
CFG=""; CMD=""; MODE=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config)       [ $# -ge 2 ] || usage; CFG=$2; shift 2 ;;
    --config=*)     CFG=${1#--config=}; shift ;;
    --mode)         [ $# -ge 2 ] || usage; MODE=$2; shift 2 ;;
    --mode=*)       MODE=${1#--mode=}; shift ;;
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
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('openfold3_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: openfold3_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the line above — the pinned upstream is not installed as pinned in this environment)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then python -m openfold3_opt.weights "$WEIGHTS" || exit $?; fi   # the pinned URL into DIR (upstream's S3 client), then the sha256 check against stock/PINS.json (opt/openfold3_opt/weights.py)
  exit 0
fi
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses by name (rc 3) when openfold3_opt is not importable or the core pin gate refuses
fi
_J0="${MODEL_OPT_JIT_ROOT:-}"   # the root configs/<card>.env derived the cache dirs from, before the seed block may choose or move it
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[openfold3-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[openfold3-kit] jit cache: $J ($W)"; }
if [ -n "${MODEL_OPT_JIT_ROOT:-}" ] && [ -n "${MODEL_OPT_STACK_KEY:-}" ]; then   # the cache dirs follow a root the seed block chose or moved; a directory set elsewhere is kept
  [ -n "${TRITON_CACHE_DIR:-}" ] && [ "$TRITON_CACHE_DIR" != "$_J0/$MODEL_OPT_STACK_KEY/triton" ] || export TRITON_CACHE_DIR="$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/triton"
  [ -n "${TORCH_EXTENSIONS_DIR:-}" ] && [ "$TORCH_EXTENSIONS_DIR" != "$_J0/$MODEL_OPT_STACK_KEY/torch_extensions" ] || export TORCH_EXTENSIONS_DIR="$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/torch_extensions"
fi
unset _J0
FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
ENVMODE=${OPENFOLD3_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with OPENFOLD3_OPT=$ENVMODE from $FROM; one invocation runs one mode — drop one of them" >&2; exit 2
fi
ARGMODE=$MODE; MODE=${MODE:-$ENVMODE}                             # the mode name is the package's to validate (modes.py: rc 2 on an unknown one)
if [ -z "$ARGMODE" ] && [ -n "$ENVMODE" ] && [ "$ENVMODE" != off ]; then   # RF6 (env route only): OPENFOLD3_OPT names a mode and no --mode was given — the operator is on the
  KIT_PTH="$HERE/opt/openfold3_opt_autoload.pth" env -u OPENFOLD3_OPT python - <<'PY' || exit 3                 # env route, where the stock CLI hooks through the kit's .pth as site.py processes it at interpreter start; the gate
import os, site, sys                                              # asks a FRESH interpreter (the route's own `python`, user site as the route sees it) whether the hook is live
name = "openfold3_opt_autoload.pth"; kit_pth = os.environ["KIT_PTH"]        # (openfold3_opt._autoload in sys.modules) — a file proves nothing; the probe runs with the variable unset (presence, not
if "openfold3_opt._autoload" in sys.modules:                     # activation). The --mode route is ungated (python -m openfold3_opt activates by construction; its ACTIVE
    sys.exit(0)                                                   # line is the evidence); `off` is the stock route
sites = list(site.getsitepackages()) + ([site.getusersitepackages()] if site.ENABLE_USER_SITE else [])   # the user site is processed when enabled (a --user / PYTHONUSERBASE install is live)
def text(p):
    try:
        return open(p, encoding="utf-8").read()
    except OSError:
        return ""
in_site = [os.path.join(d, name) for d in sites if os.path.isfile(os.path.join(d, name))]
beside = [os.path.join(d, name) for d in (os.environ.get("PYTHONPATH") or "").split(os.pathsep) if d and os.path.isfile(os.path.join(d, name))]
if in_site and text(in_site[0]) != text(kit_pth):
    diag = f"a stale copy: {in_site[0]} differs from the tree's {kit_pth} (the file the build backend generates) — reinstall the package"
elif in_site:
    diag = f"present but not processed: {in_site[0]} is in site but its import did not run (site.py reported an error running it — the package or the core is not importable from that interpreter's site)"
elif beside:
    diag = f"present but not processed: {beside[0]} sits beside a PYTHONPATH entry, which site.py never processes (a copy on PYTHONPATH proves nothing)"
else:
    diag = f"absent from the searched sites ({', '.join(sites)}; user site {'enabled' if site.ENABLE_USER_SITE else 'disabled'})"
sys.stderr.write(f"[openfold3-opt] NOT ACTIVE: the kit's hook is not live at interpreter start (a fresh {sys.executable}: openfold3_opt._autoload not in sys.modules) — {diag}; the env route "
                 f"OPENFOLD3_OPT=<mode> + the stock CLI would run STOCK silently — install the package into this interpreter: pip install -e {os.environ['MODEL_OPT']}/../common/opt_core -e {os.environ['MODEL_OPT']}/opt\n")
sys.exit(3)
PY
fi
python -I "$HERE/stock/check_pins.py" --quiet || { echo "run.sh: openfold3 is not installed at its pin, file for file (stock/PINS.json)" >&2; exit 3; }
if [ "$MODE" = off ]; then                                        # stock route: `pred --mode off` = the package's stock caller in a clean subprocess
  case "$CMD" in pred|check) ;; *) echo "run.sh: --mode off has no $CMD (the stock arm has no warm; its check is the stock caller's proof)" >&2; exit 2 ;; esac
fi
_of3_err=$(python -c "import openfold3_opt" 2>&1 >/dev/null) || { _of3_rc=$?; case "$_of3_err" in *"ModuleNotFoundError: No module named 'openfold3_opt'"*) echo "[openfold3-opt] NOT ACTIVE: reason=package_missing: openfold3_opt is not importable on $(command -v python || echo 'python (not on PATH)') (${_of3_err##*$'\n'}): pip install -e $HERE/../common/opt_core -e $HERE/opt" >&2; exit 3 ;; esac; printf '%s\n' "$_of3_err" >&2; exit "$_of3_rc"; }   # the package itself (imports nothing of the core). `package_missing` names ONLY a ModuleNotFoundError of openfold3_opt; every other failure — the installed .pth hook's by-name refusal at interpreter start under an exported OPENFOLD3_OPT* (undeclared variable / unknown mode / the core pin gate / producer probe: rc 3) or any import error — is forwarded verbatim with the interpreter's own exit code, never re-diagnosed
python -c "import openfold3_opt._core_gate as g, os; g.gate(os.path.dirname(os.path.abspath(g.__file__)), tag='openfold3-opt')" || { exit 3; }   # the core pin gate (openfold3_opt/_core_gate.py = the house template): an absent or mismatched opt_core is its NOT ACTIVE line, rc 3, before anything else runs
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m openfold3_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
