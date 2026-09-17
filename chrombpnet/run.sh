#!/bin/bash
# ChromBPNet optimization kit — single entry point: a thin wrapper over `python -m chrombpnet_opt` (the chrombpnet_opt package this kit installs: `pip install -e opt`).
#   run.sh pred_bw [--config h100|h200|a100] [--mode off|exact|fast] [--det 0|1] [--items FILE] <the stock pred_bw arguments>   exact | fast: the kit's entry script as a subprocess; off: the stock CLI in a clean subprocess;
#                                                                                    --items FILE: N regions files in one process (exact | fast; the stock arguments minus -r/-op/-os)
#   run.sh check   [--config CFG] [--mode off|exact|fast] [--det 0|1 (off only)]   dry run: the kit's route resolved for this machine and the mode line printed; nothing runs
#   run.sh warm    [--config CFG] [--mode exact|fast] [--regions BED] [--n N] [--out DIR] one small job through the kit's entry script: the kernel / driver caches warmed
#   run.sh install [--weights DIR]                          the install step: this kit installed editable into the python on PATH, then the pin check (stock/check_pins.py);
#                                                          --weights DIR also checks the three stock weight files under DIR against stock/PINS.json (sha256) — DIR is then your CHROMBPNET_OPT_WEIGHTS
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (target GPU, weights root, the driver-cache tarball, the default data paths).
# Mode = --mode when given, else CHROMBPNET_OPT from the environment, else the package default (fast); a --mode that disagrees with a set
# CHROMBPNET_OPT is refused. Modes (off|exact|fast) are resolved by the package from the kit's own tables (opt/kit_ho/tf/chrombpnet_fastkit/fastdefault.py);
# this script validates none of them; any other word is a usage error. `--mode off` is the stock route: the stock console script
# `chrombpnet pred_bw` in a clean subprocess that re-checks its own environment; its commands are `pred_bw` and `check` (the stock route has no
# warm). A config whose MODEL_OPT_TARGET_GPU is not the GPU nvidia-smi reports is noted on stderr (kit routes; never a refusal). Every route first names
# this environment against the pinned stack (stock/check_pins.py: drift in a companion package, the Python version, a CUDA library or /opt/torch is printed,
# never a refusal) and refuses (rc 3) only when the installed `chrombpnet` is not the pinned stock — that would change what "stock" means.
# Exit codes: 0 ok, 1 failed or incomplete (outputs short of the request; the install; a weight file off its pin), 2 usage, 3 not active (the mode was refused by name: an unknown mode, the kit's files missing) / stock not the pinned stock.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,16p' "$0" >&2; exit 2; }
CFG=""; CMD=""; MODE=""; DET=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config)    [ $# -ge 2 ] || usage; CFG=$2; shift 2 ;;
    --config=*)  CFG=${1#--config=}; shift ;;
    --mode)      [ $# -ge 2 ] || usage; MODE=$2; shift 2 ;;
    --mode=*)    MODE=${1#--mode=}; shift ;;
    --det)       [ $# -ge 2 ] || { echo "run.sh: --det takes 0 or 1 (--mode off only: stock as shipped / with TensorFlow's determinism settings)" >&2; exit 2; }; case "$2" in 0|1) DET=$2 ;; *) echo "run.sh: --det takes 0 or 1 (the deterministic recipe off / on), got '$2'" >&2; exit 2 ;; esac; shift 2 ;;
    --det=*)     DET=${1#--det=}; case "$DET" in 0|1) ;; *) echo "run.sh: --det takes 0 or 1, got '$DET'" >&2; exit 2 ;; esac; shift ;;
    *) if [ -z "$CMD" ]; then CMD=$1; else ARGS+=("$1"); fi; shift ;;
  esac
done
case "$CMD" in pred_bw|check|warm|install) ;; *) usage ;; esac
if [ "$CMD" = install ]; then                                       # the install step: everything below it presupposes the installed package
  WEIGHTS=""
  [ -z "$CFG" ] && [ -z "$MODE" ] && [ -z "$DET" ] || { echo "run.sh: install takes no --config / --mode / --det (usage: run.sh install [--weights DIR])" >&2; exit 2; }
  set -- ${ARGS[@]+"${ARGS[@]}"}
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
      --weights=*) WEIGHTS=${1#*=}; [ -n "$WEIGHTS" ] || { echo "run.sh: install --weights= takes a directory" >&2; exit 2; }; shift ;;
      *) echo "run.sh: install takes no argument '$1' (usage: run.sh install [--weights DIR])" >&2; exit 2 ;;
    esac
  done
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md Install)" >&2; exit 3; }
  if python -I -c "import os,sys,importlib.util as u; d=os.path.realpath(sys.argv[1]); x=u.find_spec('chrombpnet_opt'); sys.exit(0 if x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) else 1)" "$HERE/opt" 2>/dev/null; then
    echo "run.sh: chrombpnet_opt is installed from this tree already ($HERE/opt) — the pip step is skipped"   # the container image ships it installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the kit package is expected at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but the `chrombpnet` package in this environment is not the pinned stock (stock/check_pins.py: the lines above)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then python -m chrombpnet_opt.weights "$WEIGHTS" || exit $?; fi   # the three stock weight files under DIR checked against stock/PINS.json (opt/chrombpnet_opt/weights.py)
  exit 0
fi
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 2) when chrombpnet_opt is not importable
fi
KIT=chrombpnet; _J0="${MODEL_OPT_JIT_ROOT:-}"   # the root as preset, before the block may move it off a read-only mount
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
[ -z "${MODEL_OPT_JIT_ROOT:-}" ] || case "${TRITON_CACHE_DIR:-}" in ""|"${_J0:-//unset//}"/*) export TRITON_CACHE_DIR="$MODEL_OPT_JIT_ROOT/torch2.4.1-cu124-$([ "${MODEL_OPT_TARGET_GPU:-}" = A100 ] && echo sm80 || echo sm90)/triton";; esac   # the Triton forward's cache under the compile-cache root when one is set — derived here when unset or when a config derived it from the preset root the block moved (a value set elsewhere is kept; no root: Triton's own default, as before)
FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
ENVMODE=${CHROMBPNET_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with CHROMBPNET_OPT=$ENVMODE from $FROM: set one of them" >&2; exit 2
fi
MODE_FROM_CLI=$MODE                                                # captured before the merge below: empty iff --mode was never given on the command line
MODE=${MODE:-$ENVMODE}
case "$MODE" in ""|fast|exact|off) ;; *) echo "run.sh: '$MODE' is not a mode (off|exact|fast)" >&2; exit 2 ;; esac
if [ -n "$DET" ] && [ "$MODE" != off ]; then                     # the kit modes carry their numerics (exact = upstream's determinism settings, fast = shipped): --det is the stock arm's setting only
  echo "run.sh: --det is only for --mode off (\`--mode off --det 1\` = stock with TensorFlow's determinism settings, the reference \`exact\` equals); \`${MODE:-fast} --det $DET\` does not exist" >&2; exit 2
fi
if [ "$MODE" != off ] && [ -n "${MODEL_OPT_TARGET_GPU:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then    # the card note (kit routes only; the stock route never reads it): the config's target vs the card present
  GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)
  if [ -n "$GPUNAME" ] && [[ "$GPUNAME" != *"$MODEL_OPT_TARGET_GPU"* ]]; then    # a name mismatch is reported, never refused: the package keys its route on the card's compute capability (opt/kit_ho/tf/chrombpnet_fastkit/fastdefault.py), not on the name
    echo "run.sh: NOTE card_name: MODEL_OPT_TARGET_GPU=$MODEL_OPT_TARGET_GPU from $FROM but nvidia-smi reports '$GPUNAME' — the kit resolves its route from the card it finds" >&2
  fi
fi
[ -f "$HERE/stock/check_pins.py" ] || { echo "run.sh: the tree does not carry stock/check_pins.py (the stock pin check)" >&2; exit 3; }
python "$HERE/stock/check_pins.py" || { echo "run.sh: the installed chrombpnet is not the pinned stock (stock/PINS.json; drift in anything else is named above, never refused)" >&2; exit 3; }
if [ "$MODE" = off ]; then                                        # stock route: `pred_bw --mode off` = the stock console script in a clean subprocess; `check --mode off` its dry run
  case "$CMD" in pred_bw|check) ;; *) echo "run.sh: --mode off has two commands, pred_bw and check (the stock arm has no warm)" >&2; exit 2 ;; esac
fi
python -c "import chrombpnet_opt" 2>/dev/null || { echo "run.sh: chrombpnet_opt is not installed on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/opt" >&2; exit 3; }
if [ -z "$MODE_FROM_CLI" ] && [ -n "$ENVMODE" ] && [ "$ENVMODE" != off ]; then
  # the stock CLI under the variable, and only that: --mode was never given on the command line, so a plain
  # `chrombpnet pred_bw` (outside this script) run right now under the same CHROMBPNET_OPT would depend entirely
  # on the env-route hook (chrombpnet_opt/_autoload.py) having actually run at interpreter start -- this script's
  # OWN invocation below always passes --mode explicitly and cannot run stock silently, so it is never gated;
  # a fresh interpreter's sys.modules is the hook's live effect, never file presence (a copy on PYTHONPATH proves
  # nothing, since neither this script nor a bare `chrombpnet` invocation sets one)
  HOOKDIAG=$(env -u CHROMBPNET_OPT python -I - "$HERE/opt/chrombpnet_opt_autoload.pth" <<'PYEOF' 2>&1
import os, site, sys
if "chrombpnet_opt._autoload" in sys.modules:
    sys.exit(0)
F = "chrombpnet_opt_autoload.pth"
want = open(sys.argv[1], "rb").read().strip() if os.path.isfile(sys.argv[1]) else None
sites = list(dict.fromkeys(site.getsitepackages() + [site.getusersitepackages()]))
present = [d for d in sites if os.path.isfile(os.path.join(d, F))]
beside = [d for d in (os.environ.get("PYTHONPATH") or "").split(os.pathsep) if d and os.path.isfile(os.path.join(d, F))]
stale = [d for d in present + beside if want is not None and open(os.path.join(d, F), "rb").read().strip() != want]
if stale:
    print(F + " is a stale copy (its line differs from the kit's " + sys.argv[1] + "): " + ", ".join(stale))
elif present:
    print(F + " present but not processed (in a site dir site.py did not process at start): " + ", ".join(present))
elif beside:
    print(F + " present but not processed (a copy beside a PYTHONPATH entry, not a site dir): " + ", ".join(beside))
else:
    print(F + " absent from the searched sites: " + ", ".join(sites))
sys.exit(1)
PYEOF
) || { echo "run.sh: the env route (CHROMBPNET_OPT=$ENVMODE) refused: the fail-loud hook chrombpnet_opt._autoload is not live in $(command -v python) — $HOOKDIAG: pip install -e $HERE/opt — or name the mode on the command line (--mode $ENVMODE)" >&2; exit 3; }
fi
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
DETARG=();  [ -n "$DET" ] && DETARG=(--det "$DET")
exec python -m chrombpnet_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${DETARG[@]+"${DETARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
