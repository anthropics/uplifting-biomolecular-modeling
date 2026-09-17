#!/bin/bash
# Chai-1 optimized — single entry point: a thin wrapper over `python -m chai1_opt` (the installed chai1_opt package: `pip install -e opt`).
#   run.sh pred   [--config h100] [--mode M] --input <items.json|PACK.json|x.fasta> --out_dir <dir> [fold flags] [--seed S|--seeds S,S] [--msa_dir|--msa-directory D] [--tag T] [--det 0|1]
#                                                          outputs per (item, seed): a kit mode needs a seed (--seed, --seeds or the items' own), off without one is stock's unseeded call
#   run.sh check  [--config h100] [--mode M] [fold flags]   dry run: resolves and gates the mode on this box; nothing is applied
#   run.sh warm   [--config h100] [--mode M] [--crops all|N,N]  one public-input prediction through pred: the route end to end; --crops also fills the kernel caches per crop size;
#                                                                 then, for a mode carrying the compiled step, that step built ahead of time for this card (built once; folds then load it instead of compiling it)
#   run.sh install [--weights DIR]                          the install step: the shared core and this kit installed editable into the python on PATH, then the pin check (stock/check_pins.py);
#                                                          --weights DIR also fetches the eight stock weight files into DIR with upstream's own downloader and checks them against stock/PINS.json (sha256) — DIR is then your CHAI_DOWNLOADS_DIR
# Fold flags = stock's own `chai-lab fold` options with stock's defaults, every mode: --num-trunk-recycles 3 --num-diffn-timesteps 200 --num-diffn-samples 5 --num-trunk-samples 1 --recycle-msa-subsample 0 --[no-]use-esm-embeddings (on) --[no-]low-memory (on) --[no-]use-msa-server (off) --msa-server-url U --[no-]use-templates-server (off) --constraint-path P --template-hits-path P --device D (STOCK.md).
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (weights directory, target GPU). Mode = --mode when given, else CHAI1_OPT
# from the environment, else the package default (opt/chai1_opt/modes.py DEFAULT_MODE, read from the package here; README.md, the Modes
# section); --mode and a set CHAI1_OPT must agree.
# Modes: off | exact | fast | big (opt/chai1_opt/modes.py MODES). Exit: 0 ok · 1 failed (a fold, the install, a weight file off its pin) · 2 usage · 3 not active (refused, partial without --allow-partial) / pins.
# --strict-stack (or CHAI1_OPT_STRICT_STACK=1): a torch/CUDA stack other than the pinned one (stock/PINS.json stacks) is refused (exit 3), not just recorded (`STACK not pinned: ...`).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
usage() { sed -n '2,15p' "$0" >&2; exit 2; }
CMD=""; CFG=""; MODE=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CFG=$2; shift 2 ;;
    --config=*) CFG=${1#*=}; shift ;;
    --mode) MODE=$2; shift 2 ;;
    --mode=*) MODE=${1#*=}; shift ;;
    --strict-stack) export CHAI1_OPT_STRICT_STACK=1; shift ;;                   # one switch underneath: stock/check_pins.py and the package read the variable
    -h|--help) usage ;;
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
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('chai1_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: chai1_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the line above — the pinned upstream is not installed as pinned in this environment)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then python -m chai1_opt.weights "$WEIGHTS" || exit $?; fi   # upstream's downloader into DIR, then the sha256 check against stock/PINS.json (opt/chai1_opt/weights.py)
  exit 0
fi
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 2) when chai1_opt is not importable
fi
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[chai1-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[chai1-kit] jit cache: $J ($W)"; }
ENVMODE=${CHAI1_OPT:-}; FROM="the environment"
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with CHAI1_OPT=$ENVMODE from $FROM; a run has one mode — drop one of them" >&2; exit 2
fi
MODE_GIVEN=$MODE; MODE=${MODE:-$ENVMODE}
command -v python >/dev/null || { echo "run.sh: chai1_opt is not installed: no python on PATH (pip install -e $HERE/opt into the environment's interpreter)" >&2; exit 3; }
_PROBE=$(python -c "import chai1_opt" 2>&1 >/dev/null); _RC=$?   # importable? any non-zero exit is the import's own words and code, relayed verbatim (a refusal the
if [ "$_RC" -ne 0 ]; then                                         # package prints at interpreter start — chai1_opt_autoload.pth: an unknown CHAI1_OPT, an undeclared CHAI1_OPT_*
  if printf '%s' "$_PROBE" | grep -q "ModuleNotFoundError: No module named 'chai1_opt'"; then   # variable — exits 3 by name; a broken import keeps its traceback and rc); 'not installed'
    echo "run.sh: chai1_opt is not installed on $(command -v python): pip install -e $HERE/opt" >&2; exit 3   # only for the package's own ModuleNotFoundError
  fi
  printf '%s\n' "$_PROBE" >&2; exit "$_RC"
fi
unset _PROBE _RC
if [ -z "$MODE" ]; then                                           # no --mode, no CHAI1_OPT: the package default, read from its one mode table
  MODE=$(python -c "from chai1_opt import modes; print(modes.DEFAULT_MODE)") || { echo "run.sh: the package default mode could not be read" >&2; exit 3; }
  echo "run.sh: no --mode and no CHAI1_OPT: the package default mode, $MODE (opt/chai1_opt/modes.py DEFAULT_MODE)" >&2
fi
case "$MODE" in off|exact|fast|big) ;; *) echo "run.sh: '$MODE' is not a mode (exact|fast|big|off, opt/chai1_opt/modes.py)" >&2; exit 2 ;; esac
# RF6 gate (env route only, hook-live): a run under CHAI1_OPT with no --mode is the env-route form — the variable that activates the package
# in ANY process of this environment through chai1_opt_autoload.pth (opt/_build_backend.py), whose EFFECT is `chai1_opt._autoload` in
# sys.modules of a fresh interpreter. A package that is merely importable runs STOCK silently under CHAI1_OPT in those processes: refused
# here by the hook's absence, the line naming the diagnostic (the .pth absent from the sites searched | present but not processed | a stale
# copy). The --mode route is not gated (it activates by construction: `python -m chai1_opt <cmd> --mode <mode>` below imports the package
# directly, its ACTIVE line is the evidence) and --mode off runs stock by design. A file on PYTHONPATH proves nothing: site.py processes
# .pth files in site directories only (the user site included when enabled — the route's own interpreter is a plain `python`, so the probe is one too).
if [ -z "$MODE_GIVEN" ] && [ -n "$ENVMODE" ] && [ "$MODE" != off ]; then
  RF6=$(env -u CHAI1_OPT python -c "import sys, os; print(('live' if 'chai1_opt._autoload' in sys.modules else 'dead') + ' ' + (os.path.dirname(os.path.realpath(sys.modules['chai1_opt'].__file__)) if 'chai1_opt' in sys.modules else '-'))" 2>/dev/null || echo "dead -")   # the variable unset: the hook's presence, not its behaviour under it; a plain python, as the route's
  case "$RF6" in
    "live $(realpath "$HERE/opt/chai1_opt")") ;;
    live\ *) echo "run.sh: CHAI1_OPT=$ENVMODE is set and chai1_opt_autoload.pth is processed, but a STALE COPY answers: chai1_opt imported from ${RF6#live } in a fresh interpreter, not $HERE/opt/chai1_opt — reinstall this tree (pip install -e <release>/common/opt_core -e $HERE/opt)" >&2; exit 3 ;;
    *) DIAG=$(python -c "
import os, site, sys, sysconfig
name = 'chai1_opt_autoload.pth'
sites = [sysconfig.get_paths()['purelib']] + list(site.getsitepackages()) + [site.getusersitepackages()]
hit = next((d for d in sites if os.path.isfile(os.path.join(d, name))), None)
if hit: print(f'the .pth is present in the site {hit} but not processed (site.py did not import chai1_opt._autoload there: a broken install, the user site disabled — PYTHONNOUSERSITE / -s — or a site-less interpreter)'); sys.exit()
other = [d for d in os.environ.get('PYTHONPATH', '').split(os.pathsep) if d and os.path.isfile(os.path.join(d, name))]
print(f'the .pth is present at {other[0]}/{name} but not in a site directory of this interpreter (site.py processes .pth files in site-packages only; a copy on PYTHONPATH is not processed)' if other else f'the .pth is absent from the sites searched: ' + ', '.join(sites))
" 2>/dev/null || echo "the .pth could not be searched for")
       echo "run.sh: CHAI1_OPT=$ENVMODE is set but the autoload hook is not live in a fresh $(command -v python) (chai1_opt._autoload not in sys.modules at start): $DIAG — the env route would run stock silently in this environment; install the package (pip install -e <release>/common/opt_core -e $HERE/opt), or unset CHAI1_OPT and pass --mode $MODE (the driver route)" >&2; exit 3 ;;
  esac
fi
# /RF6 gate
python -I "$HERE/stock/check_pins.py" --quiet || { echo "run.sh: refused by the pin check (stock/check_pins.py: the line above — the pinned upstream not installed as pinned, or a stack other than the pinned one under --strict-stack / CHAI1_OPT_STRICT_STACK=1)" >&2; exit 3; }
exec python -m chai1_opt "$CMD" --mode "$MODE" ${ARGS[@]+"${ARGS[@]}"}
