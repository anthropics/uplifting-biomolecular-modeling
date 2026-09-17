#!/bin/bash
# AlphaFold 3 open code + the ported OF3-p2 checkpoint, optimized — single entry point: a thin wrapper over `python -m af3_jax_opt` (the installed af3_jax_opt package and its core: `pip install -e ../common/opt_core -e opt`).
#   run.sh pred    [--config h100] --variant p2 [--mode M] --output_dir <out> (--json_path <json> | --input_dir <dir>) [stock flags]  one model process; the stock script's own flags (--num_recycles, --num_diffusion_samples, --buckets, --cache_dir on off, ...) pass through verbatim
#   run.sh check   [--config h100] --variant p2 [--mode M] [--json]                dry run: resolves + gates (mode, variant, the parameters' digest, the kit files) on this box; nothing is launched
#   run.sh warm    [--config h100] --variant p2 [--mode M] [--input_dir D]         builds the mode's $CACHE for this box's key — the deterministic recipe's first step (--mode big: the class of the line D sizes to, or --n_est N; neither exits 2)
#   run.sh install [--config h100] [--variant p2] [--weights DIR] [--checkpoint_dir D]   the install step: the shared core and this kit installed editable into the python on PATH, then the pin check (stock/check_pins.py);
#                                                          with a parameters root — --weights DIR, which is then your AF3_JAX_PARAMS_ROOT, or that variable already set — also the public OpenFold3-preview2
#                                                          checkpoint fetched into DIR/checkpoints/ (--checkpoint_dir D: elsewhere; a file already there is kept) and converted with the fork's own converter into
#                                                          DIR/<variant>/, both files' sha256 compared with stock/PINS.json (python -m af3_jax_opt.convert install)
# --variant p2 (the package's variants.VARIANTS; stock/PINS.json "variants"): one variant per process. --config <cfg> sources configs/<cfg>.env:
# deployment parameters only (the fork checkout and interpreter, the parameters root, the cache root, the image's XLA environment, the
# target GPU). Mode = --mode when given, else AF3_JAX_OPT from the environment, else the package default (opt/af3_jax_opt/modes.py, the one
# mode table); a --mode that disagrees with a set AF3_JAX_OPT is refused; the same rule binds --variant and AF3_JAX_VARIANT. Modes and
# variants are validated by the package alone (modes.py resolves a mode from the add-ons' own switch rows — opt/forward/fast_inference/rows.sh
# FAST=, modes.PALLAS_LEVERS for the Pallas add-on, the FlashPairformer add-on's package switches; variants.py names the variant);
# this script carries no table of either and validates neither. `--mode off` is the stock route: the fork's run_alphafold.py in a clean
# environment that proves itself (opt/af3_jax_opt/stock_pred.py); its own flags pass through verbatim (a --cache_dir names the cache it
# shares with a kit mode's class; unstated: a fresh JAX cache per pass); `pred`, `warm` and `check` are its commands.
# Every route runs the pin check first (stock/check_pins.py): a stack other than the pin — another package or Python release, another XLA
# setting — is NAMED (`[af3-jax-opt] PINS drift …` lines) and the route runs; it refuses (rc 3) only when the kit's files, a usable interpreter or
# the pinned stock install are absent (its diagnostic line is printed above the refusal). install installs the package, then runs that same check.
# Exit codes: 0 ok (a cold cache class, or a kernel that kept stock arithmetic at some call sites, is named on the DONE line, not an exit code), 1 failed, 2 usage, 3 not active (refused by name, pins not met, or a lever of the mode engaged nowhere: DONE status=partial), 5 an attention implementation the mode counts on was absent or fell back (KERNELS ... REFUSED).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,22p' "$0" >&2; exit 2; }
CFG=""; CMD=""; MODE=""; VARIANT=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config)    [ $# -ge 2 ] || usage; CFG=$2; shift 2 ;;
    --config=*)  CFG=${1#--config=}; shift ;;
    --mode)      [ $# -ge 2 ] || usage; MODE=$2; shift 2 ;;
    --mode=*)    MODE=${1#--mode=}; shift ;;
    --variant)   [ $# -ge 2 ] || usage; VARIANT=$2; shift 2 ;;
    --variant=*) VARIANT=${1#--variant=}; shift ;;
    *) if [ -z "$CMD" ]; then CMD=$1; else ARGS+=("$1"); fi; shift ;;
  esac
done
case "$CMD" in pred|check|warm|install) ;; *) usage ;; esac
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  [ "$CMD" = install ] || { source "$HERE/configs/$CFG.env" || exit $?; }   # the config refuses (rc 2) when af3_jax_opt is not importable; install sources it after its pip step
fi
[ -n "${MODEL_OPT_JIT_ROOT:-}" ] || [ -z "${AF3_JAX_CACHE_ROOT:-}" ] || export MODEL_OPT_JIT_ROOT="$AF3_JAX_CACHE_ROOT"   # before the block: the kit's compile-cache root names the root
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[af3_jax-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[af3_jax-kit] jit cache: $J ($W)"; }
[ -z "${MODEL_OPT_JIT_ROOT:-}" ] || export AF3_JAX_CACHE_ROOT="${AF3_JAX_CACHE_ROOT:-$MODEL_OPT_JIT_ROOT}"                # after the block: the kit modes' classes (<GPU model>__jax<v>_jaxlib<v>[__mode]) live under the root; --mode off keeps its empty class
FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
ENVMODE=${AF3_JAX_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with AF3_JAX_OPT=$ENVMODE from $FROM; one mode per process — drop one of them" >&2; exit 2
fi
MODE_GIVEN=$MODE                                                   # the --mode route: activation by construction (the wrapper imports the package); the env route below is gated
MODE=${MODE:-$ENVMODE}
ENVVARIANT=${AF3_JAX_VARIANT:-}
if [ -n "$VARIANT" ] && [ -n "$ENVVARIANT" ] && [ "$VARIANT" != "$ENVVARIANT" ]; then
  echo "run.sh: --variant $VARIANT disagrees with AF3_JAX_VARIANT=$ENVVARIANT from $FROM; one variant per process — drop one of them" >&2; exit 2
fi
VARIANT=${VARIANT:-$ENVVARIANT}
if [ "$CMD" = install ]; then                                       # the install step (README.md 'Install'): everything below it presupposes the installed package
  [ -z "$MODE" ] || { echo "run.sh: install takes no --mode (usage: run.sh install [--config h100] [--variant p2] [--weights DIR] [--checkpoint_dir D])" >&2; exit 2; }
  WEIGHTS=""; STEPARGS=()
  set -- ${ARGS[@]+"${ARGS[@]}"}
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --weights takes a directory, the parameters root (usage: run.sh install [--config h100] [--variant p2] [--weights DIR] [--checkpoint_dir D])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
      --weights=*) WEIGHTS=${1#*=}; [ -n "$WEIGHTS" ] || { echo "run.sh: install --weights= takes a directory" >&2; exit 2; }; shift ;;
      *) STEPARGS+=("$1"); shift ;;                                # the parameters step's own flags (--checkpoint_dir D), handed to it verbatim
    esac
  done
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md Install)" >&2; exit 3; }
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('af3_jax_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: af3_jax_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  if [ -n "$CFG" ]; then source "$HERE/configs/$CFG.env" || exit $?; fi   # now that the package it probes is installed: AF3_JAX_REPO, AF3_JAX_PY and the pinned XLA variables, each where unset
  python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the line above — the pinned stock is not installed in the environment AF3_JAX_PY / AF3_JAX_REPO name: --config h100 or README.md Install)" >&2; exit 3; }
  [ -z "$WEIGHTS" ] || export AF3_JAX_PARAMS_ROOT=$WEIGHTS                  # --weights DIR: the parameters root of this step, and your AF3_JAX_PARAMS_ROOT afterwards
  if [ -n "${AF3_JAX_PARAMS_ROOT:-}" ]; then                          # the parameters step: the public checkpoint fetched (kept when already there), converted with the fork's own converter, both digests compared with stock/PINS.json (opt/af3_jax_opt/convert.py)
    VARARGS=(); [ -n "$VARIANT" ] && VARARGS=(--variant "$VARIANT")
    python -m af3_jax_opt.convert install ${VARARGS[@]+"${VARARGS[@]}"} ${STEPARGS[@]+"${STEPARGS[@]}"} || exit $?
  elif [ ${#STEPARGS[@]} -gt 0 ] || [ -n "$VARIANT" ]; then
    echo "run.sh: install ${VARIANT:+--variant $VARIANT }${STEPARGS[*]-}: the parameters step needs its root — --weights DIR (DIR is then your AF3_JAX_PARAMS_ROOT), or AF3_JAX_PARAMS_ROOT set" >&2; exit 2
  else
    echo "run.sh: installed and the pin check passed (PINS met, or drift named on the line above); no parameters root named (--weights DIR, or AF3_JAX_PARAMS_ROOT) — the checkpoint is not fetched: run.sh install --weights DIR does that"
  fi
  exit 0
fi
_PROBE=$(python -c "import af3_jax_opt" 2>&1) || { _RC=$?                # the import probe: the package ABSENT is named here (rc 3); any other failure of the import — a refusal at start-up, a broken dependency — keeps the interpreter's own words and exit code
  case "$_PROBE" in *"No module named 'af3_jax_opt'"*) echo "run.sh: af3_jax_opt is not installed on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/../common/opt_core -e $HERE/opt" >&2; exit 3 ;; esac
  printf '%s\n' "$_PROBE" >&2; echo "run.sh: importing af3_jax_opt failed on $(command -v python || echo 'python (not on PATH)') (rc $_RC; the interpreter's words above)" >&2; exit "$_RC"; }
python -I "$HERE/stock/check_pins.py" --quiet >&2 || { echo "run.sh: pins not met — the pinned stock is not installed where AF3_JAX_PY / AF3_JAX_REPO name, or a kit file is absent (the line above; stock/check_pins.py); stack drift alone never lands here" >&2; exit 3; }
if [ "$MODE" = off ]; then                                        # stock route: pred / warm with the fork's own script in a clean environment
  case "$CMD" in pred|warm|check) ;; *) echo "run.sh: --mode off has the commands pred, warm and check" >&2; exit 2 ;; esac
elif [ -z "$MODE_GIVEN" ] && [ -n "$ENVMODE" ]; then                # the env route (AF3_JAX_OPT names a kit mode, no --mode): the start-up hook must be LIVE in this interpreter — importable alone, that variable with the fork's run_alphafold.py would run stock silently
  case "$CMD" in pred|check)
    HOOK=$(env -u AF3_JAX_OPT python -c '
import os, site, sys
if "af3_jax_opt._autoload" in sys.modules:                         # the site-processed .pth took effect in this fresh interpreter, the form the route itself runs: the user site (when enabled) and PYTHONPATH count exactly as they do for python -m af3_jax_opt; the variable unset: the probe sees the hook, not its refusal
    print("live"); sys.exit(0)
pth = "af3_jax_opt_autoload.pth"
user = site.getusersitepackages(); sites = list(site.getsitepackages()) + [user]
here = [d for d in sites if os.path.isfile(os.path.join(d, pth))]
if here:
    import re
    got = open(os.path.join(here[0], pth)).read().strip(); kit = open(sys.argv[1]).read().strip()
    hook = lambda text: (re.search(r"import (\w+\._autoload)", text) or [None, text[:60]])[1]   # the module the guarded .pth line imports (opt/_build_backend.py pth_text)
    if got != kit:
        print(f"{pth} a stale copy: {os.path.join(here[0], pth)} imports {hook(got)!r}, the kit ships {hook(kit)!r}" + ("" if hook(got) != hook(kit) else f" with another guard text than {sys.argv[1]} (reinstall: pip install -e opt)")); sys.exit(0)
    if here[0] == user and not site.ENABLE_USER_SITE:
        print(f"{pth} present but not processed: in {user} (the user site is disabled for this interpreter: PYTHONNOUSERSITE or -s)"); sys.exit(0)
    print(f"{pth} present but not processed: in {here[0]} (site.py ran its import line and it failed: the package is not importable from that interpreter)"); sys.exit(0)
on_path = [d for d in os.environ.get("PYTHONPATH", "").split(os.pathsep) if d and os.path.isfile(os.path.join(d, pth))]
if on_path:
    print(f"{pth} present but not processed: on PYTHONPATH at {on_path[0]} (site.py processes .pth files in site-packages only)"); sys.exit(0)
print(f"{pth} absent from the searched sites: {chr(32).join(sites)}" + ("" if site.ENABLE_USER_SITE else " (the user site is disabled: PYTHONNOUSERSITE or -s)"))
' "$HERE/opt/af3_jax_opt_autoload.pth" 2>/dev/null)
    [ "$HOOK" = live ] || { echo "run.sh: the env route (AF3_JAX_OPT=$ENVMODE from $FROM, no --mode) needs the start-up hook af3_jax_opt._autoload live in $(command -v python) — $HOOK; with the package importable alone, that variable with the fork's run_alphafold.py would run stock silently — install into that interpreter: python -m pip install -e $HERE/../common/opt_core -e $HERE/opt" >&2; exit 3; } ;;
  esac
fi
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
VARARG=();  [ -n "$VARIANT" ] && VARARG=(--variant "$VARIANT")
exec python -m af3_jax_opt "$CMD" ${VARARG[@]+"${VARARG[@]}"} ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
