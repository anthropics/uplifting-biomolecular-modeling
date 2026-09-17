#!/bin/bash
# Boltz-2 optimized — single entry point: a thin wrapper over `python -m boltz2_opt` (the installed boltz2_opt package + its core: `run.sh install`).
#   run.sh install [--weights DIR]   the install step: the shared core and this package installed editable into the python on PATH (pip install -e ../common/opt_core -e opt; skipped by name when this tree is installed already), then the pin check (stock/check_pins.py); --weights DIR also fills DIR — the Boltz-2 cache, your BOLTZ_CACHE — with upstream's own downloader and checks it (opt/boltz2_opt/weights.py; README.md Setup)
#   run.sh pred   [--config h100] [--mode M] --input <yaml|fasta|dir> [...] --out_dir <dir> [--seed S | --seeds a,b] [SETTINGS] [--allow-partial] [--no-compile]   exact|fast|big: the kits' persistent worker on the YAMLs (stock's file set per (input, seed) under by_seed/)
#   run.sh pred   [--config h100] --mode off --input <yaml|fasta|dir> --out_dir <dir> [--seed S] [SETTINGS] [--no_kernels] [--det 1] [--no-compile] [-- <any other boltz predict args>]   the stock CLI in a clean subprocess
#                 SETTINGS = boltz predict's own options, upstream's spelling and defaults, the same on every mode: [--checkpoint P] [--recycling_steps 3] [--sampling_steps 200] [--diffusion_samples 1] [--max_parallel_samples 5] [--step_scale 1.5] [--write_full_pae] [--write_full_pde] [--output_format mmcif|pdb] [--num_workers N] [--override]
#                            [--use_msa_server] [--msa_server_url U] [--msa_pairing_strategy greedy] [--use_potentials] [--method M] [--preprocessing-threads T] [--affinity_mw_correction] [--sampling_steps_affinity 200] [--diffusion_samples_affinity 5] [--affinity_checkpoint P] [--max_msa_seqs 8192] [--subsample_msa] [--num_subsampled_msa 1024] [--write_embeddings]   (STOCK.md 'How stock is run')
#   run.sh check  [--config h100] [--mode M] [--allow-partial] [--no-compile]   dry run: resolves, gates and plans the partials of the mode on this box; nothing is applied
#   run.sh warm   [--config h100] [--mode M] --out <dir> [--allow-partial] [--no-compile]              the four bundled 1BRS inputs (398 / 480 / 576 / 1056 tokens: every token-count class and kernel window) through pred in one worker: the mode's Triton / NVRTC kernels compile
# The ENV route (the mode from BOLTZ2_OPT, no --mode on the command line) first requires the kit's hook LIVE in `python`'s site — boltz2_opt._autoload in
# sys.modules of a fresh interpreter, the effect of boltz2_opt_autoload.pth (opt/boltz2_opt/pth_gate.py: exit 3 by name with the diagnostic; the same hook
# makes BOLTZ2_OPT=<mode> refuse in every stock process, so without it that route runs stock silently); `run.sh install` puts it there. --mode is not gated.
# --config <cfg> sources configs/<cfg>.env: deployment parameters only (target GPU, JIT-cache keying; no path default — STOCK.md Variables). Mode = --mode when given, else
# BOLTZ2_OPT from the environment, else the package default (fast; BOLTZ2_OPT unset applies nothing); a --mode that disagrees with a set BOLTZ2_OPT is refused. Modes (off|exact|fast|big)
# are resolved by the package from its mode table (opt/boltz2_opt/modes.py); this script validates none of them. `--mode off` is the stock route:
# the package's one stock caller (opt/boltz2_opt/stock_pred.py — upstream's `boltz predict`, nothing from the kits on the path) in a clean subprocess
# that proves its environment; `pred` is its only command (the stock arm has no warm-up).
# Every route refuses (rc 3) unless boltz 2.2.1 is installed with the pinned tree (stock/check_pins.py), the package is installed and its shared
# core is the pinned release (`[boltz2-opt] NOT ACTIVE: reason=core_missing:<module>` from the real entry at an absent or older core). A lever of
# the row the kit ran without its optimization is a partial activation: `[boltz2-opt] NOT ACTIVE: partial activation — <detail>; exit 3
# (--allow-partial records and proceeds)`, rc 3, unless --allow-partial (pred, warm, check) accepts it, named on the `PARTIAL allowed:` line.
# --no-compile = MODEL_OPT_LEVERS_OFF=compile, accepted on every verb and mode: no mode of this kit engages torch.compile and stock `boltz predict`
# compiles nothing, so the ACTIVE / DRY-RUN lines print compile=off:none_in_kit (compile=off:user with the flag); nothing else changes.
# Exit codes: 0 ok, 1 failed / outputs short, 2 usage, 3 not active / pins not met / partial activation without --allow-partial, 5 an expected accelerator absent (KERNELS-REFUSED).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
usage() { sed -n '2,20p' "$0" >&2; exit 2; }
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
case "$CMD" in install|pred|check|warm) ;; *) usage ;; esac
if [ "$CMD" = install ]; then                                       # the install step: everything below it presupposes the installed package; not a route — BOLTZ2_OPT is dropped for its interpreters
  WEIGHTS=""
  [ -z "$CFG" ] && [ -z "$MODE" ] || { echo "run.sh: install takes no --config / --mode (usage: run.sh install [--weights DIR])" >&2; exit 2; }
  set -- ${ARGS[@]+"${ARGS[@]}"}
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights)   [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
      --weights=*) WEIGHTS=${1#--weights=}; [ -n "$WEIGHTS" ] || { echo "run.sh: install --weights= takes a directory (usage: run.sh install [--weights DIR])" >&2; exit 2; }; shift ;;
      *) echo "run.sh: install takes no argument '$1' (usage: run.sh install [--weights DIR])" >&2; exit 2 ;;
    esac
  done
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md Install)" >&2; exit 3; }
  if env -u BOLTZ2_OPT python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('boltz2_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: boltz2_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  env -u BOLTZ2_OPT python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but boltz is not installed at the pin (stock/check_pins.py: the line above; stock/PINS.json, README.md Install)" >&2; exit 3; }
  if [ -n "$WEIGHTS" ]; then env -u BOLTZ2_OPT python -m boltz2_opt.weights "$WEIGHTS" || exit $?; fi   # upstream's downloader into DIR, then the cache and digest check (opt/boltz2_opt/weights.py)
  exit 0
fi
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 2) when boltz2_opt is not installed; any other import-time failure or refusal (rc 3: an undeclared BOLTZ2_OPT* name) passes through with its own words and exit code
fi
KIT=boltz2   # the tag of the [install]-jitcache line below
_JIT0="${MODEL_OPT_JIT_ROOT:-}"   # the root the config saw, so a variable derived from it can follow a root the block below moves
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
if [ -n "${MODEL_OPT_JIT_ROOT:-}" ] && [ -n "${MODEL_OPT_STACK_KEY:-}" ] && { [ -z "${TRITON_CACHE_DIR:-}" ] || { [ -n "$_JIT0" ] && [ "${TRITON_CACHE_DIR:-}" = "$_JIT0/$MODEL_OPT_STACK_KEY/triton" ]; }; }; then export TRITON_CACHE_DIR=$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/triton; fi; unset _JIT0   # the configs' own derivation, repeated for a root the block above chose or moved (a TRITON_CACHE_DIR the user set elsewhere is kept)
FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
ENVMODE=${BOLTZ2_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with BOLTZ2_OPT=$ENVMODE from $FROM; a run has one mode — drop one of them" >&2; exit 2
fi
CLIMODE=$MODE; MODE=${MODE:-$ENVMODE}
if [ -z "$CLIMODE" ] && [ -n "$ENVMODE" ] && [ "$ENVMODE" != off ]; then   # the ENV route only: the selection came from BOLTZ2_OPT — the kit's hook must be LIVE in python's site (opt/boltz2_opt/pth_gate.py)
  env -u BOLTZ2_OPT python "$HERE/opt/boltz2_opt/pth_gate.py" || exit 3   # a fresh interpreter in the route's own form (no -I: the user site and PYTHONPATH count as they do for the route), the variable unset: the probe asks only whether the hook is live
fi
env -u BOLTZ2_OPT python -I "$HERE/stock/check_pins.py" --quiet || { echo "run.sh: boltz is not installed at the pin (stock/PINS.json, STOCK.md)" >&2; exit 3; }   # the pin check is not a route: the mode variable unset, so the hook's refusals never masquerade as a pin failure
if [ "$MODE" = off ]; then                                        # stock route: `pred --mode off` = the package's stock caller in a clean subprocess
  [ "$CMD" = pred ] || { echo "run.sh: --mode off has one command, pred (the stock arm has no warm-up)" >&2; exit 2; }
fi
PROBE_RC=0; PROBE_ERR=$(env -u BOLTZ2_OPT python -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('boltz2_opt') else 5)" 2>&1 >/dev/null) || PROBE_RC=$?   # is the package on python's path — the mode variable unset on the probe (the mode belongs to the real entry below, whose interpreter runs the hook and prints its own NOT ACTIVE line); no import: the core is the real entry's to judge
if [ "$PROBE_RC" = 5 ]; then echo "run.sh: boltz2_opt is not installed on $(command -v python || echo 'python (not on PATH)'): $HERE/run.sh install" >&2; exit 3; fi   # only the package absent from python's path is "not installed"
[ "$PROBE_RC" = 0 ] || { printf '%s\n' "$PROBE_ERR" >&2; exit "$PROBE_RC"; }   # any other probe failure passes through with its own words and exit code
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
exec python -m boltz2_opt "$CMD" ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}   # the real entry: an absent or older shared core refuses there by name ([boltz2-opt] NOT ACTIVE: reason=core_missing:<module>; exit 3), before anything resolves
