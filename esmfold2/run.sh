#!/bin/bash
# ESMFold2 optimized — single entry point: a thin wrapper over `python -m esmfold2_opt` (the installed esmfold2_opt package: `pip install -e opt`).
#   run.sh pred   [--config h100] --variant V [--mode M] [--n_gpu P] --input <json> --out_dir <dir> [--seeds S] [--num_loops N …fold flags] [--det 0|1|2] [--backend fused|shipped]  one process, one variant; the kit driver's file set per (input, seed, sample)
#   run.sh check  [--config h100] --variant V [--mode M]   dry run: resolves the mode against the server's own mode table on this GPU; nothing is applied
#   run.sh warm   [--config h100] --variant V [--mode M]   imports + Triton JIT to fill the caches
#   run.sh install [--weights DIR]                                            the install step: the shared core and this kit installed editable into the python on PATH, then the pin check (stock/check_pins.py);
#                                                                             --weights DIR also fetches the sixteen pinned weight files into DIR (the HF_HOME the other verbs read) with huggingface_hub's downloader at the pinned snapshots and checks each sha256 (stock/PINS.json "weights")
# --variant fast|full_msa|full_nomsa (the package's modes.VARIANTS; stock/PINS.json "variants"): one variant per process, never two. --config <cfg>
# sources configs/<cfg>.env: deployment parameters only (weights cache, CCD pickle, JIT cache, target GPU). Mode = --mode when given, else
# ESMFOLD2_OPT from the environment, else the package default (opt/esmfold2_opt/modes.py DEFAULT_MODE — the one constant; this script names no
# default); a --mode that disagrees with a set ESMFOLD2_OPT is refused; the same rule
# binds --variant and ESMFOLD2_VARIANT. `--n_gpu P` is the resource axis (opt/esmfold2_opt/tp.py; explicit, default 1, never auto-detected): this kit ships P in {1, 2, 4, 8} — P > 1 only under `--mode big` (P rank processes, the row-sharded pair stack `rowpair.py`), refused by name under off / exact / fast (exit 2) and for any P outside the shipped set (exit 3).
# Modes (off|exact|fast|big) are resolved by the package from the kit server's own table
# (opt/forward/fast_inference/driver/ef2_server.py MODES); this script validates none of them. `--mode off` is the stock route: the package's
# one stock caller (opt/esmfold2_opt/stock_fold.py — the upstream API, nothing from the kit on the path) in a clean subprocess that proves
# its environment; `pred` is its only command (the stock arm has no warm).
# Every route refuses (rc 3) unless both pinned forks are installed at their pinned commits (stock/check_pins.py) and the package is
# installed (`install` is the step that installs it). Exit codes: 0 ok, 1 failed or incomplete (a fold, the install, a weight file off its pin), 2 usage, 3 not active / pins not met / a lever of the mode's set cannot run on the card (a mode is all of its levers: refused by name, nothing folded).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export MODEL_OPT=$HERE
ef2_env_clean() { local u=() v; for v in $(compgen -e); do case "$v" in ESMFOLD2_OPT*) u+=(-u "$v") ;; esac; done; env ${u[@]+"${u[@]}"} "$@"; }   # a command with every ESMFOLD2_OPT* name removed (the .pth hook then imports nothing and refuses nothing)
ef2_pkg_absent() { local rc=0; ef2_env_clean python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('esmfold2_opt') is not None else 41)" 2>/dev/null || rc=$?; [ "$rc" -eq 41 ]; }   # true ONLY on the probe's own sentinel (41 = find_spec found no esmfold2_opt); any other exit of that interpreter (a start-up hook refusing, a crash) is NOT absence
usage() { sed -n '2,18p' "$0" >&2; exit 2; }
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
if [ "$CMD" = install ]; then                                       # the install step: everything below it presupposes the installed package
  WEIGHTS=""
  [ -z "$CFG" ] && [ -z "$MODE" ] && [ -z "$VARIANT" ] || { echo "run.sh: install takes no --config / --mode / --variant (usage: run.sh install [--weights DIR])" >&2; exit 2; }
  set -- ${ARGS[@]+"${ARGS[@]}"}
  while [ $# -gt 0 ]; do
    case "$1" in
      --weights) [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#--}" = "$2" ] || { echo "run.sh: install --weights takes a directory (usage: run.sh install [--weights DIR])" >&2; exit 2; }; WEIGHTS=$2; shift 2 ;;
      --weights=*) WEIGHTS=${1#*=}; [ -n "$WEIGHTS" ] || { echo "run.sh: install --weights= takes a directory" >&2; exit 2; }; shift ;;
      *) echo "run.sh: install takes no argument '$1' (usage: run.sh install [--weights DIR])" >&2; exit 2 ;;
    esac
  done
  command -v python >/dev/null || { echo "run.sh: no python on PATH — activate the environment this kit installs into (README.md Install)" >&2; exit 3; }
  if python -I -c "import os,sys,importlib.util as u; t=[os.path.realpath(p) for p in sys.argv[1:3]]; s=[u.find_spec(n) for n in ('esmfold2_opt','opt_core')]; sys.exit(0 if all(x and x.origin and os.path.realpath(x.origin).startswith(d+os.sep) for x,d in zip(s,t)) else 1)" "$HERE/opt" "$HERE/../common/opt_core" 2>/dev/null; then
    echo "run.sh: esmfold2_opt and opt_core are installed from this tree already ($HERE/opt, $HERE/../common/opt_core) — the pip step is skipped"   # the container image ships them installed; a read-only image cannot re-run pip
  else
    ef2_env_clean python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" || { echo "run.sh: the install failed (pip's words above): the shared core is expected at $HERE/../common/opt_core, the kit package at $HERE/opt" >&2; exit 1; }
  fi
  python -I "$HERE/stock/check_pins.py" || { echo "run.sh: installed, but refused by the pin check (stock/check_pins.py: the line above — the pinned upstream forks or the pinned stack's accelerated layer are not installed as pinned in this environment)" >&2; exit 3; }
  ef2_env_clean python -I "$HERE/opt/forward/fast_inference/driver/ef2_nvjit.py" cubins \
    || { echo "run.sh: installed, but the kit's shipped CUDA kernels (opt/forward/fast_inference/driver/prebuilt/sm_90a: the cubins of the fast / big modes' compute-capability-9.0 levers t16 / k3cute) do not match their manifest and this tree's kernel sources (the line above); rebuild them with driver/prebuilt/build_prebuilt.py" >&2; exit 1; }   # CPU-only: manifest present, cubin sha256s match, source keys == this tree's sources
  if [ -n "$WEIGHTS" ]; then ef2_env_clean python -m esmfold2_opt.weights "$WEIGHTS" || exit $?; fi   # huggingface_hub's downloader into DIR at the pinned snapshots, then the sha256 check against stock/PINS.json (opt/esmfold2_opt/weights.py)
  exit 0
fi
case "$CMD" in pred|check|warm) ;; *) usage ;; esac
# [install]-jitcache v4.3 — seed the compile caches shipped in the image, or this stack's key dir of a read-only preset root; with no preset root and no image cache, a private per-user root (identical in every kit; KIT = the kit's log tag)
J="${MODEL_OPT_JIT_ROOT:-}"; I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"; W=""; N="${MODEL_OPT_JIT_SEED_MAX_FILES:-5000}"; U=$(id -u); T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"
case "$N" in ''|*[!0-9]*) echo "run.sh: MODEL_OPT_JIT_SEED_MAX_FILES is a file count in digits, not '$N'" >&2; exit 2 ;; esac
seedroot() {   # the writable copy's root, per user (the uid ends its name): made here with mode 0700, or made so by an earlier run of this user; a path another user owns or can write, or a symbolic link, is neither written through nor read
  mkdir -p "${T%/*}" 2>/dev/null || :; mkdir -m 700 "$T" 2>/dev/null || :
  if [ -d "$T" ] && [ ! -L "$T" ] && [ -O "$T" ]; then case $(stat -c %a "$T" 2>/dev/null) in ''|*[2367]|*[2367]?) ;; *) return 0 ;; esac; fi   # group / other write bits in the mode's last two digits refuse it too
  echo "[esmfold2-kit] jit cache: $T refused (another owner, open to group or others, a symbolic link, or not creatable): nothing is seeded there" >&2; return 1
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
[ -z "$W" ] || { export MODEL_OPT_JIT_ROOT="$J"; echo "[esmfold2-kit] jit cache: $J ($W)"; }
if [ -n "$CFG" ]; then
  [ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }
  source "$HERE/configs/$CFG.env" || exit $?            # the config refuses (rc 2) when esmfold2_opt is not importable
fi
FROM="the environment"; [ -n "$CFG" ] && FROM="configs/$CFG.env"
ENVMODE=${ESMFOLD2_OPT:-}
if [ -n "$MODE" ] && [ -n "$ENVMODE" ] && [ "$MODE" != "$ENVMODE" ]; then
  echo "run.sh: --mode $MODE disagrees with ESMFOLD2_OPT=$ENVMODE from $FROM; one run has one mode — drop one of them" >&2; exit 2
fi
MODE_CLI=$MODE                                                      # the mode as given on the command line (empty: the env route)
MODE=${MODE:-$ENVMODE}
ENVVARIANT=${ESMFOLD2_VARIANT:-}
if [ -n "$VARIANT" ] && [ -n "$ENVVARIANT" ] && [ "$VARIANT" != "$ENVVARIANT" ]; then
  echo "run.sh: --variant $VARIANT disagrees with ESMFOLD2_VARIANT=$ENVVARIANT from $FROM; one variant per process — drop one of them" >&2; exit 2
fi
VARIANT=${VARIANT:-$ENVVARIANT}
case "$VARIANT" in ""|fast|full_msa|full_nomsa) ;; *) echo "run.sh: '$VARIANT' is not a variant (fast|full_msa|full_nomsa, stock/PINS.json)" >&2; exit 2 ;; esac
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}   # frozen weights: nothing is fetched at run time (huggingface_hub / transformers' documented switches; configs/h100.env sets them too)
case "$CMD" in pred|warm)                                       # the verbs that load weights refuse by name when the weights root is absent (never a silent download / default-cache read); the package's data_path_gate then checks every pinned file under it
  [ -n "${HF_HOME:-}" ] || { echo "[esmfold2-opt] NOT ACTIVE: HF_HOME (the weights root: the HuggingFace cache holding the pinned snapshots, stock/PINS.json \"weights\") is not set — with HF_HUB_OFFLINE=1 nothing is fetched and the library's default cache is never read: export HF_HOME=<that cache> or pass --config <gpu> (configs/<gpu>.env)" >&2; exit 3; }
  [ -d "$HF_HOME" ] || { echo "[esmfold2-opt] NOT ACTIVE: HF_HOME=$HF_HOME does not exist (the weights root: stock/PINS.json \"weights\")" >&2; exit 3; } ;;
esac
if [ -z "$MODE_CLI" ] && [ -n "$ENVMODE" ] && [ "$ENVMODE" != off ]; then case "$CMD" in pred|check)   # RF-6 (env route only: the mode from ESMFOLD2_OPT, none on the command line):
  # the two IMPORTABILITY probes run with the activation variables removed (`env -u`): with them set, an installed .pth runs the kit's core gate at
  # interpreter start, and a silenced probe killed by that gate would read as 'not importable' / 'predates' instead of the gate's own line; the LIVENESS
  # probe after them keeps the variables and is not silenced — a refused core prints its `NOT ACTIVE: reason=…` line there (exit 3)
  if ef2_env_clean python -c "import esmfold2_opt" 2>/dev/null; then                      # the hook must be LIVE in a fresh interpreter (the .pth's effect), else the stock CLI under the variable runs stock silently
    ef2_env_clean python -c "import esmfold2_opt.pth_gate" 2>/dev/null || { echo "[esmfold2-opt] NOT ACTIVE: the installed esmfold2_opt ($(ef2_env_clean python -c 'import esmfold2_opt, os; print(os.path.dirname(esmfold2_opt.__file__))' 2>/dev/null)) predates the autoload gate (no esmfold2_opt.pth_gate): install this tree into the interpreter (pip install -e $HERE/opt)" >&2; exit 3; }
    python -m esmfold2_opt.pth_gate || exit $?
  fi ;; esac
fi
# the kit's core pin gate (esmfold2_opt/_core_gate.py: the importable opt_core is the one opt/pyproject.toml [tool.opt_core] pins, read on disk) and its
# producer census (esmfold2_opt._producers), BEFORE anything resolves — the same two statements `python -m esmfold2_opt` and the .pth hook run first:
# `[esmfold2-opt] NOT ACTIVE: reason=core_missing:opt_core …` / `reason=core_mismatch: …` / `producer_missing:<modules> …`, the package itself absent -> named;
# exit 3 each, never a traceback, never a silent stock run; any other failure of that interpreter keeps its own text and exit code ('not installed' names only an absent package)
python -m esmfold2_opt._core_gate "$HERE/opt" >/dev/null || { rc=$?; if ef2_pkg_absent; then echo "[esmfold2-opt] NOT ACTIVE: esmfold2_opt is not installed on $(command -v python || echo 'python (not on PATH)'): pip install -e $HERE/opt" >&2; exit 3; fi; exit "$rc"; }   # a present package that refused or failed keeps ITS stderr (verbatim, above) and ITS exit code; 'not installed' names only a genuinely absent package
python -m esmfold2_opt._producers || exit 3
python -I "$HERE/stock/check_pins.py" --quiet || { echo "run.sh: the pinned upstream forks are not installed at their pinned commits (stock/PINS.json)" >&2; exit 3; }
if [ "$MODE" = off ]; then                                        # stock route: `pred --mode off` = the package's stock caller in a clean subprocess
  [ "$CMD" = pred ] || { echo "run.sh: --mode off has one command, pred (the stock arm has no warm)" >&2; exit 2; }
  [ -n "$VARIANT" ] || { echo "run.sh: --variant is required" >&2; exit 2; }
fi
MODEARG=(); [ -n "$MODE" ] && MODEARG=(--mode "$MODE")
VARARG=();  [ -n "$VARIANT" ] && VARARG=(--variant "$VARIANT")
exec python -m esmfold2_opt "$CMD" ${VARARG[@]+"${VARARG[@]}"} ${MODEARG[@]+"${MODEARG[@]}"} ${ARGS[@]+"${ARGS[@]}"}
