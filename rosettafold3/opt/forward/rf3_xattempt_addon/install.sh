#!/bin/bash
# install.sh — install / classify / check / uninstall the kit's five patched rf3 files in the active Python environment (foundry's `rf3` package).
#   bash install.sh                 apply: checks that the 5 target files in site-packages/rf3 are in a KNOWN state (pristine upstream foundry
#                                   4010e3e, or already these files), backs the originals up to <file>.hoist_orig once (the new file gets an empty
#                                   <file>.hoist_absent), copies patched/rf3/* over them, checks the installed bytes == patched/rf3/*, and writes the
#                                   overlay marker rf3/.rosettafold3_opt_overlay.json (the overlay id + the sha256 of every installed file).
#                                   Any other state -> refuses (exit 4) unless --force (the manual escape: backs those files up and installs over them).
#   bash install.sh --classify      exactly one line `OVERLAY_PRESTATE: pristine|current|previous:<id>|foreign:<files>` (exit 0), the state apply would meet:
#                                     pristine        the 4 upstream files byte-equal to the pinned upstream (STOCK_SRC), rf3/graph_flags.py absent
#                                     current         the 5 files == patched/
#                                     previous:<id>   every file not == patched/ is positively an earlier install of THESE files: recorded in the overlay
#                                                     marker a previous apply wrote, byte for byte (<id> = the first 12 hex of that apply's overlay id; a marked install
#                                                     whose file differs from its record is foreign), else — only for an install older than the marker — carrying this
#                                                     installer's own sidecar (<file>.hoist_orig holding the pinned upstream's bytes, or <file>.hoist_absent for the new
#                                                     file) AND the edits' tags ([rf3_cudagraph] / [xattempt_hoist]) (<id> = unmarked);
#                                                     <id> = partial when the files are a mix of patched/ and pristine upstream (an interrupted apply / uninstall)
#                                     foreign:<files> comma-separated target files in none of those states
#   bash install.sh --rebind        the OVERLAY_PRESTATE line, then on pristine | previous:<id> this tree's files are (re)installed (the pristine .hoist_orig
#                                   backups are kept) and one line `OVERLAY_APPLIED: 5 files id=<overlay id, 12 hex>` follows; on current nothing more happens;
#                                   on foreign exit 4 with no file touched. (The verb for a pre-built environment: whatever an earlier copy of the kit installed,
#                                   this copy's files are what the environment holds afterwards, or the command fails by name.)
#   bash install.sh --check         exit 0 iff the 5 files == patched/rf3/*
#   bash install.sh --uninstall     restore the .hoist_orig backups (back to whatever you had before) and drop the marker
#   bash install.sh --status        print which known state each file is in (patched = this directory's patched/<file>; stock = the pinned upstream's
#                                   file, STOCK_SRC/models/rf3/src/<file> with STOCK_SRC = ../../../stock/src of this kit unless set)
# The 5 files: graph_flags.py [new], inference_sampler.py, RF3_structure.py, af3_diffusion_transformer.py, loss.py — upstream's at the pin plus the
# edits marked [rf3_cudagraph] / [xattempt_hoist] in each (README.md). The overlay id = sha256 over the sorted `<sha256>  <file>` lines of patched/
# (computed here, never stored in the tree).
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); MODE="${1:-apply}"; FORCE=""; [ "${2:-}" = "--force" ] || [ "${1:-}" = "--force" ] && { FORCE=1; [ "$MODE" = "--force" ] && MODE=apply; }
case "$MODE" in apply|--classify|--rebind|--check|--uninstall|--status) ;; *) echo "usage: bash install.sh [apply|--classify|--rebind|--check|--uninstall|--status] [--force]"; exit 2 ;; esac
SP=$(python -c "import rf3, os; print(os.path.dirname(os.path.dirname(os.path.abspath(rf3.__file__))))" | tail -1)
[ -d "$SP/rf3" ] || { echo "cannot locate the rf3 package (python -c 'import rf3' failed?)"; exit 2; }
FILES="rf3/graph_flags.py rf3/diffusion_samplers/inference_sampler.py rf3/model/RF3_structure.py rf3/model/layers/af3_diffusion_transformer.py rf3/loss/loss.py"
NEW="rf3/graph_flags.py"
STOCK_SRC="${STOCK_SRC:-$HERE/../../../stock/src}"
MARKER="$SP/rf3/.rosettafold3_opt_overlay.json"          # written by apply: {"name", "overlay_id", "files": {<file>: sha256}} — the record --classify reads
NAME="rosettafold3_opt patched rf3 files"
TAG='\[rf3_cudagraph\]|\[xattempt_hoist\]'               # every edit is tagged; a file of an earlier install carries the tags too
sha(){ [ -f "$1" ] && sha256sum "$1" | awk '{print $1}' || echo "absent"; }
is_new(){ local n; for n in $NEW; do [ "$n" = "$1" ] && return 0; done; return 1; }
stock_file(){ echo "$STOCK_SRC/models/rf3/src/$1"; }
is_stock_bytes(){ # path -> 0 iff the file's bytes are the pinned upstream's for target file $1
  [ -f "$(stock_file "$1")" ] && cmp -s "$2" "$(stock_file "$1")"; }
state_of(){ # file -> patched|stock|absent-ok|unknown (byte comparison with patched/<file> and the pinned upstream's file)
  local f=$1
  [ -f "$SP/$f" ] || { is_new "$f" && { echo absent-ok; return; }; echo unknown; return; }
  cmp -s "$SP/$f" "$HERE/patched/$f" && { echo patched; return; }
  is_stock_bytes "$f" "$SP/$f" && { echo stock; return; }
  echo unknown; }
files_match(){ for f in $FILES; do [ "$(state_of $f)" = patched ] || return 1; done; return 0; }
overlay_id(){ local f; for f in $FILES; do echo "$(sha "$HERE/patched/$f")  $f"; done | sort | sha256sum | awk '{print $1}'; }
this_id(){ overlay_id | cut -c1-12; }
marker_field(){ # field -> value from the marker ("" when absent / unreadable); files.<file> -> that file's recorded sha256
  [ -f "$MARKER" ] || { echo ""; return 0; }
  python - "$MARKER" "$1" <<'PYEOF' 2>/dev/null || echo ""
import json, sys
d = json.load(open(sys.argv[1])); k = sys.argv[2]
v = (d.get("files") or {}).get(k[len("files."):]) if k.startswith("files.") else d.get(k)
print("" if v is None else v)
PYEOF
}
write_marker(){ # after a successful apply: the overlay id and every installed file's sha256
  local f lines=""; for f in $FILES; do lines="$lines$f $(sha "$SP/$f")"$'\n'; done
  python - "$MARKER" "$(overlay_id)" "$lines" "$NAME" <<'PYEOF'
import json, sys
files = dict(tuple(l.split(" ", 1)) for l in sys.argv[3].splitlines() if l.strip())
json.dump({"name": sys.argv[4], "overlay_id": sys.argv[2], "files": files}, open(sys.argv[1], "w"), indent=1, sort_keys=True)
PYEOF
}
previous_evidence(){ # file -> marker|sidecar|"" : why a file that is neither patched/ nor pristine is positively an earlier install of THESE files
  local f=$1 got rec; got=$(sha "$SP/$f")
  if [ -f "$MARKER" ]; then                                # a marked install answers from its marker only: the file must be byte for byte what that apply recorded
    rec=$(marker_field "files.$f")
    case "$(marker_field name)" in "$NAME") [ -n "$rec" ] && [ "$rec" = "$got" ] && { echo marker; return 0; } ;; esac
    echo ""; return 0; fi
  grep -qE "$TAG" "$SP/$f" 2>/dev/null || { echo ""; return 0; }   # an install older than the marker: this installer's sidecar beside the file AND the edits' tags in it
  if is_new "$f"; then [ -f "$SP/$f.hoist_absent" ] && { echo sidecar; return 0; }
  else [ -f "$SP/$f.hoist_orig" ] && is_stock_bytes "$f" "$SP/$f.hoist_orig" && { echo sidecar; return 0; }; fi
  echo ""; }
classify(){ # -> pristine | current | previous:<id> | foreign:<f1,f2,...>
  local f s e n_patched=0 n_pristine=0 n_marker=0 n_sidecar=0 foreign=""
  for f in $FILES; do s=$(state_of $f)
    case "$s" in patched) n_patched=$((n_patched+1)) ;; stock|absent-ok) n_pristine=$((n_pristine+1)) ;;
      *) e=$(previous_evidence $f); case "$e" in marker) n_marker=$((n_marker+1)) ;; sidecar) n_sidecar=$((n_sidecar+1)) ;; *) foreign="${foreign:+$foreign,}$f" ;; esac ;; esac; done
  if [ -n "$foreign" ]; then echo "foreign:$foreign"
  elif [ $n_patched -eq 5 ]; then echo current
  elif [ $n_pristine -eq 5 ]; then echo pristine
  elif [ $((n_marker+n_sidecar)) -eq 0 ]; then echo "previous:partial"                              # patched/ and pristine upstream files only: an interrupted apply or uninstall
  elif [ $n_sidecar -eq 0 ]; then local i; i=$(marker_field overlay_id); echo "previous:${i:0:12}"
  else echo "previous:unmarked"; fi; }
do_apply(){ # copy patched/ over the 5 files (backups first), check bytes == patched/, write the marker; $1 = quiet|""
  local f
  for f in $FILES; do
    if [ -f "$SP/$f" ]; then [ -f "$SP/$f.hoist_orig" ] || [ -f "$SP/$f.hoist_absent" ] || { is_new "$f" && touch "$SP/$f.hoist_absent" || cp "$SP/$f" "$SP/$f.hoist_orig"; }
    else mkdir -p "$(dirname "$SP/$f")"; touch "$SP/$f.hoist_absent"; fi
    cp "$HERE/patched/$f" "$SP/$f"; done
  find "$SP/rf3" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
  files_match || { echo "byte mismatch after install"; exit 3; }
  write_marker || { echo "patched rf3 files installed but the marker could not be written: $MARKER"; exit 3; }
  [ "${1:-}" = quiet ] || echo "patched rf3 files (RF3_CUDAGRAPH / RF3_HOIST) installed into $SP: 5 files == patched/; 'bash install.sh --uninstall' restores your previous files"; }
case "$MODE" in
  --status) echo "site-packages: $SP"; for f in $FILES; do echo "  $(state_of $f)  $f"; done; exit 0 ;;
  --classify) echo "OVERLAY_PRESTATE: $(classify)"; exit 0 ;;
  --rebind) class=$(classify); echo "OVERLAY_PRESTATE: $class"
     case "$class" in
       current) exit 0 ;;
       pristine|previous:*) do_apply quiet; echo "OVERLAY_APPLIED: 5 files id=$(this_id)"; exit 0 ;;
       *) echo "refusing: the files named are neither pristine upstream (4010e3e), nor these patched files, nor an earlier install of them — nothing was touched"; exit 4 ;;
     esac ;;
  --check) files_match && { echo "patched rf3 files (RF3_CUDAGRAPH / RF3_HOIST) installed in $SP (5 files == patched/)"; exit 0; } || { echo "patched rf3 files NOT installed (or files differ) in $SP; try: bash install.sh --status"; exit 1; } ;;
  --uninstall) n=0; for f in $FILES; do if [ -f "$SP/$f.hoist_orig" ]; then mv -f "$SP/$f.hoist_orig" "$SP/$f"; n=$((n+1)); elif [ -f "$SP/$f.hoist_absent" ]; then rm -f "$SP/$f" "$SP/$f.hoist_absent"; n=$((n+1)); fi; done
     rm -f "$MARKER"; find "$SP/rf3" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; echo "restored $n file(s) in $SP"; exit 0 ;;
  apply)
     class=$(classify)
     case "$class" in pristine|current) ;;
       *) [ -n "$FORCE" ] || { echo "refusing: OVERLAY_PRESTATE: $class — these files are not pristine upstream (4010e3e) and not this kit's patched files (an earlier install of them: bash install.sh --rebind re-installs this copy's; anything else: install.sh --status, then --force to back them up and install over them)"; exit 4; } ;;
     esac
     do_apply ;;
esac
