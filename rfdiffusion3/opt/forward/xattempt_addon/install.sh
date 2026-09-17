#!/bin/bash
# install.sh — install / classify / check / uninstall the overlay in the active Python environment (the installed `rfd3` package of foundry).
#   bash install.sh                 apply: checks that the 10 target files in site-packages/rfd3 are in a KNOWN state (pristine upstream foundry
#                                   4010e3e/faf4299, or already this overlay), backs the originals up to <file>.hoist_orig once (a new file gets an
#                                   empty <file>.hoist_absent), copies patched/rfd3/* over them, checks sha256 == patched/, and writes the overlay
#                                   marker rfd3/.xattempt_addon_overlay.json (kit name, version, overlay id, the sha256 of every installed file).
#                                   Any other state -> refuses (exit 4) unless --force (the manual escape: backs those files up and installs over them).
#   bash install.sh --classify      exactly one line `OVERLAY_PRESTATE: pristine|current|previous:<id>|foreign:<files>` (exit 0), the state apply would meet:
#                                     pristine        the 8 upstream files at 4010e3e/faf4299, the 2 new files absent
#                                     current         the 10 files == patched/
#                                     previous:<id>   every file not == patched/ is positively an earlier install of THIS overlay: recorded in the overlay
#                                                     marker a previous apply wrote, byte for byte (<id> = <marker version>/<first 12 hex of its overlay id>; a marked
#                                                     install whose file differs from its record is foreign), else — only for an install older than the marker —
#                                                     carrying this installer's own sidecar (<file>.hoist_orig holding the pristine upstream bytes, or
#                                                     <file>.hoist_absent for a new file) AND the overlay's `[RFD3_` hunk tags (<id> = unmarked);
#                                                     <id> = partial when the files are a mix of patched/ and pristine upstream (an interrupted apply / uninstall)
#                                     foreign:<files> comma-separated target files in none of those states
#   bash install.sh --rebind        the box-start verb: the OVERLAY_PRESTATE line, then on pristine | previous:<id> this tree's overlay is (re)installed (the
#                                   pristine .hoist_orig backups are kept) and one line `OVERLAY_APPLIED: <n> files id=<version>/<overlay id, 12 hex>` follows;
#                                   on current nothing more happens; on foreign exit 4 with no file touched.
#   bash install.sh --check-only    exit 0 iff the 10 files match patched/
#   bash install.sh --uninstall     restore the .hoist_orig backups (back to whatever you had before) and drop the marker
#   bash install.sh --status        print which known state each file is in
# The 10 files = 8 edited upstream files (inference_sampler.py, RFD3_diffusion_module.py, layers/{attention,blocks,block_utils,encoders,layer_utils,
# pairformer_layers}.py) + 2 new ones (hoist.py, cudagraph_sampler.py). NOTE: at upstream HEAD faf4299 layer_utils.py differs from 4010e3e by one upstream
# line; install.sh accepts either as 'stock' and installs the 4010e3e-based version of that file (the faf4299 change - a static RMSNorm size for
# torch.compile - is then reverted).
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); MODE="${1:-apply}"; FORCE=""; [ "${2:-}" = "--force" ] || [ "${1:-}" = "--force" ] && { FORCE=1; [ "$MODE" = "--force" ] && MODE=apply; }
case "$MODE" in apply|--classify|--rebind|--check-only|--uninstall|--status) ;; *) echo "usage: bash install.sh [apply|--classify|--rebind|--check-only|--uninstall|--status] [--force]"; exit 2 ;; esac
SP=$(python -c "import rfd3, os; print(os.path.dirname(os.path.dirname(os.path.abspath(rfd3.__file__))))" | tail -1)
[ -d "$SP/rfd3" ] || { echo "cannot locate the rfd3 package (python -c \"import rfd3\" failed?)"; exit 2; }
FILES="rfd3/model/cudagraph_sampler.py rfd3/model/hoist.py rfd3/model/inference_sampler.py rfd3/model/layers/attention.py rfd3/model/layers/blocks.py rfd3/model/layers/block_utils.py rfd3/model/layers/encoders.py rfd3/model/layers/layer_utils.py rfd3/model/layers/pairformer_layers.py rfd3/model/RFD3_diffusion_module.py"
NEW="rfd3/model/hoist.py rfd3/model/cudagraph_sampler.py"
MARKER="$SP/rfd3/.xattempt_addon_overlay.json"           # written by apply: {"name", "version", "overlay_id", "files": {<file>: sha256}} — the record --classify reads
TAG='\[RFD3_'                                             # every hunk of the overlay is tagged `[RFD3_<LEVER>]`; a file of an earlier install carries the tags too
sha(){ [ -f "$1" ] && sha256sum "$1" | awk '{print $1}' || echo "absent"; }
want(){ grep " $1\$" "$2" 2>/dev/null | awk '{print $1}'; }
is_new(){ local n; for n in $NEW; do [ "$n" = "$1" ] && return 0; done; return 1; }
is_stock_sha(){ # file sha -> 0 iff sha is upstream's (4010e3e or faf4299) for that file
  { [ -n "$(want $1 $HERE/STOCK_SHA256_upstream_4010e3e.txt)" ] && [ "$2" = "$(want $1 $HERE/STOCK_SHA256_upstream_4010e3e.txt)" ]; } ||
  { [ -n "$(want $1 $HERE/STOCK_SHA256_upstream_faf4299_extra.txt)" ] && [ "$2" = "$(want $1 $HERE/STOCK_SHA256_upstream_faf4299_extra.txt)" ]; }; }
state_of(){ # file -> patched|stock|stock-head|absent-ok|unknown
  local f=$1 got; got=$(sha "$SP/$f")
  [ "$got" = "$(sha "$HERE/patched/$f")" ] && { echo patched; return; }
  [ -n "$(want $f $HERE/STOCK_SHA256_upstream_4010e3e.txt)" ] && [ "$got" = "$(want $f $HERE/STOCK_SHA256_upstream_4010e3e.txt)" ] && { echo stock; return; }
  [ -n "$(want $f $HERE/STOCK_SHA256_upstream_faf4299_extra.txt)" ] && [ "$got" = "$(want $f $HERE/STOCK_SHA256_upstream_faf4299_extra.txt)" ] && { echo stock-head; return; }
  [ "$got" = "absent" ] && is_new "$f" && { echo absent-ok; return; }
  echo unknown; }
files_match(){ for f in $FILES; do [ "$(state_of $f)" = patched ] || return 1; done; return 0; }
overlay_id(){ # sha256 over the sorted `<sha256>  <file>` lines of patched/: the id of the bytes this apply installs (recorded in the marker, never stored in the tree)
  local f; for f in $FILES; do echo "$(sha "$HERE/patched/$f")  $f"; done | sort | sha256sum | awk '{print $1}'; }
marker_field(){ # field -> value from the marker ("" when absent / unreadable); files.<file> -> that file's recorded sha256
  [ -f "$MARKER" ] || { echo ""; return 0; }
  python - "$MARKER" "$1" <<'EOF' 2>/dev/null || echo ""
import json, sys
d = json.load(open(sys.argv[1])); k = sys.argv[2]
v = (d.get("files") or {}).get(k[len("files."):]) if k.startswith("files.") else d.get(k)
print("" if v is None else v)
EOF
}
write_marker(){ # after a successful apply: the kit's name/version (MANIFEST.json), the overlay id, every installed file's sha256
  local f lines=""; for f in $FILES; do lines="$lines$f $(sha "$SP/$f")"$'\n'; done
  python - "$HERE/MANIFEST.json" "$MARKER" "$(overlay_id)" "$lines" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
files = dict(tuple(l.split(" ", 1)) for l in sys.argv[4].splitlines() if l.strip())
json.dump({"name": m.get("name"), "version": str(m.get("version")), "overlay_id": sys.argv[3], "files": files}, open(sys.argv[2], "w"), indent=1, sort_keys=True)
EOF
}
previous_evidence(){ # file -> marker|sidecar|"" : why a file that is neither patched/ nor pristine is positively an earlier install of THIS overlay
  local f=$1 got rec; got=$(sha "$SP/$f")
  if [ -f "$MARKER" ]; then                                # a marked install answers from its marker only: the file must be byte for byte what that apply recorded
    rec=$(marker_field "files.$f")
    case "$(marker_field name)" in RFD3_XATTEMPT_ADDON*) [ -n "$rec" ] && [ "$rec" = "$got" ] && { echo marker; return 0; } ;; esac
    echo ""; return 0; fi
  grep -q "$TAG" "$SP/$f" 2>/dev/null || { echo ""; return 0; }   # an install older than the marker: this installer's sidecar beside the file AND the overlay's hunk tags in it
  if is_new "$f"; then [ -f "$SP/$f.hoist_absent" ] && { echo sidecar; return 0; }
  else [ -f "$SP/$f.hoist_orig" ] && is_stock_sha "$f" "$(sha "$SP/$f.hoist_orig")" && { echo sidecar; return 0; }; fi
  echo ""; }
classify(){ # -> pristine | current | previous:<id> | foreign:<f1,f2,...>
  local f s e n_patched=0 n_pristine=0 n_marker=0 n_sidecar=0 foreign=""
  for f in $FILES; do s=$(state_of $f)
    case "$s" in patched) n_patched=$((n_patched+1)) ;; stock|stock-head|absent-ok) n_pristine=$((n_pristine+1)) ;;
      *) e=$(previous_evidence $f); case "$e" in marker) n_marker=$((n_marker+1)) ;; sidecar) n_sidecar=$((n_sidecar+1)) ;; *) foreign="${foreign:+$foreign,}$f" ;; esac ;; esac; done
  if [ -n "$foreign" ]; then echo "foreign:$foreign"
  elif [ $n_patched -eq 10 ]; then echo current
  elif [ $n_pristine -eq 10 ]; then echo pristine
  elif [ $((n_marker+n_sidecar)) -eq 0 ]; then echo "previous:partial"                              # patched/ and pristine upstream files only: an interrupted apply or uninstall
  elif [ $n_sidecar -eq 0 ]; then local v i; v=$(marker_field version); i=$(marker_field overlay_id); echo "previous:${v:-unknown}/${i:0:12}"
  else echo "previous:unmarked"; fi; }
this_id(){ local v; v=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('version'))" "$HERE/MANIFEST.json" 2>/dev/null || echo unknown); echo "$v/$(overlay_id | cut -c1-12)"; }
do_apply(){ # copy patched/ over the 10 files (backups first), check sha256 == patched/, write the marker; $1 = the class met (for the closing line), $2 = quiet|""
  local f
  for f in $FILES; do
    if [ -f "$SP/$f" ]; then [ -f "$SP/$f.hoist_orig" ] || [ -f "$SP/$f.hoist_absent" ] || { is_new "$f" && touch "$SP/$f.hoist_absent" || cp "$SP/$f" "$SP/$f.hoist_orig"; }
    else mkdir -p "$(dirname "$SP/$f")"; touch "$SP/$f.hoist_absent"; fi
    cp "$HERE/patched/$f" "$SP/$f"; done
  find "$SP/rfd3" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
  files_match || { echo "sha256 mismatch after install"; exit 3; }
  write_marker || { echo "overlay installed but the marker could not be written: $MARKER"; exit 3; }
  [ "${2:-}" = quiet ] || echo "overlay installed into $SP (was OVERLAY_PRESTATE: $1): 10 files checked (sha256 == patched/), marker $MARKER, id $(this_id); 'bash install.sh --uninstall' restores your previous files"; }
case "$MODE" in
  --status) echo "site-packages: $SP"; for f in $FILES; do echo "  $(state_of $f)  $f"; done; exit 0 ;;
  --classify) echo "OVERLAY_PRESTATE: $(classify)"; exit 0 ;;
  --rebind) class=$(classify); echo "OVERLAY_PRESTATE: $class"
     case "$class" in
       current) exit 0 ;;
       pristine|previous:*) do_apply "$class" quiet; echo "OVERLAY_APPLIED: 10 files id=$(this_id)"; exit 0 ;;
       *) echo "refusing: the files named are neither pristine upstream (4010e3e/faf4299), nor this overlay, nor an earlier install of it — nothing was touched"; exit 4 ;;
     esac ;;
  --check-only) files_match && { echo "overlay in place in $SP (10 files == patched/)"; exit 0; } || { echo "overlay NOT installed (or files differ) in $SP; try: bash install.sh --status"; exit 1; } ;;
  --uninstall) n=0; for f in $FILES; do if [ -f "$SP/$f.hoist_orig" ]; then mv -f "$SP/$f.hoist_orig" "$SP/$f"; n=$((n+1)); elif [ -f "$SP/$f.hoist_absent" ]; then rm -f "$SP/$f" "$SP/$f.hoist_absent"; n=$((n+1)); fi; done
     rm -f "$MARKER"; find "$SP/rfd3" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; echo "restored $n file(s) in $SP"; exit 0 ;;
  apply)
     class=$(classify)
     case "$class" in pristine|current) ;;
       *) [ -n "$FORCE" ] || { echo "refusing: OVERLAY_PRESTATE: $class — the files named / differing are not pristine upstream (4010e3e/faf4299) and not this overlay (an earlier install of it: bash install.sh --rebind re-bases it; anything else: install into a clean foundry environment, or pass --force to back these files up and install over them)"; exit 4; } ;;
     esac
     do_apply "$class" ;;
esac
