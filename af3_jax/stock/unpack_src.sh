#!/bin/bash
# unpack_src.sh DEST — lay out the pinned stock source at DEST (a new or empty directory) with stock/patches applied: unpacked from the upstream
# source archive beside this file when the tree carries it (stock/PINS.json upstream.archive), otherwise cloned from upstream.repo at upstream.commit
# (network and git needed). One step of the install — environment/Dockerfile, README.md 'Install' route C and STOCK.md 'Install' call it; no run
# mode does. Prints one `[af3-jax-opt] STOCK SOURCE …` line naming the route taken.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=${1:?usage: unpack_src.sh DEST}
PY=$(command -v python3 || command -v python || true)
[ -n "$PY" ] || { echo "unpack_src.sh: python3 is needed to read stock/PINS.json" >&2; exit 3; }
read -r ARCHIVE REPO COMMIT < <("$PY" -c 'import json, sys; u = json.load(open(sys.argv[1]))["upstream"]; print(u["archive"]["file"], u["repo"], u["commit"])' "$HERE/PINS.json")
mapfile -t PATCHES < <("$PY" -c 'import json, sys; [print(p["file"]) for p in json.load(open(sys.argv[1]))["upstream"]["patches"]]' "$HERE/PINS.json")
mkdir -p "$DEST"
[ -z "$(ls -A "$DEST")" ] || { echo "unpack_src.sh: $DEST is not empty" >&2; exit 2; }
if [ -f "$HERE/$ARCHIVE" ]; then
  tar -xzf "$HERE/$ARCHIVE" -C "$DEST" --strip-components=1
  echo "[af3-jax-opt] STOCK SOURCE route=archive file=stock/$ARCHIVE dest=$DEST"
else
  command -v git >/dev/null || { echo "unpack_src.sh: stock/$ARCHIVE is not in this tree, so the source is cloned from $REPO at commit $COMMIT — git is needed for that" >&2; exit 3; }
  git clone --quiet --no-checkout "$REPO" "$DEST"
  git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$COMMIT"
  rm -rf "$DEST/.git"
  echo "[af3-jax-opt] STOCK SOURCE route=clone repo=$REPO commit=$COMMIT dest=$DEST"
fi
for p in "${PATCHES[@]}"; do patch -d "$DEST" -p1 --quiet < "$HERE/$p"; done
echo "[af3-jax-opt] STOCK SOURCE patches=${#PATCHES[@]} applied (stock/PINS.json upstream.patches)"
