#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HF_REPO="yeqinglin/genie3"

usage() {
    cat <<EOF
Usage:
  bash scripts/setup/download.sh [options]

Options:
  --weights   Download pretrained model weights only (pretrained/)
  --data      Download training data only (data/train/)
  -h, --help  Show this message

Default (no options): downloads both weights and data.
EOF
}

DO_WEIGHTS=false
DO_DATA=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --weights) DO_WEIGHTS=true ;;
        --data)    DO_DATA=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

# Default: both
if ! $DO_WEIGHTS && ! $DO_DATA; then
    DO_WEIGHTS=true
    DO_DATA=true
fi

if ! command -v huggingface-cli &>/dev/null; then
    echo "huggingface-cli not found. Install it with: pip install huggingface_hub" >&2
    exit 1
fi

hf_download() {
    local pattern="$1"
    hf download "$HF_REPO" \
        --include "$pattern" \
        --local-dir "$REPO_ROOT"
}

if $DO_WEIGHTS; then
    echo "Downloading pretrained weights from $HF_REPO ..."
    hf_download "pretrained/**"
    echo "Weights saved to $REPO_ROOT/pretrained/"
fi

if $DO_DATA; then
    echo "Downloading training data from $HF_REPO ..."
    hf_download "data/train/**"
    echo "Training data saved to $REPO_ROOT/data/train/"
fi

echo "Done."
