#!/bin/bash

# Download all Caliby model parameters from HuggingFace.
# Individual scripts will also auto-download only the weights they need on first run.
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ProteinDesignLab/caliby-weights', local_dir='model_params')"
