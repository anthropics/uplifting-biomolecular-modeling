#!/bin/bash
# Activate environment
ENV_DIR=envs  # set this to your environment directory
source ${ENV_DIR}/caliby/bin/activate

# Required for AtomWorks.
# We don't need these environment variables for our use case,
# but AtomWorks requires them to be set, so we set them to empty strings.
export PDB_MIRROR_PATH=""
export CCD_MIRROR_PATH=""

# Directory where model weights are stored. Weights are automatically
# downloaded from HuggingFace on first use.
export MODEL_PARAMS_DIR=model_params
