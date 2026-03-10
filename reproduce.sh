#!/bin/bash

# Ensure the conda environment exists, then run the reproduction script.

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is not available. Please install Anaconda or Miniconda."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
ENV_SCRIPT="$REPO_ROOT/reproduce/reproduce_environment.sh"

# Test whether the environment already satisfies requirements
if ! conda env list | grep -q 'estimatingmicrodsops'; then
    echo "Environment 'estimatingmicrodsops' not found. Creating it..."
    bash "$ENV_SCRIPT"
fi

# Activate the environment and run the reproduction
conda activate estimatingmicrodsops
cd "$REPO_ROOT" && ipython src/run_all.py
