#!/bin/bash

# Reproduce the conda environment. Works whether run from repo root or from reproduce/.

# Check if conda is available
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is not available. Please install Anaconda or Miniconda."
    exit 1
fi

# Resolve repo root (parent of directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if the environment exists
if conda env list | grep -q 'estimatingmicrodsops'; then
    echo "Environment 'estimatingmicrodsops' already exists. Updating it..."
    conda env update -q -f "$REPO_ROOT/environment.yml"
else
    echo "Creating environment using conda..."
    conda env create -q -f "$REPO_ROOT/environment.yml"
fi
