#!/bin/bash
set -eo pipefail

# ------------------------------------------------------------
# Mac setup script for NestedRiesz
# 1. Creates the Conda environment from environment_mac.yml
# 2. Activates the environment
# 3. Installs the R package glmnet inside that environment's R
# ------------------------------------------------------------

ENV_NAME="riesz"
ENV_YML="setup/environment_mac.yml"

# Move to repo root if script is run from setup/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "Repo root: ${REPO_ROOT}"

# Check that environment file exists
if [ ! -f "${ENV_YML}" ]; then
    echo "ERROR: ${ENV_YML} not found in repo root."
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check that conda exists
if ! command -v conda >/dev/null 2>&1; then
    if [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then
        source "${HOME}/miniforge3/etc/profile.d/conda.sh"
    else
        echo "ERROR: conda not found."
        echo "Install Miniforge first, or make sure conda is initialized."
        exit 1
    fi
fi

# Make conda activate work in non-interactive bash
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create environment only if it does not already exist
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda environment '${ENV_NAME}' already exists. Skipping creation."
else
    echo "Creating Conda environment '${ENV_NAME}' from ${ENV_YML}..."
    conda env create -f "${ENV_YML}"
fi

echo "Activating environment '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

echo "Python location:"
which python
python --version

echo "R location:"
which R
R --version | head -n 1

echo "Installing R package glmnet..."
R --vanilla -e "install.packages('glmnet', repos='https://cloud.r-project.org')"

echo "Checking glmnet installation..."
R --vanilla -e "library(glmnet); cat('glmnet installed:', as.character(packageVersion('glmnet')), '\n')"

echo "Setup complete."
echo "To use the environment later, run:"
echo "  conda activate ${ENV_NAME}"
