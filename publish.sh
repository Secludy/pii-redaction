#!/bin/bash
set -e

# Load the .env file
set -o allexport
source .env

# Check if PYPI_OPENPIPE_TOKEN is set
if [[ -z "${PYPI_OPENPIPE_TOKEN}" ]]; then
    echo "Error: PYPI_OPENPIPE_TOKEN is not set."
    exit 1
fi

uv build

# If the token is set, proceed with publishing
uv publish dist/*.whl dist/*.tar.gz --token $PYPI_OPENPIPE_TOKEN
