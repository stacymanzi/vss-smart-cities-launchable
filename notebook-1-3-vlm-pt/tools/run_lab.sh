#!/usr/bin/env bash
# Run something with the lab environment set. DLI_GPU accepts "0", "1" or "0,1".
# Credentials are NOT sourced here: the notebook reads them from the environment
# and falls back to the lab folder's .env, so both paths behave the same.
set -u
cd "$(dirname "$0")/.."
export DLI_BASE="${DLI_BASE:-/mnt/dli}"
export DLI_GPU="${DLI_GPU:-0}"
exec "$@"
