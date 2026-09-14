#!/usr/bin/env bash
# Start JupyterLab for the lab, with the environment the notebook expects.
#   ./start_jupyter.sh [port]
# Override before calling: DLI_BASE, DLI_GPU, JUPYTER_TOKEN
set -u
cd ~/lab-3-vlm-pt
set -a; . ./.env; set +a
export DLI_BASE="${DLI_BASE:-/mnt/dli}"
export DLI_GPU="${DLI_GPU:-0,1}"
PORT="${1:-8888}"
TOKEN="${JUPYTER_TOKEN:-lab3}"

echo "base : $DLI_BASE"
echo "GPUs : $DLI_GPU"
echo "URL  : http://127.0.0.1:$PORT/lab?token=$TOKEN"
echo

exec ~/labenv/bin/jupyter lab \
  --no-browser --ip=127.0.0.1 --port="$PORT" \
  --ServerApp.root_dir="$HOME/lab-3-vlm-pt" \
  --ServerApp.token="$TOKEN" \
  --ServerApp.terminado_settings="{\"shell_command\": [\"/bin/bash\"]}"
