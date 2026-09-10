#!/usr/bin/env bash
# Execute every cell of the lab through a real Jupyter kernel and keep the
# outputs in the notebook. This is the end-to-end check: it runs the same code
# path a student runs, and the resulting file is the evidence.
#
#   ./execute_notebook.sh [output.ipynb]
#
# Environment is inherited, so set DLI_BASE / DLI_GPU before calling.
set -euo pipefail
cd "$(dirname "$0")/.."   # tools/ -> lab root

OUT="${1:-lab_3.executed.ipynb}"
KERNEL="${JUPYTER_KERNEL:-lab3}"
TIMEOUT="${CELL_TIMEOUT:-7200}"      # a training cell can run for over an hour

echo "notebook : lab_3.ipynb"
echo "output   : $OUT"
echo "kernel   : $KERNEL"
echo "base dir : ${DLI_BASE:-<default>}"
echo "GPU(s)   : ${DLI_GPU:-<default>}"
echo

cp lab_3.ipynb "$OUT"
exec "${JUPYTER_BIN:-$HOME/labenv/bin/jupyter}" nbconvert \
    --to notebook --execute --inplace "$OUT" \
    --ExecutePreprocessor.kernel_name="$KERNEL" \
    --ExecutePreprocessor.timeout="$TIMEOUT" \
    --ExecutePreprocessor.allow_errors=False \
    --log-level=INFO
