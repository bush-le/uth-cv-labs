#!/usr/bin/env bash
# ==============================================================================
# Script to execute CV Labs
# Usage:
#   bash scripts/run_lab.sh [LAB_NUMBER]
# Example:
#   bash scripts/run_lab.sh 1
# ==============================================================================

set -euo pipefail

LAB="${1:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Detect Python interpreter (prefer active virtualenv)
if [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "=================================================="
echo "UTH Computer Vision Workspace - Lab Runner"
echo "Target Lab: Lab ${LAB}"
echo "Python Interpreter: $(${PYTHON} --version)"
echo "=================================================="

case "${LAB}" in
    1|"01"|"lab1"|"lab01"|"lab-01")
        echo "Executing Lab 1 Pipeline..."
        ${PYTHON} -m src.pipelines.lab01_pipeline
        echo "Lab 1 execution complete! Artifacts saved to experiments/predictions/lab01/"
        ;;
    *)
        echo "Error: Lab ${LAB} is not yet implemented or unrecognized."
        exit 1
        ;;
esac
