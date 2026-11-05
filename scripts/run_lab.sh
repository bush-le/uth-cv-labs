#!/usr/bin/env bash
# ==============================================================================
# Script to execute CV Labs (UTH Computer Vision Workspace)
# Usage:
#   bash scripts/run_lab.sh [LAB_NUMBER]
# Examples:
#   bash scripts/run_lab.sh 1
#   bash scripts/run_lab.sh 2
#   bash scripts/run_lab.sh 3
#   bash scripts/run_lab.sh 4
#   bash scripts/run_lab.sh 5
#   bash scripts/run_lab.sh all
# ==============================================================================

set -euo pipefail

LAB="${1:-all}"
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
        echo "Executing Lab 1 Pipeline (Basic Image Processing & Color Spaces)..."
        ${PYTHON} -m src.labs.lab01_basics.pipeline
        echo "Lab 1 execution complete! Artifacts saved to experiments/predictions/lab01/"
        ;;
    2|"02"|"lab2"|"lab02"|"lab-02")
        echo "Executing Lab 2 Pipeline (Point Operations & Linear/Non-linear Filtering)..."
        ${PYTHON} -m src.labs.lab02_filters.pipeline
        echo "Lab 2 execution complete! Artifacts saved to experiments/predictions/lab02/"
        ;;
    3|"03"|"lab3"|"lab03"|"lab-03")
        echo "Executing Lab 3 Pipeline (Canny Edge Detector & Morphological Segmentation)..."
        ${PYTHON} -m src.labs.lab03_canny.pipeline
        echo "Lab 3 execution complete! Artifacts saved to experiments/predictions/lab03/"
        ;;
    4|"04"|"lab4"|"lab04"|"lab-04")
        echo "Executing Lab 4 Pipeline (Wavelet Image Hashing & Similarity Retrieval)..."
        ${PYTHON} -m src.labs.lab04_wavelet.pipeline
        echo "Lab 4 execution complete! Artifacts saved to experiments/predictions/lab04/"
        ;;
    5|"05"|"lab5"|"lab05"|"lab-05")
        echo "Executing Lab 5 Pipeline (Real-Time Face Recognition with MTCNN & FaceNet)..."
        ${PYTHON} -m src.labs.lab05_facenet.pipeline --mode auto
        echo "Lab 5 execution complete! Artifacts saved to experiments/predictions/lab05/"
        ;;
    all|"ALL"|"labs")
        echo "Executing all completed labs (Lab 1 -> Lab 5)..."
        echo "--- Running Lab 1 ---"
        ${PYTHON} -m src.labs.lab01_basics.pipeline
        echo "--- Running Lab 2 ---"
        ${PYTHON} -m src.labs.lab02_filters.pipeline
        echo "--- Running Lab 3 ---"
        ${PYTHON} -m src.labs.lab03_canny.pipeline
        echo "--- Running Lab 4 ---"
        ${PYTHON} -m src.labs.lab04_wavelet.pipeline
        echo "--- Running Lab 5 ---"
        ${PYTHON} -m src.labs.lab05_facenet.pipeline --mode batch
        echo "=================================================="
        echo "All labs (1 - 5) executed successfully!"
        echo "=================================================="
        ;;
    *)
        echo "Error: Lab '${LAB}' is unrecognized. Valid options: 1, 2, 3, 4, 5, all"
        exit 1
        ;;
esac
