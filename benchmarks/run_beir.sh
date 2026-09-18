#!/usr/bin/env bash
# Reproducible BEIR benchmark runner for Sievrant.
#
# Usage:
#   bash run_beir.sh [--dataset scifact] [--seed 42]
#
# Results are written to results/<dataset>_<date>.json alongside a
# config.json that records the exact seed, model, and timestamp so
# any run can be reproduced identically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

DATASET="scifact"
SEED="42"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --seed)    SEED="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Running BEIR benchmark: dataset=${DATASET}, seed=${SEED}"
echo "Results will be saved to ${RESULTS_DIR}/"

PYTHON="${PYTHON:-python3}"

if [[ ! -f "${SCRIPT_DIR}/../.venv/bin/activate" ]]; then
    echo "No .venv found. Creating one and installing dependencies..."
    "$PYTHON" -m venv "${SCRIPT_DIR}/../.venv"
    source "${SCRIPT_DIR}/../.venv/bin/activate"
    pip install --quiet -r "${SCRIPT_DIR}/../requirements.txt"
else
    source "${SCRIPT_DIR}/../.venv/bin/activate"
fi

exec "$PYTHON" "${SCRIPT_DIR}/beir_benchmark.py" \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --results-dir "${RESULTS_DIR}"
