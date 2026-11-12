#!/usr/bin/env bash
# End-to-end Cloud GPU suite for Alpha Connectome (RTX 4090 / L40S / A100).
# Usage on the GPU host (either directly inside the container or in a Python 3.11 CUDA venv):
#   DATA_DIR=/data RUNS_DIR=/runs bash cloud/run_gpu_suite.sh
set -euo pipefail

# Deterministic CuBLAS workspace (required: seed_everything enables deterministic algorithms)
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# Prefer the host's preinstalled PyTorch interpreter when present (avoids venv shadowing).
# Override with PYTHON_BIN=/path/to/python if needed.
if [[ -z "${PYTHON_BIN:-}" && -x /opt/conda/bin/python ]]; then
  PYTHON_BIN=/opt/conda/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

BATCH_SIZE="${BATCH_SIZE:-64}"
TBPTT="${TBPTT:-2000}"
WORKERS="${WORKERS:-6}"
SGRU_STEPS="${SGRU_STEPS:-1600}"
SLSTM_STEPS="${SLSTM_STEPS:-1600}"
LT_STEPS="${LT_STEPS:-1000}"

DATA_DIR="${DATA_DIR:-/data}"
RUNS_DIR="${RUNS_DIR:-/runs}"
TRAIN_PARQUET="${DATA_DIR}/train.parquet"
VALID_PARQUET="${DATA_DIR}/valid.parquet"

VALID_ARG=()
if [[ -f "${VALID_PARQUET}" ]]; then
  echo "[suite] Found valid.parquet at ${VALID_PARQUET} — enabling live screen/dev WP evaluation."
  VALID_ARG=(--valid "${VALID_PARQUET}")
else
  echo "[suite] No valid.parquet at ${VALID_PARQUET} — training on train.parquet only (EMA checkpoints will be evaluated locally)."
fi

mkdir -p "${RUNS_DIR}"

echo "=================================================================="
echo "Arm 1/3: Full-convergence ScratchSortedGRU (hidden=160, ~10 epochs)"
echo "=================================================================="
"${PYTHON_BIN}" -m experiments.gpu_train_sorted_models \
  --train "${TRAIN_PARQUET}" "${VALID_ARG[@]}" \
  --output "${RUNS_DIR}/sgru_h160_10ep" \
  --arch sgru --mode generalist --hidden 160 \
  --batch-size "${BATCH_SIZE}" --steps "${SGRU_STEPS}" --warmup-steps 60 --tbptt "${TBPTT}" \
  --lr 2.0e-3 --mse-weight 0.5 --mask-power 1.5 --mask-floor 0.10 \
  --ema-decay 0.995 --eval-every 100 --save-every 50 --workers "${WORKERS}" --seed 20261113 --resume

echo "=================================================================="
echo "Arm 2/3: Orthogonal ScratchSortedLSTM (hidden=160, ~10 epochs)"
echo "=================================================================="
"${PYTHON_BIN}" -m experiments.gpu_train_sorted_models \
  --train "${TRAIN_PARQUET}" "${VALID_ARG[@]}" \
  --output "${RUNS_DIR}/slstm_h160_10ep" \
  --arch slstm --mode generalist --hidden 160 \
  --batch-size "${BATCH_SIZE}" --steps "${SLSTM_STEPS}" --warmup-steps 60 --tbptt "${TBPTT}" \
  --lr 1.8e-3 --mse-weight 0.5 --mask-power 1.5 --mask-floor 0.10 \
  --ema-decay 0.995 --eval-every 100 --save-every 50 --workers "${WORKERS}" --seed 20261114 --resume

echo "=================================================================="
echo "Arm 3/3: Dedicated 0.6569 + 0.4685 Large-Tick Specialist GRU (hidden=128)"
echo "=================================================================="
"${PYTHON_BIN}" -m experiments.gpu_train_sorted_models \
  --train "${TRAIN_PARQUET}" "${VALID_ARG[@]}" \
  --output "${RUNS_DIR}/sgru_largetick_h128" \
  --arch sgru --mode large_tick --hidden 128 \
  --batch-size "${BATCH_SIZE}" --steps "${LT_STEPS}" --warmup-steps 40 --tbptt "${TBPTT}" \
  --lr 2.0e-3 --mse-weight 0.6 --mask-power 1.75 --mask-floor 0.08 \
  --ema-decay 0.995 --eval-every 100 --save-every 50 --workers "${WORKERS}" --seed 20261115 --resume

echo "[suite] All 3 GPU arms complete! Copy ${RUNS_DIR}/*/final_checkpoint.pt (and report.json) back to your local machine."
