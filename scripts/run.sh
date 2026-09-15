#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-CIFAR100}"
ALPHA="${2:-0.1}"
GPU_ID="${3:-0}"
ABLATION_MODE="${4:-full}"
SEED="${5:-7}"

python "$PROJECT_DIR/FedGPLA.py" \
    --dataset "$DATASET" \
    --alpha "$ALPHA" \
    --gpu_id "$GPU_ID" \
    --ablation_mode "$ABLATION_MODE" \
    --seed "$SEED" \
    --num_clients 20 \
    --num_online_clients 8 \
    --local_epochs 5 \
    --reliability_threshold 0.95 \
    --unsup_threshold 0.95 \
    --alpha_local_prior 1.0 \
    --alpha_global_prior 0.1 \
    --my_method_prior_correction 0.25 \
    --lambda_u 1.0 \
    --lambda_kl 0.5
