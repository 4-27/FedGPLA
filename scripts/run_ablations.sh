#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-CIFAR10}"
ALPHA="${2:-0.5}"
GPU_ID="${3:-0}"
SEED="${4:-7}"

for mode in full all_hard all_soft uniform without_memory; do
    bash "$PROJECT_DIR/scripts/run.sh" \
        "$DATASET" "$ALPHA" "$GPU_ID" "$mode" "$SEED"
done
