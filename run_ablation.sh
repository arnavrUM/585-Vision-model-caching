#!/usr/bin/env bash
set -euo pipefail
SPEC_FILE="experiment/ablation_specs.json"
LOG_FILE="experiment_logs/ablation_results.csv"
SAMPLES_DIR="experiment_logs/ablation_samples"

mkdir -p $(dirname "$LOG_FILE")
CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1} python experiment/run_experiments.py \
  --specs "$SPEC_FILE" \
  --log-file "$LOG_FILE" \
  --samples-dir "$SAMPLES_DIR" \
  --purge-cache-between-runs \
  --cache-mode dry-run
