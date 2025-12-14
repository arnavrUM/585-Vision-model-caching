#!/bin/bash
# Script to run video frames pipeline: baseline (no cache) and all cached conditions
# Tests all available models with different threshold configurations

set -euo pipefail

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate 585

# Set image root
export GQA_IMAGE_ROOT=/root/585-Vision-model-caching/dataset_custom/unscrew_bottle_cap

# Create log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="video_frames_logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Running Video Frames Pipeline - Full Experiment"
echo "=========================================="
echo ""
echo "Conda env: 585"
echo "Image root: $GQA_IMAGE_ROOT"
echo "Dataset: video_frames_dataset.json"
echo "Log directory: $LOG_DIR"
echo ""

# Common arguments for all runs
COMMON_ARGS=(
  --dataset video_frames
  --video-frames-data video_frames_dataset.json
  --max-samples 64
  --chunk-source question
  --trust-remote-code
  --gpu-memory-utilization 0.9
  --summary-log video_frames_results.csv
)

# Threshold configurations
# Format: "name|vision_threshold|prompt_threshold|semantic_threshold"
THRESHOLD_CONFIGS=(
  "conservative|0.85|0.82|0.82"
  "moderate|0.82|0.80|0.80"
  "aggressive|0.80|0.78|0.78"
)

# Model configurations
# Format: "model_name|experiment_suffix|max_model_len|prompt_template"
# prompt_template can be empty for Qwen (uses default)
QWEN_MODELS=(
  "Qwen/Qwen3-VL-2B-Instruct|qwen3vl-2b|8192|"
  "Qwen/Qwen3-VL-4B-Instruct|qwen3vl-4b|4096|"
  "Qwen/Qwen3-VL-8B-Instruct|qwen3vl-8b|4096|"
)

INTERNVL_PROMPT='<image>
You are assisting with the GQA benchmark. Answer the question using the referenced image.
Image ID: {image_id}
Question: {question}
Answer:'

INTERNVL_MODELS=(
  "OpenGVLab/InternVL3_5-2B-Instruct|internvl35-2b|8192|${INTERNVL_PROMPT}"
  "OpenGVLab/InternVL3_5-4B-Instruct|internvl35-4b|4096|${INTERNVL_PROMPT}"
  "OpenGVLab/InternVL3_5-8B-Instruct|internvl35-8b|4096|${INTERNVL_PROMPT}"
)

# Function to run a single experiment
run_experiment() {
  local model="$1"
  local suffix="$2"
  local max_len="$3"
  local prompt_template="$4"
  local threshold_name="$5"
  local vision_thresh="$6"
  local prompt_thresh="$7"
  local semantic_thresh="$8"
  local use_cache="$9"
  shift 9
  local extra_args=("$@")
  
  local exp_name="${suffix}_${threshold_name}"
  if [ "$use_cache" = "baseline" ]; then
    exp_name="${suffix}_baseline"
  fi
  
  local log_file="${LOG_DIR}/${exp_name}.log"
  
  echo "=========================================="
  echo "Running: $exp_name"
  echo "Model: $model"
  echo "Threshold: $threshold_name (vision=$vision_thresh, prompt=$prompt_thresh, semantic=$semantic_thresh)"
  echo "Cache: $use_cache"
  echo "Log: $log_file"
  echo "=========================================="
  
  local args=(
    "${COMMON_ARGS[@]}"
    --model "$model"
    --max-model-len "$max_len"
    --experiment-name "$exp_name"
  )
  
  if [ -n "$prompt_template" ]; then
    args+=(--prompt-template "$prompt_template")
  fi
  
  if [ "$use_cache" = "baseline" ]; then
    args+=(
      --disable-semantic-cache
      --disable-exact-cache
      --cache-mode live
    )
  else
    args+=(
      --embedding-hook prompt_vision
      --embedding-layer "vision:512:${vision_thresh}"
      --embedding-layer "prompt:384:${prompt_thresh}"
      --similarity-threshold "$semantic_thresh"
      --cache-mode live
      --keep-cache-dirs
    )
  fi
  
  args+=("${extra_args[@]}")
  
  python experiment2/test_vllm.py "${args[@]}" 2>&1 | tee "$log_file"
  
  local exit_code=${PIPESTATUS[0]}
  if [ $exit_code -ne 0 ]; then
    echo "ERROR: Experiment $exp_name failed with exit code $exit_code"
    return $exit_code
  fi
  
  echo ""
  echo "Completed: $exp_name"
  echo ""
}

# Run experiments for Qwen models
echo "=========================================="
echo "QWEN MODELS"
echo "=========================================="
echo ""

for model_spec in "${QWEN_MODELS[@]}"; do
  IFS="|" read -r model suffix max_len prompt_template <<< "$model_spec"
  
  # Baseline (no cache)
  run_experiment "$model" "$suffix" "$max_len" "$prompt_template" "baseline" "0" "0" "0" "baseline"
  
  # Cached runs with different thresholds
  for thresh_spec in "${THRESHOLD_CONFIGS[@]}"; do
    IFS="|" read -r thresh_name vision_thresh prompt_thresh semantic_thresh <<< "$thresh_spec"
    run_experiment "$model" "$suffix" "$max_len" "$prompt_template" "$thresh_name" "$vision_thresh" "$prompt_thresh" "$semantic_thresh" "cached"
  done
done

# Run experiments for InternVL models
echo "=========================================="
echo "INTERNVL MODELS"
echo "=========================================="
echo ""

for model_spec in "${INTERNVL_MODELS[@]}"; do
  IFS="|" read -r model suffix max_len prompt_template <<< "$model_spec"
  
  # Baseline (no cache)
  run_experiment "$model" "$suffix" "$max_len" "$prompt_template" "baseline" "0" "0" "0" "baseline"
  
  # Cached runs with different thresholds
  for thresh_spec in "${THRESHOLD_CONFIGS[@]}"; do
    IFS="|" read -r thresh_name vision_thresh prompt_thresh semantic_thresh <<< "$thresh_spec"
    run_experiment "$model" "$suffix" "$max_len" "$prompt_template" "$thresh_name" "$vision_thresh" "$prompt_thresh" "$semantic_thresh" "cached"
  done
done

echo ""
echo "=========================================="
echo "All Experiments Completed Successfully!"
echo "=========================================="
echo ""
echo "=========================================="
echo "RESULTS AND LOG FILES LOCATION"
echo "=========================================="
echo ""
echo "📊 RESULTS FILE (CSV with all experiment results):"
echo "   $(pwd)/video_frames_results.csv"
echo ""
echo "📁 LOG DIRECTORY (Individual logs for each experiment):"
echo "   $(pwd)/$LOG_DIR/"
echo ""
echo "   Each experiment has its own log file:"
echo "   - Baseline: <model>_baseline.log"
echo "   - Cached:   <model>_<threshold_config>.log"
echo ""
echo "   Example log files:"
echo "   - qwen3vl-2b_baseline.log"
echo "   - qwen3vl-2b_conservative.log"
echo "   - qwen3vl-2b_moderate.log"
echo "   - qwen3vl-2b_aggressive.log"
echo ""
echo "=========================================="
echo "EXPERIMENT SUMMARY"
echo "=========================================="
echo ""
echo "Total experiments: 24"
echo "  - Baseline runs: 6 (one per model, no cache)"
echo "  - Cached runs: 18 (3 threshold configs × 6 models)"
echo ""
echo "Models tested:"
echo "  - Qwen3-VL: 2B, 4B, 8B"
echo "  - InternVL3.5: 2B, 4B, 8B"
echo ""
echo "Threshold configurations tested:"
echo "  1. conservative: vision=0.85, prompt=0.82, semantic=0.82"
echo "  2. moderate:     vision=0.82, prompt=0.80, semantic=0.80"
echo "  3. aggressive:   vision=0.80, prompt=0.78, semantic=0.78"
echo ""
echo "=========================================="
echo "To view results:"
echo "  cat video_frames_results.csv"
echo ""
echo "To view a specific log:"
echo "  cat $LOG_DIR/<experiment_name>.log"
echo ""
echo "To monitor progress (while running):"
echo "  tail -f $LOG_DIR/<experiment_name>.log"
echo "=========================================="

exit 0
