#!/bin/bash
# Script to run higher threshold experiments and append to existing CSV

set -euo pipefail

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate 585

# Set image root
export GQA_IMAGE_ROOT=/root/585-Vision-model-caching/dataset_custom/unscrew_bottle_cap

echo "=========================================="
echo "Running Higher Threshold Experiments"
echo "=========================================="
echo ""

# Higher threshold configurations (Option 1: High Precision)
# Format: "name|vision_threshold|prompt_threshold|semantic_threshold"
HIGH_THRESHOLD_CONFIGS=(
  "very_conservative|0.90|0.88|0.88"
  "ultra_conservative|0.92|0.90|0.90"
  "maximum_precision|0.95|0.93|0.93"
)

# Model configurations
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
  
  local exp_name="${suffix}_${threshold_name}"
  
  echo "=========================================="
  echo "Running: $exp_name"
  echo "Model: $model"
  echo "Thresholds: vision=$vision_thresh, prompt=$prompt_thresh, semantic=$semantic_thresh"
  echo "=========================================="
  
  local args=(
    --dataset video_frames
    --video-frames-data video_frames_dataset.json
    --max-samples 64
    --chunk-source question
    --trust-remote-code
    --gpu-memory-utilization 0.9
    --summary-log video_frames_results.csv
    --model "$model"
    --max-model-len "$max_len"
    --experiment-name "$exp_name"
    --embedding-hook prompt_vision
    --embedding-layer "vision:512:${vision_thresh}"
    --embedding-layer "prompt:384:${prompt_thresh}"
    --similarity-threshold "$semantic_thresh"
    --cache-mode live
    --keep-cache-dirs
  )
  
  if [ -n "$prompt_template" ]; then
    args+=(--prompt-template "$prompt_template")
  fi
  
  python experiment2/test_vllm.py "${args[@]}"
  
  local exit_code=$?
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
echo "QWEN MODELS - HIGHER THRESHOLDS"
echo "=========================================="
echo ""

for model_spec in "${QWEN_MODELS[@]}"; do
  IFS="|" read -r model suffix max_len prompt_template <<< "$model_spec"
  
  for thresh_spec in "${HIGH_THRESHOLD_CONFIGS[@]}"; do
    IFS="|" read -r thresh_name vision_thresh prompt_thresh semantic_thresh <<< "$thresh_spec"
    run_experiment "$model" "$suffix" "$max_len" "$prompt_template" "$thresh_name" "$vision_thresh" "$prompt_thresh" "$semantic_thresh"
  done
done

# Run experiments for InternVL models
echo "=========================================="
echo "INTERNVL MODELS - HIGHER THRESHOLDS"
echo "=========================================="
echo ""

for model_spec in "${INTERNVL_MODELS[@]}"; do
  IFS="|" read -r model suffix max_len prompt_template <<< "$model_spec"
  
  for thresh_spec in "${HIGH_THRESHOLD_CONFIGS[@]}"; do
    IFS="|" read -r thresh_name vision_thresh prompt_thresh semantic_thresh <<< "$thresh_spec"
    run_experiment "$model" "$suffix" "$max_len" "$prompt_template" "$thresh_name" "$vision_thresh" "$prompt_thresh" "$semantic_thresh"
  done
done

echo ""
echo "=========================================="
echo "All Higher Threshold Experiments Completed!"
echo "=========================================="
echo ""
echo "Results appended to: video_frames_results.csv"
echo ""
echo "Total experiments: 18 (3 threshold configs × 6 models)"
echo "Threshold configurations tested:"
echo "  1. very_conservative: vision=0.90, prompt=0.88, semantic=0.88"
echo "  2. ultra_conservative: vision=0.92, prompt=0.90, semantic=0.90"
echo "  3. maximum_precision: vision=0.95, prompt=0.93, semantic=0.93"
echo "=========================================="

exit 0


