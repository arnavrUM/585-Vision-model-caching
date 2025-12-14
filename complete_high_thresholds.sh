#!/bin/bash
# Complete remaining higher threshold experiments

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate 585

export GQA_IMAGE_ROOT=/root/585-Vision-model-caching/dataset_custom/unscrew_bottle_cap

# Remaining experiments to run
run_exp() {
  local model="$1"
  local exp_name="$2"
  local vision="$3"
  local prompt="$4"
  local semantic="$5"
  local max_len="$6"
  local prompt_template="$7"
  
  echo "Running: $exp_name"
  
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
    --embedding-layer "vision:512:${vision}"
    --embedding-layer "prompt:384:${prompt}"
    --similarity-threshold "$semantic"
    --cache-mode live
    --keep-cache-dirs
  )
  
  if [ -n "$prompt_template" ]; then
    args+=(--prompt-template "$prompt_template")
  fi
  
  python experiment2/test_vllm.py "${args[@]}"
  echo "Completed: $exp_name"
  echo ""
}

INTERNVL_PROMPT='<image>
You are assisting with the GQA benchmark. Answer the question using the referenced image.
Image ID: {image_id}
Question: {question}
Answer:'

# Qwen3-VL-8B
run_exp "Qwen/Qwen3-VL-8B-Instruct" "qwen3vl-8b_very_conservative" "0.90" "0.88" "0.88" "4096" ""
run_exp "Qwen/Qwen3-VL-8B-Instruct" "qwen3vl-8b_ultra_conservative" "0.92" "0.90" "0.90" "4096" ""
run_exp "Qwen/Qwen3-VL-8B-Instruct" "qwen3vl-8b_maximum_precision" "0.95" "0.93" "0.93" "4096" ""

# InternVL3.5-2B
run_exp "OpenGVLab/InternVL3_5-2B-Instruct" "internvl35-2b_very_conservative" "0.90" "0.88" "0.88" "8192" "$INTERNVL_PROMPT"
run_exp "OpenGVLab/InternVL3_5-2B-Instruct" "internvl35-2b_ultra_conservative" "0.92" "0.90" "0.90" "8192" "$INTERNVL_PROMPT"
run_exp "OpenGVLab/InternVL3_5-2B-Instruct" "internvl35-2b_maximum_precision" "0.95" "0.93" "0.93" "8192" "$INTERNVL_PROMPT"

# InternVL3.5-4B
run_exp "OpenGVLab/InternVL3_5-4B-Instruct" "internvl35-4b_very_conservative" "0.90" "0.88" "0.88" "4096" "$INTERNVL_PROMPT"
run_exp "OpenGVLab/InternVL3_5-4B-Instruct" "internvl35-4b_ultra_conservative" "0.92" "0.90" "0.90" "4096" "$INTERNVL_PROMPT"
run_exp "OpenGVLab/InternVL3_5-4B-Instruct" "internvl35-4b_maximum_precision" "0.95" "0.93" "0.93" "4096" "$INTERNVL_PROMPT"

# InternVL3.5-8B
run_exp "OpenGVLab/InternVL3_5-8B-Instruct" "internvl35-8b_very_conservative" "0.90" "0.88" "0.88" "4096" "$INTERNVL_PROMPT"
run_exp "OpenGVLab/InternVL3_5-8B-Instruct" "internvl35-8b_ultra_conservative" "0.92" "0.90" "0.90" "4096" "$INTERNVL_PROMPT"
run_exp "OpenGVLab/InternVL3_5-8B-Instruct" "internvl35-8b_maximum_precision" "0.95" "0.93" "0.93" "4096" "$INTERNVL_PROMPT"

echo "All remaining higher threshold experiments completed!"


