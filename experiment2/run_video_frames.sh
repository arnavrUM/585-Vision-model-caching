#!/bin/bash
# Run the custom video-frames experiment (baseline + cached configs) on Qwen3-VL and InternVL.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_FILE="${DATA_FILE:-${ROOT_DIR}/dataset_custom/video_frames_labels.json}"
FRAMES_ROOT="${FRAMES_ROOT:-${ROOT_DIR}/dataset_custom}"
LOG_DIR="${LOG_DIR:-video_frames_logs_$(date +%Y%m%d_%H%M%S)}"

USE_HF="${USE_HF:-1}"
HF_DEVICE="${HF_DEVICE:-cuda}"  # set HF_DEVICE=cpu to force CPU
ENCODER_DEVICE="cuda"
HF_ARGS=()
if [[ "${USE_HF}" == "1" ]]; then
  if [[ "${HF_DEVICE}" == "cpu" ]]; then
    ENCODER_DEVICE="cpu"
    HF_ARGS=(--use-hf --index-encoder-device cpu)
    export CUDA_VISIBLE_DEVICES=""
    echo "[info] USE_HF=1 on CPU (HF_DEVICE=cpu). CUDA_VISIBLE_DEVICES cleared."
  else
    ENCODER_DEVICE="cuda"
    HF_ARGS=(--use-hf --index-encoder-device cuda)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    echo "[info] USE_HF=1 on GPU (HF_DEVICE=${HF_DEVICE}). CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  fi
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  echo "[info] Using vLLM with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Running Video Frames Pipeline"
echo "=========================================="
echo "Dataset file : $DATA_FILE"
echo "Frames root  : $FRAMES_ROOT"
echo "Log directory: $LOG_DIR"
echo ""

COMMON_ARGS=(
  --dataset video_frames
  --video-frames-data "$DATA_FILE"
  --video-frames-root "$FRAMES_ROOT"
  --max-samples 64
  --chunk-source question
  --trust-remote-code
  --gpu-memory-utilization 0.9
  --summary-log experiment2_frames_nokv_results.csv
)

# Threshold configurations: "name|vision_threshold|prompt_threshold|semantic_threshold"
THRESHOLD_CONFIGS=(
  "conservative|0.85|0.82|0.82"
  "moderate|0.82|0.80|0.80"
  "aggressive|0.80|0.78|0.78"
)

INTERNVL_PROMPT='<image>
You are assisting with the GQA benchmark. Answer the question using the referenced image.
Image ID: {image_id}
Question: {question}
Answer:'

# Model configurations: "model|suffix|max_model_len|prompt_template"
QWEN_MODELS=(
  "Qwen/Qwen3-VL-2B-Instruct|qwen3vl-2b|8192|"
  "Qwen/Qwen3-VL-4B-Instruct|qwen3vl-4b|4096|"
  "Qwen/Qwen3-VL-8B-Instruct|qwen3vl-8b|4096|"
)

INTERNVL_MODELS=(
  "OpenGVLab/InternVL3_5-2B-Instruct|internvl35-2b|8192|${INTERNVL_PROMPT}"
  "OpenGVLab/InternVL3_5-4B-Instruct|internvl35-4b|4096|${INTERNVL_PROMPT}"
  "OpenGVLab/InternVL3_5-8B-Instruct|internvl35-8b|4096|${INTERNVL_PROMPT}"
)

run_experiment() {
  local model="$1"
  local suffix="$2"
  local max_len="$3"
  local prompt_template="$4"
  local threshold_name="$5"
  local vision_thresh="$6"
  local prompt_thresh="$7"
  local semantic_thresh="$8"
  local mode="$9"
  shift 9
  local extra_args=("$@")

  local exp_name="${suffix}_${threshold_name}"
  if [ "$mode" = "baseline" ]; then
    exp_name="${suffix}_baseline"
  fi

  local log_file="${LOG_DIR}/${exp_name}.log"

  echo "=== ${exp_name} ==="
  echo "model=${model}"
  echo "thresholds: vision=${vision_thresh}, prompt=${prompt_thresh}, semantic=${semantic_thresh}"
  echo "cache mode=${mode}"
  echo "log=${log_file}"

  local args=(
    "${COMMON_ARGS[@]}"
    --model "$model"
    --max-model-len "$max_len"
    --experiment-name "$exp_name"
  )

  if [ -n "$prompt_template" ]; then
    args+=(--prompt-template "$prompt_template")
  fi

  if [ "$mode" = "baseline" ]; then
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

  args+=(
    --index-encoder-device "$ENCODER_DEVICE"
    "${HF_ARGS[@]}"
    "${extra_args[@]}"
  )

  CUDA_LAUNCH_BLOCKING=1 python experiment2/run_benchmark.py "${args[@]}" 2>&1 | tee "$log_file"
  local exit_code=${PIPESTATUS[0]}
  if [ $exit_code -ne 0 ]; then
    echo "ERROR: Experiment $exp_name failed with exit code $exit_code"
    exit $exit_code
  fi
}

echo ""
echo "=== Qwen3-VL models ==="
for model_spec in "${QWEN_MODELS[@]}"; do
  IFS="|" read -r model suffix max_len prompt_template <<< "$model_spec"
  run_experiment "$model" "$suffix" "$max_len" "$prompt_template" "baseline" "0" "0" "0" "baseline"
  for thresh_spec in "${THRESHOLD_CONFIGS[@]}"; do
    IFS="|" read -r thresh_name vision_thresh prompt_thresh semantic_thresh <<< "$thresh_spec"
    run_experiment "$model" "$suffix" "$max_len" "$prompt_template" "$thresh_name" "$vision_thresh" "$prompt_thresh" "$semantic_thresh" "cached"
  done
done

echo ""
echo "=== InternVL3.5 models ==="
for model_spec in "${INTERNVL_MODELS[@]}"; do
  IFS="|" read -r model suffix max_len prompt_template <<< "$model_spec"
  run_experiment "$model" "$suffix" "$max_len" "$prompt_template" "baseline" "0" "0" "0" "baseline"
  for thresh_spec in "${THRESHOLD_CONFIGS[@]}"; do
    IFS="|" read -r thresh_name vision_thresh prompt_thresh semantic_thresh <<< "$thresh_spec"
    run_experiment "$model" "$suffix" "$max_len" "$prompt_template" "$thresh_name" "$vision_thresh" "$prompt_thresh" "$semantic_thresh" "cached"
  done
done

echo ""
echo "All experiments complete. Results: experiment2_frames_nokv_results.csv"
echo "Logs: ${LOG_DIR}/"
