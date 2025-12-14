#!/usr/bin/env bash

set -euo pipefail

export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CACHE_MODE="${CACHE_MODE:-live}"

EXPERIMENT_ROOT="experiment2"
GQA_IMAGE_DIR="${GQA_IMAGE_DIR:-${EXPERIMENT_ROOT}/gqa_images}"
CACHE_MAX_SIZE_GB="${CACHE_MAX_SIZE_GB:-64}"
CACHE_DIR="${EXPERIMENT_ROOT}/kv_chunks_ablation"
FUSION_DIR="${EXPERIMENT_ROOT}/fusion_chunks_ablation"
LOG_DIR="${EXPERIMENT_ROOT}/experiment_logs"
SAMPLES_DIR="${LOG_DIR}/ablation_samples"

mkdir -p "${SAMPLES_DIR}"
mkdir -p "${GQA_IMAGE_DIR}"

# Cap the model context length so 32B+/38B+ variants fit in 48 GB VRAM.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_MODEL_LEN_ARGS=()
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  MAX_MODEL_LEN_ARGS=(--max-model-len "${MAX_MODEL_LEN}")
fi

COMMON_ARGS=(
  --dataset gqa
  --dataset-config val_balanced_instructions
  --split val
  --max-samples 512
  --shuffle-seed 42
  --chunk-source semantic
  --similarity-threshold 0.8
  --cache-dir "${CACHE_DIR}"
  --fusion-cache-dir "${FUSION_DIR}"
  --enable-fusion-cache
  --index-encoder sentence-transformers/all-MiniLM-L6-v2
  --index-encoder-device cuda
  --embedding-layer prompt:384:0.8
  --embedding-layer vision:512:0.8
  --embedding-hook prompt_vision
  --temperature 0.0
  --max-tokens 64
  --tensor-parallel-size 1
  --gpu-memory-utilization 0.9
  --cache-mode "${CACHE_MODE}"
  --gqa-image-dir "${GQA_IMAGE_DIR}"
  --cache-max-size-gb "${CACHE_MAX_SIZE_GB}"
  --trust-remote-code
  --summary-log "${LOG_DIR}/ablation_results.csv"
)

run_experiment() {
  local name="$1"
  shift
  echo "=== ${name} ==="
  rm -rf "${FUSION_DIR}" "${CACHE_DIR}"
  CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python "${EXPERIMENT_ROOT}/test_vllm.py" \
    --experiment-name "${name}" \
    "${COMMON_ARGS[@]}" \
    "${MAX_MODEL_LEN_ARGS[@]}" \
    --samples-jsonl "${SAMPLES_DIR}/${name}.jsonl" \
    "$@"
}

QWEN_MODELS=(
  "Qwen/Qwen3-VL-8B-Instruct|qwen3vl-8b-allcache"
  "Qwen/Qwen3-VL-4B-Instruct|qwen3vl-4b-allcache"
  "Qwen/Qwen3-VL-2B-Instruct|qwen3vl-2b-allcache"
)

for spec in "${QWEN_MODELS[@]}"; do
  IFS="|" read -r model slug max_len <<<"$spec"
  args=(--model "${model}")
  if [[ -n "${max_len:-}" ]]; then
    args+=(--max-model-len "${max_len}")
  fi
  run_experiment "${slug}" "${args[@]}"
done

INTERN_PROMPT=$'<image>\nYou are assisting with the GQA benchmark. Answer the question using the referenced image.\nImage ID: {image_id}\nQuestion: {question}\nAnswer:'
INTERN_MODELS=(
  "OpenGVLab/InternVL3_5-8B-Instruct|internvl35-8b-allcache"
  "OpenGVLab/InternVL3_5-4B-Instruct|internvl35-4b-allcache"
  "OpenGVLab/InternVL3_5-2B-Instruct|internvl35-2b-allcache"
)

for spec in "${INTERN_MODELS[@]}"; do
  IFS="|" read -r model slug max_len <<<"$spec"
  args=(--model "${model}" --prompt-template "${INTERN_PROMPT}")
  if [[ -n "${max_len:-}" ]]; then
    args+=(--max-model-len "${max_len}")
  fi
  run_experiment "${slug}" \
    "${args[@]}"
done
