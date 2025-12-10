#!/bin/bash
# Script to run video frames pipeline with caching

# Set image root
export GQA_IMAGE_ROOT=/root/585-Vision-model-caching/dataset_custom/unscrew_bottle_cap

echo "=========================================="
echo "Running Video Frames Pipeline with Cache"
echo "=========================================="
echo ""
echo "Image root: $GQA_IMAGE_ROOT"
echo "Dataset: video_frames_dataset.json"
echo ""

# Run with all cache layers enabled
python experiment2/test_vllm.py \
  --dataset video_frames \
  --video-frames-data video_frames_dataset.json \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --max-samples 64 \
  --chunk-source combined \
  --embedding-hook prompt_vision \
  --embedding-layer vision:512:0.85 \
  --embedding-layer prompt:384:0.8 \
  --similarity-threshold 0.8 \
  --cache-mode live \
  --trust-remote-code \
  --summary-log video_frames_results.csv \
  --experiment-name video_frames_cached

echo ""
echo "=========================================="
echo "Results saved to: video_frames_results.csv"
echo "=========================================="
