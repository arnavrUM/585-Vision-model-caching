# Running the Pipeline on Video Frames Dataset

## Quick Start

### Step 1: Set Image Root Environment Variable

```bash
export GQA_IMAGE_ROOT=/root/585-Vision-model-caching/dataset_custom/unscrew_bottle_cap
```

### Step 2: Run with Cache (Recommended)

```bash
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
```

### Step 3: Run Baseline (Without Cache) for Comparison

```bash
python experiment2/test_vllm.py \
  --dataset video_frames \
  --video-frames-data video_frames_dataset.json \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --max-samples 64 \
  --chunk-source combined \
  --disable-semantic-cache \
  --disable-exact-cache \
  --cache-mode live \
  --trust-remote-code \
  --summary-log video_frames_results.csv \
  --experiment-name video_frames_baseline
```

---

## Complete Command Explanation

### Required Flags

| Flag | Value | Purpose |
|------|-------|---------|
| `--dataset` | `video_frames` | Use video frames dataset loader |
| `--video-frames-data` | `video_frames_dataset.json` | Path to your dataset JSON |
| `--model` | `Qwen/Qwen3-VL-2B-Instruct` | Vision-language model |
| `--max-samples` | `64` | Process all 64 frames |
| `--cache-mode` | `live` | **Enable actual caching** (not dry-run) |
| `--trust-remote-code` | (flag) | Required for some models |

### Cache Configuration Flags

| Flag | Value | Purpose |
|------|-------|---------|
| `--chunk-source` | `combined` | Use semantic + groups for cache keys (best for video) |
| `--embedding-hook` | `prompt_vision` | Extract both text and image embeddings |
| `--embedding-layer vision` | `512:0.85` | Vision cache: 512-dim, 0.85 similarity threshold |
| `--embedding-layer prompt` | `384:0.8` | Prompt cache: 384-dim, 0.8 similarity threshold |
| `--similarity-threshold` | `0.8` | Semantic text cache threshold |

### Logging Flags (Optional)

| Flag | Value | Purpose |
|------|-------|---------|
| `--summary-log` | `video_frames_results.csv` | Save aggregate metrics |
| `--experiment-name` | `video_frames_cached` | Label for this run |
| `--samples-jsonl` | `video_frames_samples.jsonl` | Per-sample detailed logs |

---

## Expected Performance

With your dataset structure:
- **Frames 1-11**: Same question/answer → **High exact cache hits**
- **Frames 12-16**: Same question/answer → **High exact cache hits**
- **Frames 17-43**: Same question/answer → **High exact cache hits**
- **Frames 44-64**: Same question/answer → **High exact cache hits**

**Expected cache hit rate**: 80-95% (depending on vision similarity)

**Expected speedup**: 2-5x faster than baseline

---

## Full Example: Baseline vs Cached

### 1. Baseline Run (No Cache)

```bash
export GQA_IMAGE_ROOT=/root/585-Vision-model-caching/dataset_custom/unscrew_bottle_cap

python experiment2/test_vllm.py \
  --dataset video_frames \
  --video-frames-data video_frames_dataset.json \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --max-samples 64 \
  --chunk-source combined \
  --disable-semantic-cache \
  --disable-exact-cache \
  --cache-mode live \
  --trust-remote-code \
  --summary-log video_frames_comparison.csv \
  --experiment-name baseline_no_cache \
  --samples-jsonl video_frames_baseline.jsonl
```

### 2. Cached Run (All Layers Enabled)

```bash
export GQA_IMAGE_ROOT=/root/585-Vision-model-caching/dataset_custom/unscrew_bottle_cap

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
  --summary-log video_frames_comparison.csv \
  --experiment-name cached_all_layers \
  --samples-jsonl video_frames_cached.jsonl
```

### 3. Compare Results

Check the CSV file for metrics:
- `total_latency`: Total time taken
- `cache_hit_rate`: Percentage of cache hits
- `avg_latency_per_sample`: Average time per frame

---

## Troubleshooting

### Error: "Video frames dataset not found"

**Solution**: Use absolute path:
```bash
--video-frames-data /root/585-Vision-model-caching/video_frames_dataset.json
```

### Error: "vision embedding hook could not resolve image file"

**Solution**: Verify environment variable:
```bash
echo $GQA_IMAGE_ROOT
# Should output: /root/585-Vision-model-caching/dataset_custom/unscrew_bottle_cap

# Verify files exist
ls $GQA_IMAGE_ROOT/output_0001.png
```

### Cache Not Working

**Solution**: Make sure `--cache-mode live` (not `dry-run`):
```bash
--cache-mode live  # ✅ Actually uses cache
--cache-mode dry-run  # ❌ Only simulates (no speedup)
```

### Low Cache Hit Rate

**Possible causes**:
1. Vision embeddings too different → Lower vision threshold: `--embedding-layer vision:512:0.75`
2. Questions too different → Lower prompt threshold: `--embedding-layer prompt:384:0.7`
3. Semantic threshold too high → Lower: `--similarity-threshold 0.7`

---

## Advanced: Testing Different Cache Configurations

### Test Only Exact Cache

```bash
python experiment2/test_vllm.py \
  --dataset video_frames \
  --video-frames-data video_frames_dataset.json \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --max-samples 64 \
  --disable-semantic-cache \
  --cache-mode live \
  --trust-remote-code \
  --experiment-name exact_only
```

### Test Only Semantic Text Cache

```bash
python experiment2/test_vllm.py \
  --dataset video_frames \
  --video-frames-data video_frames_dataset.json \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --max-samples 64 \
  --disable-exact-cache \
  --similarity-threshold 0.8 \
  --cache-mode live \
  --trust-remote-code \
  --experiment-name semantic_only
```

### Test Only Embedding Cache

```bash
python experiment2/test_vllm.py \
  --dataset video_frames \
  --video-frames-data video_frames_dataset.json \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --max-samples 64 \
  --chunk-source combined \
  --embedding-hook prompt_vision \
  --embedding-layer vision:512:0.85 \
  --embedding-layer prompt:384:0.8 \
  --disable-semantic-cache \
  --disable-exact-cache \
  --cache-mode live \
  --trust-remote-code \
  --experiment-name embedding_only
```

---

## Summary

**Minimum command to run**:
```bash
export GQA_IMAGE_ROOT=/root/585-Vision-model-caching/dataset_custom/unscrew_bottle_cap

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
  --trust-remote-code
```

**Key points**:
1. ✅ Set `GQA_IMAGE_ROOT` environment variable
2. ✅ Use `--cache-mode live` (not `dry-run`)
3. ✅ Use `--dataset video_frames` with `--video-frames-data`
4. ✅ Enable all cache layers for maximum speedup
5. ✅ Run baseline first to compare performance

