# Ablation Study Script Explanation

## What is `ablation.sh`?

`ablation.sh` is a shell script that runs a **comprehensive ablation study** to systematically test different cache techniques and threshold combinations. It evaluates how each caching method performs in isolation and in combination.

## Purpose

The script systematically tests:
- **Which cache techniques work best**
- **How different similarity thresholds affect performance**
- **How techniques compare across different models**

## Experiment Structure

The script runs **32 total experiments** (16 per model × 2 models):

### Models Tested
1. **Qwen/Qwen3-VL-2B-Instruct** (lines 5-50)
2. **OpenGVLab/InternVL3_5-2B-Instruct** (lines 52-162)

### Experiment Categories (per model)

#### 1. Baseline Techniques (2 experiments)
- **exact-only**: Only exact text cache enabled
  - `--disable-semantic-cache` (disables L1 semantic text cache)
  - Exact cache still works (L0.5)
  
- **fusion-only**: Only fusion cache enabled
  - `--enable-fusion-cache`
  - `--disable-semantic-cache --disable-exact-cache`
  - Tests fusion cache in isolation

#### 2. Semantic Text Cache Thresholds (5 experiments)
Tests semantic text cache (L1) with different similarity thresholds:
- `semantic-0.5`: `--similarity-threshold 0.5`
- `semantic-0.6`: `--similarity-threshold 0.6`
- `semantic-0.7`: `--similarity-threshold 0.7`
- `semantic-0.8`: `--similarity-threshold 0.8`
- `semantic-0.9`: `--similarity-threshold 0.9`

All with `--disable-exact-cache` to test semantic cache in isolation.

#### 3. Embedding Cache Threshold Grid (9 experiments)
Tests embedding cache (L2) with different threshold combinations:
- **3×3 grid**: prompt threshold (0.7, 0.8, 0.9) × vision threshold (0.7, 0.8, 0.9)
- Format: `embed-p0.7-v0.7`, `embed-p0.7-v0.8`, etc.
- Uses `--embedding-layer prompt:384:THRESH --embedding-layer vision:512:THRESH`
- With `--disable-semantic-cache --disable-exact-cache` to test embedding cache in isolation

## Key Features

### 1. Cache Isolation
Each experiment **cleans cache directories** before running:
```bash
rm -rf experiment2/fusion_chunks_ablation && rm -rf experiment2/kv_chunks_ablation
```
This ensures no cross-contamination between experiments.

### 2. Consistent Configuration
All experiments use the same base configuration:
- **Dataset**: GQA `val_balanced_instructions` split
- **Samples**: 1024 samples
- **Shuffle seed**: 42 (ensures same sample order)
- **Chunk source**: `semantic`
- **Temperature**: 0.0 (deterministic)
- **Max tokens**: 64
- **Cache mode**: `dry-run` (simulates caching without actual GPU operations)

### 3. Logging
Results are logged to:
- **Summary CSV**: `experiment2/experiment_logs/ablation_results.csv`
  - Aggregated metrics per experiment
- **Per-sample JSONL**: `experiment2/experiment_logs/ablation_samples/{experiment-name}.jsonl`
  - Detailed results for each sample

### 4. Model-Specific Settings
- **Qwen**: Uses default prompt template
- **InternVL**: Uses custom prompt template (lines 53-57, 60-64, etc.)

## Experiment Naming Convention

- `{model}-{technique}-{params}`
  - `qwen-exact-only`
  - `qwen-semantic-0.8`
  - `qwen-embed-p0.8-v0.9` (prompt threshold 0.8, vision threshold 0.9)
  - `internvl-fusion-only`

## What Gets Measured

Each experiment logs:
- **Cache hit rate**: Percentage of prompts that hit cache
- **Latency**: Average response time (hits vs misses)
- **Answer accuracy**: Whether cached responses match expected answers
- **Hit breakdown**: Which cache layer provided the hit
- **Technique status**: Hit/miss/skip for each cache layer

## Running the Ablation Study

```bash
chmod +x experiment2/ablation.sh
./experiment2/ablation.sh
```

**Note**: This will take a long time (32 experiments × time per experiment). Each experiment processes 1024 samples.

## Expected Output

After completion, you'll have:
- `experiment2/experiment_logs/ablation_results.csv` - Summary table with all experiments
- `experiment2/experiment_logs/ablation_samples/*.jsonl` - Detailed per-sample results
- You can analyze which techniques and thresholds work best

## Analysis Example

You can compare:
- **Which threshold gives best hit rate**: Compare semantic-0.5 through semantic-0.9
- **Embedding vs semantic**: Compare embedding experiments vs semantic experiments
- **Model differences**: Compare qwen-* vs internvl-* results
- **Technique effectiveness**: Compare exact-only, fusion-only, semantic, embedding

## Summary Table

| Category | Experiments | What It Tests |
|----------|-------------|---------------|
| Baseline | 2 per model | Exact cache only, Fusion cache only |
| Semantic | 5 per model | Text similarity thresholds (0.5-0.9) |
| Embedding | 9 per model | Embedding threshold grid (3×3) |
| **Total** | **32** | **16 per model × 2 models** |

This systematic approach allows you to understand the contribution of each cache layer and find optimal threshold settings for your use case.

