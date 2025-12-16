# Vision Model Caching Experiments

This project contains two experiments for evaluating semantic caching strategies in vision models:

- **Experiment 1**: ViT-based image classification (only L2 embedding)
- **Experiment 2**: VLM scene understanding (using all cache layers)

## Prerequisites

- Python 3.11

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```


## Experiment 1: Image Classification

### Step 1: Fine-tune the Model

First, train the ViT model on the rice dataset:

```bash
cd experiment1
python train.py
```

This will:
- Load the Rice Disease Classification Dataset
- Fine-tune a ViT-base model for classification
- Save the fine-tuned model to `./vit_rice_finetuned`


### Step 2: Run Cache Comparison Experiments

After training, run the cache comparison experiments:

```bash
python test_compare.py
```

This script will:
- Run baseline inference (no cache)
- Run with cache inference, on multiple encoders (ResNet18, MobileNetV2, EfficientNet-B0) with different similarity thresholds

**Results**: Experiment data is saved to `experiment1_results.csv` in the `experiment1/` directory.

## Experiment 2: VLM Scene Understanding with Hierarchical Caching

Experiment 2 evaluates multi-level caching (MLC) on vision-language models using the GQA dataset. The experiment implements a hierarchical cache with multiple levels:
- **L0.5**: Exact text matching
- **L1**: Semantic text similarity
- **L2**: Multimodal embeddings (prompt + vision)

### Running Ablation Study

Run ablation experiments to test individual cache techniques across different thresholds:

```bash
chmod +x ./experiment2/ablation.sh
./experiment2/ablation.sh
```

This will test:
- **Qwen3-VL-2B-Instruct** and **InternVL3.5-2B-Instruct**
- Each cache level in isolation (exact-only, fusion-only, semantic with thresholds 0.5-0.9)
- Embedding cache with various threshold combinations
- Native VLM embeddings at different similarity levels

**Results**: Saved to `experiment2/experiment_logs/ablation_results.csv`

**Sample outputs**: Per-sample traces saved to `experiment2/experiment_logs/ablation_samples/`

### Running Model Comparison

Compare cache performance across different model sizes with all techniques enabled:

```bash
chmod +x ./experiment2/model_comparison.sh
./experiment2/model_comparison.sh
```

This tests:
- **Qwen3-VL**: 2B, 4B, 8B variants
- **InternVL3.5**: 2B, 4B, 8B variants
- Both with and without caching enabled
- All cache levels active (exact text + semantic text + prompt/vision embeddings)

**Results**: Saved to `experiment2/experiment_logs/model_comparison_results.csv`

### Experiment 2 Custom: Video Frames (No KV Injection)

A custom experiment for testing on video frame sequences without KV chunk injection.

#### Dataset

The video frames dataset is located in the `dataset_custom/` folder. The dataset contains sequential video frames (e.g., `unscrew_bottle_cap/` with frames `output_0001.png` through `output_0064.png`). Metadata lives at `dataset_custom/video_frames_labels.json`.

#### Running the Experiment

```bash
chmod +x ./experiment2/run_video_frames.sh
./experiment2/run_video_frames.sh
```

This script will:
- Run experiments on multiple vision-language models:
  - **Qwen3-VL**: 2B, 4B, 8B Instruct variants
  - **InternVL3.5**: 2B, 4B, 8B Instruct variants
- Test different cache configurations:
  - **Baseline**: No caching
  - **Conservative**: vision=0.85, prompt=0.82, semantic=0.82
  - **Moderate**: vision=0.82, prompt=0.80, semantic=0.80
  - **Aggressive**: vision=0.80, prompt=0.78, semantic=0.78
- Enable hierarchical caching with:
  - L0.5: Exact text cache
  - L1: Semantic text cache
  - L2: Prompt and vision embedding caches

**Results**: Experiment result is saved to `experiment2_frames_nokv_results.csv` in the project root.

**Logs**: Individual experiment logs are saved to `video_frames_logs_<timestamp>/` directory with one log file per experiment run.

You can override the dataset or frame root if you relocate the assets:
- `DATA_FILE=/path/to/video_frames_labels.json ./experiment2/run_video_frames.sh`
- `FRAMES_ROOT=/data/custom_frames ./experiment2/run_video_frames.sh`

The same paths can be passed directly to `experiment2/run_benchmark.py` via `--video-frames-data` and `--video-frames-root`.

## Output Files

- **Experiment 1**: `experiment1/experiment1_results.csv`
- **Experiment 2 Ablation**: `experiment2/experiment_logs/ablation_results.csv`
- **Experiment 2 Model Comparison**: `experiment2/experiment_logs/model_comparison_results.csv`
- **Experiment 2 Custom (Video Frames)**: `experiment2_frames_nokv_results.csv`
- **Experiment 2 Custom Logs**: `video_frames_logs_<timestamp>/`
