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

## Experiment 2: MLC (Video Frames, No KV)

### Dataset

The video frames dataset is located in the `dataset_custom/` folder. The dataset contains sequential video frames (e.g., `unscrew_bottle_cap/` with frames `output_0001.png` through `output_0064.png`).

### Running the Experiment

Run MLC pipeline on video frames data (No KV chunk injection):

```bash
./run_video_frames.sh
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
- (Testing on sample dataset `dataset_custom/unscrew_bottle_cap`)

**Results**: Experiment result is saved to `experiment2_frames_nokv_results.csv` in the project root.

**Logs**: Individual experiment logs are saved to `video_frames_logs_<timestamp>/` directory with one log file per experiment run.

## Output Files

- **Experiment 1**: `experiment1/experiment1_results.csv`
- **Experiment 2**: `experiment2_frames_nokv_results.csv`
- **Experiment 2 Logs**: `video_frames_logs_<timestamp>/`
