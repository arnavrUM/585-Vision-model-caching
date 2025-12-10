# Dataset Configuration Guide

This guide explains where datasets are located and how to change them.

## Current Dataset Options

The system supports **3 built-in dataset types**:

1. **GQA** (`gqa`) - Default, loaded from HuggingFace
2. **LLaVA-Instruct-150K** (`llava150k`) - Loaded from HuggingFace or local JSON
3. **Synthetic** (`synthetic`) - Built-in test dataset for cache validation

## Where Datasets Are Located

### 1. GQA Dataset

**Source**: HuggingFace Hub (`lmms-lab/GQA`)

**Location**: Automatically downloaded and cached by HuggingFace `datasets` library
- Default cache: `~/.cache/huggingface/datasets/`
- No manual download needed - it's fetched automatically

**Code**: `experiment2/test_vllm.py` lines 267-274

```python
def load_gqa_dataset(config: str, split: str, limit: int | None, seed: int | None) -> Dataset:
    dataset = load_dataset("lmms-lab/GQA", config, split=split)
    # ... shuffling and limiting ...
    return dataset
```

### 2. LLaVA-Instruct-150K Dataset

**Source**: HuggingFace Hub or local JSON file

**Location**: 
- **Remote**: Downloaded from HuggingFace to `experiment2/dataset_cache/llava_instruct_150k.json`
- **Local cache**: `experiment2/dataset_cache/` (configurable via `VMC_CACHE_ROOT` or `SCRATCH_DIR`)

**Code**: `experiment2/test_vllm.py` lines 277-289

```python
def load_llava_dataset(data_url: str, limit: int | None, seed: int | None) -> Dataset:
    local_path = _ensure_llava_file(data_url)  # Downloads if needed
    # ... loads from JSON ...
    return dataset
```

**Default URL**: `https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json`

### 3. Synthetic Dataset

**Source**: Built-in function that generates test samples

**Location**: Generated in-memory, no files

**Code**: `experiment2/test_vllm.py` lines 338-432

---

## How to Change the Dataset

### Method 1: Command Line (Easiest)

#### Switch Between Built-in Datasets

```bash
# Use GQA (default)
python experiment2/test_vllm.py --dataset gqa --dataset-config val_balanced_instructions --split val ...

# Use LLaVA
python experiment2/test_vllm.py --dataset llava150k ...

# Use synthetic test dataset
python experiment2/test_vllm.py --dataset synthetic ...
```

#### GQA Configuration Options

```bash
# Different GQA configs
--dataset-config val_balanced_instructions  # Default
--dataset-config train_balanced_instructions
--dataset-config val_all

# Different splits
--split val
--split val[:256]  # First 256 samples
--split train
```

#### LLaVA Configuration

```bash
# Use custom LLaVA data URL
python experiment2/test_vllm.py \
  --dataset llava150k \
  --llava-data-url https://your-custom-url.com/data.json \
  ...
```

### Method 2: Experiment Spec JSON

When using `run_experiments.py`:

```json
{
  "defaults": {
    "dataset_config": "val_balanced_instructions",
    "split": "val",
    "max_samples": 128
  },
  "experiments": [
    {
      "name": "gqa-experiment",
      "dataset_config": "val_balanced_instructions",
      "split": "val"
    },
    {
      "name": "llava-experiment",
      "dataset": "llava150k"
    }
  ]
}
```

**Note**: The `dataset` field in ExperimentSpec is not directly supported - you need to modify `run_experiments.py` or use CLI.

### Method 3: Add a Custom Dataset (Code Changes)

To add your own dataset, you need to:

1. **Create a dataset loader function** in `test_vllm.py`:

```python
def load_custom_dataset(data_path: str, limit: int | None, seed: int | None) -> Dataset:
    # Load your dataset from file or API
    # Format: list of dicts with fields like:
    #   - "id": sample ID
    #   - "imageId": image identifier
    #   - "question": question text
    #   - "answer": answer text
    #   - "fullAnswer": full answer (optional)
    #   - "groups": dict (optional)
    #   - "semantic": list (optional)
    
    data = load_your_data(data_path)
    dataset = Dataset.from_list(data)
    
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    if limit is not None:
        limit = min(limit, len(dataset))
        dataset = dataset.select(range(limit))
    
    return dataset
```

2. **Add dataset choice** in `parse_args()` (around line 873):

```python
parser.add_argument(
    "--dataset",
    choices=["gqa", "llava150k", "synthetic", "custom"],  # Add "custom"
    default="gqa",
    help="Dataset to evaluate...",
)
```

3. **Add dataset loading logic** in `main()` (around line 1066):

```python
if args.dataset == "gqa":
    dataset = load_gqa_dataset(args.dataset_config, args.split, limit, shuffle_seed)
    base_samples: Sequence[dict] = dataset
    dataset_label = f"lmms-lab/GQA ({args.dataset_config}, split={args.split})"
elif args.dataset == "llava150k":
    dataset = load_llava_dataset(args.llava_data_url, limit, shuffle_seed)
    base_samples = build_llava_records(dataset)
    dataset_label = "LLaVA-Instruct-150K"
elif args.dataset == "custom":
    dataset = load_custom_dataset(args.custom_data_path, limit, shuffle_seed)
    base_samples: Sequence[dict] = dataset
    dataset_label = "Custom Dataset"
else:
    dataset = build_synthetic_samples()
    base_samples = dataset if limit is None else dataset[:limit]
    dataset_label = "synthetic-cache-validation"
```

4. **Add command-line argument** for your dataset path:

```python
parser.add_argument(
    "--custom-data-path",
    help="Path to custom dataset file",
)
```

---

## Dataset Format Requirements

Your dataset must provide samples in this format:

```python
{
    "id": "sample_001",                    # Required: unique sample ID
    "imageId": "image_123",                # Required: image identifier
    "question": "What color is the car?",  # Required: question text
    "answer": "red",                       # Required: short answer
    "fullAnswer": "The car is red.",      # Optional: full answer
    "groups": {                            # Optional: grouping metadata
        "global": "vehicles",
        "local": "cars"
    },
    "semantic": [                          # Optional: semantic program
        {"operation": "query", "argument": "color"},
        {"operation": "filter", "argument": "car"}
    ],
    "semanticStr": "query:color | filter:car"  # Optional: semantic string
}
```

The `build_prompts()` function (line 435) processes these fields to create prompts.

---

## Image Files Location

### For Vision Embedding Hooks

If you're using vision embedding hooks (`--embedding-hook vision` or `prompt_vision`), you need to provide image files.

**Set environment variable**:
```bash
export GQA_IMAGE_ROOT=/path/to/gqa/images
# or
export LLAVA_IMAGE_ROOT=/path/to/llava/images
# or
export SEMANTIC_CACHE_IMAGE_ROOT=/path/to/images
```

**Image file naming**:
- The system looks for: `<image_id>.jpg`, `<image_id>.jpeg`, `<image_id>.png`
- Also checks: `<first_3_chars>/<image_id>.jpg` (for sharded directories)

**Code**: `experiment2/semantic_cache/embedding_hooks.py` lines 77-109

### Example Image Directory Structure

```
/path/to/images/
├── 000/
│   ├── 000123.jpg
│   └── 000456.jpg
├── 001/
│   └── 001789.jpg
└── image_abc.jpg
```

---

## Dataset Configuration Examples

### Example 1: Use GQA with Custom Split

```bash
python experiment2/test_vllm.py \
  --dataset gqa \
  --dataset-config val_balanced_instructions \
  --split "val[:512]" \
  --max-samples 512 \
  --shuffle-seed 42 \
  ...
```

### Example 2: Use LLaVA with Local JSON

```bash
# First, download or create your JSON file
# Then point to it:
python experiment2/test_vllm.py \
  --dataset llava150k \
  --llava-data-url file:///path/to/local/llava_data.json \
  --max-samples 128 \
  ...
```

### Example 3: Use Synthetic Dataset for Testing

```bash
python experiment2/test_vllm.py \
  --dataset synthetic \
  --max-samples 10 \
  ...
```

### Example 4: Custom Dataset with Images

```bash
# Set image root
export GQA_IMAGE_ROOT=/data/my_images

# Run with vision embeddings
python experiment2/test_vllm.py \
  --dataset gqa \
  --embedding-hook prompt_vision \
  --embedding-layer vision:512:0.85 \
  ...
```

---

## Dataset Cache Locations

### HuggingFace Datasets Cache

Default location: `~/.cache/huggingface/datasets/`

To change:
```bash
export HF_DATASETS_CACHE=/path/to/custom/cache
```

### LLaVA JSON Cache

Default location: `experiment2/dataset_cache/llava_instruct_150k.json`

To change: Set environment variable:
```bash
export VMC_CACHE_ROOT=/path/to/cache
# or
export SCRATCH_DIR=/path/to/cache
```

**Code**: `experiment2/test_vllm.py` lines 69-86

---

## Troubleshooting

### Dataset Not Found

**Error**: `DatasetNotFoundError` or similar

**Solution**: 
- Check internet connection (for HuggingFace datasets)
- Verify dataset name: `lmms-lab/GQA` (not `gqa` or `GQA`)
- Check HuggingFace cache: `~/.cache/huggingface/datasets/`

### Images Not Found

**Error**: `[warn] vision embedding hook could not resolve image file`

**Solution**:
1. Set image root environment variable:
   ```bash
   export GQA_IMAGE_ROOT=/path/to/images
   ```
2. Verify image files exist: `<image_id>.jpg` in the root directory
3. Check image ID format matches your dataset

### Wrong Dataset Format

**Error**: Missing fields or incorrect structure

**Solution**: Ensure your dataset has required fields:
- `id`, `imageId`, `question`, `answer`
- Optional but recommended: `fullAnswer`, `groups`, `semantic`

---

## Summary

| Dataset | Source | Location | How to Use |
|---------|--------|----------|------------|
| **GQA** | HuggingFace | Auto-downloaded | `--dataset gqa --dataset-config val_balanced_instructions` |
| **LLaVA** | HuggingFace/JSON | `experiment2/dataset_cache/` | `--dataset llava150k` |
| **Synthetic** | Built-in | In-memory | `--dataset synthetic` |
| **Custom** | Your files | Your path | Add loader function (see Method 3) |

The easiest way to change datasets is via the `--dataset` command-line argument. For custom datasets, you'll need to add a loader function following the pattern of `load_gqa_dataset()` or `load_llava_dataset()`.

