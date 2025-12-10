# Video Frames Dataset - Quick Start

## 🚀 Quick Setup (3 Steps)

### 1. Create Dataset JSON

**Option A: Use helper script**
```bash
python experiment2/create_video_frames_dataset.py \
  --output video_frames_dataset.json \
  --num-frames 100 \
  --video-id video_001
```

**Option B: Create manually** (see `VIDEO_FRAMES_DATASET_GUIDE.md` for format)

### 2. Set Image Directory

```bash
export GQA_IMAGE_ROOT=/path/to/your/video/frames
```

### 3. Run Experiment

```bash
python experiment2/test_vllm.py \
  --dataset video_frames \
  --video-frames-data video_frames_dataset.json \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --max-samples 1000 \
  --chunk-source combined \
  --embedding-hook prompt_vision \
  --embedding-layer vision:512:0.85 \
  --embedding-layer prompt:384:0.8 \
  --similarity-threshold 0.8 \
  --trust-remote-code
```

---

## 📋 Dataset Format (Minimal)

**File**: `video_frames_dataset.json`

```json
[
  {
    "id": "video_001_frame_0100",
    "imageId": "video_001_frame_0100",
    "question": "What objects are visible?",
    "answer": "car, person"
  }
]
```

**Required fields**: `id`, `imageId`, `question`, `answer`

---

## 📁 Directory Structure

```
/data/video_experiment/
├── frames/
│   ├── video_001_frame_0100.jpg  ← Image files
│   ├── video_001_frame_0101.jpg
│   └── ...
└── dataset.json                   ← Dataset JSON
```

**Important**: `imageId` in JSON must match image filename (without extension)

---

## 🔧 Full Example

```bash
# 1. Create dataset
python experiment2/create_video_frames_dataset.py \
  --output /data/video_experiment/dataset.json \
  --num-frames 500 \
  --video-id video_001

# 2. Set image root
export GQA_IMAGE_ROOT=/data/video_experiment/frames

# 3. Run
python experiment2/test_vllm.py \
  --dataset video_frames \
  --video-frames-data /data/video_experiment/dataset.json \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --max-samples 500 \
  --chunk-source combined \
  --embedding-hook prompt_vision \
  --embedding-layer vision:512:0.85 \
  --embedding-layer prompt:384:0.8 \
  --similarity-threshold 0.8 \
  --trust-remote-code
```

---

## 📖 Full Documentation

See **`VIDEO_FRAMES_DATASET_GUIDE.md`** for:
- Complete dataset format specification
- Optional fields (groups, semantic, metadata)
- Image file naming conventions
- Troubleshooting
- Python scripts for dataset creation

---

## ✅ Checklist

- [ ] Created JSON file with frame data
- [ ] Required fields present: `id`, `imageId`, `question`, `answer`
- [ ] Image files placed in directory
- [ ] `GQA_IMAGE_ROOT` environment variable set
- [ ] `imageId` matches image filenames
- [ ] Run with `--dataset video_frames --video-frames-data <path>`

