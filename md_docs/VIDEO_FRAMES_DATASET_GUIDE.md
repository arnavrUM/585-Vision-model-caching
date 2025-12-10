# Video Frames Dataset Setup Guide

This guide explains how to prepare and use your video frames dataset with the caching pipeline.

## Overview

The system now supports custom video frames datasets. You provide a JSON file with frame metadata, and the system processes it with full caching support.

---

## Dataset Format

### JSON File Structure

Create a JSON file with a **list of sample objects**. Each sample represents one frame with its question/answer.

**File**: `video_frames_dataset.json`

```json
[
  {
    "id": "video_001_frame_0100",
    "imageId": "video_001_frame_0100",
    "question": "What objects are visible in this frame?",
    "answer": "car, person, building",
    "fullAnswer": "A red car, a person walking, and a building in the background.",
    "groups": {
      "global": "outdoor",
      "local": "street"
    },
    "semantic": [
      {"operation": "query", "argument": "objects"},
      {"operation": "filter", "argument": "visible"}
    ],
    "semanticStr": "query:objects | filter:visible",
    "metadata": {
      "video_id": "video_001",
      "frame_number": 100,
      "timestamp": 3.33,
      "scene_id": "scene_05"
    }
  },
  {
    "id": "video_001_frame_0101",
    "imageId": "video_001_frame_0101",
    "question": "What color is the car?",
    "answer": "red",
    "fullAnswer": "The car is red.",
    "groups": {
      "global": "outdoor",
      "local": "street"
    },
    "semantic": [
      {"operation": "query", "argument": "color"},
      {"operation": "filter", "argument": "car"}
    ],
    "semanticStr": "query:color | filter:car",
    "metadata": {
      "video_id": "video_001",
      "frame_number": 101,
      "timestamp": 3.37,
      "scene_id": "scene_05"
    }
  }
]
```

### Required Fields

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `id` | string | ✅ Yes | Unique sample identifier | `"video_001_frame_0100"` |
| `imageId` | string | ✅ Yes | Frame identifier (used for image lookup) | `"video_001_frame_0100"` |
| `question` | string | ✅ Yes | Question text | `"What color is the car?"` |
| `answer` | string | ✅ Yes | Short answer | `"red"` |

### Optional Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `fullAnswer` | string | Full answer text | `"The car is red."` |
| `groups` | object | Scene/location grouping | `{"global": "outdoor", "local": "street"}` |
| `semantic` | array | Semantic program steps | `[{"operation": "query", "argument": "color"}]` |
| `semanticStr` | string | Semantic string representation | `"query:color | filter:car"` |
| `metadata` | object | Additional metadata | `{"video_id": "video_001", "frame_number": 100}` |

---

## Image Files Setup

### Directory Structure

Place your frame images in a directory. The system will look for images matching the `imageId` field.

**Recommended structure**:
```
/path/to/video/frames/
├── video_001_frame_0100.jpg
├── video_001_frame_0101.jpg
├── video_001_frame_0102.jpg
├── video_002_frame_0000.jpg
└── ...
```

**Alternative (sharded)**:
```
/path/to/video/frames/
├── 000/
│   ├── 000100.jpg
│   └── 000101.jpg
├── 001/
│   └── 001200.jpg
└── ...
```

### Image File Naming

The system looks for images using the `imageId` field:

1. **Direct match**: `<imageId>.jpg`, `<imageId>.jpeg`, `<imageId>.png`
2. **Sharded**: `<first_3_chars>/<imageId>.jpg` (if imageId is long)

**Example**:
- `imageId: "video_001_frame_0100"` → Looks for:
  - `video_001_frame_0100.jpg`
  - `video_001_frame_0100.jpeg`
  - `video_001_frame_0100.png`
  - `vid/video_001_frame_0100.jpg` (sharded)

### Set Image Root Environment Variable

```bash
export GQA_IMAGE_ROOT=/path/to/video/frames
# or
export LLAVA_IMAGE_ROOT=/path/to/video/frames
# or
export SEMANTIC_CACHE_IMAGE_ROOT=/path/to/video/frames
```

The system checks these environment variables in order.

---

## Creating Your Dataset JSON

### Method 1: Manual Creation

Create a JSON file with your frame data:

```json
[
  {
    "id": "frame_001",
    "imageId": "frame_001",
    "question": "What is in this frame?",
    "answer": "car"
  },
  {
    "id": "frame_002",
    "imageId": "frame_002",
    "question": "What color is the car?",
    "answer": "red"
  }
]
```

### Method 2: Python Script

```python
import json
from pathlib import Path

# Your frame data
frames = [
    {
        "id": f"video_001_frame_{i:04d}",
        "imageId": f"video_001_frame_{i:04d}",
        "question": "What objects are visible?",
        "answer": "car, person",
        "fullAnswer": "A car and a person are visible.",
        "groups": {
            "global": "outdoor",
            "local": "street"
        },
        "metadata": {
            "video_id": "video_001",
            "frame_number": i,
            "timestamp": i * 0.033,  # 30 FPS
            "scene_id": "scene_01"
        }
    }
    for i in range(100, 200)  # Frames 100-199
]

# Save to JSON
output_path = Path("video_frames_dataset.json")
with output_path.open("w") as f:
    json.dump(frames, f, indent=2)

print(f"Created dataset with {len(frames)} frames")
```

### Method 3: From Video File

```python
import cv2
import json
from pathlib import Path

def extract_frames_to_dataset(video_path: str, output_json: str, questions: list[str]):
    """Extract frames from video and create dataset JSON."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame image
        frame_id = f"video_001_frame_{frame_num:04d}"
        frame_path = f"/path/to/frames/{frame_id}.jpg"
        cv2.imwrite(frame_path, frame)
        
        # Create dataset entry
        frames.append({
            "id": frame_id,
            "imageId": frame_id,
            "question": questions[frame_num % len(questions)],  # Cycle through questions
            "answer": "placeholder",  # You'll need to provide answers
            "groups": {
                "global": "video",
                "local": f"scene_{frame_num // 100}"
            },
            "metadata": {
                "video_id": "video_001",
                "frame_number": frame_num,
                "timestamp": frame_num / 30.0  # Assuming 30 FPS
            }
        })
        
        frame_num += 1
    
    cap.release()
    
    # Save dataset
    with open(output_json, "w") as f:
        json.dump(frames, f, indent=2)
    
    print(f"Extracted {len(frames)} frames to {output_json}")

# Usage
extract_frames_to_dataset(
    video_path="my_video.mp4",
    output_json="video_frames_dataset.json",
    questions=["What objects are visible?", "What is happening?", "Who is in the frame?"]
)
```

---

## Running with Video Frames Dataset

### Basic Command

```bash
# Set image root
export GQA_IMAGE_ROOT=/path/to/video/frames

# Run with video frames dataset
python experiment2/test_vllm.py \
  --dataset video_frames \
  --video-frames-data /path/to/video_frames_dataset.json \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --max-samples 1000 \
  --chunk-source combined \
  --embedding-hook prompt_vision \
  --embedding-layer vision:512:0.85 \
  --embedding-layer prompt:384:0.8 \
  --similarity-threshold 0.8 \
  --trust-remote-code
```

### Full Example with All Options

```bash
# Set image root
export GQA_IMAGE_ROOT=/data/video_frames

# Run experiment
python experiment2/test_vllm.py \
  --dataset video_frames \
  --video-frames-data /data/video_frames_dataset.json \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --max-samples 1000 \
  --shuffle-seed 42 \
  --chunk-source combined \
  --embedding-hook prompt_vision \
  --embedding-layer vision:512:0.85 \
  --embedding-layer prompt:384:0.8 \
  --similarity-threshold 0.8 \
  --cache-dir experiment2/kv_chunks_video \
  --fusion-cache-dir experiment2/fusion_chunks_video \
  --temperature 0.0 \
  --max-tokens 64 \
  --trust-remote-code \
  --summary-log video_experiment_results.csv \
  --samples-jsonl video_experiment_samples.jsonl
```

---

## Complete Setup Example

### Step 1: Prepare Your Data

**Directory structure**:
```
/data/video_experiment/
├── frames/
│   ├── video_001_frame_0100.jpg
│   ├── video_001_frame_0101.jpg
│   └── ...
└── dataset.json
```

**dataset.json**:
```json
[
  {
    "id": "video_001_frame_0100",
    "imageId": "video_001_frame_0100",
    "question": "What objects are visible?",
    "answer": "car, person",
    "fullAnswer": "A red car and a person walking.",
    "groups": {
      "global": "outdoor",
      "local": "street"
    },
    "metadata": {
      "video_id": "video_001",
      "frame_number": 100,
      "scene_id": "scene_01"
    }
  }
]
```

### Step 2: Set Environment Variables

```bash
export GQA_IMAGE_ROOT=/data/video_experiment/frames
```

### Step 3: Run the Experiment

```bash
python experiment2/test_vllm.py \
  --dataset video_frames \
  --video-frames-data /data/video_experiment/dataset.json \
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

## Dataset Format Examples

### Minimal Format (Required Fields Only)

```json
[
  {
    "id": "frame_001",
    "imageId": "frame_001",
    "question": "What is in this frame?",
    "answer": "car"
  }
]
```

### With Scene Grouping (Recommended for Video)

```json
[
  {
    "id": "video_001_frame_0100",
    "imageId": "video_001_frame_0100",
    "question": "What objects are visible?",
    "answer": "car, person",
    "fullAnswer": "A red car and a person.",
    "groups": {
      "global": "outdoor",
      "local": "street_scene_01"
    },
    "metadata": {
      "video_id": "video_001",
      "frame_number": 100,
      "scene_id": "scene_01"
    }
  }
]
```

### With Semantic Programs

```json
[
  {
    "id": "video_001_frame_0100",
    "imageId": "video_001_frame_0100",
    "question": "What color is the car?",
    "answer": "red",
    "fullAnswer": "The car is red.",
    "groups": {
      "global": "outdoor",
      "local": "street"
    },
    "semantic": [
      {"operation": "query", "argument": "color"},
      {"operation": "filter", "argument": "car"}
    ],
    "semanticStr": "query:color | filter:car"
  }
]
```

---

## Tips for Video Frames

### 1. Use Scene Grouping

**Add `groups` field** to help semantic cache:
```json
{
  "groups": {
    "global": "indoor",  // Scene type
    "local": "kitchen_scene_1"  // Specific scene
  }
}
```

**Use `--chunk-source combined`** to leverage grouping:
```bash
--chunk-source combined  # Uses semantic + groups for cache keys
```

### 2. Include Frame Metadata

**Add metadata for tracking**:
```json
{
  "metadata": {
    "video_id": "video_001",
    "frame_number": 100,
    "timestamp": 3.33,
    "scene_id": "scene_05"
  }
}
```

### 3. Consistent Image IDs

**Make `imageId` match your image filenames**:
- If image is `video_001_frame_0100.jpg`, use `imageId: "video_001_frame_0100"`
- If image is `frame_0100.jpg`, use `imageId: "frame_0100"`

### 4. Leverage Temporal Redundancy

**For consecutive frames**, use similar questions to maximize cache hits:
- Frame 100: "What objects are visible?"
- Frame 101: "What objects are visible?" (same question → exact cache hit)
- Frame 102: "What's in the frame?" (similar question → semantic cache hit)

---

## Troubleshooting

### Error: "Video frames dataset not found"

**Solution**: Check the path to your JSON file:
```bash
# Verify file exists
ls -l /path/to/video_frames_dataset.json

# Use absolute path
--video-frames-data /absolute/path/to/video_frames_dataset.json
```

### Error: "Expected video frames JSON to contain a list of samples"

**Solution**: Ensure your JSON is a list, not an object:
```json
// ✅ Correct
[
  {"id": "frame_001", ...},
  {"id": "frame_002", ...}
]

// ❌ Wrong
{
  "samples": [
    {"id": "frame_001", ...}
  ]
}
```

### Error: "vision embedding hook could not resolve image file"

**Solution**: 
1. Set image root environment variable:
   ```bash
   export GQA_IMAGE_ROOT=/path/to/video/frames
   ```

2. Verify image files exist:
   ```bash
   ls /path/to/video/frames/video_001_frame_0100.jpg
   ```

3. Check `imageId` matches filename:
   - JSON: `"imageId": "video_001_frame_0100"`
   - File: `video_001_frame_0100.jpg` ✅

### Missing Required Fields

**Error**: Missing `id`, `imageId`, `question`, or `answer`

**Solution**: Ensure all required fields are present in each sample:
```json
{
  "id": "frame_001",        // ✅ Required
  "imageId": "frame_001",   // ✅ Required
  "question": "What is...", // ✅ Required
  "answer": "car"           // ✅ Required
}
```

---

## Quick Start Checklist

- [ ] Create JSON file with frame data (list of samples)
- [ ] Ensure required fields: `id`, `imageId`, `question`, `answer`
- [ ] Place frame images in a directory
- [ ] Set `GQA_IMAGE_ROOT` environment variable
- [ ] Verify `imageId` matches image filenames
- [ ] Run with `--dataset video_frames --video-frames-data <path>`

---

## Example: Complete Video Frames Dataset

**File**: `video_frames_dataset.json`

```json
[
  {
    "id": "video_001_frame_0100",
    "imageId": "video_001_frame_0100",
    "question": "What objects are visible in this frame?",
    "answer": "car, person, building",
    "fullAnswer": "A red car, a person walking, and a building in the background.",
    "groups": {
      "global": "outdoor",
      "local": "street_scene_01"
    },
    "semantic": [
      {"operation": "query", "argument": "objects"},
      {"operation": "filter", "argument": "visible"}
    ],
    "semanticStr": "query:objects | filter:visible",
    "metadata": {
      "video_id": "video_001",
      "frame_number": 100,
      "timestamp": 3.33,
      "scene_id": "scene_01"
    }
  },
  {
    "id": "video_001_frame_0101",
    "imageId": "video_001_frame_0101",
    "question": "What color is the car?",
    "answer": "red",
    "fullAnswer": "The car is red.",
    "groups": {
      "global": "outdoor",
      "local": "street_scene_01"
    },
    "semantic": [
      {"operation": "query", "argument": "color"},
      {"operation": "filter", "argument": "car"}
    ],
    "semanticStr": "query:color | filter:car",
    "metadata": {
      "video_id": "video_001",
      "frame_number": 101,
      "timestamp": 3.37,
      "scene_id": "scene_01"
    }
  }
]
```

**Image directory**: `/data/video_frames/`
```
/data/video_frames/
├── video_001_frame_0100.jpg
├── video_001_frame_0101.jpg
└── ...
```

**Command**:
```bash
export GQA_IMAGE_ROOT=/data/video_frames

python experiment2/test_vllm.py \
  --dataset video_frames \
  --video-frames-data /data/video_frames_dataset.json \
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

## Summary

1. **Create JSON file** with list of frame samples
2. **Place images** in a directory
3. **Set `GQA_IMAGE_ROOT`** to image directory
4. **Run with** `--dataset video_frames --video-frames-data <json_path>`

The system will:
- Load your video frames dataset
- Extract vision and prompt embeddings
- Use all cache layers (exact, semantic, embedding, fusion)
- Achieve high cache hit rates due to temporal redundancy!

