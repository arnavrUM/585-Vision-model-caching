# Command Explanation: Video Frames Caching Setup

This document explains each part of the recommended command for video frames caching.

## Full Command

```bash
python experiment2/test_vllm.py \
  --dataset custom \
  --chunk-source combined \
  --embedding-hook prompt_vision \
  --embedding-layer vision:512:0.85 \
  --embedding-layer prompt:384:0.8 \
  --similarity-threshold 0.8 \
  --enable-exact-text-cache \
  --enable-semantic-text-cache
```

**Note**: The `--enable-exact-text-cache` and `--enable-semantic-text-cache` flags don't actually exist in the codebase. These caches are **enabled by default**. If you want to disable them, use `--disable-exact-cache` and `--disable-semantic-cache`. The corrected command is shown below.

---

## Corrected Command

```bash
python experiment2/test_vllm.py \
  --dataset custom \
  --chunk-source combined \
  --embedding-hook prompt_vision \
  --embedding-layer vision:512:0.85 \
  --embedding-layer prompt:384:0.8 \
  --similarity-threshold 0.8
```

(The exact and semantic text caches are enabled by default, so no flags needed)

---

## Flag-by-Flag Explanation

### 1. `python experiment2/test_vllm.py`

**What it does**: Runs the main test script that orchestrates dataset loading, cache configuration, and inference.

**Location**: `experiment2/test_vllm.py`

---

### 2. `--dataset custom`

**What it does**: Specifies which dataset to use.

**Options**:
- `gqa` (default) - GQA benchmark from HuggingFace
- `llava150k` - LLaVA-Instruct-150K dataset
- `synthetic` - Built-in test dataset
- `custom` - **Note**: This option doesn't exist in the current codebase!

**Issue**: The `custom` option isn't actually implemented. You would need to:
1. Add a custom dataset loader function
2. Add `"custom"` to the choices in `parse_args()`
3. Add loading logic in `main()`

**For video frames, you should**:
- Either use `gqa` or `llava150k` format
- Or modify the code to add a custom dataset loader
- Or use `synthetic` for testing

**Code location**: `experiment2/test_vllm.py` line 873-876

---

### 3. `--chunk-source combined`

**What it does**: Determines what text is used as the cache key (for exact and semantic text caches).

**Options**:
- `question` - Uses only the question text
- `semantic` - Uses semantic program (e.g., "query:color | filter:car")
- `answer` - Uses the answer text
- `group` - Uses grouping categories (global + local)
- `image` - Uses image ID
- `combined` - **Combines semantic program + global group + local group**

**For `combined` mode**:
```python
# Combines these fields:
semantic_program = "query:color | filter:car"
global_group = "vehicles"
local_group = "cars"
# Result: "query:color | filter:car vehicles cars"
```

**Why use `combined` for video frames**:
- Captures both semantic structure AND scene grouping
- Frames in same scene with similar queries → cache hits
- More informative cache keys than just question text

**Code location**: `experiment2/test_vllm.py` lines 241-264 (`render_chunk_text()`)

---

### 4. `--embedding-hook prompt_vision`

**What it does**: Specifies which embedding hook to use for extracting latent embeddings from the model.

**Options**:
- `none` - No embedding extraction (disables embedding cache)
- `prompt` - Extracts prompt/text embeddings only
- `vision` - Extracts vision/image embeddings only
- `prompt_vision` - **Extracts both prompt AND vision embeddings**

**What `prompt_vision` does**:
1. **Prompt embedding**: Encodes the text prompt/question using sentence-transformers
2. **Vision embedding**: Loads the actual image file and encodes it using CLIP-style encoder

**Why use `prompt_vision` for video frames**:
- **Vision embeddings** capture visual similarity between frames (critical!)
- **Prompt embeddings** capture question similarity
- Both together maximize cache hit opportunities

**Requirements**:
- Must set `GQA_IMAGE_ROOT` or `LLAVA_IMAGE_ROOT` environment variable
- Image files must exist: `<image_id>.jpg` in the image root directory

**Code location**: `experiment2/semantic_cache/embedding_hooks.py`

---

### 5. `--embedding-layer vision:512:0.85`

**What it does**: Registers a vision embedding layer for the embedding cache (L2).

**Format**: `NAME:DIM[:THRESHOLD]`
- `vision` - Layer name (used by the hook)
- `512` - Embedding dimension (must match what the hook produces)
- `0.85` - Similarity threshold (cosine similarity, 0.0-1.0)

**What it means**:
- Creates a FAISS index for vision embeddings (512-dimensional vectors)
- Two frames are considered similar if cosine similarity ≥ 0.85
- When similarity is high enough, reuses cached KV blocks

**Why `0.85` threshold**:
- Balanced: catches similar frames without too many false positives
- For video: consecutive frames typically have 0.90-0.99 similarity
- Lower (0.8): More aggressive, higher hit rate, some false positives
- Higher (0.9): More conservative, fewer false positives, lower hit rate

**Code location**: `experiment2/test_vllm.py` lines 229-238 (`parse_embedding_layer_spec()`)

---

### 6. `--embedding-layer prompt:384:0.8`

**What it does**: Registers a prompt embedding layer for the embedding cache (L2).

**Format**: `NAME:DIM[:THRESHOLD]`
- `prompt` - Layer name (used by the hook)
- `384` - Embedding dimension (sentence-transformers default)
- `0.8` - Similarity threshold

**What it means**:
- Creates a FAISS index for prompt/text embeddings (384-dimensional vectors)
- Two questions are considered similar if cosine similarity ≥ 0.8
- When questions are similar enough, reuses cached KV blocks

**Why `0.8` threshold**:
- Slightly lower than vision (0.85) because text can vary more
- "What color is the car?" vs "What's the color of the vehicle?" → ~0.85 similarity
- Lower threshold catches more question variations

**Multiple embedding layers**:
- You can register multiple layers
- System searches all layers and uses the best match
- Vision and prompt layers work independently

---

### 7. `--similarity-threshold 0.8`

**What it does**: Sets the similarity threshold for the **semantic text cache** (L1 layer).

**Range**: 0.0 to 1.0 (cosine similarity)

**What it means**:
- Controls when two text chunks are similar enough to reuse KV cache
- Uses sentence-transformers to encode text, then FAISS to find similar chunks
- If similarity ≥ 0.8, reuses cached KV blocks

**Different from embedding thresholds**:
- This is for **semantic text cache** (L1)
- Embedding thresholds are for **embedding cache** (L2)
- They work independently

**Why `0.8`**:
- Balanced threshold for semantic similarity
- Catches paraphrases and similar questions
- Lower (0.7): More aggressive, higher hit rate
- Higher (0.9): More conservative, fewer false positives

**Code location**: `experiment2/test_vllm.py` line 931-935

---

### 8. `--enable-exact-text-cache` ❌

**Issue**: This flag **doesn't exist** in the codebase!

**Reality**: Exact text cache is **enabled by default**.

**To disable it**: Use `--disable-exact-cache`

**What exact text cache does** (L0.5):
- Normalizes text (lowercase, whitespace collapse)
- Stores hash → chunk_id mapping
- Instant lookup for identical text (modulo formatting)
- Fastest cache layer, but only catches exact matches

**Code location**: `experiment2/semantic_cache/semantic_cache.py` line 37
```python
enable_exact_text_cache: bool = True  # Enabled by default
```

---

### 9. `--enable-semantic-text-cache` ❌

**Issue**: This flag **doesn't exist** in the codebase!

**Reality**: Semantic text cache is **enabled by default**.

**To disable it**: Use `--disable-semantic-cache`

**What semantic text cache does** (L1):
- Uses sentence-transformers to encode text
- FAISS index for similarity search
- Reuses KV cache when text is semantically similar (above threshold)
- Catches paraphrases and similar questions

**Code location**: `experiment2/semantic_cache/semantic_cache.py` line 36
```python
enable_semantic_text_cache: bool = True  # Enabled by default
```

---

## What the Command Actually Does

When you run this command (corrected version):

1. **Loads dataset**: Tries to load "custom" dataset (will fail - needs implementation)

2. **Configures chunk source**: Uses `combined` mode to create cache keys from semantic + groups

3. **Sets up embedding hooks**: 
   - Extracts both prompt and vision embeddings
   - Requires image files to be accessible

4. **Registers embedding layers**:
   - Vision: 512-dim, threshold 0.85 (for similar frames)
   - Prompt: 384-dim, threshold 0.8 (for similar questions)

5. **Sets semantic threshold**: 0.8 for semantic text cache

6. **Enables all cache layers** (by default):
   - L0.5: Exact text cache ✅
   - L1: Semantic text cache ✅
   - L2: Embedding cache (vision + prompt) ✅

---

## Corrected Command for Video Frames

```bash
# Set image root for vision embeddings
export GQA_IMAGE_ROOT=/path/to/video/frames

# Run with video frames dataset
python experiment2/test_vllm.py \
  --dataset gqa \
  --dataset-config val_balanced_instructions \
  --split val \
  --chunk-source combined \
  --embedding-hook prompt_vision \
  --embedding-layer vision:512:0.85 \
  --embedding-layer prompt:384:0.8 \
  --similarity-threshold 0.8 \
  --max-samples 1000
```

**Or if you have a custom dataset loader**:
```bash
python experiment2/test_vllm.py \
  --dataset custom \
  --custom-data-path /path/to/video_frames.json \
  --chunk-source combined \
  --embedding-hook prompt_vision \
  --embedding-layer vision:512:0.85 \
  --embedding-layer prompt:384:0.8 \
  --similarity-threshold 0.8
```

---

## Summary Table

| Flag | Purpose | Value | Effect |
|------|---------|-------|--------|
| `--dataset` | Dataset source | `custom` (needs implementation) | Loads your data |
| `--chunk-source` | Cache key source | `combined` | Uses semantic + groups for cache keys |
| `--embedding-hook` | Embedding extractor | `prompt_vision` | Extracts both text and image embeddings |
| `--embedding-layer vision` | Vision cache config | `512:0.85` | 512-dim, 0.85 similarity threshold |
| `--embedding-layer prompt` | Prompt cache config | `384:0.8` | 384-dim, 0.8 similarity threshold |
| `--similarity-threshold` | Semantic text cache | `0.8` | 0.8 similarity for text cache |
| `--enable-*-cache` | ❌ Doesn't exist | N/A | Caches are enabled by default |

---

## Key Takeaways

1. **`--enable-*-cache` flags don't exist** - caches are enabled by default
2. **`--dataset custom` doesn't exist** - you need to implement it or use existing datasets
3. **`--chunk-source combined`** - best for video frames (uses semantic + scene grouping)
4. **`--embedding-hook prompt_vision`** - critical for video (captures visual similarity)
5. **Vision embedding threshold 0.85** - good balance for similar frames
6. **Prompt embedding threshold 0.8** - catches question variations
7. **Semantic threshold 0.8** - balanced for semantic similarity

The command is designed to maximize cache hits for video frames by leveraging:
- Visual similarity (vision embeddings)
- Question similarity (prompt embeddings)
- Scene grouping (combined chunk source)
- Semantic similarity (semantic text cache)
- Exact matches (exact text cache)

