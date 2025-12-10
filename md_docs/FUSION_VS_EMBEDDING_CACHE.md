# Fusion Cache vs Embedding Cache: Key Differences

This document explains what fusion cache stores versus embedding cache, and how they differ.

## Quick Summary

| Aspect | Fusion Cache (L1.5) | Embedding Cache (L2) |
|--------|---------------------|---------------------|
| **What it stores** | Fusion module output tensors | Lightweight embedding vectors |
| **Purpose** | Direct reuse of fusion computation | Similarity search to find cached KV blocks |
| **Size** | Large (full fusion tensors) | Small (compressed embeddings) |
| **When used** | After KV cache hit | Before KV cache lookup |
| **What it skips** | Fusion module computation | Used to find what to skip |

---

## Fusion Cache (L1.5)

### What It Stores

**FusionState** - Contains the actual output tensors from the fusion module:

```python
@dataclass
class FusionState:
    chunk_id: str
    tensors: dict[str, torch.Tensor]  # ← Actual fusion module outputs
    metadata: dict[str, Any]
```

**Example tensors stored**:
```python
{
    "fusion_output": <torch.Tensor shape=[8, 1024]>,  # Fused image+text features
    "image_features": <torch.Tensor shape=[1, 1024]>,  # Processed image embeddings
    "projected_text": <torch.Tensor shape=[7, 1024]>   # Processed text embeddings
}
```

### Where It Comes From

**Fusion module output** - The intermediate computation results from Step 3 of the pipeline:

```
Image Embeddings (1×768)
    ↓
Image Projector
    ↓
Projected Image (1×1024) ──┐
                           ├─→ Fusion Module ──→ Fused Features (8×1024)
Text Embeddings (7×768) ──→ Text Projector ──→ Projected Text (7×1024) ──┘
                                                      ↑
                                              Fusion Cache stores this!
```

### What It Does

1. **Captures**: Fusion module output tensors after processing image + text
2. **Stores**: Full tensor values (can be large, e.g., 8×1024 = 8K values per tensor)
3. **Reuses**: Directly injects these tensors back into the model
4. **Skips**: The entire fusion computation (projection + fusion layers)

### When It's Used

**After a KV cache hit** - Fusion cache is applied when KV blocks are injected:

```python
# 1. KV cache hit occurs (from any layer)
match = cache.try_reuse(request_id, chunk_text, embeddings)
# Returns: CacheHit(chunk_id="chunk_123")

# 2. Inject KV blocks
adapter.inject(request_id, stored_chunk)

# 3. Check for fusion state
fusion_state = fusion_cache.load("chunk_123")
if fusion_state:
    # 4. Inject fusion tensors directly into model
    fusion_cache.inject(llm=llm, request_id=request_id, state=fusion_state)
    # Model now has pre-computed fusion features, skips fusion computation
```

**Code location**: `experiment2/semantic_cache/semantic_cache.py` lines 121-130

### Storage

- **Format**: Pickled `FusionState` objects
- **Location**: `fusion_chunks/{chunk_id}.fusion`
- **Size**: Large (full tensors, e.g., 1-5 MB per state)

---

## Embedding Cache (L2)

### What It Stores

**Lightweight embedding vectors** - Compressed representations used for similarity search:

```python
# Vision embedding: 512-dimensional vector
vision_embedding = np.array([0.12, -0.45, 0.78, ..., 0.23])  # shape: (512,)

# Prompt embedding: 384-dimensional vector  
prompt_embedding = np.array([0.31, 0.39, ..., 0.81])  # shape: (384,)
```

### Where It Comes From

**External embedding extractors** - Not from the model itself, but from separate encoders:

```
Image File (frame_001.jpg)
    ↓
CLIP Vision Encoder (sentence-transformers)
    ↓
Vision Embedding (512-dim vector) ──→ Stored in FAISS index
                                      ↑
                              Embedding Cache stores this!

Text Prompt ("What color is the car?")
    ↓
Sentence Transformer (all-MiniLM-L6-v2)
    ↓
Prompt Embedding (384-dim vector) ──→ Stored in FAISS index
                                       ↑
                               Embedding Cache stores this!
```

**Code location**: `experiment2/semantic_cache/embedding_hooks.py`

### What It Does

1. **Extracts**: Lightweight embeddings using external encoders (CLIP, sentence-transformers)
2. **Stores**: Small vectors in FAISS indices (for fast similarity search)
3. **Searches**: Finds similar inputs by comparing embedding vectors
4. **Finds**: Cached KV blocks that match similar embeddings

### When It's Used

**Before KV cache lookup** - Embedding cache is checked first to find similar inputs:

```python
# 1. Extract embeddings from current input
embeddings = embedding_hook(llm=llm, sample=sample)
# Returns: {"vision": <512-dim vector>, "prompt": <384-dim vector>}

# 2. Search FAISS indices for similar embeddings
match = embedding_cache.search(embeddings)
# FAISS finds: chunk_id="chunk_456" with similarity 0.92

# 3. If match found, load and inject KV blocks
if match:
    stored = kv_store.load(match.chunk_id)
    adapter.inject(request_id, stored)
    # Now we can reuse the cached KV blocks
```

**Code location**: `experiment2/semantic_cache/semantic_cache.py` lines 167-186

### Storage

- **Format**: FAISS indices (in-memory) + ID mappings
- **Location**: In-memory (FAISS `IndexFlatIP`)
- **Size**: Small (compressed vectors, e.g., 512 floats = 2 KB per embedding)

---

## Key Differences

### 1. **What They Store**

**Fusion Cache**:
- **Actual model intermediate states** (tensors)
- Full fusion module outputs
- Large: 1-5 MB per state

**Embedding Cache**:
- **Lightweight similarity vectors** (embeddings)
- Compressed representations for search
- Small: ~2 KB per embedding

### 2. **Purpose**

**Fusion Cache**:
- **Direct reuse** of fusion computation
- Injects pre-computed fusion tensors
- Skips fusion module entirely

**Embedding Cache**:
- **Similarity search** to find cached KV blocks
- Uses embeddings to find similar inputs
- Indirect: finds what to reuse, doesn't directly reuse

### 3. **When They're Used**

**Fusion Cache**:
- **After** KV cache hit
- Applied when KV blocks are injected
- Works alongside KV cache

**Embedding Cache**:
- **Before** KV cache lookup
- Used to find which KV blocks to reuse
- Triggers KV cache injection

### 4. **What They Skip**

**Fusion Cache**:
- ✅ Fusion module computation (projection + fusion layers)
- ✅ Image-text integration step
- **Saves**: ~20-100ms per request

**Embedding Cache**:
- ✅ Transformer forward passes (via KV cache)
- ✅ Attention computation
- **Saves**: ~500-2000ms per request (much more!)

### 5. **Dependency**

**Fusion Cache**:
- **Depends on** KV cache hit
- Only used if KV blocks are injected
- Cannot work independently

**Embedding Cache**:
- **Independent** - finds KV cache entries
- Can trigger KV cache injection
- Works as a search mechanism

---

## Visual Comparison

### Fusion Cache Flow

```
Request arrives
    ↓
KV cache hit (from any layer)
    ↓
Inject KV blocks
    ↓
Load fusion state ──→ FusionState(tensors={...})
    ↓
Inject fusion tensors into model
    ↓
Skip fusion computation ✅
    ↓
Continue generation
```

### Embedding Cache Flow

```
Request arrives
    ↓
Extract embeddings (vision + prompt)
    ↓
Search FAISS indices
    ↓
Find similar embedding (similarity ≥ threshold)
    ↓
Get chunk_id from match
    ↓
Load KV blocks for that chunk_id
    ↓
Inject KV blocks
    ↓
Skip transformer forward passes ✅
    ↓
Continue generation
```

---

## Example: Both Working Together

**Scenario**: Frame 100 and Frame 101 are very similar

### Frame 100 (First Request)

```python
# 1. Extract embeddings
embeddings = {
    "vision": np.array([0.1, 0.2, ..., 0.9]),  # 512-dim
    "prompt": np.array([0.3, 0.4, ..., 0.8])   # 384-dim
}

# 2. No cache hit (first time)
# Full pipeline runs:
#   - Image encoding
#   - Text encoding
#   - Fusion module → fusion_output tensor
#   - Transformer layers → KV blocks

# 3. Cache everything
embedding_cache.add("chunk_100", embeddings)  # Store embeddings
kv_store.save(kv_chunk)                      # Store KV blocks
fusion_cache.capture(fusion_state)           # Store fusion tensors
```

### Frame 101 (Second Request - Similar)

```python
# 1. Extract embeddings
embeddings = {
    "vision": np.array([0.11, 0.19, ..., 0.91]),  # Very similar (0.95 similarity)
    "prompt": np.array([0.31, 0.39, ..., 0.81])   # Very similar (0.92 similarity)
}

# 2. Embedding cache search
match = embedding_cache.search(embeddings)
# Finds: chunk_id="chunk_100", similarity=0.95

# 3. Load and inject KV blocks
stored = kv_store.load("chunk_100")
adapter.inject(request_id, stored)  # Inject KV blocks

# 4. Load and inject fusion state
fusion_state = fusion_cache.load("chunk_100")
fusion_cache.inject(llm=llm, request_id=request_id, state=fusion_state)

# 5. Skip computation
# ✅ Skipped: Fusion module (via fusion cache)
# ✅ Skipped: Transformer forward passes (via KV cache)
# Only generation happens
```

---

## Summary Table

| Feature | Fusion Cache | Embedding Cache |
|---------|-------------|----------------|
| **Stores** | Fusion module output tensors | Lightweight embedding vectors |
| **Size** | Large (1-5 MB) | Small (~2 KB) |
| **Format** | `FusionState` (pickled) | FAISS indices (in-memory) |
| **Source** | Model's fusion module | External encoders (CLIP, sentence-transformers) |
| **Purpose** | Direct reuse of fusion | Similarity search for KV cache |
| **When used** | After KV cache hit | Before KV cache lookup |
| **What it skips** | Fusion computation | Transformer forward passes (via KV cache) |
| **Dependency** | Requires KV cache hit | Independent, finds KV cache entries |
| **Savings** | ~20-100ms | ~500-2000ms (via KV cache) |

---

## Key Takeaways

1. **Fusion Cache** = Stores actual model intermediate states (fusion tensors)
   - Direct reuse of fusion computation
   - Works after KV cache hit
   - Saves fusion module computation

2. **Embedding Cache** = Stores lightweight similarity vectors
   - Used for similarity search
   - Finds which KV blocks to reuse
   - Saves transformer forward passes (via KV cache)

3. **They work together**:
   - Embedding cache finds similar inputs
   - KV cache stores transformer states
   - Fusion cache stores fusion module outputs
   - All three can be used together for maximum savings

4. **Different roles**:
   - Embedding cache: **Search mechanism** (finds what to reuse)
   - Fusion cache: **Direct reuse** (injects pre-computed tensors)
   - KV cache: **Transformer state reuse** (injects attention states)

The embedding cache is a **search tool** to find cached KV blocks, while fusion cache is a **direct reuse** of fusion computation. They serve different purposes and can work together!

