# Cache Layer Usage Examples at Inference Time

This document provides concrete examples of how each cache layer is used during inference, showing the data flow and how cached data skips computation.

## Inference Flow Overview

When a new prompt arrives, the system checks cache layers in this order:
1. **L0.5: Exact Text Cache** (fastest, exact matches)
2. **L2: Embedding Cache** (highest precision, semantic similarity)
3. **L1: Semantic Text Cache** (text similarity fallback)
4. **L1.5: Fusion Cache** (applied when KV cache is injected)

If any layer hits, the cached KV blocks are injected into vLLM's GPU memory, skipping forward passes through the model.

---

## Example 1: L0.5 - Exact Normalized Text Cache

### Scenario
**First Request:**
- Input prompt: `"What color is the car?"`
- Chunk text: `"What color is the car?"`

**Second Request (cache hit):**
- Input prompt: `"What color is the car?"` (identical)
- Chunk text: `"what color is the car?"` (different casing/whitespace)

### Cache Lookup Process

```python
# 1. Normalize the incoming text
normalized = exact_cache.normalize("what color is the car?")
# Result: "what color is the car?"

# 2. Look up in text_index.json
chunk_ids = exact_cache.candidates(normalized)
# Result: ["chunk_abc123"]  # from previous request

# 3. Load the cached KV chunk from disk
stored = kv_store.load("chunk_abc123")
# Result: KVChunk(
#   chunk_id="chunk_abc123",
#   block_ids=[42, 43, 44],  # vLLM block IDs
#   tensors={
#     "layer_0": <torch.Tensor shape=[1, 3, 16, 128]>,  # Key cache
#     "layer_1": <torch.Tensor shape=[1, 3, 16, 128]>,  # Value cache
#     ...
#   },
#   num_tokens=48
# )

# 4. Inject KV blocks into vLLM's GPU memory
adapter.inject(request_id="req_xyz", chunk=stored)
# This copies the cached tensors into vLLM's KV cache blocks [42, 43, 44]
# vLLM now has the pre-computed attention states ready

# 5. Skip forward pass - only decode from the cached position
# vLLM generates response starting from token 48 instead of token 0
```

### What Gets Skipped
- ✅ No tokenization needed (vLLM handles this, but prefix is cached)
- ✅ No forward pass through transformer layers for the cached prefix
- ✅ No attention computation for the cached tokens
- ✅ Only decoding happens from the cached position

### Data Structures
- **Cache Index**: `text_index.json` → `{"what color is the car?": ["chunk_abc123"]}`
- **KV Chunk File**: `kv_chunks/chunk_abc123.pkl` → Serialized KVChunk with tensors

---

## Example 2: L1 - Semantic Text Cache

### Scenario
**First Request:**
- Chunk text: `"What color is the vehicle?"`
- Response generated and cached

**Second Request (semantic hit):**
- Chunk text: `"What color is the car?"` (semantically similar, different words)

### Cache Lookup Process

```python
# 1. Encode the incoming text using sentence-transformers
query_embedding = semantic_cache._encode("What color is the car?")
# Result: np.array([0.12, -0.45, 0.78, ..., 0.23])  # 384-dim vector

# 2. Search FAISS index for similar embeddings
match = semantic_cache.search("What color is the car?")
# FAISS computes cosine similarity:
#   - "What color is the vehicle?" → score: 0.92 (above threshold 0.85)
#   - Returns: SemanticTextMatch(chunk_id="chunk_def456", score=0.92)

# 3. Load the cached KV chunk
stored = kv_store.load("chunk_def456")
# Result: KVChunk with pre-computed KV blocks from the similar question

# 4. Inject into vLLM
adapter.inject(request_id="req_new", chunk=stored)
# vLLM now has attention states from "What color is the vehicle?"
# These states are semantically similar enough to reuse

# 5. Continue generation from cached position
# Model generates response, potentially with slight variations due to semantic similarity
```

### What Gets Skipped
- ✅ Forward pass for the semantically similar prefix
- ✅ Attention computation for cached tokens
- ⚠️ May need minor adjustments, but most computation is saved

### Data Structures
- **FAISS Index**: In-memory `IndexFlatIP` with normalized embeddings
- **ID Mapping**: `self.ids = ["chunk_def456", ...]` maps FAISS index → chunk_id
- **KV Chunk**: Same as exact cache - serialized tensors on disk

---

## Example 3: L2 - Latent Embedding Cache

### Scenario
**First Request:**
- Image: `gqa_image_001.jpg` (red car)
- Question: `"What color is this?"`
- Vision encoder produces: `vision_embedding = [0.1, 0.2, ..., 0.9]` (512-dim)
- Prompt encoder produces: `prompt_embedding = [0.3, 0.4, ..., 0.8]` (384-dim)
- Response generated and cached

**Second Request (embedding hit):**
- Image: `gqa_image_001.jpg` (same image)
- Question: `"Tell me the color"` (different wording, same intent)
- Vision encoder produces: `vision_embedding = [0.11, 0.19, ..., 0.91]` (very similar)
- Prompt encoder produces: `prompt_embedding = [0.31, 0.39, ..., 0.81]` (very similar)

### Cache Lookup Process

```python
# 1. Extract embeddings from model via hooks
embeddings = embedding_hook(llm=llm, sample=sample)
# Result: {
#   "vision": np.array([0.11, 0.19, ..., 0.91]),  # 512-dim
#   "prompt": np.array([0.31, 0.39, ..., 0.81])   # 384-dim
# }

# 2. Search each embedding layer's FAISS index
embed_match = embedding_cache.search(embeddings)
# For "vision" layer:
#   - Normalize L2: vector / ||vector||
#   - FAISS search: finds chunk_ghi789 with score 0.94 (above threshold 0.85)
# For "prompt" layer:
#   - Finds chunk_ghi789 with score 0.89
# Returns best match: EmbeddingMatch(
#   layer="vision",
#   chunk_id="chunk_ghi789",
#   score=0.94
# )

# 3. Load cached KV chunk
stored = kv_store.load("chunk_ghi789")
# Contains KV blocks from the previous similar embedding state

# 4. Inject KV blocks
adapter.inject(request_id="req_new", chunk=stored)
# vLLM now has attention states from the similar embedding configuration

# 5. Continue generation
# Model generates response based on cached attention states
```

### What Gets Skipped
- ✅ Forward pass through transformer layers for the cached prefix
- ✅ Attention computation for cached tokens
- ✅ Vision encoder processing (if vision embedding matches)
- ⚠️ Embedding extraction still happens (needed for lookup), but much cheaper than full forward pass

### Data Structures
- **Per-Layer FAISS Indices**: 
  - `embedding_cache.layers["vision"].index` → FAISS IndexFlatIP(512-dim)
  - `embedding_cache.layers["prompt"].index` → FAISS IndexFlatIP(384-dim)
- **ID Mappings**: `self.ids = ["chunk_ghi789", ...]` per layer
- **KV Chunk**: Same serialized format on disk

---

## Example 4: L1.5 - Fusion Cache

### Scenario
**First Request (multimodal):**
- Image: `gqa_image_002.jpg`
- Text: `"Describe this scene"`
- Model processes through fusion layer (image-text projector)
- Fusion tensors captured: `fusion_state.tensors = {"fusion_output": <Tensor>}`

**Second Request (fusion hit):**
- Same image and similar text
- KV cache hit occurs, fusion state also available

### Cache Lookup Process

```python
# 1. KV cache hit occurs (via any of L0.5, L1, or L2)
match = semantic_cache.try_reuse(request_id, chunk_text, embeddings)
# Returns: ReuseReport(hit=CacheHit(chunk_id="chunk_jkl012", ...))

# 2. KV blocks are injected
adapter.inject(request_id, stored_chunk)

# 3. Check for fusion state
fusion_state = fusion_cache.load("chunk_jkl012")
# Result: FusionState(
#   chunk_id="chunk_jkl012",
#   tensors={
#     "fusion_output": <torch.Tensor>,  # Pre-computed fusion features
#     "image_features": <torch.Tensor>  # Processed image embeddings
#   }
# )

# 4. Inject fusion tensors into model
fusion_cache.inject(
    llm=llm,
    request_id=request_id,
    state=fusion_state
)
# This writes the fusion tensors back into the model's fusion module
# Model can now skip the image-text projection step

# 5. Continue generation with both KV cache and fusion state
# Model generates response using cached attention + cached fusion features
```

### What Gets Skipped
- ✅ Image-text fusion computation (projection layers)
- ✅ Vision encoder processing (if fusion state includes it)
- ✅ KV cache computation (from previous layers)
- ✅ Full forward pass through transformer

### Data Structures
- **Fusion State File**: `fusion_chunks/chunk_jkl012.fusion` → Pickled FusionState
- **FusionState**: Contains dict of tensors from fusion modules
- **KV Chunk**: Separate from fusion, stored in `kv_chunks/`

---

## Example 5: Complete Inference Flow with Cache Miss

### Scenario
**New Request (no cache hit):**
- Chunk text: `"What is the weather today?"`
- No matches in any cache layer

### Inference Process

```python
# 1. Try all cache layers
reuse = cache.try_reuse(request_id, "What is the weather today?", embeddings={})
# Checks:
#   - Exact cache: normalized lookup → miss
#   - Embedding cache: FAISS search → miss
#   - Semantic cache: FAISS search → miss (score 0.72 < threshold 0.85)
# Returns: ReuseReport(hit=None, statuses={"kv_cache": "miss", ...})

# 2. Add request to vLLM engine
engine.add_request(request_id, prompt, sampling_params)

# 3. Register callback to capture KV blocks after generation
cache.add_observation(
    request_id,
    "What is the weather today?",
    embeddings=embeddings,  # Extract and store embeddings
    metadata={"dataset_id": "sample_001"}
)

# 4. Run full inference (no cache)
response = drain_request(engine, request_id)
# vLLM performs:
#   - Tokenization
#   - Full forward pass through all layers
#   - Attention computation
#   - Generation

# 5. After generation completes, callback fires:
def _capture_and_store():
    # Capture KV blocks from GPU
    chunk = adapter.capture(
        request_id=request_id,
        chunk_id="chunk_new_789",
        num_blocks=3
    )
    # Result: KVChunk with block_ids=[100, 101, 102] and tensors
    
    # Store in all cache layers
    exact_cache.record("What is the weather today?", "chunk_new_789")
    semantic_cache.add("chunk_new_789", "What is the weather today?")
    embedding_cache.add("chunk_new_789", embeddings)
    kv_store.save(chunk)  # Persist to disk
    
    # Capture fusion state if enabled
    fusion_cache.capture(llm=llm, request_id=request_id, chunk_id="chunk_new_789")

# 6. Future requests can now reuse this cached data
```

### What Happens
- ❌ No computation skipped (cache miss)
- ✅ Full model forward pass
- ✅ KV blocks captured for future reuse
- ✅ All cache layers updated

---

## Key Implementation Details

### KV Block Injection (`kv_adapter.py`)

```python
def inject(self, request_id: str, chunk: KVChunk) -> bool:
    # 1. Get vLLM's KV cache manager
    kv_manager = scheduler.kv_cache_manager
    blocks = kv_manager.get_blocks(request_id)
    dst_block_ids = blocks.get_block_ids()[0]  # e.g., [50, 51, 52]
    
    # 2. For each transformer layer
    for layer_name, tensor in self._layer_kv_map().items():
        cached = chunk.tensors[layer_name]  # CPU tensor from disk
        
        # 3. Copy cached blocks into GPU memory
        platform.insert_blocks_to_device(
            cached,           # Source: CPU cache
            tensor,           # Destination: GPU KV cache
            src_index=[0,1,2], # Indices in cached tensor
            dst_index=[50,51,52] # vLLM block IDs
        )
    
    # vLLM now has pre-computed attention states ready
    return True
```

### Cache Hit Statistics

When a cache hit occurs, the system tracks:
- `statuses["exact_text"] = "hit"` or `"miss"` or `"skip"`
- `statuses["semantic_text"] = "hit"` or `"miss"` or `"skip"`
- `statuses["embedding:vision"] = "hit"` or `"miss"` or `"skip"`
- `statuses["kv_cache"] = "hit"` (if any layer succeeded)

Example log output:
```
[sample_001] hit (text:exact) | latency=0.032s | answer match=True | 
  exact_text=hit, semantic_text=skip, embedding:vision=skip, kv_cache=hit
```

---

## Performance Impact

### Latency Comparison (typical values)

| Cache Layer | Lookup Time | Injection Time | Total Saved |
|------------|-------------|---------------|-------------|
| L0.5 (Exact) | ~0.001s | ~0.010s | ~0.5-2.0s (full forward pass) |
| L1 (Semantic) | ~0.005s | ~0.010s | ~0.5-2.0s |
| L2 (Embedding) | ~0.010s | ~0.010s | ~0.5-2.0s |
| L1.5 (Fusion) | ~0.001s | ~0.005s | Additional ~0.1-0.5s (fusion computation) |

### Memory Usage

- **KV Chunks**: ~1-10 MB per chunk (depends on model size and sequence length)
- **FAISS Indices**: ~100 KB - 1 MB per index (depends on number of cached items)
- **Fusion States**: ~1-5 MB per state (depends on fusion module size)

---

## Summary

Each cache layer provides a different trade-off between:
- **Speed**: How fast the lookup is
- **Precision**: How semantically similar matches need to be
- **Coverage**: What types of inputs can be cached

The hierarchical approach ensures the fastest possible hit while maintaining high precision through deeper semantic matching.

