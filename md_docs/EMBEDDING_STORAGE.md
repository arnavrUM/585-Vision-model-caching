# Embedding Storage: Vision vs Prompt - Separate Indices

Yes, you're correct! Vision and prompt embeddings are **stored separately** in individual FAISS indices.

## How They're Stored

### Separate FAISS Indices

Each embedding layer has its **own FAISS index**:

```python
# In embedding_cache.py, line 72-73
def __init__(self, layers: Iterable[EmbeddingLayerConfig]):
    self.layers = {config.name: _EmbeddingLayerIndex(config) for config in layers}
    # Creates separate index for each layer:
    #   - self.layers["vision"] → _EmbeddingLayerIndex with FAISS index (512-dim)
    #   - self.layers["prompt"] → _EmbeddingLayerIndex with FAISS index (384-dim)
```

### Each Layer Has Its Own Index

**`_EmbeddingLayerIndex` class** (lines 32-66):
```python
class _EmbeddingLayerIndex:
    def __init__(self, config: EmbeddingLayerConfig):
        self.config = config
        self.index = faiss.IndexFlatIP(config.dim)  # ← Separate FAISS index
        self.ids: list[str] = []  # ← Separate ID mapping
```

**Example**:
- `vision` layer: `IndexFlatIP(512)` + `ids = ["chunk_1", "chunk_2", ...]`
- `prompt` layer: `IndexFlatIP(384)` + `ids = ["chunk_1", "chunk_2", ...]`

---

## Storage Structure

### When Embeddings Are Added

```python
# In embedding_cache.py, line 75-82
def add(self, chunk_id: str, embeddings: dict[str, np.ndarray]):
    for name, vector in embeddings.items():
        layer = self.layers.get(name)  # Get layer by name
        if layer is None or vector is None:
            continue
        layer.add(chunk_id, vector)  # Add to that layer's index
```

**What happens**:
```python
# Input embeddings:
embeddings = {
    "vision": np.array([0.1, 0.2, ..., 0.9]),  # 512-dim
    "prompt": np.array([0.3, 0.4, ..., 0.8])   # 384-dim
}

# Vision embedding → Added to vision layer's FAISS index
self.layers["vision"].index.add(vision_vector)  # IndexFlatIP(512)
self.layers["vision"].ids.append("chunk_123")

# Prompt embedding → Added to prompt layer's FAISS index
self.layers["prompt"].index.add(prompt_vector)  # IndexFlatIP(384)
self.layers["prompt"].ids.append("chunk_123")
```

**Result**: 
- Vision embeddings stored in **vision layer's FAISS index**
- Prompt embeddings stored in **prompt layer's FAISS index**
- Both point to the same `chunk_id` (same KV blocks)

---

## Search Process

### Independent Search Per Layer

```python
# In embedding_cache.py, line 84-97
def search(self, embeddings: dict[str, np.ndarray]) -> EmbeddingMatch | None:
    best: EmbeddingMatch | None = None
    for name, vector in embeddings.items():
        layer = self.layers.get(name)  # Get layer
        if layer is None or vector is None:
            continue
        match = layer.search(vector)  # ← Search that layer's index independently
        if match is None:
            continue
        if best is None or match.score > best.score:
            best = match  # Keep best match across all layers
    return best
```

**What happens**:
```python
# Input embeddings:
embeddings = {
    "vision": np.array([0.11, 0.19, ..., 0.91]),  # 512-dim
    "prompt": np.array([0.31, 0.39, ..., 0.81])   # 384-dim
}

# 1. Search vision layer's index
vision_match = self.layers["vision"].search(vision_vector)
# Searches IndexFlatIP(512) → finds chunk_456 with score 0.94

# 2. Search prompt layer's index
prompt_match = self.layers["prompt"].search(prompt_vector)
# Searches IndexFlatIP(384) → finds chunk_789 with score 0.89

# 3. Return best match
# Returns: vision_match (score 0.94 > 0.89)
```

---

## Data Structure

### Separate Storage Per Layer

```
EmbeddingCache
├── layers["vision"]
│   ├── index: FAISS IndexFlatIP(512)  ← Vision embeddings only
│   └── ids: ["chunk_1", "chunk_2", ...]  ← Maps to chunk_ids
│
└── layers["prompt"]
    ├── index: FAISS IndexFlatIP(384)  ← Prompt embeddings only
    └── ids: ["chunk_1", "chunk_2", ...]  ← Maps to chunk_ids
```

### Same chunk_id, Different Indices

**Important**: Both layers can point to the **same chunk_id**, but they're stored in **separate indices**:

```python
# When adding embeddings for chunk_123:
embeddings = {"vision": vision_vec, "prompt": prompt_vec}

# Vision layer:
self.layers["vision"].index.add(vision_vec)  # Added to vision index
self.layers["vision"].ids.append("chunk_123")  # Points to chunk_123

# Prompt layer:
self.layers["prompt"].index.add(prompt_vec)  # Added to prompt index
self.layers["prompt"].ids.append("chunk_123")  # Points to same chunk_123
```

**Result**: 
- Vision index: `[vision_vec_1, vision_vec_2, ...]` → `["chunk_1", "chunk_2", ...]`
- Prompt index: `[prompt_vec_1, prompt_vec_2, ...]` → `["chunk_1", "chunk_2", ...]`
- Both can point to the same chunk_id, but stored separately

---

## Why Separate Storage?

### 1. Different Dimensions

- **Vision**: 512-dimensional vectors
- **Prompt**: 384-dimensional vectors

**Cannot be in same index** - FAISS indices require fixed dimensions.

### 2. Independent Thresholds

- **Vision**: Threshold 0.85 (for similar images)
- **Prompt**: Threshold 0.8 (for similar questions)

**Each layer can have different thresholds**.

### 3. Independent Search

- Search vision layer for similar images
- Search prompt layer for similar questions
- Pick the best match across both

**Allows flexible matching** - can match on vision alone, prompt alone, or both.

---

## Example: How They Work Together

### Scenario: Frame 100 and Frame 101

**Frame 100** (first request):
```python
# Extract embeddings
embeddings = {
    "vision": np.array([0.1, 0.2, ..., 0.9]),  # 512-dim
    "prompt": np.array([0.3, 0.4, ..., 0.8])   # 384-dim
}

# Add to separate indices
embedding_cache.add("chunk_100", embeddings)

# Vision layer:
self.layers["vision"].index.add([0.1, 0.2, ..., 0.9])  # Added to vision index
self.layers["vision"].ids.append("chunk_100")

# Prompt layer:
self.layers["prompt"].index.add([0.3, 0.4, ..., 0.8])  # Added to prompt index
self.layers["prompt"].ids.append("chunk_100")
```

**Frame 101** (second request - similar):
```python
# Extract embeddings
embeddings = {
    "vision": np.array([0.11, 0.19, ..., 0.91]),  # Very similar (0.95)
    "prompt": np.array([0.35, 0.42, ..., 0.79])   # Less similar (0.82)
}

# Search both indices independently
match = embedding_cache.search(embeddings)

# Vision layer search:
vision_match = self.layers["vision"].search([0.11, 0.19, ..., 0.91])
# Finds: chunk_100, score=0.95 (above threshold 0.85) ✅

# Prompt layer search:
prompt_match = self.layers["prompt"].search([0.35, 0.42, ..., 0.79])
# Finds: chunk_100, score=0.82 (above threshold 0.8) ✅

# Return best match
# Returns: vision_match (score 0.95 > 0.82)
# Result: EmbeddingMatch(layer="vision", chunk_id="chunk_100", score=0.95)
```

---

## Key Points

### ✅ Separate Storage

- **Vision embeddings**: Stored in `layers["vision"].index` (FAISS IndexFlatIP(512))
- **Prompt embeddings**: Stored in `layers["prompt"].index` (FAISS IndexFlatIP(384))
- **Separate ID mappings**: Each layer has its own `ids` list

### ✅ Independent Search

- Vision layer searched independently
- Prompt layer searched independently
- Best match returned across all layers

### ✅ Same chunk_id

- Both layers can point to the same `chunk_id`
- Same KV blocks, but found via different embedding types
- Allows matching on vision alone, prompt alone, or both

### ✅ Flexible Matching

- Can match on vision similarity (similar images)
- Can match on prompt similarity (similar questions)
- Can match on both (best score wins)

---

## Code Evidence

### Separate Indices Created

```python
# embedding_cache.py, line 72-73
self.layers = {config.name: _EmbeddingLayerIndex(config) for config in layers}
# Creates:
#   self.layers["vision"] = _EmbeddingLayerIndex(EmbeddingLayerConfig("vision", 512, 0.85))
#   self.layers["prompt"] = _EmbeddingLayerIndex(EmbeddingLayerConfig("prompt", 384, 0.8))
```

### Separate FAISS Indices

```python
# embedding_cache.py, line 32-36
class _EmbeddingLayerIndex:
    def __init__(self, config: EmbeddingLayerConfig):
        self.config = config
        self.index = faiss.IndexFlatIP(config.dim)  # ← Separate index per layer
        self.ids: list[str] = []  # ← Separate ID list per layer
```

### Separate Storage

```python
# embedding_cache.py, line 75-82
def add(self, chunk_id: str, embeddings: dict[str, np.ndarray]):
    for name, vector in embeddings.items():
        layer = self.layers.get(name)  # Get specific layer
        layer.add(chunk_id, vector)  # Add to that layer's index
```

### Independent Search

```python
# embedding_cache.py, line 84-97
def search(self, embeddings: dict[str, np.ndarray]):
    for name, vector in embeddings.items():
        layer = self.layers.get(name)  # Get specific layer
        match = layer.search(vector)  # Search that layer's index
        # Compare matches, return best
```

---

## Summary

**Yes, vision and prompt embeddings are stored separately!**

- **Vision embeddings**: Separate FAISS index (512-dim)
- **Prompt embeddings**: Separate FAISS index (384-dim)
- **Independent search**: Each layer searched separately
- **Best match**: System returns best match across all layers
- **Same chunk_id**: Both can point to same KV blocks, but found via different paths

This design allows:
- Matching on vision similarity alone (similar images, different questions)
- Matching on prompt similarity alone (similar questions, different images)
- Matching on both (best score wins)

The separation is necessary because:
1. Different dimensions (512 vs 384)
2. Different thresholds (can be configured independently)
3. Different use cases (vision for images, prompt for text)

