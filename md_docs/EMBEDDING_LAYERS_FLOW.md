# Embedding Layers Flow: What is `layers`?

This document traces the code to show exactly what `layers` is in `EmbeddingCache.__init__(layers)`.

## Complete Flow

### Step 1: Command Line Arguments

**User provides**:
```bash
--embedding-layer vision:512:0.85 --embedding-layer prompt:384:0.8
```

**Code**: `test_vllm.py` line 911-917
```python
parser.add_argument(
    "--embedding-layer",
    action="append",  # ← Creates a list
    default=[],
    metavar="NAME:DIM[:THRESH]",
    help="Register latent embedding layers...",
)
```

**Result**: `args.embedding_layer = ["vision:512:0.85", "prompt:384:0.8"]` (list of strings)

---

### Step 2: Parse String Specs into Config Objects

**Code**: `test_vllm.py` line 1093
```python
layer_configs = [parse_embedding_layer_spec(spec) for spec in args.embedding_layer]
```

**Parse function**: `test_vllm.py` lines 229-238
```python
def parse_embedding_layer_spec(spec: str) -> EmbeddingLayerConfig:
    parts = spec.split(":")
    name = parts[0].strip()      # "vision" or "prompt"
    dim = int(parts[1])           # 512 or 384
    threshold = float(parts[2]) if len(parts) > 2 else 0.85
    return EmbeddingLayerConfig(name=name, dim=dim, similarity_threshold=threshold)
```

**Result**: `layer_configs = [
    EmbeddingLayerConfig(name="vision", dim=512, similarity_threshold=0.85),
    EmbeddingLayerConfig(name="prompt", dim=384, similarity_threshold=0.8)
]`

**Type**: `list[EmbeddingLayerConfig]`

---

### Step 3: Pass to SemanticCacheConfig

**Code**: `test_vllm.py` line 1103
```python
cache_config = SemanticCacheConfig(
    ...
    embedding_layers=layer_configs,  # ← List of EmbeddingLayerConfig objects
    ...
)
```

**Config definition**: `semantic_cache.py` line 34
```python
@dataclass
class SemanticCacheConfig:
    ...
    embedding_layers: list[EmbeddingLayerConfig] = field(default_factory=list)
```

**Result**: `cache_config.embedding_layers = [
    EmbeddingLayerConfig(name="vision", dim=512, similarity_threshold=0.85),
    EmbeddingLayerConfig(name="prompt", dim=384, similarity_threshold=0.8)
]`

---

### Step 4: Pass to EmbeddingCache

**Code**: `semantic_cache.py` line 84
```python
self.embedding_cache: EmbeddingCache | None = (
    EmbeddingCache(self.config.embedding_layers) if self.config.embedding_layers else None
)
```

**What gets passed**: `self.config.embedding_layers`
- Type: `list[EmbeddingLayerConfig]`
- Value: `[EmbeddingLayerConfig(name="vision", dim=512, ...), EmbeddingLayerConfig(name="prompt", dim=384, ...)]`

---

### Step 5: EmbeddingCache.__init__ Receives `layers`

**Code**: `embedding_cache.py` line 72-73
```python
def __init__(self, layers: Iterable[EmbeddingLayerConfig]) -> None:
    self.layers = {config.name: _EmbeddingLayerIndex(config) for config in layers}
```

**What `layers` is**:
- Type: `Iterable[EmbeddingLayerConfig]` (in practice, a `list[EmbeddingLayerConfig]`)
- Value: `[
    EmbeddingLayerConfig(name="vision", dim=512, similarity_threshold=0.85),
    EmbeddingLayerConfig(name="prompt", dim=384, similarity_threshold=0.8)
]`

**What happens**:
```python
# Dict comprehension creates:
self.layers = {
    "vision": _EmbeddingLayerIndex(EmbeddingLayerConfig(name="vision", dim=512, ...)),
    "prompt": _EmbeddingLayerIndex(EmbeddingLayerConfig(name="prompt", dim=384, ...))
}
```

---

## EmbeddingLayerConfig Structure

**Definition**: `embedding_cache.py` lines 16-22
```python
@dataclass
class EmbeddingLayerConfig:
    """Configuration describing a latent embedding layer."""
    
    name: str                    # e.g., "vision" or "prompt"
    dim: int                     # e.g., 512 or 384
    similarity_threshold: float = 0.85  # e.g., 0.85 or 0.8
```

**Example objects**:
```python
EmbeddingLayerConfig(
    name="vision",
    dim=512,
    similarity_threshold=0.85
)

EmbeddingLayerConfig(
    name="prompt",
    dim=384,
    similarity_threshold=0.8
)
```

---

## Complete Example

### Input (Command Line)
```bash
--embedding-layer vision:512:0.85 --embedding-layer prompt:384:0.8
```

### Step-by-Step Transformation

1. **Args parsing**:
   ```python
   args.embedding_layer = ["vision:512:0.85", "prompt:384:0.8"]
   ```

2. **Parse to configs**:
   ```python
   layer_configs = [
       EmbeddingLayerConfig(name="vision", dim=512, similarity_threshold=0.85),
       EmbeddingLayerConfig(name="prompt", dim=384, similarity_threshold=0.8)
   ]
   ```

3. **Pass to config**:
   ```python
   cache_config.embedding_layers = layer_configs
   ```

4. **Pass to EmbeddingCache**:
   ```python
   EmbeddingCache(self.config.embedding_layers)
   # Where self.config.embedding_layers = layer_configs
   ```

5. **In EmbeddingCache.__init__**:
   ```python
   def __init__(self, layers: Iterable[EmbeddingLayerConfig]):
       # layers = [
       #     EmbeddingLayerConfig(name="vision", dim=512, similarity_threshold=0.85),
       #     EmbeddingLayerConfig(name="prompt", dim=384, similarity_threshold=0.8)
       # ]
       
       self.layers = {config.name: _EmbeddingLayerIndex(config) for config in layers}
       # Creates:
       # {
       #     "vision": _EmbeddingLayerIndex(EmbeddingLayerConfig(...)),
       #     "prompt": _EmbeddingLayerIndex(EmbeddingLayerConfig(...))
       # }
   ```

---

## Code References

| Step | Code Location | What It Does |
|------|---------------|--------------|
| 1. CLI args | `test_vllm.py:911-917` | Parses `--embedding-layer` flags into list of strings |
| 2. Parse specs | `test_vllm.py:229-238` | Converts strings to `EmbeddingLayerConfig` objects |
| 3. Create configs | `test_vllm.py:1093` | Creates list of `EmbeddingLayerConfig` objects |
| 4. Store in config | `test_vllm.py:1103` | Stores in `SemanticCacheConfig.embedding_layers` |
| 5. Pass to cache | `semantic_cache.py:84` | Passes `self.config.embedding_layers` to `EmbeddingCache()` |
| 6. Create indices | `embedding_cache.py:72-73` | Creates separate `_EmbeddingLayerIndex` for each config |

---

## Summary

**What `layers` is in `EmbeddingCache.__init__(layers)`**:

- **Type**: `Iterable[EmbeddingLayerConfig]` (typically `list[EmbeddingLayerConfig]`)
- **Content**: List of `EmbeddingLayerConfig` dataclass objects
- **Each config contains**:
  - `name`: Layer name (e.g., "vision", "prompt")
  - `dim`: Embedding dimension (e.g., 512, 384)
  - `similarity_threshold`: Similarity threshold (e.g., 0.85, 0.8)

**Example**:
```python
layers = [
    EmbeddingLayerConfig(name="vision", dim=512, similarity_threshold=0.85),
    EmbeddingLayerConfig(name="prompt", dim=384, similarity_threshold=0.8)
]
```

**What happens**:
```python
for config in layers:  # Iterates over EmbeddingLayerConfig objects
    # Creates separate _EmbeddingLayerIndex for each
    self.layers[config.name] = _EmbeddingLayerIndex(config)
```

So `layers` is a **list of configuration objects** that define each embedding layer (name, dimension, threshold), and the dict comprehension creates separate FAISS indices for each one.

