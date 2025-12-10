# Cache Hit Threshold Configuration Guide

This document shows all the places where you can configure cache hit thresholds in the system.

## Overview

There are **two types of thresholds** in the system:

1. **Semantic Text Cache Threshold** - Controls when text embeddings are similar enough to reuse KV cache
2. **Embedding Cache Threshold** - Controls when latent embeddings (vision, prompt, etc.) are similar enough to reuse

---

## 1. Semantic Text Cache Threshold

Controls the L1 (semantic text) cache layer. This threshold determines the minimum cosine similarity required for two text chunks to be considered similar enough to reuse KV cache.

### Configuration Locations

#### A. Command Line Argument (Recommended)

**Location**: `experiment2/test_vllm.py` line 931-935

```bash
python experiment2/test_vllm.py \
  --similarity-threshold 0.85 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  ...
```

**Default**: `0.8` (from CLI), `0.85` (in code default)

**Range**: Typically `0.0` to `1.0` (cosine similarity)
- Lower values (e.g., 0.5-0.7): More aggressive caching, more false positives
- Higher values (e.g., 0.85-0.95): More conservative, fewer false positives

#### B. Code Default (SemanticCacheConfig)

**Location**: `experiment2/semantic_cache/semantic_cache.py` line 26

```python
@dataclass
class SemanticCacheConfig:
    similarity_threshold: float = 0.85  # ← Change default here
```

**Usage**: This is used when creating `SemanticCacheConfig` without specifying the threshold.

#### C. Experiment Spec JSON

**Location**: `experiment2/specs.py` line 23

When using `run_experiments.py` with a JSON spec file:

```json
{
  "defaults": {
    "similarity_threshold": 0.8
  },
  "experiments": [
    {
      "name": "my-experiment",
      "similarity_threshold": 0.85
    }
  ]
}
```

**File**: `experiment2/specs.py` - `ExperimentSpec` class default is `0.8`

#### D. Model Presets

**Location**: `experiment2/model_presets.py`

Presets can override the default threshold:

```python
# Line 40 - qwen3-vl-2b preset
"similarity_threshold": 0.82,

# Line 65 - internvl3.5-2b preset  
"similarity_threshold": 0.8,
```

**Usage**: When using `--preset qwen3-vl-2b`, the threshold is set to 0.82.

#### E. Model Router Config

**Location**: `experiment2/model_router/router.py` line 24

If using the model router backend instead of semantic-kv:

```python
@dataclass
class ModelRouterConfig:
    similarity_threshold: float = 0.85  # ← Change default here
```

---

## 2. Embedding Cache Threshold

Controls the L2 (latent embedding) cache layer. Each embedding layer can have its own threshold.

### Configuration Locations

#### A. Command Line Argument (Recommended)

**Location**: `experiment2/test_vllm.py` line 912-917

```bash
python experiment2/test_vllm.py \
  --embedding-layer vision:512:0.9 \
  --embedding-layer prompt:384:0.85 \
  ...
```

**Format**: `NAME:DIM[:THRESHOLD]`
- `vision:512:0.9` - vision layer, 512 dimensions, threshold 0.9
- `prompt:384:0.85` - prompt layer, 384 dimensions, threshold 0.85
- `vision:512` - uses default threshold 0.85

**Parsing**: `experiment2/test_vllm.py` lines 230-238

```python
def _parse_embedding_layer(spec: str) -> EmbeddingLayerConfig:
    parts = spec.split(":")
    name = parts[0].strip()
    dim = int(parts[1])
    threshold = float(parts[2]) if len(parts) > 2 else 0.85  # ← Default here
    return EmbeddingLayerConfig(name=name, dim=dim, similarity_threshold=threshold)
```

#### B. Code Default (EmbeddingLayerConfig)

**Location**: `experiment2/semantic_cache/techniques/embedding_cache.py` line 22

```python
@dataclass
class EmbeddingLayerConfig:
    name: str
    dim: int
    similarity_threshold: float = 0.85  # ← Change default here
```

**Usage**: This default is used when threshold is not specified in the CLI argument.

#### C. Experiment Spec JSON

When using `run_experiments.py`:

```json
{
  "experiments": [
    {
      "name": "my-experiment",
      "embedding_layers": [
        "vision:512:0.9",
        "prompt:384:0.85"
      ]
    }
  ]
}
```

---

## 3. Where Thresholds Are Applied

### Semantic Text Cache Threshold

**Applied in**: `experiment2/semantic_cache/semantic_cache.py` line 195

```python
match = self.semantic_text_cache.search(chunk_text)
if match is None or match.score < self.config.similarity_threshold:  # ← Applied here
    statuses["semantic_text"] = "miss"
    return ReuseReport(hit=None, statuses=dict(statuses))
```

**Also in**: `experiment2/model_router/router.py` line 166

```python
if match and match.score >= self.config.similarity_threshold:  # ← Applied here
    # Cache hit
```

### Embedding Cache Threshold

**Applied in**: `experiment2/semantic_cache/techniques/embedding_cache.py` line 64

```python
def search(self, embedding: np.ndarray) -> EmbeddingMatch | None:
    # ... FAISS search ...
    score = float(scores[0][0])
    if score < self.config.similarity_threshold:  # ← Applied here
        return None
    return EmbeddingMatch(layer=self.config.name, chunk_id=self.ids[idx], score=score)
```

---

## 4. Quick Reference: How to Change Thresholds

### Method 1: Command Line (Easiest)

```bash
# Change semantic text cache threshold
python experiment2/test_vllm.py \
  --similarity-threshold 0.9 \
  ...

# Change embedding cache thresholds
python experiment2/test_vllm.py \
  --embedding-layer vision:512:0.95 \
  --embedding-layer prompt:384:0.9 \
  ...
```

### Method 2: Experiment Spec JSON

Create/edit `experiments.json`:

```json
{
  "defaults": {
    "similarity_threshold": 0.85,
    "embedding_layers": ["vision:512:0.9", "prompt:384:0.85"]
  },
  "experiments": [
    {
      "name": "high-precision",
      "similarity_threshold": 0.95,
      "embedding_layers": ["vision:512:0.95"]
    }
  ]
}
```

Then run:
```bash
python experiment2/run_experiments.py --specs experiments.json
```

### Method 3: Modify Code Defaults

**For semantic text cache**:
- Edit `experiment2/semantic_cache/semantic_cache.py` line 26
- Change `similarity_threshold: float = 0.85` to your desired value

**For embedding cache**:
- Edit `experiment2/semantic_cache/techniques/embedding_cache.py` line 22
- Change `similarity_threshold: float = 0.85` to your desired value

**For CLI default**:
- Edit `experiment2/test_vllm.py` line 933
- Change `default=0.8` to your desired value

---

## 5. Threshold Selection Guidelines

### Semantic Text Cache Threshold

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.5-0.7 | Very aggressive, many matches | Testing, high redundancy datasets |
| 0.75-0.85 | Balanced (recommended) | General use, GQA benchmark |
| 0.85-0.95 | Conservative, high precision | Production, quality-critical |
| 0.95+ | Very strict, few matches | Research, precision studies |

### Embedding Cache Threshold

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.7-0.8 | Aggressive matching | Similar images/questions |
| 0.8-0.9 | Balanced (recommended) | General vision-language tasks |
| 0.9-0.95 | High precision | Critical accuracy requirements |
| 0.95+ | Very strict | Research, exact matches only |

### Typical Values from Ablation Studies

Looking at `experiment2/ablation.sh`, the system tests:
- Semantic thresholds: `0.5, 0.6, 0.7, 0.8, 0.9`
- Embedding thresholds: `0.7, 0.8, 0.9` (grid of 3×3 combinations)

---

## 6. Examples

### Example 1: High Precision Setup

```bash
python experiment2/test_vllm.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --similarity-threshold 0.9 \
  --embedding-layer vision:512:0.95 \
  --embedding-layer prompt:384:0.9 \
  --embedding-hook prompt_vision \
  ...
```

### Example 2: Aggressive Caching Setup

```bash
python experiment2/test_vllm.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --similarity-threshold 0.7 \
  --embedding-layer vision:512:0.8 \
  --embedding-layer prompt:384:0.75 \
  --embedding-hook prompt_vision \
  ...
```

### Example 3: Using Presets with Override

```bash
python experiment2/test_vllm.py \
  --preset qwen3-vl-2b \
  --similarity-threshold 0.9 \
  ...
```

The preset sets threshold to 0.82, but `--similarity-threshold 0.9` overrides it.

---

## 7. Important Notes

1. **Exact Text Cache (L0.5) has no threshold** - it's binary (exact match or not)

2. **Fusion Cache (L1.5) has no threshold** - it's tied to KV cache hits

3. **Threshold precedence**:
   - CLI argument > Experiment spec > Code default
   - For embedding layers: CLI format > Code default

4. **Thresholds are cosine similarity** - range is typically -1.0 to 1.0, but normalized embeddings usually produce 0.0 to 1.0

5. **Different thresholds for different layers** - You can set different thresholds for `vision` vs `prompt` embedding layers

---

## 8. Files Summary

| File | Line | What It Controls |
|------|------|------------------|
| `experiment2/test_vllm.py` | 933 | CLI default for semantic text threshold |
| `experiment2/test_vllm.py` | 237 | Default for embedding layer threshold (when not specified) |
| `experiment2/semantic_cache/semantic_cache.py` | 26 | Code default for semantic text threshold |
| `experiment2/semantic_cache/techniques/embedding_cache.py` | 22 | Code default for embedding layer threshold |
| `experiment2/specs.py` | 23 | Experiment spec default for semantic text threshold |
| `experiment2/model_router/router.py` | 24 | Model router default for semantic text threshold |
| `experiment2/model_presets.py` | 40, 65 | Preset-specific thresholds |

