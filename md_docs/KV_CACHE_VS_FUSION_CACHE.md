# KV Cache (Baseline) vs Fusion Cache: What's the Difference?

This document clarifies the difference between vLLM's built-in KV cache and what fusion cache adds.

## Quick Answer

**Yes, KV cache is baseline!** But there are **two types** of KV cache:

1. **vLLM's built-in KV cache (L0)**: Token-level prefix matching **within the same request**
2. **Semantic KV cache (this system)**: Cross-request KV block reuse for **similar/different prompts**

**Fusion cache** adds something completely different: it caches **fusion module outputs** (BEFORE transformer layers).

---

## vLLM's Built-in KV Cache (L0) - Baseline

### What It Is

**Token-level prefix caching** - vLLM's built-in feature that works **within a single request**.

### How It Works

**Within the same request**, if tokens share a prefix, vLLM reuses KV cache:

```
Request: "What color is the car? It is"
    ↓
Token 1-7: "What color is the car?" → Compute KV cache
Token 8: "It" → Reuse KV cache from tokens 1-7
Token 9: "is" → Reuse KV cache from tokens 1-8
```

**Example**:
- Generating token 8: Reuses KV cache from tokens 1-7
- Generating token 9: Reuses KV cache from tokens 1-8
- **Within same request only**

### Limitations

- ❌ Only works **within the same request**
- ❌ Cannot reuse KV cache **across different requests**
- ❌ Cannot reuse KV cache for **similar but different prompts**

**Example where it fails**:
```
Request 1: "What color is the car?" → Computes KV cache
Request 2: "What color is the car?" → Computes KV cache AGAIN (can't reuse!)
```

---

## Semantic KV Cache (This System) - What's Added

### What It Is

**Cross-request KV block reuse** - This system adds the ability to reuse KV cache **across different requests** for similar prompts.

### How It Works

**Across different requests**, if prompts are similar, reuse KV blocks:

```
Request 1: "What color is the car?" 
    ↓
Compute KV cache → Store in kv_chunks/chunk_123.pkl

Request 2: "What color is the car?" (identical)
    ↓
Find cached KV blocks → Inject into vLLM → Skip computation ✅

Request 3: "What color is the vehicle?" (semantically similar)
    ↓
Find similar cached KV blocks → Inject → Skip computation ✅
```

### What It Adds

- ✅ **Cross-request reuse**: Reuse KV cache across different requests
- ✅ **Semantic matching**: Reuse for similar prompts (not just identical)
- ✅ **Persistent storage**: KV blocks saved to disk, can be reused later

**Layers added**:
- **L0.5**: Exact text cache (finds identical prompts)
- **L1**: Semantic text cache (finds semantically similar prompts)
- **L2**: Embedding cache (finds similar inputs via embeddings)

---

## Fusion Cache - What It Adds

### What It Is

**Fusion module output caching** - Caches the outputs of the fusion module (image-text projector), which happens **BEFORE transformer layers**.

### Pipeline Position

```
Image → Vision Encoder → Image Embeddings
                              ↓
Text → Tokenizer → Text Embeddings → Fusion Module → Fused Features
                                                          ↑
                                                    Fusion Cache stores this!
                                                          ↓
                                              Transformer Layers → KV Cache
                                                          ↑
                                                    Semantic KV Cache stores this!
```

### What Fusion Cache Stores

**Fusion module outputs** - The tensors produced by the fusion module:

```python
FusionState(
    tensors={
        "fusion_output": <Tensor [8, 1024]>,  # Fused image+text features
        "image_features": <Tensor [1, 1024]>,  # Processed image
        "projected_text": <Tensor [7, 1024]>   # Processed text
    }
)
```

### What It Adds

- ✅ **Caches fusion computation**: Skips fusion module (image-text projection)
- ✅ **Works before transformer**: Fusion happens BEFORE KV cache is computed
- ✅ **Independent of KV cache**: Can work even if KV cache misses

### Why It's Needed

**Fusion module is expensive**:
- Image projection: 1×768 → 1×1024
- Text projection: 7×768 → 7×1024
- Fusion layers: Additional computation
- **Cost**: ~20-100ms per request

**Without fusion cache**:
- Even if KV cache hits, fusion module still runs
- Fusion computation happens every time

**With fusion cache**:
- If fusion state is cached, skip fusion module entirely
- Saves ~20-100ms per request

---

## Comparison Table

| Feature | vLLM KV Cache (L0) | Semantic KV Cache (L0.5-L2) | Fusion Cache (L1.5) |
|---------|-------------------|---------------------------|---------------------|
| **What it caches** | KV blocks (K, V tensors) | KV blocks (K, V tensors) | Fusion module outputs |
| **Scope** | Within same request | Across different requests | Across different requests |
| **Matching** | Token prefix matching | Semantic similarity | Exact/semantic matching |
| **When computed** | During transformer forward pass | During transformer forward pass | During fusion module |
| **Pipeline position** | After fusion | After fusion | Before transformer |
| **What it skips** | Attention computation (within request) | Attention computation (across requests) | Fusion computation |
| **Storage** | In-memory (vLLM) | On-disk (kv_chunks/) | On-disk (fusion_chunks/) |

---

## What Each Cache Adds

### vLLM KV Cache (Baseline - L0)

**What it does**:
- Caches KV blocks during generation
- Reuses within the same request
- Built into vLLM

**What it skips**:
- Attention computation for already-processed tokens (within same request)

**Example**:
```
Request: "What color is the car? It is red."
Token 1-7: Compute KV cache
Token 8-11: Reuse KV cache from tokens 1-7 ✅
```

### Semantic KV Cache (This System - L0.5, L1, L2)

**What it adds**:
- Cross-request KV block reuse
- Semantic matching (similar prompts)
- Persistent storage

**What it skips**:
- Attention computation for similar prompts (across requests)

**Example**:
```
Request 1: "What color is the car?" → Compute KV cache, store
Request 2: "What color is the car?" → Reuse KV cache from Request 1 ✅
Request 3: "What color is the vehicle?" → Reuse KV cache (semantic match) ✅
```

### Fusion Cache (This System - L1.5)

**What it adds**:
- Fusion module output caching
- Skips fusion computation entirely

**What it skips**:
- Fusion module computation (image-text projection)

**Example**:
```
Request 1: Image + "What color is the car?"
    ↓
Fusion module: Computes fusion_output tensor
    ↓
Store fusion_output in fusion cache
    ↓
Request 2: Same/similar image + "What color is the car?"
    ↓
Load fusion_output from cache
    ↓
Skip fusion module computation ✅
```

---

## How They Work Together

### Complete Pipeline with All Caches

```
Request arrives
    ↓
1. Extract embeddings (for embedding cache search)
    ↓
2. Embedding cache search → Find similar cached entry
    ↓
3. Load KV blocks from disk (semantic KV cache)
    ↓
4. Load fusion state from disk (fusion cache)
    ↓
5. Inject KV blocks into vLLM
    ↓
6. Inject fusion tensors into model
    ↓
7. Forward pass:
   ├─→ Fusion module: Sees pre-injected tensors → SKIPS ✅
   ├─→ Transformer: Sees pre-injected KV blocks → SKIPS ✅
   └─→ vLLM KV cache: Reuses within generation → SKIPS ✅
    ↓
8. Generation continues
```

### What Gets Skipped

- ✅ **Fusion module**: Skipped via fusion cache
- ✅ **Transformer forward passes**: Skipped via semantic KV cache
- ✅ **Attention within generation**: Skipped via vLLM KV cache

---

## Key Insights

### 1. KV Cache Has Two Types

**vLLM's built-in (L0)**:
- Works within same request
- Token-level prefix matching
- In-memory only

**Semantic KV cache (this system)**:
- Works across different requests
- Semantic similarity matching
- Persistent on-disk storage

### 2. Fusion Cache is Different

**Fusion cache**:
- Caches fusion module outputs (BEFORE transformer)
- Independent of KV cache
- Skips fusion computation

**KV cache**:
- Caches transformer attention states (AFTER fusion)
- Works at transformer layer level
- Skips attention computation

### 3. They Complement Each Other

- **Fusion cache**: Skips fusion module (Step 3 of pipeline)
- **Semantic KV cache**: Skips transformer layers (Step 4 of pipeline)
- **vLLM KV cache**: Skips attention within generation (Step 5 of pipeline)

All three work together to maximize savings!

---

## Summary

### Your Question
> "I thought KV cache should already be there as the baseline? What fusion cache has added?"

### Answer

**Yes, KV cache is baseline!** But:

1. **vLLM's KV cache (L0)**: 
   - ✅ Already there (baseline)
   - Works within same request only
   - Token-level prefix matching

2. **Semantic KV cache (this system)**:
   - ✅ Adds cross-request reuse
   - Works across different requests
   - Semantic similarity matching

3. **Fusion cache (this system)**:
   - ✅ Adds fusion module output caching
   - Caches fusion computation (BEFORE transformer)
   - Skips fusion module entirely

**What fusion cache adds**:
- Caches fusion module outputs (image-text projection results)
- Skips fusion computation (~20-100ms savings)
- Works independently of KV cache
- Caches a different part of the pipeline (before transformer)

**Bottom line**: 
- KV cache (baseline) = Transformer attention states
- Fusion cache (added) = Fusion module outputs
- They cache different parts of the pipeline and work together!

