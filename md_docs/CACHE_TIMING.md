# Cache Timing: When Embedding Cache vs Fusion Cache Are Used

This document clarifies the exact timing of when embedding cache and fusion cache are used relative to the forward pass.

## Quick Answer

**Both are used BEFORE the forward pass**, but they serve different purposes:

- **Embedding Cache**: Used BEFORE forward pass to **find** what to reuse (search phase)
- **Fusion Cache**: Injected BEFORE forward pass to **skip** fusion computation during forward pass

---

## Detailed Timing

### Embedding Cache: Search Phase (Before Forward Pass)

**When**: BEFORE forward pass starts

**Code flow**:
```python
# In test_vllm.py, line 479-492
for sample in prompts:
    # 1. Extract embeddings (external encoders, not model forward pass)
    embeddings = embedding_hook(llm=llm, sample=sample)  # ← BEFORE forward pass
    
    # 2. Add request to vLLM engine
    engine.add_request(request_id, sample.prompt, sampling_params)
    
    # 3. Try to reuse (checks embedding cache)
    reuse = cache.try_reuse(request_id, sample.chunk_text, embeddings=embeddings)
    # ← This happens BEFORE forward pass
    
    # 4. If cache hit, KV blocks are injected
    # 5. Forward pass happens (or is skipped if cache hit)
```

**In `try_reuse()` (semantic_cache.py, line 167-186)**:
```python
if self.embedding_cache:
    if embeddings:
        # Search FAISS indices for similar embeddings
        embed_match = self.embedding_cache.search(embeddings)  # ← Search phase
        if embed_match is not None:
            # Inject KV blocks (happens BEFORE forward pass)
            injected = self._maybe_inject(request_id, embed_match)
            # If injected, forward pass will skip transformer layers
```

**Timing**: 
- ✅ Used BEFORE forward pass
- Purpose: **Search** to find similar cached entries
- Result: Triggers KV cache injection

---

### Fusion Cache: Injection Phase (Before Forward Pass)

**When**: BEFORE forward pass starts (but AFTER KV cache injection)

**Code flow**:
```python
# In semantic_cache.py, line 104-122
def _maybe_inject(self, request_id, match):
    # 1. Load KV blocks
    stored = self.store.load(match.chunk_id)
    
    # 2. Inject KV blocks into vLLM
    injected = self.adapter.inject(request_id, stored)  # ← BEFORE forward pass
    
    # 3. If successful, inject fusion state
    if injected:
        self._inject_fusion_state(request_id, match.chunk_id)  # ← BEFORE forward pass
        return True

def _inject_fusion_state(self, request_id, chunk_id):
    # Load fusion state
    state = self.fusion_cache.load(chunk_id)
    
    # Inject fusion tensors into model
    self.fusion_cache.inject(llm=self.llm, request_id=request_id, state=state)
    # ← This happens BEFORE forward pass
    # When forward pass runs, fusion module sees pre-computed tensors
```

**Timing**:
- ✅ Injected BEFORE forward pass
- Purpose: **Skip** fusion computation during forward pass
- Result: Forward pass skips fusion module

---

## Complete Timeline

### Request Processing Flow

```
1. Request arrives
   ↓
2. Extract embeddings (external encoders)
   ├─→ Vision embedding: CLIP encoder
   └─→ Prompt embedding: Sentence transformer
   ↓
3. Embedding cache search ← BEFORE forward pass
   ├─→ Search FAISS indices
   ├─→ Find similar embedding (if exists)
   └─→ Get chunk_id
   ↓
4. Load KV blocks from disk
   ↓
5. Inject KV blocks into vLLM ← BEFORE forward pass
   ↓
6. Load fusion state from disk
   ↓
7. Inject fusion tensors into model ← BEFORE forward pass
   ↓
8. Forward pass starts
   ├─→ Fusion module: Sees pre-injected tensors, SKIPS computation ✅
   ├─→ Transformer layers: Sees pre-injected KV blocks, SKIPS attention ✅
   └─→ Generation: Continues from cached position
```

---

## Key Insight: Both Are Pre-Forward-Pass

**Important**: Both embedding cache and fusion cache are used **BEFORE** the forward pass starts:

1. **Embedding cache** (search): Finds what to reuse
2. **KV cache injection**: Injects transformer states
3. **Fusion cache injection**: Injects fusion tensors
4. **Forward pass**: Runs with pre-injected states, skips computation

---

## What Happens During Forward Pass

### Without Cache

```
Forward Pass:
1. Image encoding
2. Text encoding
3. Fusion module computation ← Computes fusion tensors
4. Transformer layers computation ← Computes KV blocks
5. Generation
```

### With Cache (Both Injected)

```
Forward Pass:
1. Image encoding (still needed for embedding extraction)
2. Text encoding (still needed)
3. Fusion module: Sees pre-injected tensors → SKIPS computation ✅
4. Transformer layers: Sees pre-injected KV blocks → SKIPS attention ✅
5. Generation: Continues from cached position
```

---

## Clarification

### Your Question
> "embedding cache is used before running the forward pass, while fusion cache is used during the forward pass"

### Corrected Understanding

**Both are used BEFORE the forward pass**, but:

1. **Embedding cache**: 
   - Used BEFORE forward pass to **search** for similar entries
   - Triggers KV cache injection
   - Does NOT directly skip computation (it finds what to skip)

2. **Fusion cache**:
   - Injected BEFORE forward pass
   - Stores outputs FROM a previous forward pass
   - When injected, allows forward pass to **skip** fusion computation
   - The skipping happens DURING forward pass, but injection happens BEFORE

### More Precise Statement

- **Embedding cache**: Used BEFORE forward pass (search phase)
- **Fusion cache**: Injected BEFORE forward pass, enables skipping DURING forward pass

---

## Code Evidence

### Embedding Cache (Before Forward Pass)

```python
# test_vllm.py, line 482-492
embeddings = embedding_hook(llm=llm, sample=sample)  # Extract embeddings
engine.add_request(request_id, sample.prompt, sampling_params)  # Add to engine
reuse = cache.try_reuse(request_id, sample.chunk_text, embeddings=embeddings)
# ↑ This happens BEFORE forward pass

# Inside try_reuse():
embed_match = self.embedding_cache.search(embeddings)  # Search
injected = self._maybe_inject(request_id, embed_match)  # Inject KV blocks
# ↑ Both happen BEFORE forward pass
```

### Fusion Cache (Before Forward Pass)

```python
# semantic_cache.py, line 121
self._inject_fusion_state(request_id, match.chunk_id)
# ↑ Called during _maybe_inject(), which happens BEFORE forward pass

# semantic_cache.py, line 124-130
def _inject_fusion_state(self, request_id, chunk_id):
    state = self.fusion_cache.load(chunk_id)
    self.fusion_cache.inject(llm=self.llm, request_id=request_id, state=state)
    # ↑ Injection happens BEFORE forward pass
```

### Forward Pass (After Injection)

```python
# test_vllm.py, line 517
response = drain_request(engine, request_id)
# ↑ Forward pass happens HERE
# At this point, KV blocks and fusion tensors are already injected
# Forward pass sees pre-injected states and skips computation
```

---

## Summary

| Cache | When Used | Purpose | What It Does |
|-------|-----------|---------|--------------|
| **Embedding Cache** | BEFORE forward pass | Search | Finds similar cached entries |
| **Fusion Cache** | BEFORE forward pass (injected) | Skip fusion | Injects fusion tensors to skip fusion computation |
| **Forward Pass** | AFTER injection | Execute | Sees pre-injected states, skips computation |

**Corrected statement**:
- Embedding cache: Used BEFORE forward pass (search phase)
- Fusion cache: Injected BEFORE forward pass (enables skipping DURING forward pass)

Both happen before the forward pass, but fusion cache's effect (skipping fusion computation) happens during the forward pass when the model sees the pre-injected tensors.

