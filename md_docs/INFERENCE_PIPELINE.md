# Vision-Language Model Inference Pipeline (Without Caching)

This document explains the standard inference pipeline for a vision-language model, step by step.

## Overview

A vision-language model (like Qwen-VL, InternVL) processes both images and text to generate responses. Here's what happens **without caching**:

---

## Standard Inference Pipeline (No Cache)

### Step 1: Input Preparation

**Inputs**:
- **Image**: Raw image file (e.g., `frame_001.jpg`)
- **Text Prompt**: "What color is the car?"

**What happens**:
```python
# Load image
image = Image.open("frame_001.jpg")

# Format prompt
prompt = "You are assisting with the GQA benchmark. Answer the question based on the referenced image.\nImage ID: frame_001\nQuestion: What color is the car?\nAnswer:"
```

---

### Step 2: Image Preprocessing & Encoding

**What happens**:
1. **Image preprocessing**: Resize, normalize, convert to tensor
2. **Vision encoder**: Process image through vision transformer (ViT) or CNN
3. **Image embeddings**: Extract visual features

**Pipeline**:
```
Raw Image (H×W×3)
    ↓
Image Preprocessor
    ↓
Image Patches/Tokens (N×patch_size)
    ↓
Vision Encoder (ViT/CNN)
    ↓
Image Embeddings (N×d_vision)
    ↓
Pooled/Projected Features (1×d_vision)
```

**Example**:
- Input: 224×224×3 RGB image
- After preprocessing: 196 patches of 16×16 (for ViT-B/16)
- After vision encoder: 196×768 feature vectors
- After pooling: 1×768 image embedding

**Computation cost**: High (vision encoder forward pass)

**Code location**: Handled by vLLM internally, or by embedding hooks for extraction

---

### Step 3: Text Tokenization & Encoding

**What happens**:
1. **Tokenization**: Convert text to token IDs
2. **Text embedding**: Map tokens to embedding vectors
3. **Position encoding**: Add positional information

**Pipeline**:
```
Text Prompt (string)
    ↓
Tokenizer
    ↓
Token IDs [101, 2023, 2003, ...]
    ↓
Token Embeddings (N×d_text)
    ↓
Position Embeddings
    ↓
Text Embeddings (N×d_text)
```

**Example**:
- Input: "What color is the car?"
- After tokenization: `[101, 2023, 2003, 3231, 2003, 2007, 102]` (7 tokens)
- After embedding: 7×768 embedding vectors

**Computation cost**: Low (lookup + addition)

**Code location**: Handled by vLLM tokenizer

---

### Step 4: Fusion (Image-Text Integration)

**What happens**:
1. **Projection**: Align image and text embeddings to same dimension
2. **Concatenation**: Combine image and text embeddings
3. **Fusion layers**: Process combined embeddings through fusion modules

**Pipeline**:
```
Image Embeddings (1×d_vision)
    ↓
Image Projector
    ↓
Projected Image Features (1×d_fusion)
    ↓
    └─→ Concatenate
Text Embeddings (N×d_text) ──→ Text Projector ──→ Projected Text (N×d_fusion)
    ↓
Combined Embeddings ((1+N)×d_fusion)
    ↓
Fusion Module (Image-Text Projector)
    ↓
Fused Features ((1+N)×d_fusion)
```

**Example**:
- Image: 1×768 → projected to 1×1024
- Text: 7×768 → projected to 7×1024
- Combined: 8×1024 (1 image token + 7 text tokens)
- After fusion: 8×1024 fused features

**Computation cost**: High (projection matrices + fusion layers)

**Code location**: Model's fusion module (e.g., Qwen-VL's image projector)

**Note**: This is what **fusion cache** (L1.5) captures and reuses!

---

### Step 5: Transformer Forward Passes

**What happens**:
For each transformer layer (typically 24-32 layers):

1. **Self-Attention**:
   - Compute Query (Q), Key (K), Value (V) from embeddings
   - Attention scores: `Q @ K^T / sqrt(d)`
   - Attention weights: `softmax(scores)`
   - Attention output: `weights @ V`
   - **KV Cache**: Store K and V for future reuse (this is what gets cached!)

2. **Feedforward Network (FFN)**:
   - Two linear layers with activation
   - Process attention output

3. **Residual connections & Layer normalization**

**Pipeline** (for one layer):
```
Fused Features (N×d)
    ↓
Layer Normalization
    ↓
Self-Attention
    ├─→ Q, K, V computation
    ├─→ Attention(Q, K, V)
    └─→ KV Cache Storage (K, V tensors)
    ↓
Residual Connection
    ↓
Layer Normalization
    ↓
Feedforward Network
    ↓
Residual Connection
    ↓
Next Layer...
```

**Example** (24-layer model):
- Input: 8×1024 (8 tokens: 1 image + 7 text)
- Layer 1: 8×1024 → attention → 8×1024 → FFN → 8×1024
- Layer 2: 8×1024 → ... (repeat)
- ...
- Layer 24: 8×1024 → final hidden states

**KV Cache per layer**:
- Key cache: 8×1024 tensor (one per layer)
- Value cache: 8×1024 tensor (one per layer)
- Total: 24 layers × 2 (K+V) × 8 tokens × 1024 dims = ~400K values

**Computation cost**: Very high (most expensive step)
- Attention: O(N²) complexity
- FFN: O(N×d²) complexity
- 24-32 layers × multiple operations

**Code location**: vLLM's transformer layers

**Note**: This is what **KV cache** captures and reuses!

---

### Step 6: Generation/Decoding

**What happens**:
1. **Output projection**: Map hidden states to vocabulary
2. **Sampling**: Select next token (greedy, top-k, top-p, etc.)
3. **Iterative generation**: Repeat until EOS token or max length

**Pipeline**:
```
Final Hidden States (N×d)
    ↓
Output Projection (d×vocab_size)
    ↓
Logits (N×vocab_size)
    ↓
Softmax
    ↓
Token Probabilities (N×vocab_size)
    ↓
Sampling (greedy/top-k/top-p)
    ↓
Next Token ID
    ↓
Append to sequence
    ↓
Repeat (if not EOS)
```

**Example**:
- Input: 8 tokens (1 image + 7 text)
- Generate token 9: "red"
- Generate token 10: "."
- Generate token 11: EOS
- Final output: "red."

**Computation cost**: Moderate (projection + sampling, repeated per token)

**Code location**: vLLM's generation loop

---

## Complete Pipeline Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT PREPARATION                         │
│  Image: frame_001.jpg                                        │
│  Text: "What color is the car?"                             │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 1: IMAGE ENCODING                          │
│  • Image preprocessing (resize, normalize)                   │
│  • Vision encoder (ViT/CNN forward pass)                     │
│  • Image embeddings: 1×768                                   │
│  Cost: HIGH (vision transformer)                             │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 2: TEXT ENCODING                            │
│  • Tokenization                                              │
│  • Token embeddings: 7×768                                   │
│  Cost: LOW (lookup)                                          │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 3: FUSION                                  │
│  • Image projection: 1×768 → 1×1024                         │
│  • Text projection: 7×768 → 7×1024                          │
│  • Concatenation: 8×1024                                     │
│  • Fusion module forward pass                                │
│  Cost: HIGH (projection + fusion)                            │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│         STEP 4: TRANSFORMER FORWARD PASSES                   │
│  Layer 1:                                                    │
│    • Self-attention (Q, K, V computation)                    │
│    • Store K, V in KV cache                                  │
│    • Feedforward network                                     │
│  Layer 2: (repeat)                                           │
│  ...                                                         │
│  Layer 24: (repeat)                                         │
│  Cost: VERY HIGH (24 layers × attention + FFN)               │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 5: GENERATION                               │
│  • Output projection                                          │
│  • Token sampling                                             │
│  • Iterative decoding                                         │
│  Cost: MODERATE (repeated per token)                         │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
                   OUTPUT: "red."
```

---

## Computation Breakdown (Typical)

For a single inference request:

| Step | Operation | Cost | Time (approx) |
|------|-----------|------|---------------|
| **1. Image Encoding** | Vision encoder forward pass | High | 50-200ms |
| **2. Text Encoding** | Tokenization + embedding lookup | Low | 1-5ms |
| **3. Fusion** | Projection + fusion layers | High | 20-100ms |
| **4. Transformer** | 24 layers × (attention + FFN) | Very High | 500-2000ms |
| **5. Generation** | Output projection + sampling | Moderate | 50-200ms per token |

**Total (no cache)**: ~600-2500ms for first token, +50-200ms per additional token

---

## What Gets Cached (With Caching System)

### L0.5: Exact Text Cache
- **Caches**: Nothing (just lookup)
- **Saves**: Steps 2-5 (if exact match)

### L1: Semantic Text Cache
- **Caches**: KV blocks from transformer layers
- **Saves**: Step 4 (transformer forward passes)

### L1.5: Fusion Cache
- **Caches**: Fusion module outputs
- **Saves**: Step 3 (fusion computation)

### L2: Embedding Cache
- **Caches**: KV blocks from transformer layers
- **Saves**: Step 4 (transformer forward passes)
- **Uses**: Vision/prompt embeddings to find similar inputs

---

## With Cache Hit: What Gets Skipped

### Example: Vision Embedding Cache Hit

**Scenario**: Frame 100 and Frame 101 are very similar (0.95 cosine similarity)

**Without cache**:
- Frame 100: Full pipeline (Steps 1-5) → 1500ms
- Frame 101: Full pipeline (Steps 1-5) → 1500ms
- **Total**: 3000ms

**With cache**:
- Frame 100: Full pipeline (Steps 1-5) → 1500ms, **cache KV blocks**
- Frame 101: 
  - Step 1: Image encoding → 100ms (still needed for embedding extraction)
  - Step 2: Text encoding → 2ms
  - **Step 3: Fusion** → **SKIPPED** (if fusion cache hit)
  - **Step 4: Transformer** → **SKIPPED** (KV cache injection) ✅
  - Step 5: Generation → 50ms
  - **Total**: ~150ms

**Savings**: 1350ms (90% reduction!)

---

## Key Insights

1. **Image encoding** (Step 1) is expensive but usually still needed for embedding extraction
2. **Fusion** (Step 3) can be cached if fusion cache is enabled
3. **Transformer forward passes** (Step 4) are the most expensive and benefit most from caching
4. **KV cache** stores the Key and Value tensors from attention layers
5. **Generation** (Step 5) still happens but is much faster when starting from cached position

---

## Summary

**Standard pipeline (no cache)**:
1. Image → Vision encoder → Image embeddings
2. Text → Tokenizer → Text embeddings
3. Image + Text → Fusion module → Fused features
4. Fused features → Transformer layers (24×) → Hidden states
5. Hidden states → Output projection → Token generation

**With caching**:
- Steps 3-4 can be **skipped** if cache hit
- KV blocks from Step 4 are **reused** instead of recomputed
- Only generation (Step 5) continues from cached position

The caching system optimizes the **most expensive steps** (fusion and transformer forward passes), which can save 80-90% of computation time on cache hits!

