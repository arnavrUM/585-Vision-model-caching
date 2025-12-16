# Expected Outcomes for GQA Scene Description Caching

## Dataset Context: GQA Visual Question Answering

The GQA dataset contains questions about scenes in images. From the sample data:
- **Question types**: "Is the fence in the bottom or top?", "What is on the shelf?", "Who is wearing goggles?"
- **Answer format**: Short factual responses (e.g., "The fence is in the bottom part.")
- **Total samples**: 1024 questions per experiment, shuffled with seed=42

## Caching Technique Analysis

### 1. **Exact Text Cache** (qwen-exact-only, internvl-exact-only)
**Expected Hit Rate**: **~0%** ❌

**Reasoning**:
- Requires **identical** text prompts after normalization
- GQA questions are diverse: "What is on the shelf?" ≠ "What is on the bike?"
- With 1024 shuffled samples from a large validation set, duplicates are rare
- Exact matches only occur if dataset has repeated question-image pairs

**Current Results**: *(not yet run)*

**Accuracy Impact**: Neutral - exact matches should return correct cached responses

---

### 2. **Fusion Cache Only** (qwen-fusion-only, internvl-fusion-only)
**Expected Hit Rate**: **~0%** ❌

**Reasoning**:
- Fusion cache combines text + vision embeddings
- Requires similar text AND similar image
- GQA has diverse images (different scenes, objects, compositions)
- Threshold 0.8 is strict - needs near-identical prompt + image pairs
- No duplicate images in shuffled 1024-sample subset

**Current Results**:
```
qwen-fusion-only:  0/1024 hits (0.0%) - Accuracy: 12.79%
internvl-fusion-only: 0/1024 hits (0.0%) - Accuracy: 5.57%
```

**Accuracy Impact**: Neutral (no cache hits to evaluate)

**Note**: Low baseline accuracy (12.79% Qwen, 5.57% InternVL) suggests model struggles with GQA or evaluation criteria is strict

---

### 3. **Semantic Text Cache** (qwen-semantic-0.5 to 0.9)
**Expected Hit Rate**: **60-97%** ✅ (threshold dependent)

**Reasoning**:
- Matches semantically similar questions even with different wording
- GQA has question patterns that repeat:
  - "What is on X?" → matches "What is on Y?"
  - "Is the X on the left or right?" → similar structure
  - "Who is wearing X?" → common pattern
- **Lower thresholds** (0.5) = more aggressive matching = **higher hit rate** but **lower accuracy**
- **Higher thresholds** (0.9) = stricter matching = **lower hit rate** but **higher accuracy**

**Current Results**:
```
qwen-semantic-0.5: 993/1024 hits (96.97%) - Accuracy: 1.07%
internvl-semantic-0.5: 993/1024 hits (96.97%) - Accuracy: 0.39%
```

**Critical Issue**: ⚠️ **Accuracy collapsed to ~1%** with semantic caching!

**Analysis**: 
- Cache is matching questions with similar patterns but **different answers**
- Example: "What is on the shelf?" → cached as "The shelf has books on it." → reused for "What is on the bike?" (WRONG!)
- Semantic similarity at 0.5 threshold is TOO AGGRESSIVE for GQA
- Questions have similar syntactic structure but require image-specific answers

**Expected Threshold Behavior**:
- **0.5-0.6**: 90-97% hit rate, **<5% accuracy** (over-caching)
- **0.7-0.8**: 40-70% hit rate, **10-20% accuracy** (still problematic)
- **0.9**: 10-30% hit rate, **closer to baseline accuracy** (conservative)

**Latency**: Cache hits = 0ms, misses = ~100-290ms → **30x speedup** when hitting

---

### 4. **Embedding Cache** (qwen-embed-p0.5-v0.5 to p0.9-v0.9)
**Expected Hit Rate**: **Variable** (5-60% depending on p/v balance)

**Reasoning**:
- Combines prompt embedding (p) + vision embedding (v) similarity
- **Prompt embedding (p)**: Text similarity (like semantic cache)
- **Vision embedding (v)**: CLIP image similarity
  - High v threshold (0.9) = only similar images match (rare in GQA)
  - Low v threshold (0.5) = broader image matching
- **Both must exceed thresholds** → harder to hit than text-only

**Expected Configurations**:
- **p0.9-v0.9**: <10% hit rate, baseline accuracy (very strict)
- **p0.7-v0.7**: 20-40% hit rate, degraded accuracy (moderate)
- **p0.5-v0.5**: 50-70% hit rate, **poor accuracy** (too aggressive)

**Accuracy Impact**: Similar problem to semantic text cache - matching similar questions/images but wrong answers

---

## Key Insights for GQA Scene Description

### Why Caching is Problematic:
1. **Question-specific answers**: "What is on the shelf?" vs "What is on the bike?" are semantically similar but have completely different correct answers
2. **Image-specific content**: Even visually similar scenes have different objects
3. **No duplicate queries**: Real-world GQA usage has diverse, non-repeating questions
4. **Strict evaluation**: GQA expects exact object/attribute identification

### Where Caching Could Work:
1. **Repeated identical questions**: System responding to same query multiple times (not in this dataset)
2. **Similar images with same question type**: If dataset had near-duplicate images
3. **Higher-level tasks**: Summarization, general description (not fine-grained VQA)

### Optimal Settings for GQA (if caching is required):
- **Semantic threshold ≥ 0.95**: Very conservative matching
- **Vision threshold ≥ 0.95**: Near-identical images only
- **Exact cache only**: Safe but minimal benefit

### Recommended Approach:
For GQA and similar VQA tasks, **response-level caching may harm accuracy more than it helps latency**. Better alternatives:
- **KV-cache at model level**: Share prefix computation (vLLM already does this)
- **Batch processing**: Process similar questions together
- **No semantic caching**: Only cache exact duplicates (rare but safe)

---

## Expected Final Results Summary

| Technique | Hit Rate | Accuracy | Latency Speedup | Verdict |
|-----------|----------|----------|-----------------|---------|
| Exact only | ~0% | Baseline (~12%) | None | ✅ Safe but useless |
| Fusion only | ~0% | Baseline | None | ❌ Ineffective |
| Semantic 0.5 | 97% | **1%** ❌ | 30x | ❌ Destroys accuracy |
| Semantic 0.6 | 85% | **2-3%** | 30x | ❌ Still harmful |
| Semantic 0.7 | 60% | **5-8%** | 30x | ❌ Marginal |
| Semantic 0.8 | 30% | **8-10%** | 30x | ⚠️ Questionable |
| Semantic 0.9 | 10% | **~11%** | 30x | ⚠️ Slight benefit |
| Embed p0.9-v0.9 | <5% | Baseline | None | ✅ Safe but minimal |
| Embed p0.5-v0.5 | 60% | **<5%** ❌ | 30x | ❌ Harmful |

**Conclusion**: For GQA visual question answering, semantic/embedding caching trades massive accuracy loss for latency gains. Only conservative thresholds (≥0.9) or exact matching are viable.
