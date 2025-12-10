# Data Characteristics for High Cache Hit Rates

This guide explains what types of data work best with the hierarchical caching system to achieve high cache hit rates.

## Overview

The system is designed for **"any prompt set that exhibits overlap"** (as stated in the README). High cache hit rates occur when there's redundancy, similarity, or repetition in your data.

---

## Key Characteristics for High Cache Hit Rates

### 1. **Data with Overlap/Redundancy** ⭐⭐⭐⭐⭐

**Best for**: Exact Text Cache (L0.5)

**What it means**: Same questions appearing multiple times

**Examples**:
- Customer support: "How do I reset my password?" (asked 100+ times)
- FAQ systems: Repeated questions from different users
- Batch processing: Same queries processed multiple times
- API endpoints: Identical requests

**Expected hit rate**: 30-80%+ (depending on redundancy)

**Configuration**:
```bash
--chunk-source question  # Use question text for caching
--enable-exact-text-cache  # Enable exact matching
```

---

### 2. **Semantically Similar Questions** ⭐⭐⭐⭐⭐

**Best for**: Semantic Text Cache (L1)

**What it means**: Questions that mean the same thing but are worded differently

**Examples**:
- "What color is the car?" vs "What's the color of the vehicle?"
- "How old is this person?" vs "What is the age of this individual?"
- "Is the dog sitting?" vs "Is the dog in a seated position?"
- "Describe the scene" vs "What do you see in this image?"

**Expected hit rate**: 20-60%+ (depending on similarity threshold)

**Configuration**:
```bash
--chunk-source semantic  # Use semantic program for caching
--similarity-threshold 0.8  # Adjust based on your data
```

**Threshold guidance**:
- **0.7-0.8**: More aggressive, catches more variations (higher hit rate, some false positives)
- **0.85-0.9**: Balanced (recommended)
- **0.9+**: Conservative, only very similar questions

---

### 3. **Repeated Images with Different Questions** ⭐⭐⭐⭐

**Best for**: Embedding Cache (L2) - Vision layer

**What it means**: Same images appearing with different questions

**Examples**:
- Product catalog: Same product image, different questions ("What's the price?", "Is it in stock?", "What color is it?")
- Medical imaging: Same X-ray, different diagnostic questions
- Security footage: Same frame, different analysis questions
- Educational content: Same diagram, different quiz questions

**Expected hit rate**: 15-50%+ (depending on image reuse)

**Configuration**:
```bash
--embedding-hook prompt_vision
--embedding-layer vision:512:0.85
--chunk-source question  # Questions differ, but images repeat
```

**Requirements**:
- Set `GQA_IMAGE_ROOT` or `LLAVA_IMAGE_ROOT` environment variable
- Image files must be accessible: `<image_id>.jpg`

---

### 4. **Structured/Grouped Queries** ⭐⭐⭐⭐

**Best for**: Semantic Text Cache (L1) with semantic programs

**What it means**: Questions that can be grouped into categories or have semantic structure

**Examples**:
- GQA benchmark: Questions with semantic programs like `query:color | filter:car`
- Database queries: Similar SQL-like structures
- API calls: Similar request patterns
- Form submissions: Similar field combinations

**Expected hit rate**: 25-70%+ (depending on grouping)

**Configuration**:
```bash
--chunk-source semantic  # Use semantic program
--chunk-source group  # Use grouping categories
--chunk-source combined  # Combine multiple fields
```

**GQA Example**:
```python
{
    "question": "What color is the car?",
    "semantic": [
        {"operation": "query", "argument": "color"},
        {"operation": "filter", "argument": "car"}
    ],
    "groups": {
        "global": "vehicles",
        "local": "cars"
    }
}
```

---

### 5. **Limited Question Space** ⭐⭐⭐⭐

**What it means**: Not too many unique questions (finite vocabulary)

**Examples**:
- Multiple choice questions: Limited set of questions
- Template-based queries: Questions follow patterns
- Domain-specific: Medical, legal, technical (limited terminology)
- Classification tasks: Fixed set of categories

**Expected hit rate**: 40-90%+ (depending on vocabulary size)

**Strategy**: 
- Smaller vocabulary → Higher hit rate
- Larger vocabulary → Lower hit rate (but still beneficial)

---

### 6. **Repeated Prompt Patterns** ⭐⭐⭐

**Best for**: Exact Text Cache + Semantic Cache

**What it means**: Similar prompt structures with different values

**Examples**:
- "Analyze image {image_id} for {feature}"
- "What is the {attribute} of the {object}?"
- "Is there a {object} in the {location}?"

**Expected hit rate**: 10-40%+ (depending on pattern reuse)

---

## Data Types That Work Well

### ✅ Excellent Candidates

1. **Question-Answering Benchmarks** (like GQA)
   - Structured questions
   - Semantic programs
   - Grouped by categories
   - **Expected hit rate**: 30-60%

2. **Customer Support Systems**
   - Repeated questions
   - Similar intents
   - FAQ-style queries
   - **Expected hit rate**: 50-80%+

3. **Product Catalogs**
   - Same images, different questions
   - Limited question types
   - Structured attributes
   - **Expected hit rate**: 40-70%

4. **Medical Imaging Analysis**
   - Same images, different diagnostic questions
   - Structured medical queries
   - Limited terminology
   - **Expected hit rate**: 30-60%

5. **Educational Content**
   - Same diagrams/images, different questions
   - Template-based quizzes
   - Limited question space
   - **Expected hit rate**: 40-70%

### ⚠️ Moderate Candidates

1. **Creative Writing Prompts**
   - High diversity
   - Low redundancy
   - **Expected hit rate**: 5-20%

2. **Open-Ended Conversations**
   - Unique contexts
   - Low repetition
   - **Expected hit rate**: 5-15%

3. **Research Queries**
   - Diverse topics
   - Unique questions
   - **Expected hit rate**: 10-30%

### ❌ Poor Candidates

1. **Completely Random Data**
   - No patterns
   - No redundancy
   - **Expected hit rate**: <5%

2. **One-Time Queries**
   - Each query is unique
   - No repetition
   - **Expected hit rate**: ~0%

---

## Optimizing Your Data for Caching

### Strategy 1: Increase Redundancy

**If you control data generation**:
- Reuse common questions
- Use templates for similar queries
- Batch similar requests together

**Example**: Instead of unique questions, use:
- "What color is {object}?" (template)
- "Is there a {object} in the {location}?" (template)

### Strategy 2: Use Semantic Grouping

**Add metadata to your data**:
```python
{
    "question": "What color is the car?",
    "groups": {
        "global": "vehicles",
        "local": "cars"
    },
    "semantic": [
        {"operation": "query", "argument": "color"},
        {"operation": "filter", "argument": "car"}
    ]
}
```

**Configuration**:
```bash
--chunk-source semantic  # Use semantic programs
# or
--chunk-source group  # Use grouping
```

### Strategy 3: Leverage Image Reuse

**If you have repeated images**:
- Use vision embedding cache
- Set up image root directory
- Enable vision hooks

**Configuration**:
```bash
export GQA_IMAGE_ROOT=/path/to/images
--embedding-hook prompt_vision
--embedding-layer vision:512:0.85
```

### Strategy 4: Adjust Thresholds

**For high redundancy data**:
```bash
--similarity-threshold 0.7  # More aggressive
```

**For diverse data**:
```bash
--similarity-threshold 0.9  # More conservative
```

---

## Measuring Cache Effectiveness

### Check Hit Rates

The system logs hit rates by source:
```
[sample_001] hit (text:exact) | latency=0.032s
[sample_002] hit (embedding:vision) | latency=0.045s
[sample_003] miss | latency=1.234s
```

### Analyze Results

After running, check:
- **Total hit rate**: Percentage of prompts that hit cache
- **Hit by source**: Which cache layer provided hits
- **Latency improvement**: Average latency reduction

**Example output**:
```
=== Experiment summary ===
Total prompts: 1024
Cache hits: 512 (50.0%)
Cache misses: 512 (50.0%)
Average latency: 0.623s
Average latency (hit): 0.032s
Average latency (miss): 1.214s
Hit rate by cache:
  - text:exact: 256 hits (25.0% of prompts)
  - embedding:vision: 128 hits (12.5% of prompts)
  - text: 128 hits (12.5% of prompts)
```

---

## Real-World Examples

### Example 1: E-commerce Product Q&A

**Data characteristics**:
- Same product images
- Limited question types: "What's the price?", "Is it in stock?", "What size is it?"
- High redundancy

**Configuration**:
```bash
--chunk-source question
--embedding-hook prompt_vision
--embedding-layer vision:512:0.85
--similarity-threshold 0.8
```

**Expected hit rate**: 60-80%

### Example 2: Medical Image Analysis

**Data characteristics**:
- Same X-rays/scans
- Structured diagnostic questions
- Limited medical terminology

**Configuration**:
```bash
--chunk-source semantic
--embedding-hook prompt_vision
--embedding-layer vision:512:0.9  # Higher threshold for medical accuracy
--similarity-threshold 0.85
```

**Expected hit rate**: 40-60%

### Example 3: Educational Quiz System

**Data characteristics**:
- Same diagrams/images
- Template-based questions
- Limited question space

**Configuration**:
```bash
--chunk-source question
--embedding-hook prompt_vision
--embedding-layer vision:512:0.85
--similarity-threshold 0.75  # More aggressive for educational content
```

**Expected hit rate**: 50-70%

---

## Summary: Data Characteristics Checklist

For high cache hit rates, your data should have:

- ✅ **Redundancy**: Same questions appearing multiple times
- ✅ **Semantic similarity**: Questions that mean the same thing
- ✅ **Image reuse**: Same images with different questions
- ✅ **Structured queries**: Questions with patterns or groups
- ✅ **Limited vocabulary**: Not too many unique questions
- ✅ **Metadata**: Semantic programs, groups, categories

**Best case scenario**: Data with all of the above → **70-90%+ hit rate**

**Typical scenario**: Data with 2-3 characteristics → **30-60% hit rate**

**Worst case**: Completely unique, random data → **<5% hit rate**

---

## Quick Reference

| Data Characteristic | Best Cache Layer | Expected Hit Rate |
|---------------------|------------------|-------------------|
| Exact duplicates | L0.5 (Exact) | 50-90% |
| Semantic similarity | L1 (Semantic) | 20-60% |
| Repeated images | L2 (Vision Embedding) | 15-50% |
| Structured queries | L1 (Semantic) | 25-70% |
| Limited vocabulary | All layers | 40-90% |
| High diversity | L1 (Semantic, low threshold) | 5-20% |

The system is most effective when your data exhibits **overlap, redundancy, or semantic similarity**. The more repetition and structure, the higher your cache hit rates will be!

