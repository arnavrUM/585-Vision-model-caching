# Video Frames as Dataset: High Cache Hit Rate Use Case

Video frames are an **excellent candidate** for high cache hit rates due to temporal redundancy and visual similarity.

## Why Video Frames Work Well

### 1. **Temporal Redundancy** ⭐⭐⭐⭐⭐

**Consecutive frames are very similar**:
- Frame N and Frame N+1 share 90-99% of pixels
- Same objects, same scenes, minimal changes
- Questions about consecutive frames are semantically identical

**Example**:
- Frame 100: "What object is in the center?"
- Frame 101: "What object is in the center?" (same scene, 0.1s later)
- **Cache hit**: Vision embedding cache recognizes similar visual content

**Expected contribution**: 20-40% hit rate from temporal redundancy

### 2. **Repeated Scenes** ⭐⭐⭐⭐⭐

**Same scenes appear multiple times**:
- Long shots: Same scene for many frames
- Recurring locations: Same room, same street, etc.
- Static backgrounds: Same environment, different foregrounds

**Example**:
- Frames 50-200: Same room, different people
- Question: "What room is this?" → Same answer for 150 frames
- **Cache hit**: Exact text cache or semantic cache

**Expected contribution**: 30-50% hit rate from scene repetition

### 3. **Similar Visual Content** ⭐⭐⭐⭐⭐

**Vision embedding cache excels here**:
- Similar frames have similar visual embeddings
- Even if questions differ, visual similarity triggers cache
- Works across different frame numbers

**Example**:
- Frame 100: "What color is the car?" (red car visible)
- Frame 250: "Is there a red vehicle?" (same red car, different angle)
- **Cache hit**: Vision embedding similarity (0.92 cosine similarity)

**Expected contribution**: 25-45% hit rate from visual similarity

### 4. **Repeated Questions** ⭐⭐⭐⭐

**Same questions about different frames**:
- "What is happening in this frame?"
- "Who is in the scene?"
- "What objects are visible?"

**Example**:
- Frame 10: "What objects are visible?"
- Frame 50: "What objects are visible?" (different frame, same question)
- **Cache hit**: Exact text cache (if same question) or semantic cache

**Expected contribution**: 15-30% hit rate from question repetition

---

## Expected Overall Cache Hit Rate

### Conservative Estimate: **50-70%**

**Breakdown**:
- Temporal redundancy (consecutive frames): 20-40%
- Scene repetition (same locations): 30-50%
- Visual similarity (similar frames): 25-45%
- Question repetition: 15-30%

**Note**: These overlap, so total is not additive, but combined effect is strong.

### Best Case Scenario: **70-90%**

**When**:
- Long static shots (security cameras, surveillance)
- Repetitive content (sports, manufacturing)
- Limited question vocabulary
- High frame rate (more temporal redundancy)

### Worst Case Scenario: **30-50%**

**When**:
- Fast cuts, rapid scene changes
- Highly diverse questions
- Low frame rate
- Unique content per frame

---

## Optimal Configuration for Video Frames

### Recommended Setup

```bash
python experiment2/test_vllm.py \
  --dataset custom \
  --chunk-source combined \
  --embedding-hook prompt_vision \
  --embedding-layer vision:512:0.85 \
  --embedding-layer prompt:384:0.8 \
  --similarity-threshold 0.8 \
  --enable-exact-text-cache \
  --enable-semantic-text-cache
```

### Why This Configuration?

1. **`--chunk-source combined`**: Uses both semantic and grouping info
   - Captures frame relationships
   - Groups similar scenes

2. **`--embedding-hook prompt_vision`**: Enables both vision and prompt embeddings
   - Vision: Captures visual similarity between frames
   - Prompt: Captures question similarity

3. **`--embedding-layer vision:512:0.85`**: Vision embedding cache
   - **Critical for video frames** - similar frames trigger hits
   - Threshold 0.85 balances precision and recall

4. **`--similarity-threshold 0.8`**: Semantic text cache
   - Catches similar questions about similar frames
   - Moderate threshold for video content

5. **Enable all caches**: Maximize hit opportunities
   - Exact cache: Same questions
   - Semantic cache: Similar questions
   - Embedding cache: Similar frames

---

## Video Frame Dataset Format

### Recommended Structure

```python
{
    "id": "video_001_frame_0100",
    "imageId": "video_001_frame_0100",  # Frame identifier
    "question": "What objects are visible in this frame?",
    "answer": "car, person, building",
    "fullAnswer": "A red car, a person walking, and a building in the background.",
    "groups": {
        "global": "outdoor",  # Scene category
        "local": "street"     # Specific location
    },
    "semantic": [
        {"operation": "query", "argument": "objects"},
        {"operation": "filter", "argument": "visible"}
    ],
    "metadata": {
        "video_id": "video_001",
        "frame_number": 100,
        "timestamp": 3.33,  # seconds
        "scene_id": "scene_05"  # Scene grouping
    }
}
```

### Key Fields for Video Caching

1. **`imageId`**: Should be frame identifier
   - Format: `{video_id}_frame_{number}`
   - Used for vision embedding cache

2. **`groups`**: Scene/location grouping
   - `global`: Scene type (indoor/outdoor, day/night)
   - `local`: Specific location (room_1, street_corner)
   - Helps semantic cache group similar frames

3. **`metadata.scene_id`**: Scene grouping
   - Frames in same scene → high cache hit rate
   - Use for `--chunk-source group`

4. **`metadata.frame_number`**: Temporal information
   - Consecutive frames → temporal redundancy
   - Can be used for frame proximity matching

---

## Cache Layer Effectiveness for Video

### L0.5: Exact Text Cache

**Effectiveness**: ⭐⭐⭐⭐ (High)

**When it hits**:
- Same question about different frames
- "What objects are visible?" asked multiple times

**Expected hit rate**: 15-30%

### L1: Semantic Text Cache

**Effectiveness**: ⭐⭐⭐⭐⭐ (Very High)

**When it hits**:
- Similar questions about similar frames
- "What's in the frame?" vs "What objects are visible?"
- Questions about same scene with different wording

**Expected hit rate**: 25-40%

### L2: Vision Embedding Cache

**Effectiveness**: ⭐⭐⭐⭐⭐ (Very High - **Most Important**)

**When it hits**:
- Similar frames (even with different questions)
- Consecutive frames (temporal redundancy)
- Same scene, different angles
- Similar visual content

**Expected hit rate**: 30-50% (**Highest contributor**)

**Configuration**:
```bash
--embedding-hook prompt_vision
--embedding-layer vision:512:0.85  # Adjust threshold based on frame similarity
```

**Threshold guidance**:
- **0.8-0.85**: For similar frames (recommended)
- **0.85-0.9**: For very similar frames (consecutive frames)
- **0.9+**: For nearly identical frames (only exact matches)

### L1.5: Fusion Cache

**Effectiveness**: ⭐⭐⭐ (Moderate)

**When it hits**:
- Same frame processed multiple times
- Requires fusion provider implementation

**Expected hit rate**: 10-20%

---

## Strategies for Maximum Cache Hits

### Strategy 1: Frame Grouping

**Group frames by scene**:
```python
{
    "groups": {
        "global": "indoor",
        "local": "kitchen_scene_1"
    },
    "metadata": {
        "scene_id": "scene_05"
    }
}
```

**Configuration**:
```bash
--chunk-source group  # Use scene grouping for cache keys
```

**Benefit**: Frames in same scene share cache entries

### Strategy 2: Temporal Proximity

**Process consecutive frames together**:
- Frame 100, 101, 102 processed sequentially
- Frame 101 hits cache from Frame 100 (vision embedding)
- Frame 102 hits cache from Frame 101

**Benefit**: Temporal redundancy captured automatically

### Strategy 3: Question Templates

**Use structured questions**:
- Template: "What {attribute} of {object} is visible in frame {frame_number}?"
- Creates semantic similarity across frames

**Benefit**: Semantic cache catches template variations

### Strategy 4: Lower Vision Threshold

**For high temporal redundancy**:
```bash
--embedding-layer vision:512:0.8  # Lower threshold
```

**Benefit**: More aggressive matching for similar frames

**Trade-off**: May have some false positives, but higher hit rate

---

## Example: Video Analysis Pipeline

### Scenario: Security Camera Footage

**Characteristics**:
- 30 FPS video
- Static camera (same scene)
- Questions: "Who is in the frame?", "What objects are visible?"

**Expected Performance**:
- **Frame 0**: Cache miss (first frame)
- **Frame 1**: Cache hit (vision embedding, 0.95 similarity) ✅
- **Frame 2**: Cache hit (vision embedding, 0.96 similarity) ✅
- **Frame 30**: Cache hit (same scene, vision embedding) ✅
- **Frame 100**: Cache hit (same question, exact text cache) ✅

**Overall hit rate**: **70-85%**

### Configuration:
```bash
python experiment2/test_vllm.py \
  --dataset custom \
  --chunk-source combined \
  --embedding-hook prompt_vision \
  --embedding-layer vision:512:0.85 \
  --similarity-threshold 0.8 \
  --max-samples 1000
```

---

## Challenges and Solutions

### Challenge 1: Fast Scene Changes

**Problem**: Rapid cuts reduce temporal redundancy

**Solution**:
- Use scene detection to group frames
- Lower vision threshold: `--embedding-layer vision:512:0.8`
- Focus on semantic cache for question similarity

### Challenge 2: Unique Questions Per Frame

**Problem**: Each frame has different questions

**Solution**:
- Vision embedding cache still works (similar frames)
- Use question templates for semantic similarity
- Group by scene for better caching

### Challenge 3: Frame Storage

**Problem**: Need to store frame images for vision embedding

**Solution**:
```bash
export GQA_IMAGE_ROOT=/path/to/video/frames
# Structure: /path/to/video/frames/video_001_frame_0100.jpg
```

**Frame naming**: Match `imageId` from dataset

---

## Performance Expectations

### Typical Video Dataset

**Characteristics**:
- 1000 frames
- 10-20 unique scenes
- 5-10 question templates
- 30 FPS (high temporal redundancy)

**Expected Results**:
```
Total prompts: 1000
Cache hits: 650 (65.0%)
Cache misses: 350 (35.0%)
Average latency: 0.45s
Average latency (hit): 0.035s
Average latency (miss): 1.2s

Hit breakdown:
  - embedding:vision: 400 hits (40.0%)  ← Highest contributor
  - text:exact: 150 hits (15.0%)
  - text: 100 hits (10.0%)
```

### Best Case: Static Camera

**Characteristics**:
- Static security camera
- Same scene for entire video
- Limited question set

**Expected Results**:
```
Cache hits: 850 (85.0%)  ← Very high!
  - embedding:vision: 600 hits (60.0%)
  - text:exact: 200 hits (20.0%)
  - text: 50 hits (5.0%)
```

---

## Summary

### ✅ Video Frames Are Excellent for Caching

**Why**:
1. **Temporal redundancy**: Consecutive frames are very similar
2. **Scene repetition**: Same scenes appear multiple times
3. **Visual similarity**: Vision embedding cache is highly effective
4. **Question patterns**: Similar questions across frames

### Expected Hit Rates

- **Conservative**: 50-70%
- **Typical**: 60-80%
- **Best case** (static camera): 70-90%

### Key Configuration

**Most important**: Vision embedding cache
```bash
--embedding-hook prompt_vision
--embedding-layer vision:512:0.85
```

**Also enable**:
- Exact text cache (for repeated questions)
- Semantic text cache (for similar questions)
- Combined chunk source (for scene grouping)

### Bottom Line

**Yes, video frames should yield high cache hit rates!** The temporal redundancy and visual similarity make video frames one of the best use cases for this caching system. Expect **60-80% hit rates** in typical scenarios, potentially **80-90%** for static cameras or repetitive content.

