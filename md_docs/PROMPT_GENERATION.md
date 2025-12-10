# Prompt Generation Guide

This guide explains where prompts come from and how they're constructed.

## Overview

Prompts are **dynamically generated** from dataset samples using a **prompt template**. The system:

1. Loads dataset samples (from GQA, LLaVA, or synthetic)
2. Extracts fields from each sample (question, image ID, etc.)
3. Fills in a prompt template with those fields
4. Creates the final prompt text

## Prompt Generation Flow

```
Dataset Sample
    ↓
format_prompt(template, sample)
    ↓
Final Prompt Text
```

### Step-by-Step Process

1. **Dataset Loading**: Samples are loaded from the dataset (GQA, LLaVA, or synthetic)
2. **Template Selection**: A prompt template is chosen (default or custom)
3. **Field Extraction**: Fields are extracted from each sample
4. **Template Filling**: The template is filled with sample values
5. **Prompt Creation**: The final prompt is created

**Code**: `experiment2/test_vllm.py` lines 435-462 (`build_prompts()`)

---

## Where Prompts Come From

### 1. Dataset Samples

The dataset provides the **raw data** that gets formatted into prompts:

**GQA Dataset** (from HuggingFace `lmms-lab/GQA`):
- Each sample contains: `question`, `imageId`, `answer`, `fullAnswer`, `groups`, `semantic`, etc.

**LLaVA Dataset**:
- Each sample contains: `conversations` (with question/answer pairs), `image`, etc.

**Synthetic Dataset**:
- Built-in test samples with predefined questions and answers

### 2. Prompt Template

The template defines the **structure** of the prompt. It uses Python string formatting with placeholders.

**Default Template** (`BASE_PROMPT_TEMPLATE`):

**Location**: `experiment2/model_presets.py` lines 7-13

```python
BASE_PROMPT_TEMPLATE = (
    "You are assisting with the GQA benchmark. "
    "Answer the question based on the referenced image.\n"
    "Image ID: {image_id}\n"
    "Question: {question}\n"
    "Answer:"
)
```

**Example Output**:
```
You are assisting with the GQA benchmark. Answer the question based on the referenced image.
Image ID: 2407890
Question: What color is the car?
Answer:
```

### 3. Template Filling Process

**Function**: `format_prompt(template, sample)` 

**Location**: `experiment2/test_vllm.py` lines 193-210

```python
def format_prompt(template: str, sample: dict) -> str:
    groups = sample.get("groups") or {}
    semantic = sample.get("semantic") or []
    semantic_program = " | ".join(
        f"{step.get('operation', '')}:{step.get('argument', '')}" for step in semantic
    )
    context = {
        "question": sample.get("question", ""),
        "answer": sample.get("answer", ""),
        "full_answer": sample.get("fullAnswer", ""),
        "image_id": sample.get("imageId", ""),
        "dataset_id": sample.get("id", ""),
        "global_group": groups.get("global", ""),
        "local_group": groups.get("local", ""),
        "semantic_str": sample.get("semanticStr", ""),
        "semantic_program": semantic_program,
    }
    return template.format_map(_SafeDict(context)).strip()
```

**Available Template Variables**:
- `{question}` - The question text from the dataset
- `{image_id}` - Image identifier
- `{answer}` - Short answer
- `{full_answer}` - Full answer text
- `{dataset_id}` - Sample ID
- `{global_group}` - Global grouping category
- `{local_group}` - Local grouping category
- `{semantic_str}` - Semantic string representation
- `{semantic_program}` - Formatted semantic program

---

## How to Change Prompts

### Method 1: Command Line (Easiest)

Use `--prompt-template` to provide a custom template:

```bash
python experiment2/test_vllm.py \
  --prompt-template "Question: {question}\nImage: {image_id}\nAnswer:" \
  ...
```

**Multi-line templates**:
```bash
python experiment2/test_vllm.py \
  --prompt-template "You are a helpful assistant.
Image ID: {image_id}
Question: {question}
Please answer:" \
  ...
```

### Method 2: Model Presets

Presets can include custom templates:

**InternVL Preset** (`experiment2/model_presets.py` lines 54-61):
```python
"prompt_template": (
    "<image>\n"
    "You are assisting with the GQA benchmark. "
    "Answer the question using the referenced image.\n"
    "Image ID: {image_id}\n"
    "Question: {question}\n"
    "Answer:"
)
```

**Usage**:
```bash
python experiment2/test_vllm.py --preset internvl3.5-2b ...
```

### Method 3: Experiment Spec JSON

```json
{
  "experiments": [
    {
      "name": "custom-prompt",
      "prompt_template": "Question: {question}\nAnswer:"
    }
  ]
}
```

### Method 4: Modify Code Default

Edit `experiment2/model_presets.py` line 7:

```python
BASE_PROMPT_TEMPLATE = (
    "Your custom prompt template here.\n"
    "Question: {question}\n"
    "Image: {image_id}\n"
    "Answer:"
)
```

---

## Example: Complete Prompt Generation

### Input: Dataset Sample

```python
{
    "id": "sample_001",
    "imageId": "2407890",
    "question": "What color is the car?",
    "answer": "red",
    "fullAnswer": "The car is red.",
    "groups": {
        "global": "vehicles",
        "local": "cars"
    }
}
```

### Template

```
You are assisting with the GQA benchmark. Answer the question based on the referenced image.
Image ID: {image_id}
Question: {question}
Answer:
```

### Output: Final Prompt

```
You are assisting with the GQA benchmark. Answer the question based on the referenced image.
Image ID: 2407890
Question: What color is the car?
Answer:
```

This prompt is then sent to the model for inference.

---

## Model-Specific Templates

Different models may require different prompt formats:

### Qwen Models

**Default template** (works for most Qwen models):
```
You are assisting with the GQA benchmark. Answer the question based on the referenced image.
Image ID: {image_id}
Question: {question}
Answer:
```

### InternVL Models

**Custom template** (includes `<image>` token):
```
<image>
You are assisting with the GQA benchmark. Answer the question using the referenced image.
Image ID: {image_id}
Question: {question}
Answer:
```

The `<image>` token tells InternVL where to insert the image in the conversation.

---

## Chunk Text vs Prompt Text

**Important distinction**:

- **Prompt Text**: The full text sent to the model (includes instructions, question, etc.)
- **Chunk Text**: The text used for cache matching (can be just the question, semantic program, etc.)

**Chunk text** is determined by `--chunk-source`:
- `question`: Uses only the question text
- `semantic`: Uses semantic program
- `group`: Uses grouping categories
- `combined`: Combines multiple fields

**Code**: `experiment2/test_vllm.py` lines 241-264 (`render_chunk_text()`)

**Example**:
- **Prompt**: Full formatted text with instructions
- **Chunk Text** (if `--chunk-source question`): Just "What color is the car?"

The chunk text is what gets cached and matched, while the prompt text is what the model sees.

---

## Custom Template Examples

### Example 1: Simple Q&A Format

```bash
--prompt-template "Q: {question}\nA:"
```

**Output**:
```
Q: What color is the car?
A:
```

### Example 2: Include Semantic Information

```bash
--prompt-template "Question: {question}
Semantic: {semantic_program}
Answer:"
```

**Output**:
```
Question: What color is the car?
Semantic: query:color | filter:car
Answer:
```

### Example 3: Include Grouping

```bash
--prompt-template "Category: {global_group} / {local_group}
Question: {question}
Answer:"
```

### Example 4: Minimal Template

```bash
--prompt-template "{question}"
```

**Output**: Just the question text

---

## Prompt Template Best Practices

1. **Include clear instructions**: Tell the model what to do
2. **Use model-specific tokens**: Some models need special tokens (e.g., `<image>` for InternVL)
3. **Keep it consistent**: Use the same template across experiments for fair comparison
4. **Test your template**: Make sure the model understands your format

---

## Summary

| Component | Source | Location |
|-----------|--------|----------|
| **Raw Data** | Dataset samples | HuggingFace or local files |
| **Template** | Code default or CLI argument | `model_presets.py` or `--prompt-template` |
| **Filling** | `format_prompt()` function | `test_vllm.py` line 193 |
| **Final Prompt** | Generated from template + sample | Created in `build_prompts()` |

**Key Points**:
- Prompts are **generated**, not stored
- They come from **dataset samples** + **prompt template**
- The template can be **customized** via CLI, presets, or code
- Different models may need **different templates**

The system is flexible - you can use any template format that works with your model!

