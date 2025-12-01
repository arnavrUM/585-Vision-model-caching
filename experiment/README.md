# Hierarchical Semantic Cache Experiment

This project builds a multi-level caching stack on top of vLLM to reduce redundant compute for the GQA benchmark (or any prompt set that exhibits overlap). vLLM already performs token-level prefix caching; we now add four higher-level layers:

1. **Exact normalized text cache** – stores a cheap hash of the normalized chunk text so identical questions/prompts can be served instantly without touching FAISS.
2. **Textual chunk similarity** – reuse KV pages when the first `chunk_window` characters line up semantically (via sentence-transformers + FAISS).
3. **Multimodal fusion cache** – optional layer that captures tensors emitted by fusion modules (e.g., image-text projectors) and re-injects them when the same sample recurs.
4. **Latent embedding similarity** – reuse when deeper embeddings (e.g., pooled vision encoder features, prompt embeddings) match above a configurable threshold.

Each level is optional and can be extended independently, giving you a hierarchical cache that escalates from cheap heuristics to richer embeddings.

## Installation (via `uv`)

[uv](https://github.com/astral-sh/uv) provides fast Python environment + dependency management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # install uv if you don't have it
uv venv                                         # creates .venv
source .venv/bin/activate
uv pip install -r requirements.txt              # pulls vllm, sentence-transformers, faiss-cpu, etc.
```

## Repository Layout

- `experiment/test_vllm.py` – CLI orchestrating dataset prep, cache configuration, and logging.
- `experiment/semantic_cache/` – modular cache layers:
  - `techniques/` – houses individual cache techniques (`exact_text_cache.py`, `semantic_text_cache.py`, `fusion_cache.py`, `embedding_cache.py`).
  - `kv_adapter.py` – bridges vLLM’s scheduler to capture/inject KV blocks.
  - `embedding_hooks.py` – pluggable hooks that extract model-specific embeddings (prompt text, vision encoder latents, etc.).
  - `kv_store.py` / `kv_protocols.py` – serialization helpers for cached blocks.

## Running the Hierarchical Cache

A minimal run (textual chunk cache only):

```bash
python experiment/test_vllm.py \
  --model facebook/opt-125m \
  --max-samples 64 \
  --chunk-source semantic
```

Enable latent-level reuse by stacking `--embedding-layer` and a corresponding `--embedding-hook`:

```bash
python experiment/test_vllm.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --max-samples 128 \
  --chunk-source group \
  --embedding-layer prompt:384:0.8 \
  --embedding-layer vision:512:0.82 \
  --embedding-hook prompt_vision
```

### Model Presets (InternVL3.5-2B included)

Some model families require a consistent bundle of settings (model id, `trust_remote_code`, embedding layers, etc.). Use `--preset` to hydrate those recommended values and then override whichever flags you care about:

```bash
python experiment/test_vllm.py \
  --preset internvl3.5-2b \
  --dataset gqa \
  --max-samples 64 \
  --cache-mode dry-run
```

The `internvl3.5-2b` preset loads `OpenGVLab/InternVL3_5-2B-Instruct`, enables `trust_remote_code`, and registers prompt + vision-id embeddings (`--embedding-layer prompt:384:0.8` and `vision:512:0.82`) via the `prompt_vision` hook. You can still add or override any flag explicitly on the CLI. The original `qwen3-vl-2b` configuration is also available via `--preset qwen3-vl-2b`.

> **Vision image roots**: the `vision` embedding layer now reads the *actual image file* (through a CLIP sentence-transformer). Point the hook at your image directory via one of `SEMANTIC_CACHE_IMAGE_ROOT`, `GQA_IMAGE_ROOT`, or `LLAVA_IMAGE_ROOT` (e.g., `export GQA_IMAGE_ROOT=/data/gqa/images`). The hook looks for `<image_id>.jpg/.png` under those roots and falls back gracefully if the file is missing.

### Ablation sweep helper

Need to benchmark each cache layer in isolation? `experiment/ablation_specs.json` enumerates all 32 combinations requested (2 models × {[exact only], [fusion only], semantic thresholds {0.5…0.9}, embedding threshold grid 3×3}). Launch the full sweep with:

```bash
./run_ablation.sh
```

The helper script just invokes `experiment/run_experiments.py` with that spec, logging results under `experiment_logs/ablation_results.csv` and dumping per-sample traces into `experiment_logs/ablation_samples/`. Edit the JSON if you want to tweak thresholds, sample counts, or add/remove experiments before re-running the script.
By default the script passes `--purge-cache-between-runs`, so each experiment wipes its `cache_dir`/`fusion_cache_dir` before executing, preventing cross-run reuse.

## Batch Sweeps & Logging

Use `experiment/run_experiments.py` to schedule many runs (different models/datasets/hyperparameters) and log the outcome to CSV/JSON automatically. Create a JSON spec that declares defaults plus a list of experiments:

```json
{
  "defaults": {
    "dataset_config": "val_balanced_instructions",
    "split": "val",
    "max_samples": 128,
    "chunk_source": "semantic"
  },
  "experiments": [
    {
      "name": "opt125m-semantic",
      "model": "facebook/opt-125m",
      "similarity_threshold": 0.8
    },
    {
      "name": "qwen-prompts",
      "model": "Qwen/Qwen2.5-VL-7B-Instruct",
      "embedding_layers": ["prompt:384:0.8", "vision:512:0.85"],
      "embedding_hook": "prompt_vision",
      "max_samples": 64
    },
    {
      "name": "internvl35-2b",
      "preset": "internvl3.5-2b",
      "max_samples": 64,
      "notes": "InternVL3.5-2B with prompt-semantic cache."
    }
  ]
}
```

Then run:

```bash
python experiment/run_experiments.py \
  --specs experiments.json \
  --log-file sweep_logs.csv \
  --samples-dir sweep_samples
```

Each experiment is executed sequentially, the aggregated metrics are appended to `sweep_logs.csv`, and (optionally) per-sample JSONL files land in `sweep_samples/`. Use `--resume` to skip experiment names that already appear in the log.

Just like the CLI, a spec entry can set `"preset": "internvl3.5-2b"` (or any other preset) to adopt the recommended defaults before applying per-experiment overrides.

### Key Flags

- `--preset` – load a prebundled configuration. Currently available: `qwen3-vl-2b`, `internvl3.5-2b`.
- `--model` – HuggingFace/vLLM identifier.
- `--dataset-config`, `--split`, `--max-samples`, `--shuffle-seed` – control the GQA subset.
- `--chunk-source` – textual key (`question`, `semantic`, `group`, `combined`, etc.).
- `--similarity-threshold` – minimum cosine similarity for the textual FAISS index.
- `--embedding-layer NAME:DIM[:THRESH]` – registers one or more latent layers (e.g., `vision:1024:0.9`). You can add multiple `--embedding-layer` flags.
- `--embedding-hook` – selects the hook that retrieves embeddings (`none`, `prompt`, `vision`, `prompt_vision`, or dotted path `package.module:Factory`).
- `--disable-semantic-cache` – completely bypass the semantic text cache (useful for ablations).
- `--max-cached-blocks`, `--cache-dir`, `--index-encoder` – storage and encoder tweaks.

Logs include hit provenance:

```
[05515938] hit (embedding:prompt) | latency=0.032s | answer match=True
```

so you can tell whether a reuse came from textual chunks or an embedding layer.

## Hierarchical Flow

1. **L0 (vLLM prefix cache)** – handled internally by vLLM (token-level).
2. **L0.5 (exact chunk cache)** – normalized text lookups stored in `text_index.json` hit instantly when the incoming chunk matches a previous one verbatim/modulo whitespace + casing.
3. **L1 (text chunk cache)** – `SemanticTextCache` searches normalized text / semantic programs.
4. **L1.5 (fusion cache)** – `FusionCache` captures multimodal fusion tensors (if your provider exposes them) and replays them before decoding.
5. **L2 (latent embedding cache)** – `EmbeddingCache` queries FAISS per registered layer. Hooks (e.g., prompt encoder, vision encoder tap) produce the embeddings on demand.

To enable the fusion layer, provide a `FusionProvider` implementation that knows how to read fusion tensors from your model (for capture) and how to write them back (for injection), then set `SemanticCacheConfig.enable_fusion_cache=True`. Captured states are persisted under `fusion_cache_dir`, mirroring how KV chunks live inside `kv_chunks/`.

The driver first tries L2 matches (highest cost but highest precision), falls back to L1, and only then pays the full generation cost. On misses, KV blocks (and any embeddings) are committed for future reuse.

## Extending the Hierarchy

- **Custom hooks**: subclass/compose an embedding hook that reaches into your model (vision encoder output, mid-layer hidden state, etc.) and expose it via `pkg.module:factory`. Return `{layer_name: np.ndarray}` matching the `--embedding-layer` specs.
- **Additional levels**: add new caches (e.g., compressed KV tiers, hidden-state caches) by following the pattern in `semantic_cache/` and wiring them in `test_vllm.py`.
- **Different datasets/models**: adjust the prompt template and dataset loader; the cache layers are agnostic.

Because each layer is modular, you can selectively enable/disable them, experiment with different thresholds, or integrate the package into other inference servers as a reusable hierarchical caching solution.
