# Hierarchical Semantic Cache Experiment

This project builds a multi-level caching stack on top of vLLM to reduce redundant compute for the GQA benchmark (or any prompt set that exhibits overlap). vLLM already performs token-level prefix caching; we now add three higher-level layers:

1. **Exact normalized text cache** – stores a cheap hash of the normalized chunk text so identical questions/prompts can be served instantly without touching FAISS.
2. **Textual chunk similarity** – reuse KV pages when the first `chunk_window` characters line up semantically (via sentence-transformers + FAISS).
3. **Latent embedding similarity** – reuse when deeper embeddings (e.g., pooled vision encoder features, prompt embeddings) match above a configurable threshold.

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
  - `techniques/` – houses individual cache techniques (`exact_text_cache.py`, `semantic_text_cache.py`, `embedding_cache.py`).
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
  --embedding-hook prompt
```

### Key Flags

- `--model` – HuggingFace/vLLM identifier.
- `--dataset-config`, `--split`, `--max-samples`, `--shuffle-seed` – control the GQA subset.
- `--chunk-source` – textual key (`question`, `semantic`, `group`, `combined`, etc.).
- `--similarity-threshold` – minimum cosine similarity for the textual FAISS index.
- `--embedding-layer NAME:DIM[:THRESH]` – registers one or more latent layers (e.g., `vision:1024:0.9`). You can add multiple `--embedding-layer` flags.
- `--embedding-hook` – selects the hook that retrieves embeddings (`none`, `prompt`, or dotted path `package.module:Factory`).
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
4. **L2 (latent embedding cache)** – `EmbeddingCache` queries FAISS per registered layer. Hooks (e.g., prompt encoder, vision encoder tap) produce the embeddings on demand.

The driver first tries L2 matches (highest cost but highest precision), falls back to L1, and only then pays the full generation cost. On misses, KV blocks (and any embeddings) are committed for future reuse.

## Extending the Hierarchy

- **Custom hooks**: subclass/compose an embedding hook that reaches into your model (vision encoder output, mid-layer hidden state, etc.) and expose it via `pkg.module:factory`. Return `{layer_name: np.ndarray}` matching the `--embedding-layer` specs.
- **Additional levels**: add new caches (e.g., compressed KV tiers, hidden-state caches) by following the pattern in `semantic_cache/` and wiring them in `test_vllm.py`.
- **Different datasets/models**: adjust the prompt template and dataset loader; the cache layers are agnostic.

Because each layer is modular, you can selectively enable/disable them, experiment with different thresholds, or integrate the package into other inference servers as a reusable hierarchical caching solution.
