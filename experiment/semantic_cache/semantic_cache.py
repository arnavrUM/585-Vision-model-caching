from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from .kv_adapter import VLLMEngineAdapter
from .kv_protocols import KVChunk
from .kv_store import KVStore
from .techniques import (
    EmbeddingCache,
    EmbeddingLayerConfig,
    EmbeddingMatch,
    ExactTextCache,
    SemanticTextCache,
    SemanticTextMatch,
)


@dataclass
class SemanticCacheConfig:
    similarity_threshold: float = 0.85
    max_cached_blocks: int | None = None
    index_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: str = "kv_chunks"
    text_cache_index: str = "text_index.json"
    embedding_layers: list[EmbeddingLayerConfig] = field(default_factory=list)


@dataclass
class CacheHit:
    chunk_id: str
    score: float
    source: str = "text"


class SemanticCache:
    """Semantic chunk reuse built on top of the vLLM execution engine."""

    def __init__(
        self,
        llm: Any,
        *,
        config: SemanticCacheConfig | None = None,
        index: SemanticTextCache | None = None,
        store: KVStore | None = None,
    ) -> None:
        self.config = config or SemanticCacheConfig()
        self.adapter = VLLMEngineAdapter(llm)
        self.semantic_text_cache = index or SemanticTextCache(
            encoder_name=self.config.index_encoder
        )
        self.store = store or KVStore(self.config.cache_dir)
        self.exact_cache = ExactTextCache(self.config.cache_dir, self.config.text_cache_index)
        self.embedding_cache: EmbeddingCache | None = (
            EmbeddingCache(self.config.embedding_layers) if self.config.embedding_layers else None
        )

    def _record_chunk(self, chunk_text: str, chunk: KVChunk) -> None:
        chunk.metadata.setdefault("text", chunk_text)
        self.semantic_text_cache.add(chunk.chunk_id, chunk_text)
        self.store.save(chunk)
        self.exact_cache.record(chunk_text, chunk.chunk_id)

    def _maybe_inject(
        self,
        request_id: str,
        match: SemanticTextMatch | EmbeddingMatch,
    ) -> bool:
        stored = self.store.load(match.chunk_id)
        if stored is None:
            return False
        return self.adapter.inject(request_id, stored)

    # ------------------------------------------------------------------ public
    def try_reuse(
        self,
        request_id: str,
        chunk_text: str,
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> CacheHit | None:
        normalized = self.exact_cache.normalize(chunk_text)
        if normalized:
            exact_hit = self._exact_text_match(request_id, normalized)
            if exact_hit:
                return exact_hit
        if self.embedding_cache and embeddings:
            embed_match = self.embedding_cache.search(embeddings)
            if embed_match is not None:
                injected = self._maybe_inject(request_id, embed_match)
                if injected:
                    return CacheHit(
                        chunk_id=embed_match.chunk_id,
                        score=embed_match.score,
                        source=f"embedding:{embed_match.layer}",
                    )

        match = self.semantic_text_cache.search(chunk_text)
        if match is None or match.score < self.config.similarity_threshold:
            return None
        injected = self._maybe_inject(request_id, match)
        if not injected:
            return None
        return CacheHit(chunk_id=match.chunk_id, score=match.score, source="text")

    def _exact_text_match(self, request_id: str, normalized: str) -> CacheHit | None:
        chunk_ids = self.exact_cache.candidates(normalized)
        for chunk_id in chunk_ids:
            stored = self.store.load(chunk_id)
            if stored is None:
                self.exact_cache.remove(normalized, chunk_id)
                continue
            injected = self.adapter.inject(request_id, stored)
            if injected:
                return CacheHit(chunk_id=chunk_id, score=1.0, source="text:exact")
        return None

    def add_observation(
        self,
        request_id: str,
        chunk_text: str,
        chunk_id: str | None = None,
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> KVChunk | None:
        chunk = self.adapter.capture(
            request_id=request_id,
            chunk_id=chunk_id or uuid.uuid4().hex,
            num_blocks=self.config.max_cached_blocks,
        )
        if chunk is None:
            return None
        self._record_chunk(chunk_text, chunk)
        if self.embedding_cache and embeddings:
            self.embedding_cache.add(chunk.chunk_id, embeddings)
            chunk.metadata.setdefault("embeddings", list(embeddings))
        return chunk
