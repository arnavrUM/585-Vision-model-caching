from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from .kv_adapter import VLLMEngineAdapter
from .kv_protocols import KVChunk
from .kv_store import KVStore
from .semantic_index import SemanticIndex, SemanticMatch


@dataclass
class SemanticCacheConfig:
    similarity_threshold: float = 0.85
    max_cached_blocks: int | None = None
    index_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: str = "kv_chunks"


@dataclass
class CacheHit:
    chunk_id: str
    score: float


class SemanticCache:
    """Semantic chunk reuse built on top of the vLLM execution engine."""

    def __init__(
        self,
        llm: Any,
        *,
        config: SemanticCacheConfig | None = None,
        index: SemanticIndex | None = None,
        store: KVStore | None = None,
    ) -> None:
        self.config = config or SemanticCacheConfig()
        self.adapter = VLLMEngineAdapter(llm)
        self.index = index or SemanticIndex(encoder_name=self.config.index_encoder)
        self.store = store or KVStore(self.config.cache_dir)

    # ----------------------------------------------------------------- helpers
    def _record_chunk(self, chunk_text: str, chunk: KVChunk) -> None:
        chunk.metadata.setdefault("text", chunk_text)
        self.index.add(chunk.chunk_id, chunk_text)
        self.store.save(chunk)

    def _maybe_inject(self, request_id: str, match: SemanticMatch) -> bool:
        stored = self.store.load(match.chunk_id)
        if stored is None:
            return False
        return self.adapter.inject(request_id, stored)

    # ------------------------------------------------------------------ public
    def try_reuse(self, request_id: str, chunk_text: str) -> CacheHit | None:
        match = self.index.search(chunk_text)
        if match is None or match.score < self.config.similarity_threshold:
            return None
        injected = self._maybe_inject(request_id, match)
        if not injected:
            return None
        return CacheHit(chunk_id=match.chunk_id, score=match.score)

    def add_observation(
        self,
        request_id: str,
        chunk_text: str,
        chunk_id: str | None = None,
    ) -> KVChunk | None:
        chunk = self.adapter.capture(
            request_id=request_id,
            chunk_id=chunk_id or uuid.uuid4().hex,
            num_blocks=self.config.max_cached_blocks,
        )
        if chunk is None:
            return None
        self._record_chunk(chunk_text, chunk)
        return chunk
