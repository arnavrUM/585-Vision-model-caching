from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from .kv_adapter import VLLMEngineAdapter
from .kv_protocols import KVChunk
from .kv_store import KVStore
from .techniques import (
    EmbeddingCache,
    EmbeddingLayerConfig,
    EmbeddingMatch,
    ExactTextCache,
    FusionCache,
    FusionProvider,
    NullFusionProvider,
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
    enable_fusion_cache: bool = False
    fusion_cache_dir: str = "fusion_chunks"
    embedding_layers: list[EmbeddingLayerConfig] = field(default_factory=list)
    dry_run: bool = False
    enable_semantic_text_cache: bool = True


@dataclass
class CacheHit:
    chunk_id: str
    score: float
    source: str = "text"


@dataclass
class ReuseReport:
    hit: CacheHit | None
    statuses: dict[str, str]


class SemanticCache:
    """Semantic chunk reuse built on top of the vLLM execution engine."""

    def __init__(
        self,
        llm: Any,
        *,
        config: SemanticCacheConfig | None = None,
        index: SemanticTextCache | None = None,
        store: KVStore | None = None,
        fusion_cache: FusionCache | None = None,
        fusion_provider: FusionProvider | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or SemanticCacheConfig()
        self.adapter = VLLMEngineAdapter(llm)
        if self.config.enable_semantic_text_cache:
            self.semantic_text_cache = index or SemanticTextCache(
                encoder_name=self.config.index_encoder
            )
        else:
            self.semantic_text_cache = None
        self.store = store or KVStore(self.config.cache_dir)
        self.exact_cache = ExactTextCache(self.config.cache_dir, self.config.text_cache_index)
        self.embedding_cache: EmbeddingCache | None = (
            EmbeddingCache(self.config.embedding_layers) if self.config.embedding_layers else None
        )
        self.fusion_cache = fusion_cache
        if self.fusion_cache is None and self.config.enable_fusion_cache:
            provider = fusion_provider or NullFusionProvider()
            self.fusion_cache = FusionCache(provider, root=self.config.fusion_cache_dir)
        self._closed = False

    def _record_chunk(self, chunk_text: str, chunk: KVChunk) -> None:
        chunk.metadata.setdefault("text", chunk_text)
        if self.semantic_text_cache:
            self.semantic_text_cache.add(chunk.chunk_id, chunk_text)
        self.store.save(chunk)
        self.exact_cache.record(chunk_text, chunk.chunk_id)
    def _capture_fusion_state(self, request_id: str, chunk_id: str) -> None:
        if not self.fusion_cache:
            return
        self.fusion_cache.capture(llm=self.llm, request_id=request_id, chunk_id=chunk_id)

    def _maybe_inject(
        self,
        request_id: str,
        match: SemanticTextMatch | EmbeddingMatch,
    ) -> bool:
        if self.config.dry_run:
            return True
        stored = self.store.load(match.chunk_id)
        if stored is None:
            return False
        try:
            injected = self.adapter.inject(request_id, stored)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] semantic cache inject failed for {match.chunk_id}: {exc}")
            return False
        if not injected:
            return False
        self._inject_fusion_state(request_id, match.chunk_id)
        return True

    def _inject_fusion_state(self, request_id: str, chunk_id: str) -> None:
        if not self.fusion_cache:
            return
        state = self.fusion_cache.load(chunk_id)
        if state is None:
            return
        self.fusion_cache.inject(llm=self.llm, request_id=request_id, state=state)

    # ------------------------------------------------------------------ public
    def try_reuse(
        self,
        request_id: str,
        chunk_text: str,
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> ReuseReport:
        statuses: dict[str, str] = {"kv_cache": "miss"}
        normalized = self.exact_cache.normalize(chunk_text)
        if normalized:
            statuses["exact_text"] = "miss"
        else:
            statuses["exact_text"] = "skip"

        if self.semantic_text_cache:
            statuses["semantic_text"] = "miss"
        else:
            statuses["semantic_text"] = "skip"
        if self.embedding_cache:
            for layer_name in self.embedding_cache.layers:
                statuses[f"embedding:{layer_name}"] = "skip"

        if normalized:
            exact_hit = self._exact_text_match(request_id, normalized)
            if exact_hit:
                statuses["exact_text"] = "hit"
                statuses["semantic_text"] = "skip"
                for key in list(statuses):
                    if key.startswith("embedding:"):
                        statuses[key] = "skip"
                statuses["kv_cache"] = "hit"
                return ReuseReport(hit=exact_hit, statuses=dict(statuses))
            statuses["exact_text"] = "miss"

        if self.embedding_cache:
            if embeddings:
                for layer_name in embeddings:
                    key = f"embedding:{layer_name}"
                    if key in statuses:
                        statuses[key] = "miss"
                embed_match = self.embedding_cache.search(embeddings)
                if embed_match is not None:
                    key = f"embedding:{embed_match.layer}"
                    statuses[key] = "hit"
                    injected = self._maybe_inject(request_id, embed_match)
                    if injected:
                        statuses["semantic_text"] = "skip"
                        statuses["kv_cache"] = "hit"
                        hit = CacheHit(
                            chunk_id=embed_match.chunk_id,
                            score=embed_match.score,
                            source=f"embedding:{embed_match.layer}",
                        )
                        return ReuseReport(hit=hit, statuses=dict(statuses))
                    statuses[key] = "miss"
            else:
                for layer_name in self.embedding_cache.layers:
                    statuses[f"embedding:{layer_name}"] = "skip"

        if not self.semantic_text_cache:
            return ReuseReport(hit=None, statuses=dict(statuses))
        match = self.semantic_text_cache.search(chunk_text)
        if match is None or match.score < self.config.similarity_threshold:
            statuses["semantic_text"] = "miss"
            return ReuseReport(hit=None, statuses=dict(statuses))
        injected = self._maybe_inject(request_id, match)
        if not injected:
            statuses["semantic_text"] = "miss"
            return ReuseReport(hit=None, statuses=dict(statuses))
        statuses["semantic_text"] = "hit"
        statuses["kv_cache"] = "hit"
        hit = CacheHit(chunk_id=match.chunk_id, score=match.score, source="text")
        return ReuseReport(hit=hit, statuses=dict(statuses))

    def _exact_text_match(self, request_id: str, normalized: str) -> CacheHit | None:
        chunk_ids = self.exact_cache.candidates(normalized)
        for chunk_id in chunk_ids:
            stored = self.store.load(chunk_id)
            if stored is None:
                self.exact_cache.remove(normalized, chunk_id)
                continue
            if self.config.dry_run:
                return CacheHit(chunk_id=chunk_id, score=1.0, source="text:exact")
            try:
                injected = self.adapter.inject(request_id, stored)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] semantic cache inject failed for {chunk_id}: {exc}")
                continue
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
        if self.config.dry_run:
            stub = KVChunk(
                chunk_id=chunk_id or uuid.uuid4().hex,
                block_ids=[],
                tensors={},
                num_tokens=0,
            )
            self._record_chunk(chunk_text, stub)
            if self.embedding_cache and embeddings:
                self.embedding_cache.add(stub.chunk_id, embeddings)
                stub.metadata.setdefault("embeddings", list(embeddings))
            return stub

        def _capture_and_store() -> None:
            try:
                chunk = self.adapter.capture(
                    request_id=request_id,
                    chunk_id=chunk_id or uuid.uuid4().hex,
                    num_blocks=self.config.max_cached_blocks,
                )
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] semantic cache capture failed for {request_id}: {exc}")
                return
            if chunk is None:
                return
            self._record_chunk(chunk_text, chunk)
            self._capture_fusion_state(request_id, chunk.chunk_id)
            if self.embedding_cache and embeddings:
                self.embedding_cache.add(chunk.chunk_id, embeddings)
                chunk.metadata.setdefault("embeddings", list(embeddings))

        self.adapter.register_on_free(request_id, _capture_and_store)
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.adapter:
                self.adapter.close()
        except Exception:
            pass
        self.adapter = None
        self.llm = None
