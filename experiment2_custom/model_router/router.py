from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from experiment2.semantic_cache.semantic_cache import CacheHit, ReuseReport
from experiment2.semantic_cache.techniques import (
    EmbeddingCache,
    EmbeddingLayerConfig,
    EmbeddingMatch,
    ExactTextCache,
    SemanticTextCache,
    SemanticTextMatch,
)

from .response_store import ResponseStore, RouteRecord


@dataclass
class ModelRouterConfig:
    similarity_threshold: float = 0.85
    response_cache_dir: str = "model_shortcuts"
    text_cache_index: str = "text_index.json"
    index_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_encoder_device: str = "cuda"
    embedding_layers: list[EmbeddingLayerConfig] = field(default_factory=list)
    enable_semantic_text_cache: bool = True
    enable_exact_text_cache: bool = True


@dataclass
class _PendingObservation:
    chunk_id: str
    chunk_text: str
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRouter:
    """Multi-level semantic router that shortcuts at the response level."""

    requires_llm_request = False

    def __init__(self, llm: Any | None = None, *, config: ModelRouterConfig | None = None) -> None:
        self.llm = llm
        self.config = config or ModelRouterConfig()
        self.store = ResponseStore(self.config.response_cache_dir)
        if self.config.enable_semantic_text_cache:
            self.semantic_text_cache = SemanticTextCache(
                encoder_name=self.config.index_encoder,
                device=self.config.index_encoder_device,
            )
        else:
            self.semantic_text_cache = None
        self.exact_cache: ExactTextCache | None = None
        if self.config.enable_exact_text_cache:
            self.exact_cache = ExactTextCache(
                self.config.response_cache_dir,
                self.config.text_cache_index,
            )
        self.embedding_cache: EmbeddingCache | None = (
            EmbeddingCache(self.config.embedding_layers) if self.config.embedding_layers else None
        )
        self._pending: dict[str, _PendingObservation] = {}

    # ----------------------------------------------------------------- helpers
    def _load_record(self, chunk_id: str) -> RouteRecord | None:
        return self.store.load(chunk_id)

    def _build_report(
        self,
        *,
        chunk_id: str,
        score: float,
        source: str,
        response: str,
        statuses: dict[str, str],
    ) -> ReuseReport:
        statuses["kv_cache"] = "hit"
        statuses["model_router"] = "hit"
        hit = CacheHit(chunk_id=chunk_id, score=score, source=source)
        return ReuseReport(hit=hit, statuses=dict(statuses), response=response)

    def _record_entry(self, record: RouteRecord, embeddings: dict[str, np.ndarray]) -> None:
        self.store.save(record)
        if self.semantic_text_cache:
            self.semantic_text_cache.add(record.chunk_id, record.chunk_text)
        if self.exact_cache:
            self.exact_cache.record(record.chunk_text, record.chunk_id)
        if self.embedding_cache and embeddings:
            self.embedding_cache.add(record.chunk_id, embeddings)

    # ------------------------------------------------------------------ public
    def try_reuse(
        self,
        request_id: str,
        chunk_text: str,
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> ReuseReport:
        statuses: dict[str, str] = {"kv_cache": "miss", "model_router": "miss"}
        embeddings = embeddings or {}
        if self.exact_cache:
            normalized = self.exact_cache.normalize(chunk_text)
            statuses["exact_text"] = "miss" if normalized else "skip"
        else:
            normalized = ""
            statuses["exact_text"] = "skip"
        if self.semantic_text_cache:
            statuses["semantic_text"] = "miss"
        else:
            statuses["semantic_text"] = "skip"
        if self.embedding_cache:
            for layer_name in self.embedding_cache.layers:
                statuses[f"embedding:{layer_name}"] = "skip"
        else:
            for layer_name in embeddings:
                statuses[f"embedding:{layer_name}"] = "skip"

        if self.exact_cache and normalized:
            for chunk_id in self.exact_cache.candidates(normalized):
                record = self._load_record(chunk_id)
                if record is None:
                    self.exact_cache.remove(normalized, chunk_id)
                    continue
                statuses["exact_text"] = "hit"
                statuses["semantic_text"] = "skip"
                for key in list(statuses):
                    if key.startswith("embedding:"):
                        statuses[key] = "skip"
                return self._build_report(
                    chunk_id=record.chunk_id,
                    score=1.0,
                    source="text:exact",
                    response=record.response,
                    statuses=statuses,
                )

        if self.embedding_cache:
            if embeddings:
                for layer_name in embeddings:
                    key = f"embedding:{layer_name}"
                    if key in statuses:
                        statuses[key] = "miss"
                match = self.embedding_cache.search(embeddings)
            else:
                match = None
            if match:
                record = self._load_record(match.chunk_id)
                if record:
                    statuses[f"embedding:{match.layer}"] = "hit"
                    statuses["semantic_text"] = "skip"
                    return self._build_report(
                        chunk_id=match.chunk_id,
                        score=match.score,
                        source=f"embedding:{match.layer}",
                        response=record.response,
                        statuses=statuses,
                    )
                statuses[f"embedding:{match.layer}"] = "miss"

        if self.semantic_text_cache:
            match = self.semantic_text_cache.search(chunk_text)
            if match and match.score >= self.config.similarity_threshold:
                record = self._load_record(match.chunk_id)
                if record:
                    statuses["semantic_text"] = "hit"
                    return self._build_report(
                        chunk_id=match.chunk_id,
                        score=match.score,
                        source="text",
                        response=record.response,
                        statuses=statuses,
                    )
                statuses["semantic_text"] = "miss"

        return ReuseReport(hit=None, statuses=dict(statuses), response=None)

    def add_observation(
        self,
        request_id: str,
        chunk_text: str,
        chunk_id: str | None = None,
        embeddings: dict[str, np.ndarray] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if request_id in self._pending:
            return
        pending = _PendingObservation(
            chunk_id=chunk_id or uuid.uuid4().hex,
            chunk_text=chunk_text,
            embeddings=dict(embeddings or {}),
            metadata=dict(metadata or {}),
        )
        self._pending[request_id] = pending

    def finalize_observation(
        self,
        request_id: str,
        *,
        response: str,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pending = self._pending.pop(request_id, None)
        if pending is None:
            return
        record = RouteRecord(
            chunk_id=pending.chunk_id,
            chunk_text=pending.chunk_text,
            response=response,
            model_name=model_name or "",
            metadata={**pending.metadata, **(metadata or {})},
        )
        self._record_entry(record, embeddings=pending.embeddings)

    def close(self) -> None:
        self._pending.clear()
        self.llm = None
