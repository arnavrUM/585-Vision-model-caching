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
    index_encoder_device: str = "cuda"
    cache_dir: str = "experiment2/kv_chunks"
    text_cache_index: str = "text_index.json"
    enable_fusion_cache: bool = False
    fusion_cache_dir: str = "experiment2/fusion_chunks"
    embedding_layers: list[EmbeddingLayerConfig] = field(default_factory=list)
    dry_run: bool = False
    enable_semantic_text_cache: bool = True
    enable_exact_text_cache: bool = True


@dataclass
class CacheHit:
    chunk_id: str
    score: float
    source: str = "text"


@dataclass
class ReuseReport:
    hit: CacheHit | None
    statuses: dict[str, str]
    response: str | None = None


class SemanticCache:
    """Semantic chunk reuse built on top of the vLLM execution engine."""

    requires_llm_request = True

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
                encoder_name=self.config.index_encoder,
                device=self.config.index_encoder_device,
            )
        else:
            self.semantic_text_cache = None
        self.store = store or KVStore(self.config.cache_dir)
        print(f"[DEBUG] SemanticCache.__init__: cache_dir={self.config.cache_dir}, exact_cache enabled={self.config.enable_exact_text_cache}")
        self.exact_cache: ExactTextCache | None = None
        if self.config.enable_exact_text_cache:
            self.exact_cache = ExactTextCache(self.config.cache_dir, self.config.text_cache_index)
            print(f"[DEBUG] ExactTextCache created: index_path={self.exact_cache.index_path}, existing_entries={len(self.exact_cache._index)}")
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
            print(f"[DEBUG] _record_chunk: Added to semantic_text_cache, chunk_id={chunk.chunk_id}, chunk_text='{chunk_text[:50]}...', cache_size={len(self.semantic_text_cache.ids)}")
        self.store.save(chunk)
        if self.exact_cache:
            self.exact_cache.record(chunk_text, chunk.chunk_id)
            normalized = self.exact_cache.normalize(chunk_text)
            candidates = self.exact_cache.candidates(normalized)
            print(f"[DEBUG] _record_chunk: Added to exact_cache, chunk_id={chunk.chunk_id}, normalized='{normalized[:50]}...', total_candidates_for_this_text={len(candidates)}")
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

        if normalized:
            exact_hit = self._exact_text_match(request_id, normalized)
            if exact_hit:
                statuses["exact_text"] = "hit"
                statuses["semantic_text"] = "skip"
                for key in list(statuses):
                    if key.startswith("embedding:"):
                        statuses[key] = "skip"
                # Note: _exact_text_match returns hit even if injection fails (for cache hit rate tracking)
                # We mark kv_cache as miss if injection failed, but still count as cache hit
                candidates = self.exact_cache.candidates(normalized) if self.exact_cache else []
                print(f"[DEBUG] exact_text: HIT for normalized='{normalized[:50]}...' (candidates: {len(candidates)})")
                return ReuseReport(hit=exact_hit, statuses=dict(statuses))
            statuses["exact_text"] = "miss"
            candidates = self.exact_cache.candidates(normalized) if self.exact_cache else []
            print(f"[DEBUG] exact_text: MISS for normalized='{normalized[:50]}...' (candidates: {len(candidates)})")

        if self.embedding_cache:
            if embeddings:
                print(f"[DEBUG] embedding_cache: checking embeddings for layers: {list(embeddings.keys())}")
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
                        print(f"[DEBUG] embedding_cache: HIT for layer={embed_match.layer}, score={embed_match.score:.4f}")
                        return ReuseReport(hit=hit, statuses=dict(statuses))
                    # Match found but injection failed - still count as hit for statistics
                    statuses["kv_cache"] = "miss"  # Injection failed
                    hit = CacheHit(
                        chunk_id=embed_match.chunk_id,
                        score=embed_match.score,
                        source=f"embedding:{embed_match.layer}",
                    )
                    print(f"[DEBUG] embedding_cache: HIT (match found, injection failed) for layer={embed_match.layer}, score={embed_match.score:.4f}")
                    return ReuseReport(hit=hit, statuses=dict(statuses))
            else:
                print(f"[DEBUG] embedding_cache: no embeddings provided (expected layers: {list(self.embedding_cache.layers.keys())})")
                for layer_name in self.embedding_cache.layers:
                    statuses[f"embedding:{layer_name}"] = "skip"

        if not self.semantic_text_cache:
            return ReuseReport(hit=None, statuses=dict(statuses))
        match = self.semantic_text_cache.search(chunk_text)
        if match is None:
            statuses["semantic_text"] = "miss"
            print(f"[DEBUG] semantic_text: no match found for chunk_text='{chunk_text[:50]}...'")
            return ReuseReport(hit=None, statuses=dict(statuses))
        # Print similarity score even if below threshold
        print(f"[DEBUG] semantic_text: score={match.score:.4f} (threshold={self.config.similarity_threshold:.4f}) for chunk_text='{chunk_text[:50]}...'")
        if match.score < self.config.similarity_threshold:
            statuses["semantic_text"] = "miss"
            return ReuseReport(hit=None, statuses=dict(statuses))
        injected = self._maybe_inject(request_id, match)
        statuses["semantic_text"] = "hit"
        # Count as hit even if injection failed (for cache hit rate tracking)
        hit = CacheHit(chunk_id=match.chunk_id, score=match.score, source="text")
        if injected:
            statuses["kv_cache"] = "hit"
        else:
            statuses["kv_cache"] = "miss"  # Injection failed but match found
            print(f"[DEBUG] semantic_text: HIT (match found, injection failed) for chunk_text='{chunk_text[:50]}...'")
        return ReuseReport(hit=hit, statuses=dict(statuses))

    def _exact_text_match(self, request_id: str, normalized: str) -> CacheHit | None:
        if not self.exact_cache:
            return None
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
                # Still return hit even if injection failed (for cache hit rate tracking)
                return CacheHit(chunk_id=chunk_id, score=1.0, source="text:exact")
            if injected:
                return CacheHit(chunk_id=chunk_id, score=1.0, source="text:exact")
            # Injection failed but match found - still return hit for statistics
            return CacheHit(chunk_id=chunk_id, score=1.0, source="text:exact")
        return None

    def add_observation(
        self,
        request_id: str,
        chunk_text: str,
        chunk_id: str | None = None,
        embeddings: dict[str, np.ndarray] | None = None,
        metadata: dict[str, Any] | None = None,
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
                print(f"[DEBUG] add_observation: chunk is None for request_id={request_id}")
                return
            print(f"[DEBUG] add_observation: Capturing chunk_id={chunk.chunk_id} for chunk_text='{chunk_text[:50]}...'")
            self._record_chunk(chunk_text, chunk)
            self._capture_fusion_state(request_id, chunk.chunk_id)
            if self.embedding_cache and embeddings:
                self.embedding_cache.add(chunk.chunk_id, embeddings)
                print(f"[DEBUG] add_observation: Added embeddings to embedding_cache for chunk_id={chunk.chunk_id}, layers={list(embeddings.keys())}")
                chunk.metadata.setdefault("embeddings", list(embeddings))

        self.adapter.register_on_free(request_id, _capture_and_store)
        return None

    def finalize_observation(
        self,
        request_id: str,
        *,
        response: str | None = None,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # The semantic cache captures KV blocks asynchronously; nothing to finalize here.
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
