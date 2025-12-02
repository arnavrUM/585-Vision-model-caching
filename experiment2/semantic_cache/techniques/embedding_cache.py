from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "faiss is required for embedding-based caching. Install it with `pip install faiss-cpu`."
    ) from exc


@dataclass
class EmbeddingLayerConfig:
    """Configuration describing a latent embedding layer."""

    name: str
    dim: int
    similarity_threshold: float = 0.85


@dataclass
class EmbeddingMatch:
    layer: str
    chunk_id: str
    score: float


class _EmbeddingLayerIndex:
    def __init__(self, config: EmbeddingLayerConfig) -> None:
        self.config = config
        self.index = faiss.IndexFlatIP(config.dim)
        self.ids: list[str] = []

    def add(self, chunk_id: str, embedding: np.ndarray) -> None:
        vector = np.asarray(embedding, dtype="float32").reshape(1, -1)
        if vector.shape[1] != self.config.dim:
            raise ValueError(
                f"Embedding for layer '{self.config.name}' has dim {vector.shape[1]} "
                f"but expected {self.config.dim}."
            )
        faiss.normalize_L2(vector)
        self.index.add(vector)
        self.ids.append(chunk_id)

    def search(self, embedding: np.ndarray) -> EmbeddingMatch | None:
        if len(self.ids) == 0:
            return None
        vector = np.asarray(embedding, dtype="float32").reshape(1, -1)
        if vector.shape[1] != self.config.dim:
            raise ValueError(
                f"Embedding for layer '{self.config.name}' has dim {vector.shape[1]} "
                f"but expected {self.config.dim}."
            )
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, 1)
        score = float(scores[0][0])
        idx = int(indices[0][0])
        if idx < 0 or idx >= len(self.ids):
            return None
        if score < self.config.similarity_threshold:
            return None
        return EmbeddingMatch(layer=self.config.name, chunk_id=self.ids[idx], score=score)


class EmbeddingCache:
    """Maintains FAISS indices for multiple latent embedding layers."""

    def __init__(self, layers: Iterable[EmbeddingLayerConfig]) -> None:
        self.layers = {config.name: _EmbeddingLayerIndex(config) for config in layers}

    def add(self, chunk_id: str, embeddings: dict[str, np.ndarray]) -> None:
        if not embeddings:
            return
        for name, vector in embeddings.items():
            layer = self.layers.get(name)
            if layer is None or vector is None:
                continue
            layer.add(chunk_id, vector)

    def search(self, embeddings: dict[str, np.ndarray]) -> EmbeddingMatch | None:
        if not embeddings:
            return None
        best: EmbeddingMatch | None = None
        for name, vector in embeddings.items():
            layer = self.layers.get(name)
            if layer is None or vector is None:
                continue
            match = layer.search(vector)
            if match is None:
                continue
            if best is None or match.score > best.score:
                best = match
        return best
