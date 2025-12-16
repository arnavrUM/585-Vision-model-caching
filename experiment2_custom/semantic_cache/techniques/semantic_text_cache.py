from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "faiss is required for semantic caching experiments. "
        "Install it with `pip install faiss-cpu`."
    ) from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "sentence-transformers is required for semantic caching experiments. "
        "Install it with `pip install sentence-transformers`."
    ) from exc


_ENCODER_CACHE: dict[tuple[str, str], SentenceTransformer] = {}


def _resolve_encoder(name: str, device: str) -> SentenceTransformer:
    attempts = [device]
    normalized = device.lower()
    if normalized.startswith("cuda") and "cpu" not in attempts:
        attempts.append("cpu")
    last_error: Exception | None = None
    for target in attempts:
        key = (name, target)
        encoder = _ENCODER_CACHE.get(key)
        if encoder is not None:
            if target != device:
                _ENCODER_CACHE[(name, device)] = encoder
            return encoder
        try:
            encoder = SentenceTransformer(name, device=target)
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            continue
        _ENCODER_CACHE[key] = encoder
        if target != device:
            _ENCODER_CACHE[(name, device)] = encoder
            print(
                f"[warn] semantic text encoder could not use device '{device}'; falling back to '{target}'."
            )
        return encoder
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to initialize sentence-transformer '{name}' for device '{device}'.")


@dataclass
class SemanticTextMatch:
    chunk_id: str
    score: float


class SemanticTextCache:
    """FAISS-backed semantic text index used for chunk-level reuse."""

    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.encoder = _resolve_encoder(encoder_name, device)
        self.dim = dim or self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.ids: list[str] = []

    def _encode(self, text: str | Iterable[str]) -> np.ndarray:
        sentences = [text] if isinstance(text, str) else list(text)
        embeddings = self.encoder.encode(sentences, normalize_embeddings=True)
        return np.asarray(embeddings, dtype="float32")

    def add(self, chunk_id: str, text: str) -> None:
        vec = self._encode(text)
        self.index.add(vec)
        self.ids.append(chunk_id)

    def search(self, text: str, k: int = 1) -> SemanticTextMatch | None:
        if len(self.ids) == 0:
            print(f"[DEBUG] semantic_text_cache: empty cache (0 entries)")
            return None
        vec = self._encode(text)
        scores, indices = self.index.search(vec, k)
        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])
        if best_idx < 0 or best_idx >= len(self.ids):
            print(f"[DEBUG] semantic_text_cache: invalid index {best_idx} (total: {len(self.ids)})")
            return None
        print(f"[DEBUG] semantic_text_cache: found match with score={best_score:.4f} (cache size: {len(self.ids)})")
        return SemanticTextMatch(chunk_id=self.ids[best_idx], score=best_score)
