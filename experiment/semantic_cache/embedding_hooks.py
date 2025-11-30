from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]


class EmbeddingHook(Protocol):
    """Produces latent embeddings for a request/sample before generation."""

    def __call__(
        self,
        *,
        llm: Any,
        sample: Any,
    ) -> dict[str, np.ndarray]:
        ...


class NullEmbeddingHook:
    """Fallback hook that does nothing."""

    def __call__(self, *, llm: Any, sample: Any) -> dict[str, np.ndarray]:
        return {}


class PromptEmbeddingHook:
    """Encodes the textual prompt via sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str = "cpu",
    ) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required for PromptEmbeddingHook. "
                "Install it with `pip install sentence-transformers`."
            )
        self._encoder = SentenceTransformer(model_name)
        self.layer_name = "prompt"

    def __call__(self, *, llm: Any, sample: Any) -> dict[str, np.ndarray]:
        text = getattr(sample, "prompt", None) or getattr(sample, "chunk_text", "")
        if not text:
            return {}
        vec = self._encoder.encode([text], normalize_embeddings=True)
        return {self.layer_name: np.asarray(vec[0], dtype="float32")}


def load_embedding_hook(identifier: str) -> EmbeddingHook:
    """Load an embedding hook given a shorthand or dotted path."""

    if not identifier or identifier.lower() in {"none", "disable"}:
        return NullEmbeddingHook()
    if identifier.lower() == "prompt":
        return PromptEmbeddingHook()
    if ":" in identifier:
        module_name, attr = identifier.rsplit(":", 1)
    elif "." in identifier:
        module_name, attr = identifier.rsplit(".", 1)
    else:
        raise ValueError(
            f"Invalid embedding hook '{identifier}'. Use 'prompt', 'none', or module:attr."
        )
    module = import_module(module_name)
    hook_factory = getattr(module, attr)
    hook = hook_factory() if callable(hook_factory) else hook_factory
    if not callable(hook):
        raise TypeError(f"Embedding hook '{identifier}' is not callable.")
    return hook  # type: ignore[return-value]
