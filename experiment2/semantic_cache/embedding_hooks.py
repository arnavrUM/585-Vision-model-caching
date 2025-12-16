from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
from PIL import Image

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


class _SentenceTransformerEncoder:
    """Thin wrapper so multiple hooks can share the same initialization logic."""

    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        if SentenceTransformer is None:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for semantic embedding hooks. "
                "Install it with `pip install sentence-transformers`."
            )
        self._encoder = SentenceTransformer(model_name, device=device)

    def encode(self, items: Iterable[Any]) -> np.ndarray:
        batch = [item for item in items if item is not None]
        if not batch:
            return np.zeros((0, 0), dtype="float32")
        vectors = self._encoder.encode(batch, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")


class PromptEmbeddingHook:
    """Encodes the textual prompt via sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str = "cpu",
    ) -> None:
        self._encoder = _SentenceTransformerEncoder(model_name, device=device)
        self.layer_name = "prompt"

    def __call__(self, *, llm: Any, sample: Any) -> dict[str, np.ndarray]:
        text = getattr(sample, "prompt", None) or getattr(sample, "chunk_text", "")
        if not text:
            return {}
        vec = self._encoder.encode([text])
        if vec.size == 0:
            return {}
        return {self.layer_name: np.asarray(vec[0], dtype="float32")}


def _default_image_roots() -> list[Path]:
    env_vars = ["SEMANTIC_CACHE_IMAGE_ROOT", "GQA_IMAGE_ROOT", "LLAVA_IMAGE_ROOT"]
    roots = []
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            roots.append(Path(value))
    return roots


def _candidate_image_paths(sample: Any, roots: list[Path]) -> list[Path]:
    metadata = getattr(sample, "metadata", {}) or {}
    extra = metadata.get("extra") or {}
    fields = ["image_path", "image", "path"]
    candidates: list[Path] = []
    for field in fields:
        value = extra.get(field)
        if not value:
            continue
        path = Path(value)
        candidates.append(path)
        for root in roots:
            candidates.append(root / value)
    image_id = getattr(sample, "image_id", "") or ""
    if image_id:
        suffixes = [".jpg", ".jpeg", ".png"]
        for root in roots:
            for suffix in suffixes:
                candidates.append(root / f"{image_id}{suffix}")
                if len(image_id) > 3:
                    shard = image_id[:3]
                    candidates.append(root / shard / f"{image_id}{suffix}")
    return candidates


class VisionImageEmbeddingHook:
    """Encodes the underlying image via a CLIP-style sentence-transformer."""

    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        *,
        device: str = "cpu",
        image_roots: list[str] | None = None,
    ) -> None:
        self.layer_name = "vision"
        self._encoder = _SentenceTransformerEncoder(model_name, device=device)
        roots = [Path(root) for root in (image_roots or []) if root]
        roots.extend(path for path in _default_image_roots() if path)
        self.image_roots = roots
        self._warned_missing = False

    def __call__(self, *, llm: Any, sample: Any) -> dict[str, np.ndarray]:
        image = self._load_image(sample)
        if image is None:
            return {}
        vec = self._encoder.encode([image])
        if vec.size == 0:
            return {}
        return {self.layer_name: np.asarray(vec[0], dtype="float32")}

    def _load_image(self, sample: Any) -> Image.Image | None:
        candidates = _candidate_image_paths(sample, self.image_roots)
        for path in candidates:
            if not path:
                continue
            try_path = path.expanduser()
            if not try_path.is_file():
                continue
            try:
                return Image.open(try_path).convert("RGB")
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] vision embedding hook failed to load {try_path}: {exc}")
                continue
        if not self._warned_missing:
            dataset_id = getattr(sample, "dataset_id", "")
            image_id = getattr(sample, "image_id", "")
            print(
                f"[warn] vision embedding hook could not resolve image file "
                f"(dataset_id={dataset_id}, image_id={image_id}). "
                "Set GQA_IMAGE_ROOT / LLAVA_IMAGE_ROOT to point at your image directory."
            )
            self._warned_missing = True
        return None


class PromptVisionEmbeddingHook:
    """Produces both prompt and vision/image-id embeddings in one pass."""

    def __init__(
        self,
        prompt_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vision_model: str = "clip-ViT-B-32",
        *,
        device: str = "cpu",
        image_roots: list[str] | None = None,
    ) -> None:
        self.prompt = PromptEmbeddingHook(model_name=prompt_model, device=device)
        self.vision = VisionImageEmbeddingHook(
            model_name=vision_model, device=device, image_roots=image_roots
        )

    def __call__(self, *, llm: Any, sample: Any) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {}
        payload.update(self.prompt(llm=llm, sample=sample))
        payload.update(self.vision(llm=llm, sample=sample))
        return payload


def load_embedding_hook(identifier: str) -> EmbeddingHook:
    """Load an embedding hook given a shorthand or dotted path."""

    if not identifier or identifier.lower() in {"none", "disable"}:
        return NullEmbeddingHook()
    lowered = identifier.lower()
    if lowered == "prompt":
        return PromptEmbeddingHook()
    if lowered in {"vision", "image", "image_id"}:
        return VisionImageEmbeddingHook()
    if lowered in {"prompt_vision", "prompt+vision", "prompt-image", "prompt_image"}:
        return PromptVisionEmbeddingHook()
    # Native VLM embeddings
    if lowered in {"native_encoder", "native-encoder", "encoder"}:
        from experiment2.semantic_cache.native_embedding_hooks import NativeEncoderEmbeddingHook
        return NativeEncoderEmbeddingHook()
    if lowered in {"native_text", "native-text", "text"}:
        from experiment2.semantic_cache.native_embedding_hooks import NativeTextEmbeddingHook
        return NativeTextEmbeddingHook()
    if lowered in {"native_text_vision", "native-text-vision", "text_vision", "text+vision"}:
        from experiment2.semantic_cache.native_embedding_hooks import NativeTextVisionEmbeddingHook
        return NativeTextVisionEmbeddingHook()
    if lowered in {"native_decoder", "native-decoder", "decoder"}:
        from experiment2.semantic_cache.native_embedding_hooks import NativeDecoderEmbeddingHook
        return NativeDecoderEmbeddingHook()
    if lowered in {"native", "native_both", "native-both", "encoder_decoder", "encoder-decoder"}:
        from experiment2.semantic_cache.native_embedding_hooks import NativeEncoderDecoderEmbeddingHook
        return NativeEncoderDecoderEmbeddingHook()
    if ":" in identifier:
        module_name, attr = identifier.rsplit(":", 1)
    elif "." in identifier:
        module_name, attr = identifier.rsplit(".", 1)
    else:
        raise ValueError(
            f"Invalid embedding hook '{identifier}'. "
            "Use 'prompt', 'vision', 'prompt_vision', 'none', or module:attr."
        )
    module = import_module(module_name)
    hook_factory = getattr(module, attr)
    hook = hook_factory() if callable(hook_factory) else hook_factory
    if not callable(hook):
        raise TypeError(f"Embedding hook '{identifier}' is not callable.")
    return hook  # type: ignore[return-value]
