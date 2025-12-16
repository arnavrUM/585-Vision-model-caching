"""Storage for pre-extracted embeddings to avoid re-computation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _compute_cache_key(text: str, image_id: str | None) -> str:
    """Generate a deterministic key for text + image combination."""
    parts = [text]
    if image_id:
        parts.append(image_id)
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


class EmbeddingStore:
    """Caches extracted embeddings to avoid re-extraction overhead."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "embedding_cache_index.json"
        self.index: dict[str, dict[str, str]] = {}
        self._load_index()

    def _load_index(self) -> None:
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    self.index = json.load(f)
            except Exception:
                self.index = {}

    def _save_index(self) -> None:
        try:
            with open(self.index_path, "w") as f:
                json.dump(self.index, f, indent=2)
        except Exception as exc:
            print(f"[warn] failed to save embedding index: {exc}")

    def _embedding_path(self, layer_name: str, cache_key: str) -> Path:
        return self.cache_dir / f"emb_{layer_name}_{cache_key}.npy"

    def get(self, text: str, image_id: str | None = None) -> dict[str, np.ndarray] | None:
        """Retrieve cached embeddings for text+image combination."""
        cache_key = _compute_cache_key(text, image_id)
        if cache_key not in self.index:
            return None

        embeddings: dict[str, np.ndarray] = {}
        layer_files = self.index[cache_key]
        
        for layer_name, filename in layer_files.items():
            path = self.cache_dir / filename
            if not path.exists():
                continue
            try:
                embeddings[layer_name] = np.load(str(path))
            except Exception as exc:
                print(f"[warn] failed to load {layer_name} embedding: {exc}")
                continue
        
        return embeddings if embeddings else None

    def put(
        self,
        text: str,
        embeddings: dict[str, np.ndarray],
        image_id: str | None = None,
    ) -> None:
        """Store embeddings for text+image combination."""
        if not embeddings:
            return
        
        cache_key = _compute_cache_key(text, image_id)
        layer_files: dict[str, str] = {}
        
        for layer_name, embedding in embeddings.items():
            path = self._embedding_path(layer_name, cache_key)
            try:
                np.save(str(path), embedding)
                layer_files[layer_name] = path.name
            except Exception as exc:
                print(f"[warn] failed to save {layer_name} embedding: {exc}")
        
        if layer_files:
            self.index[cache_key] = layer_files
            self._save_index()

    def clear(self) -> None:
        """Clear all cached embeddings."""
        for cache_key in list(self.index.keys()):
            for filename in self.index[cache_key].values():
                path = self.cache_dir / filename
                if path.exists():
                    path.unlink()
        self.index.clear()
        self._save_index()
