from __future__ import annotations

import pickle
from pathlib import Path

from .kv_protocols import KVChunk


class KVStore:
    """Disk-backed KV chunk storage."""

    def __init__(self, root: str | Path = "kv_chunks") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, chunk_id: str) -> Path:
        return self.root / f"{chunk_id}.pkl"

    def save(self, chunk: KVChunk) -> None:
        path = self._path(chunk.chunk_id)
        with path.open("wb") as f:
            pickle.dump(chunk.cpu(), f)

    def load(self, chunk_id: str) -> KVChunk | None:
        path = self._path(chunk_id)
        if not path.exists():
            return None
        with path.open("rb") as f:
            return pickle.load(f)
