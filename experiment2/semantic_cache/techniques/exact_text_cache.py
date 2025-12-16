from __future__ import annotations

import json
import re
from pathlib import Path


class ExactTextCache:
    """Cheap normalized text lookup to skip semantic search when exact matches recur."""

    def __init__(self, root: str = "experiment2/kv_chunks", index_path: str = "text_index.json") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = Path(index_path)
        if not self.index_path.is_absolute():
            self.index_path = self.root / index_path
        self._index = self._load_index()

    def _load_index(self) -> dict[str, list[str]]:
        if not self.index_path.exists():
            return {}
        try:
            with self.index_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
                if isinstance(payload, dict):
                    return {key: list(value or []) for key, value in payload.items()}
        except Exception:
            pass
        return {}

    def _persist(self) -> None:
        try:
            with self.index_path.open("w", encoding="utf-8") as fh:
                json.dump(self._index, fh, indent=2)
        except Exception as exc:  # pragma: no cover - filesystem issues are environment-specific
            print(f"[warn] exact text index persistence failed: {exc}")

    def normalize(self, text: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()
        return normalized

    def record(self, text: str, chunk_id: str) -> None:
        normalized = self.normalize(text)
        bucket = self._index.setdefault(normalized, [])
        if chunk_id not in bucket:
            bucket.append(chunk_id)
            self._persist()

    def candidates(self, normalized_text: str) -> list[str]:
        if not normalized_text:
            return []
        return list(self._index.get(normalized_text, []))

    def remove(self, normalized_text: str, chunk_id: str) -> None:
        if normalized_text not in self._index:
            return
        candidates = self._index[normalized_text]
        self._index[normalized_text] = [candidate for candidate in candidates if candidate != chunk_id]
        if not self._index[normalized_text]:
            del self._index[normalized_text]
        self._persist()
