from __future__ import annotations

import json
import re
from pathlib import Path


class ExactTextCache:
    """Caches normalized text hashes for instant exact-match lookups."""

    def __init__(self, cache_dir: str | Path, index_filename: str = "text_index.json") -> None:
        self.cache_dir = Path(cache_dir)
        self.index_path = self.cache_dir / index_filename
        self._space_re = re.compile(r"\s+")``
        self._index = self._load()

    # ------------------------------------------------------------------ utils
    def normalize(self, text: str | None) -> str:
        normalized = (text or "").strip().lower()
        if not normalized:
            return ""
        return self._space_re.sub(" ", normalized)

    def _load(self) -> dict[str, list[str]]:
        if not self.index_path.exists():
            return {}
        try:
            data = json.loads(self.index_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        fixed: dict[str, list[str]] = {}
        for key, chunk_ids in data.items():
            if not isinstance(key, str):
                continue
            filtered = [chunk_id for chunk_id in chunk_ids if isinstance(chunk_id, str)]
            if filtered:
                fixed[key] = filtered
        return fixed

    def _persist(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: ids for key, ids in self._index.items() if ids}
        self.index_path.write_text(json.dumps(payload))

    # ----------------------------------------------------------------- public
    def record(self, text: str, chunk_id: str) -> None:
        normalized = self.normalize(text)
        if not normalized:
            return
        entries = self._index.setdefault(normalized, [])
        if chunk_id in entries:
            return
        entries.append(chunk_id)
        self._persist()

    def candidates(self, normalized: str) -> list[str]:
        return list(self._index.get(normalized, ()))

    def remove(self, normalized: str, chunk_id: str) -> None:
        entries = self._index.get(normalized)
        if not entries:
            return
        try:
            entries.remove(chunk_id)
        except ValueError:
            return
        if not entries:
            self._index.pop(normalized, None)
        self._persist()
