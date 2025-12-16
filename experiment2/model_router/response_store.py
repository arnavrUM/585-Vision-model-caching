from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RouteRecord:
    """Serializable representation of a routed response."""

    chunk_id: str
    chunk_text: str
    response: str
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ResponseStore:
    """Disk-backed store for response-level routing."""

    def __init__(self, root: str | Path = "model_shortcuts") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "metadata.json"
        self._metadata: dict[str, dict[str, Any]] = {}
        self._load_metadata()

    def _record_path(self, chunk_id: str) -> Path:
        return self.root / f"{chunk_id}.pkl"

    def _load_metadata(self) -> None:
        if not self.meta_path.exists():
            return
        try:
            self._metadata = json.loads(self.meta_path.read_text())
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] response store metadata load failed: {exc}")
            self._metadata = {}

    def _persist_metadata(self) -> None:
        try:
            with self.meta_path.open("w", encoding="utf-8") as fh:
                json.dump(self._metadata, fh, indent=2)
        except OSError as exc:  # pragma: no cover - filesystem issues are environment-specific
            print(f"[warn] response store metadata save failed: {exc}")

    def save(self, record: RouteRecord) -> None:
        path = self._record_path(record.chunk_id)
        tmp_path: Path | None = None
        try:
            with open(path, "wb") as fh:
                pickle.dump(record, fh)
            self._metadata[record.chunk_id] = {
                "chunk_text": record.chunk_text[:120],
                "model_name": record.model_name,
                "metadata": record.metadata,
            }
            self._persist_metadata()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] response store save failed for {record.chunk_id}: {exc}")
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def load(self, chunk_id: str) -> RouteRecord | None:
        path = self._record_path(chunk_id)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            print(f"[warn] response store load failed for {chunk_id}: {exc}")
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            self._metadata.pop(chunk_id, None)
            self._persist_metadata()
            return None

    def clear(self) -> None:
        for path in self.root.glob("*.pkl"):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
        self._metadata = {}
        self._persist_metadata()

    def size_bytes(self) -> int:
        total = 0
        for path in self.root.glob("*.pkl"):
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total
