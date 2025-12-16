from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class RouteRecord:
    """Serialized model response tied to a cached chunk."""

    chunk_id: str
    chunk_text: str
    response: str
    model_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ResponseStore:
    """Persistence helper for model-level shortcut records."""

    def __init__(self, root: str) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _record_path(self, chunk_id: str) -> Path:
        return self.root / f"{chunk_id}.json"

    def save(self, record: RouteRecord) -> None:
        path = self._record_path(record.chunk_id)
        payload = asdict(record)
        path.write_text(json.dumps(payload, ensure_ascii=False))

    def load(self, chunk_id: str) -> RouteRecord | None:
        path = self._record_path(chunk_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return RouteRecord(
            chunk_id=data["chunk_id"],
            chunk_text=data["chunk_text"],
            response=data["response"],
            model_name=data.get("model_name", ""),
            metadata=data.get("metadata", {}),
        )
