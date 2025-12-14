from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

from .kv_protocols import KVChunk


class KVStore:
    """Disk-backed KV chunk storage with optional size capping."""

    def __init__(
        self,
        root: str | Path = "experiment2/kv_chunks",
        *,
        max_bytes: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def _path(self, chunk_id: str) -> Path:
        return self.root / f"{chunk_id}.pkl"

    def _iter_chunk_files(self) -> list[Path]:
        return [path for path in self.root.glob("*.pkl") if path.is_file()]

    def _prune_if_necessary(self) -> None:
        if self.max_bytes is None:
            return
        files: list[tuple[float, int, Path]] = []
        total = 0
        for path in self._iter_chunk_files():
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((stat.st_mtime, stat.st_size, path))
            total += stat.st_size
        if total <= self.max_bytes:
            return
        files.sort(key=lambda entry: entry[0])
        pruned = 0
        for _, size, path in files:
            try:
                path.unlink(missing_ok=True)
                total -= size
                pruned += 1
            except OSError as exc:  # pragma: no cover - disk errors are environment-specific
                print(f"[warn] semantic-cache prune failed for {path.name}: {exc}")
            if total <= self.max_bytes:
                break
        if pruned:
            print(
                f"[info] pruned {pruned} cached chunks to honor a {self.max_bytes} byte disk budget."
            )

    def save(self, chunk: KVChunk) -> None:
        path = self._path(chunk.chunk_id)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", delete=False, dir=self.root, suffix=".tmp"
            ) as tmp_handle:
                tmp_path = Path(tmp_handle.name)
                pickle.dump(chunk.cpu(), tmp_handle)
                tmp_handle.flush()
                os.fsync(tmp_handle.fileno())
            os.replace(tmp_path, path)
            self._prune_if_necessary()
        except OSError as exc:  # pragma: no cover - disk errors are environment-specific
            print(f"[warn] semantic-cache store failed for {chunk.chunk_id}: {exc}")
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def load(self, chunk_id: str) -> KVChunk | None:
        path = self._path(chunk_id)
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            print(f"[warn] semantic-cache load failed for {chunk_id}: {exc}")
            path.unlink(missing_ok=True)
            return None
