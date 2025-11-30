from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch


@dataclass
class FusionState:
    """Serializable representation of multimodal fusion tensors."""

    chunk_id: str
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)

    def cpu(self) -> FusionState:
        cpu_tensors = {name: tensor.cpu() for name, tensor in self.tensors.items()}
        return FusionState(chunk_id=self.chunk_id, tensors=cpu_tensors, metadata=dict(self.metadata))


class FusionProvider(Protocol):
    """Captures and injects fusion tensors from/to the host LLM."""

    def capture(self, *, llm: Any, request_id: str, chunk_id: str) -> FusionState | None:
        ...

    def inject(self, *, llm: Any, request_id: str, state: FusionState) -> bool:
        ...


class NullFusionProvider:
    """No-op provider when the backend exposes no fusion state."""

    def capture(self, *, llm: Any, request_id: str, chunk_id: str) -> FusionState | None:  # noqa: D401
        return None

    def inject(self, *, llm: Any, request_id: str, state: FusionState) -> bool:  # noqa: D401
        return False


class FusionStateStore:
    """Disk-backed persistence for fusion states."""

    def __init__(self, root: str | Path = "fusion_chunks") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, chunk_id: str) -> Path:
        return self.root / f"{chunk_id}.fusion"

    def save(self, state: FusionState) -> None:
        path = self._path(state.chunk_id)
        with path.open("wb") as fh:
            pickle.dump(state.cpu(), fh)

    def load(self, chunk_id: str) -> FusionState | None:
        path = self._path(chunk_id)
        if not path.exists():
            return None
        with path.open("rb") as fh:
            return pickle.load(fh)


class FusionCache:
    """High-level helper that wraps a provider + on-disk store."""

    def __init__(
        self,
        provider: FusionProvider,
        *,
        root: str | Path = "fusion_chunks",
    ) -> None:
        self.provider = provider
        self.store = FusionStateStore(root)

    def capture(self, *, llm: Any, request_id: str, chunk_id: str) -> FusionState | None:
        state = self.provider.capture(llm=llm, request_id=request_id, chunk_id=chunk_id)
        if state is None:
            return None
        self.store.save(state)
        return state

    def load(self, chunk_id: str) -> FusionState | None:
        return self.store.load(chunk_id)

    def inject(self, *, llm: Any, request_id: str, state: FusionState) -> bool:
        return self.provider.inject(llm=llm, request_id=request_id, state=state)
