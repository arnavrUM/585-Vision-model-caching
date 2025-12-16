from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class KVChunk:
    """Serializable representation of cached KV blocks."""

    chunk_id: str
    block_ids: list[int]
    tensors: dict[str, torch.Tensor]
    num_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def cpu(self) -> "KVChunk":
        """Return a CPU copy to make serialization cheaper."""
        cpu_tensors = {name: tensor.cpu() for name, tensor in self.tensors.items()}
        return KVChunk(
            chunk_id=self.chunk_id,
            block_ids=list(self.block_ids),
            tensors=cpu_tensors,
            num_tokens=self.num_tokens,
            metadata=dict(self.metadata),
        )
