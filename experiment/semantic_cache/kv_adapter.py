from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .kv_protocols import KVChunk


def _resolve_model_runner(llm: Any):
    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        raise RuntimeError("LLM.llm_engine is not initialized yet.")
    executor = getattr(engine, "model_executor", None)
    if executor is None:
        raise RuntimeError("Engine is missing model_executor.")
    driver_worker = getattr(executor, "driver_worker", executor)
    runner = getattr(driver_worker, "model_runner", None)
    if runner is None:
        raise RuntimeError("Failed to resolve GPUModelRunner from vLLM engine.")
    return engine, runner


@dataclass
class VLLMEngineAdapter:
    """Allows reading and writing KV pages for custom experiments."""

    llm: Any

    def __post_init__(self) -> None:
        self.engine, self.runner = _resolve_model_runner(self.llm)

    # ------------------------------------------------------------------ utils
    def _layer_kv_map(self) -> dict[str, torch.Tensor]:
        forward_context = getattr(
            getattr(self.runner, "compilation_config", None), "static_forward_context", {}
        )
        kv_map: dict[str, torch.Tensor] = {}
        for layer_name, attn in (forward_context or {}).items():
            cache = getattr(attn, "kv_cache", None)
            if cache:
                kv_map[layer_name] = cache[0]
        if not kv_map:
            for idx, tensor in enumerate(getattr(self.runner, "kv_caches", [])):
                kv_map[f"layer_{idx}"] = tensor
        if not kv_map:
            raise RuntimeError("No KV cache tensors discovered.")
        return kv_map

    def _select_blocks(self, tensor: torch.Tensor, block_ids: list[int]) -> torch.Tensor:
        block_index = torch.as_tensor(block_ids, device=tensor.device, dtype=torch.long)
        return tensor.index_select(0, block_index).detach().cpu().contiguous()

    def _scatter_blocks(
        self,
        tensor: torch.Tensor,
        block_ids: list[int],
        values: torch.Tensor,
    ) -> None:
        block_index = torch.as_tensor(block_ids, device=tensor.device, dtype=torch.long)
        tensor.index_copy_(0, block_index, values.to(tensor.device))

    # ------------------------------------------------------------------ public
    def capture(
        self,
        request_id: str,
        chunk_id: str,
        *,
        num_blocks: int | None = None,
    ) -> KVChunk | None:
        kv_manager = getattr(self.engine.scheduler, "kv_cache_manager", None)
        if kv_manager is None:
            raise RuntimeError("Scheduler has no kv_cache_manager (kv cache disabled?).")
        blocks = kv_manager.get_blocks(request_id)
        block_groups = blocks.get_block_ids(allow_none=True)
        if block_groups is None:
            return None
        block_ids = list(block_groups[0])
        if not block_ids:
            return None
        if num_blocks is not None:
            block_ids = block_ids[:num_blocks]

        tensors = {
            layer_name: self._select_blocks(tensor, block_ids)
            for layer_name, tensor in self._layer_kv_map().items()
        }
        block_size = getattr(self.engine.scheduler, "block_size", None)
        num_tokens = len(block_ids) * block_size if block_size else len(block_ids)
        return KVChunk(
            chunk_id=chunk_id,
            block_ids=list(block_ids),
            tensors=tensors,
            num_tokens=num_tokens,
        )

    def inject(self, request_id: str, chunk: KVChunk) -> bool:
        kv_manager = getattr(self.engine.scheduler, "kv_cache_manager", None)
        if kv_manager is None:
            raise RuntimeError("Scheduler has no kv_cache_manager.")
        blocks = kv_manager.get_blocks(request_id)
        block_groups = blocks.get_block_ids(allow_none=True)
        if block_groups is None:
            return False
        dst_block_ids = list(block_groups[0])
        if len(dst_block_ids) < len(chunk.block_ids):
            return False

        available_layers = self._layer_kv_map()
        for layer_name, tensor in available_layers.items():
            cached = chunk.tensors.get(layer_name)
            if cached is None:
                continue
            self._scatter_blocks(tensor, chunk.block_ids, cached)
        return True
