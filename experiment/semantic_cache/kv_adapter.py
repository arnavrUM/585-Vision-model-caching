from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .kv_protocols import KVChunk


def _try_resolve_engine_components(llm: Any):
    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        raise RuntimeError("LLM.llm_engine is not initialized yet.")
    scheduler = getattr(engine, "scheduler", None)
    engine_core_client = getattr(engine, "engine_core", None)
    if scheduler is None and engine_core_client is not None:
        scheduler = getattr(getattr(engine_core_client, "engine_core", None), "scheduler", None)
    if scheduler is None:
        raise RuntimeError("Failed to resolve Scheduler from vLLM engine.")
    executor = getattr(engine, "model_executor", None)
    if executor is None and engine_core_client is not None:
        executor = getattr(getattr(engine_core_client, "engine_core", None), "model_executor", None)
    if executor is None:
        raise RuntimeError("Engine is missing model_executor (multiprocessing disabled?).")
    driver_worker = getattr(executor, "driver_worker", executor)
    runner = getattr(driver_worker, "model_runner", None)
    if runner is None:
        worker = getattr(driver_worker, "worker", None)
        if worker is not None:
            runner = getattr(worker, "model_runner", None)
    if runner is None:
        raise RuntimeError("Failed to resolve GPUModelRunner from vLLM engine.")
    return engine, scheduler, runner


def _resolve_engine_components(llm: Any, retries: int = 10, delay: float = 1.0):
    last_error: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            return _try_resolve_engine_components(llm)
        except RuntimeError as exc:  # pragma: no cover - timing dependent
            last_error = exc
            time.sleep(delay)
    assert last_error is not None
    raise last_error


@dataclass
class VLLMEngineAdapter:
    """Allows reading and writing KV pages for custom experiments."""

    llm: Any

    def __post_init__(self) -> None:
        self.engine, self.scheduler, self.runner = _resolve_engine_components(self.llm)
        self._pending_callbacks: dict[str, list[Callable[[], None]]] = {}
        self._wrapped_free_request: Callable[[Any], Any] | None = None
        if not hasattr(self.scheduler, "_semantic_cache_wrapped"):
            original_free = self.scheduler._free_request

            def wrapped_free(request):
                callbacks = self._pending_callbacks.pop(request.request_id, [])
                for callback in callbacks:
                    try:
                        callback()
                    except Exception as exc:  # pragma: no cover - defensive
                        print(f"[warn] semantic-cache callback failed: {exc}")
                return original_free(request)

            self.scheduler._free_request = wrapped_free  # type: ignore[attr-defined]
            self.scheduler._semantic_cache_wrapped = True  # type: ignore[attr-defined]
            self._wrapped_free_request = wrapped_free
        else:
            original_free = getattr(self.scheduler, "_free_request")
        self._original_free_request = original_free

    # ------------------------------------------------------------------ utils
    def register_on_free(self, request_id: str, callback: Callable[[], None]) -> None:
        callbacks = self._pending_callbacks.setdefault(request_id, [])
        callbacks.append(callback)

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
        kv_manager = getattr(self.scheduler, "kv_cache_manager", None)
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
        block_size = getattr(self.scheduler, "block_size", None)
        num_tokens = len(block_ids) * block_size if block_size else len(block_ids)
        return KVChunk(
            chunk_id=chunk_id,
            block_ids=list(block_ids),
            tensors=tensors,
            num_tokens=num_tokens,
        )

    def inject(self, request_id: str, chunk: KVChunk) -> bool:
        kv_manager = getattr(self.scheduler, "kv_cache_manager", None)
        if kv_manager is None:
            raise RuntimeError("Scheduler has no kv_cache_manager.")
        blocks = kv_manager.get_blocks(request_id)
        block_groups = blocks.get_block_ids(allow_none=True)
        if not block_groups:
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

    def close(self) -> None:
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None and self._wrapped_free_request is not None:
            current_free = getattr(scheduler, "_free_request", None)
            if current_free is self._wrapped_free_request:
                scheduler._free_request = self._original_free_request  # type: ignore[attr-defined]
                if hasattr(scheduler, "_semantic_cache_wrapped"):
                    delattr(scheduler, "_semantic_cache_wrapped")
        self._pending_callbacks.clear()
        self.llm = None
        self.engine = None
        self.scheduler = None
        self.runner = None
        self._wrapped_free_request = None
        self._original_free_request = None
