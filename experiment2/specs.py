from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiment2.model_presets import BASE_PROMPT_TEMPLATE, apply_preset_mapping

DEFAULT_PROMPT_TEMPLATE = BASE_PROMPT_TEMPLATE


@dataclass
class ExperimentSpec:
    name: str
    model: str
    dataset_config: str = "val_balanced_instructions"
    split: str = "val"
    max_samples: int | None = 128
    shuffle_seed: int | None = 7
    chunk_source: str = "semantic"
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    similarity_threshold: float = 0.8
    embedding_layers: list[str] = field(default_factory=list)
    embedding_hook: str = "none"
    max_cached_blocks: int | None = None
    cache_dir: str = "experiment2/kv_chunks"
    index_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_encoder_device: str = "cuda"
    temperature: float = 0.0
    max_tokens: int = 64
    trust_remote_code: bool = False
    notes: str | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    enable_fusion_cache: bool = False
    fusion_cache_dir: str = "experiment2/fusion_chunks"
    enable_semantic_text_cache: bool = True
    enable_exact_text_cache: bool = True
    preset: str | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any], defaults: dict[str, Any] | None = None) -> "ExperimentSpec":
        merged: dict[str, Any] = {}
        if defaults:
            merged.update(defaults)
        preset_name = merged.get("preset")
        if "preset" in row:
            preset_name = row["preset"]
        if preset_name:
            apply_preset_mapping(merged, preset_name)
        merged.update(row)
        layers = merged.get("embedding_layers")
        if isinstance(layers, str):
            merged["embedding_layers"] = [layers]
        elif layers is None:
            merged["embedding_layers"] = []
        return cls(**merged)

    def limit(self) -> int | None:
        if self.max_samples is None or self.max_samples < 0:
            return None
        return self.max_samples

    def to_row(self) -> dict[str, Any]:
        return {
            "experiment": self.name,
            "model": self.model,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "max_samples": self.max_samples if self.max_samples is not None else -1,
            "shuffle_seed": self.shuffle_seed if self.shuffle_seed is not None else "",
            "chunk_source": self.chunk_source,
            "prompt_template": self.prompt_template.replace("\n", "\\n"),
            "similarity_threshold": self.similarity_threshold,
            "embedding_layers": ",".join(self.embedding_layers),
            "embedding_hook": self.embedding_hook,
            "max_cached_blocks": self.max_cached_blocks if self.max_cached_blocks is not None else "",
            "cache_dir": self.cache_dir,
            "index_encoder": self.index_encoder,
            "index_encoder_device": self.index_encoder_device,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "trust_remote_code": self.trust_remote_code,
            "notes": self.notes or "",
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_fusion_cache": self.enable_fusion_cache,
            "fusion_cache_dir": self.fusion_cache_dir,
            "enable_semantic_text_cache": self.enable_semantic_text_cache,
            "enable_exact_text_cache": self.enable_exact_text_cache,
            "preset": self.preset or "",
        }


def load_spec_file(path: str | Path) -> list[ExperimentSpec]:
    data = json.loads(Path(path).read_text())
    defaults: dict[str, Any] = {}
    if isinstance(data, dict):
        defaults = data.get("defaults", {})
        experiments = data.get("experiments", [])
    else:
        experiments = data
    specs: list[ExperimentSpec] = []
    for entry in experiments:
        if not isinstance(entry, dict):
            raise ValueError("Each experiment spec must be a JSON object.")
        specs.append(ExperimentSpec.from_dict(entry, defaults))
    return specs
