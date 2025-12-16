from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

BASE_PROMPT_TEMPLATE = (
    "You are assisting with the GQA benchmark. "
    "Answer the question based on the referenced image.\n"
    "Image ID: {image_id}\n"
    "Question: {question}\n"
    "Answer:"
)


@dataclass(frozen=True)
class ModelPreset:
    """Lightweight bundle of recommended settings for an experiment."""

    name: str
    description: str
    values: Mapping[str, Any]

    def materialize(self) -> Dict[str, Any]:
        """Return a detached copy of the preset values."""
        return {key: _clone(value) for key, value in self.values.items()}


_PRESETS: dict[str, ModelPreset] = {
    "qwen3-vl-2b": ModelPreset(
        name="qwen3-vl-2b",
        description="Qwen3-VL-2B-Instruct baseline used in initial cache sweeps.",
        values={
            "model": "Qwen/Qwen3-VL-2B-Instruct",
            "trust_remote_code": True,
            "prompt_template": BASE_PROMPT_TEMPLATE,
            "chunk_source": "semantic",
            "embedding_layers": ("prompt:384:0.82", "vision:512:0.85"),
            "embedding_hook": "prompt_vision",
            "similarity_threshold": 0.82,
            "temperature": 0.0,
            "max_tokens": 64,
            "notes": (
                "Prompt + vision-id embeddings for Qwen3-VL-2B using sentence-transformers."
            ),
        },
    ),
    "internvl3.5-2b": ModelPreset(
        name="internvl3.5-2b",
        description="InternVL3.5-2B Instruct with prompt embeddings and remote code.",
        values={
            "model": "OpenGVLab/InternVL3_5-2B-Instruct",
            "trust_remote_code": True,
            "prompt_template": (
                "<image>\n"
                "You are assisting with the GQA benchmark. "
                "Answer the question using the referenced image.\n"
                "Image ID: {image_id}\n"
                "Question: {question}\n"
                "Answer:"
            ),
            "chunk_source": "semantic",
            "embedding_layers": ("prompt:384:0.8", "vision:512:0.82"),
            "embedding_hook": "prompt_vision",
            "similarity_threshold": 0.8,
            "temperature": 0.0,
            "max_tokens": 64,
            "notes": (
                "Validated with InternVL3.5-2B Instruct; requires trust_remote_code "
                "and piggybacks on prompt + vision-id embeddings via sentence-transformers."
            ),
        },
    ),
}

_UNSET = object()


def list_model_presets() -> list[str]:
    """Return the available preset identifiers."""
    return sorted(_PRESETS)


def get_model_preset(name: str) -> ModelPreset:
    """Lookup a preset by name."""
    try:
        return _PRESETS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown model preset '{name}'. Options: {list_model_presets()}") from exc


def apply_preset_mapping(target: dict[str, Any], name: str) -> ModelPreset:
    """Materialize a preset and overlay it on top of an existing mapping."""
    preset = get_model_preset(name)
    target.update(preset.materialize())
    target["preset"] = preset.name
    return preset


def apply_preset_to_args(args: Namespace, defaults: Namespace) -> ModelPreset | None:
    """Inject preset values into an argparse namespace when fields are untouched."""
    preset_name = getattr(args, "preset", None)
    if not preset_name:
        return None
    preset = get_model_preset(preset_name)
    values = preset.materialize()
    for field, value in values.items():
        target_field = field
        if field == "embedding_layers" and hasattr(args, "embedding_layer"):
            target_field = "embedding_layer"
        if not hasattr(args, target_field):
            continue
        default_value = getattr(defaults, target_field, _UNSET)
        current_value = getattr(args, target_field, _UNSET)
        if _is_default_value(current_value, default_value):
            setattr(args, target_field, value)
    setattr(args, "preset", preset.name)
    return preset


def _clone(value: Any) -> Any:
    if isinstance(value, list):
        return [*value]
    if isinstance(value, tuple):
        return [*value]
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    return value


def _is_default_value(current: Any, default: Any) -> bool:
    if default is _UNSET:
        return False
    if isinstance(default, (list, tuple)):
        return isinstance(current, Iterable) and len(list(current)) == 0 and len(list(default)) == 0
    return current == default
