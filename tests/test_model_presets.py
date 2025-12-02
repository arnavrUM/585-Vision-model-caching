from argparse import Namespace

from experiment2.model_presets import (
    apply_preset_to_args,
    get_model_preset,
    list_model_presets,
)


def _make_namespace(**overrides) -> Namespace:
    base = dict(
        preset=None,
        model="base/model",
        trust_remote_code=False,
        prompt_template="prompt",
        chunk_source="semantic",
        embedding_layer=[],
        embedding_hook="none",
        similarity_threshold=0.8,
        index_encoder_device="cuda",
        disable_exact_cache=False,
        temperature=0.0,
        max_tokens=64,
        notes=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_model_preset_lookup_includes_internvl() -> None:
    presets = list_model_presets()
    assert "internvl3.5-2b" in presets
    preset = get_model_preset("internvl3.5-2b")
    values = preset.materialize()
    assert "InternVL3_5-2B" in values["model"]
    assert values["embedding_layers"] == ["prompt:384:0.8", "vision:512:0.82"]


def test_apply_preset_to_args_overrides_defaults() -> None:
    defaults = _make_namespace()
    args = _make_namespace(preset="internvl3.5-2b")
    preset = apply_preset_to_args(args, defaults)
    assert preset is not None
    assert args.model.startswith("OpenGVLab/InternVL3_5-2B")
    assert args.embedding_layer == ["prompt:384:0.8", "vision:512:0.82"]
    assert args.embedding_hook == "prompt_vision"
    assert args.trust_remote_code is True


def test_apply_preset_respects_manual_overrides() -> None:
    defaults = _make_namespace()
    args = _make_namespace(preset="internvl3.5-2b", model="custom/model")
    apply_preset_to_args(args, defaults)
    assert args.model == "custom/model"
