from __future__ import annotations

import os
from io import BytesIO
from typing import Iterable

from datasets import load_dataset
from PIL import Image
from vllm import LLM, SamplingParams

BASE_PROMPT_TEMPLATE = (
    "You are assisting with the GQA benchmark. "
    "Answer the question based on the referenced image.\n"
    "Image ID: {image_id}\n"
    "Question: {question}\n"
    "Answer:"
)

SAMPLE_LIMIT = 10


def _normalize_image_id(raw_id: str) -> str:
    return str(raw_id or "").strip()


def _gqa_image_config_name(config: str) -> str:
    if config.endswith("_instructions"):
        return config[: -len("_instructions")] + "_images"
    if config.endswith("_images"):
        return config
    return f"{config}_images"


def _serialize_image(image_obj: object) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, dict) and "bytes" in image_obj:
        return Image.open(BytesIO(image_obj["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported GQA image payload type: {type(image_obj)}")


def _load_gqa_subset(config: str, split: str, limit: int, seed: int) -> list[dict]:
    dataset = load_dataset("lmms-lab/GQA", config, split=split)
    dataset = dataset.shuffle(seed=seed)
    limit = min(limit, len(dataset))
    return list(dataset.select(range(limit)))


def _load_gqa_images(
    image_ids: Iterable[str],
    dataset_config: str,
    split: str,
) -> dict[str, Image.Image]:
    pending = {_normalize_image_id(image_id) for image_id in image_ids if image_id}
    images: dict[str, Image.Image] = {}
    if not pending:
        return images
    image_config = _gqa_image_config_name(dataset_config)
    stream = load_dataset("lmms-lab/GQA", image_config, split=split, streaming=True)
    for sample in stream:
        sample_id = _normalize_image_id(sample.get("id", ""))
        if sample_id not in pending:
            continue
        try:
            images[sample_id] = _serialize_image(sample.get("image"))
        except Exception as exc:
            print(f"[warn] failed to decode image {sample_id}: {exc}")
        pending.discard(sample_id)
        if not pending:
            break
    if pending:
        print(f"[warn] missing {len(pending)} images: {sorted(list(pending))[:5]}")
    return images


def _build_qwen_prompt(question: str, image_id: str) -> str:
    placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    user_block = BASE_PROMPT_TEMPLATE.format(image_id=image_id, question=question)
    user_block = f"{placeholder}\n{user_block}"
    return f"<|im_start|>user\n{user_block}\n<|im_end|>\n<|im_start|>assistant\n"


def _build_mm_payload(prompt_text: str, image: Image.Image | None) -> dict:
    if image is None:
        raise ValueError("Image payload is required for Qwen2-VL.")
    return {"prompt": prompt_text, "multi_modal_data": {"image": [image]}}


def main() -> None:
    model_name = os.environ.get("SMOKE_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
    dataset_config = "val_balanced_instructions"
    split = "val"
    samples = _load_gqa_subset(dataset_config, split, limit=SAMPLE_LIMIT, seed=42)
    images = _load_gqa_images([s.get("imageId", "") for s in samples], dataset_config, split)

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        mm_processor_kwargs={"use_fast": False},
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=64)

    for sample in samples:
        image_id = _normalize_image_id(sample.get("imageId", ""))
        question = sample.get("question", "")
        prompt = _build_qwen_prompt(question, image_id)
        image = images.get(image_id)
        if image is None:
            print(f"[warn] skipping sample {sample.get('id', '')}: missing image {image_id}")
            continue
        payload = _build_mm_payload(prompt, image)
        outputs = llm.generate([payload], sampling)
        response = outputs[0].outputs[0].text.strip()
        print("---")
        print(f"id={sample.get('id', '')} | image={image_id} | question={question}")
        print(f"model: {response}")


if __name__ == "__main__":
    main()
