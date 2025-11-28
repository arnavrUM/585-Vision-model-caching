from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image
from vllm import LLM, SamplingParams

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
CHAT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""


def build_prompt(question: str, include_image: bool) -> str:
    """Wrap a user question in the Qwen chat template."""
    user_content = question
    if include_image:
        user_content = f"{IMAGE_PLACEHOLDER}\n{question}"
    return CHAT_TEMPLATE.format(user_content=user_content)


def load_image(source: str) -> Image.Image:
    """Load an image from a URL or a local path."""
    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    path = Path(source).expanduser().resolve(strict=True)
    return Image.open(path).convert("RGB")


def run_inference(
    model: str,
    questions: Iterable[str],
    image_source: str | None,
    temperature: float,
    top_p: float,
) -> None:
    prompts = [build_prompt(q, include_image=bool(image_source)) for q in questions]
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p)
    llm = LLM(model=model, trust_remote_code=True)

    images = None
    if image_source:
        image = load_image(image_source)
        images = [[image] for _ in prompts]

    outputs = llm.generate(prompts, sampling_params, images=images)

    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        print(f"Prompt:\n{output.prompt}\n---\nGenerated:\n{generated_text}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the Qwen2.5-VL model with optional image input."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name or path. Defaults to the 8B-class Qwen2.5-VL checkpoint.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional path or URL to an image that will be included with every prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Softmax temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling cutoff.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    questions = [
        "Give me a concise caption for this image.",
        "List three visual details that stand out.",
    ]
    run_inference(
        model=args.model,
        questions=questions,
        image_source=args.image,
        temperature=args.temperature,
        top_p=args.top_p,
    )
