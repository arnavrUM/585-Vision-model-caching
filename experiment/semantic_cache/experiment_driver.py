from __future__ import annotations

import argparse
import time
import uuid
from typing import Iterable

from vllm import LLM, SamplingParams

from .semantic_cache import CacheHit, SemanticCache, SemanticCacheConfig


def chunk_prompt(prompt: str, window: int) -> str:
    if len(prompt) <= window:
        return prompt
    return prompt[:window]


def drain_request(engine, request_id: str) -> str:
    """Drive the engine loop until the request finishes."""
    response = ""
    while engine.has_unfinished_requests():
        outputs = engine.step()
        for output in outputs:
            if output.request_id != request_id:
                continue
            if output.outputs:
                response = output.outputs[-1].text
            if output.finished:
                return response
    return response


def run_prompts(
    llm: LLM,
    prompts: Iterable[str],
    cache: SemanticCache,
    sampling_params: SamplingParams,
    chunk_window: int,
):
    engine = llm.llm_engine
    stats: list[tuple[str, CacheHit | None, float]] = []

    for prompt in prompts:
        chunk_text = chunk_prompt(prompt, chunk_window)
        request_id = uuid.uuid4().hex
        hit = cache.try_reuse(request_id, chunk_text)

        start = time.perf_counter()
        engine.add_request(request_id, prompt, sampling_params)
        text = drain_request(engine, request_id)
        latency = time.perf_counter() - start
        stats.append((prompt, hit, latency))

        if hit is None:
            cache.add_observation(request_id, chunk_text)

        print(
            f"Prompt: {prompt[:30]!r}... | "
            f"{'cache hit' if hit else 'cache miss'} | "
            f"latency={latency:.3f}s\n{text}\n"
        )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run semantic KV cache experiments on top of vLLM."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model identifier understood by vLLM.",
    )
    parser.add_argument(
        "--chunk-window",
        type=int,
        default=256,
        help="Number of characters to use when computing semantic similarity.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.88,
        help="Minimum cosine similarity before a cache entry is reused.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = LLM(model=args.model, trust_remote_code=True)
    config = SemanticCacheConfig(similarity_threshold=args.similarity_threshold)
    cache = SemanticCache(llm, config=config)
    prompts = [
        "Describe the Louvre museum in Paris.",
        "Where is the capital of France located?",
        "Explain why Paris is known as the city of lights.",
    ]
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    run_prompts(llm, prompts, cache, sampling_params, args.chunk_window)


if __name__ == "__main__":
    main()
