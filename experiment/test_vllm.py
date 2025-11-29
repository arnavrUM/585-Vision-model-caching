from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import numpy as np
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams

from experiment.semantic_cache import SemanticCache, SemanticCacheConfig
from experiment.semantic_cache.techniques import EmbeddingLayerConfig
from experiment.semantic_cache.embedding_hooks import (
    EmbeddingHook,
    NullEmbeddingHook,
    load_embedding_hook,
)
from experiment.semantic_cache.experiment_driver import drain_request
from experiment.semantic_cache.semantic_cache import CacheHit


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    lowered = text.lower()
    return " ".join(re.findall(r"[a-z0-9]+", lowered))


def _answer_matches(reference: str | None, generated: str) -> bool | None:
    normalized_reference = _normalize_text(reference)
    if not normalized_reference:
        return None
    normalized_generated = _normalize_text(generated)
    return normalized_reference in normalized_generated


class _SafeDict(dict):
    def __missing__(self, key):
        return ""


@dataclass
class GQAPrompt:
    dataset_id: str
    image_id: str
    question: str
    answer: str
    full_answer: str
    prompt: str
    chunk_text: str
    metadata: dict

    @property
    def reference(self) -> str:
        return self.full_answer or self.answer or ""


@dataclass
class PromptResult:
    sample: GQAPrompt
    hit: CacheHit | None
    latency: float
    response: str
    is_correct: bool | None


def format_prompt(template: str, sample: dict) -> str:
    groups = sample.get("groups") or {}
    semantic = sample.get("semantic") or []
    semantic_program = " | ".join(
        f"{step.get('operation', '')}:{step.get('argument', '')}" for step in semantic
    )
    context = {
        "question": sample.get("question", ""),
        "answer": sample.get("answer", ""),
        "full_answer": sample.get("fullAnswer", ""),
        "image_id": sample.get("imageId", ""),
        "dataset_id": sample.get("id", ""),
        "global_group": groups.get("global", ""),
        "local_group": groups.get("local", ""),
        "semantic_str": sample.get("semanticStr", ""),
        "semantic_program": semantic_program,
    }
    return template.format_map(_SafeDict(context)).strip()


_PAREN_CONTENT = re.compile(r"\([^)]*\)")


def _semantic_signature(sample: dict) -> str:
    text = sample.get("semanticStr", "")
    if not text:
        semantic = sample.get("semantic") or []
        text = " | ".join(
            f"{step.get('operation', '').strip()}:{step.get('argument', '').strip()}"
            for step in semantic
        )
    text = _PAREN_CONTENT.sub("", text)
    text = re.sub(r"\s+", " ", text.strip())
    return text


def parse_embedding_layer_spec(spec: str) -> EmbeddingLayerConfig:
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid embedding layer spec '{spec}'. Expected format name:dim[:threshold]."
        )
    name = parts[0].strip()
    dim = int(parts[1])
    threshold = float(parts[2]) if len(parts) > 2 else 0.85
    return EmbeddingLayerConfig(name=name, dim=dim, similarity_threshold=threshold)


def render_chunk_text(sample: dict, mode: str) -> str:
    if mode == "question":
        text = sample.get("question", "")
    elif mode == "semantic":
        text = _semantic_signature(sample)
    elif mode == "answer":
        text = sample.get("fullAnswer") or sample.get("answer") or ""
    elif mode == "group":
        groups = sample.get("groups") or {}
        text = f"{groups.get('global', '')} {groups.get('local', '')}"
    elif mode == "image":
        text = f"image-{sample.get('imageId', '')}"
    elif mode == "combined":
        groups = sample.get("groups") or {}
        parts = [
            _semantic_signature(sample),
            groups.get("global", ""),
            groups.get("local", ""),
        ]
        text = " ".join(part for part in parts if part)
    else:
        raise ValueError(f"Unknown chunk source: {mode}")
    text = (text or "").strip()
    return text or sample.get("question", "")


def load_gqa_dataset(config: str, split: str, limit: int | None, seed: int | None) -> Dataset:
    dataset = load_dataset("lmms-lab/GQA", config, split=split)
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    if limit is not None:
        limit = min(limit, len(dataset))
        dataset = dataset.select(range(limit))
    return dataset


def build_prompts(
    dataset: Dataset,
    prompt_template: str,
    chunk_source: str,
) -> list[GQAPrompt]:
    prompts: list[GQAPrompt] = []
    for sample in dataset:
        prompt = format_prompt(prompt_template, sample)
        chunk_text = render_chunk_text(sample, chunk_source)
        prompts.append(
            GQAPrompt(
                dataset_id=sample.get("id", ""),
                image_id=sample.get("imageId", ""),
                question=sample.get("question", ""),
                answer=sample.get("answer", ""),
                full_answer=sample.get("fullAnswer", ""),
                prompt=prompt,
                chunk_text=chunk_text,
                metadata={
                    "groups": sample.get("groups", {}),
                    "semantic": sample.get("semanticStr", ""),
                },
            )
        )
    return prompts


def run_samples(
    llm: LLM,
    cache: SemanticCache,
    prompts: Sequence[GQAPrompt],
    sampling_params: SamplingParams,
    *,
    embedding_hook: EmbeddingHook | None = None,
) -> list[PromptResult]:
    engine = llm.llm_engine
    results: list[PromptResult] = []
    for sample in prompts:
        request_id = uuid.uuid4().hex
        embeddings: dict[str, np.ndarray] = {}
        if embedding_hook is not None:
            try:
                embeddings = embedding_hook(llm=llm, sample=sample) or {}
            except Exception as exc:
                print(f"[warn] embedding hook failed for {sample.dataset_id}: {exc}")
                embeddings = {}
        hit = cache.try_reuse(request_id, sample.chunk_text, embeddings=embeddings)
        start = time.perf_counter()
        engine.add_request(request_id, sample.prompt, sampling_params)
        response = drain_request(engine, request_id)
        latency = time.perf_counter() - start
        if hit is None:
            cache.add_observation(request_id, sample.chunk_text, embeddings=embeddings)
        is_correct = _answer_matches(sample.reference, response)
        results.append(
            PromptResult(
                sample=sample,
                hit=hit,
                latency=latency,
                response=response.strip(),
                is_correct=is_correct,
            )
        )
        source = hit.source if hit else "none"
        print(
            f"[{sample.dataset_id}] "
            f"{'hit' if hit else 'miss'} ({source}) | latency={latency:.3f}s | "
            f"answer match={is_correct}"
        )
    return results


def summarize_results(results: Sequence[PromptResult]) -> None:
    if not results:
        print("No prompts were evaluated.")
        return
    latencies = [result.latency for result in results]
    hits = [result for result in results if result.hit]
    misses = [result for result in results if not result.hit]
    matchable = [result for result in results if result.is_correct is not None]
    match_rate = (
        sum(1 for result in matchable if result.is_correct) / len(matchable) if matchable else 0.0
    )
    def _mean(values: Iterable[float]) -> float:
        return statistics.mean(values) if values else 0.0
    print("\n=== Experiment summary ===")
    print(f"Total prompts: {len(results)}")
    print(f"Cache hits: {len(hits)} ({len(hits) / len(results):.1%})")
    print(f"Cache misses: {len(misses)} ({len(misses) / len(results):.1%})")
    print(f"Average latency: {_mean(latencies):.3f}s")
    print(f"Average latency (hit): {_mean([r.latency for r in hits]):.3f}s")
    print(f"Average latency (miss): {_mean([r.latency for r in misses]):.3f}s")
    print(f"Answer match rate: {match_rate:.1%} (n={len(matchable)})")


def summarize_chunk_texts(prompts: Sequence[GQAPrompt], limit: int = 5) -> None:
    if not prompts:
        return
    counts = Counter(prompt.chunk_text for prompt in prompts)
    reused = [text for text, count in counts.items() if count > 1]
    reuse_ratio = (len(reused) / len(counts)) if counts else 0.0
    print(
        f"Chunk-key uniqueness: {len(counts)} unique keys / {len(prompts)} prompts "
        f"({reuse_ratio:.1%} reused >=2x)"
    )
    if reused:
        print("Most frequent chunk keys:")
        for text, count in counts.most_common(limit):
            if count < 2:
                break
            preview = text[:60] + ("…" if len(text) > 60 else "")
            print(f"- occurrences={count}: {preview}")


def parse_args() -> argparse.Namespace:
    default_template = (
        "You are assisting with the GQA benchmark. "
        "Answer the question based on the referenced image.\n"
        "Image ID: {image_id}\n"
        "Question: {question}\n"
        "Answer:"
    )
    parser = argparse.ArgumentParser(
        description="Benchmark semantic KV caching on the GQA dataset using vLLM."
    )
    parser.add_argument("--model", default="facebook/opt-125m", help="Model path or name for vLLM.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward to vLLM to trust remote code when loading the model.",
    )
    parser.add_argument(
        "--dataset-config",
        default="val_balanced_instructions",
        help="Configuration name from lmms-lab/GQA.",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Split spec understood by datasets.load_dataset (e.g., 'val[:256]').",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Maximum number of prompts to evaluate. Use -1 for all samples in the split.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=7,
        help="Optional shuffle seed so cache hits depend on question ordering.",
    )
    parser.add_argument(
        "--chunk-source",
        choices=["question", "semantic", "answer", "group", "image", "combined"],
        default="semantic",
        help="Field used to derive semantic similarity keys for caching.",
    )
    parser.add_argument(
        "--embedding-layer",
        action="append",
        default=[],
        metavar="NAME:DIM[:THRESH]",
        help="Register latent embedding layers for semantic reuse (e.g., 'vision:1024:0.9').",
    )
    parser.add_argument(
        "--embedding-hook",
        default="none",
        help="Embedding hook identifier ('none', 'prompt', or dotted path module:attr).",
    )
    parser.add_argument(
        "--prompt-template",
        default=default_template,
        help="Python format string used to build prompts. "
        "Supports placeholders such as {question}, {image_id}, {answer}, {full_answer}, "
        "{global_group}, {local_group}, {semantic_str}, {semantic_program}, and {dataset_id}.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Cosine similarity threshold for accepting a cached chunk.",
    )
    parser.add_argument(
        "--max-cached-blocks",
        type=int,
        default=None,
        help="Optional cap on how many KV blocks are stored per chunk.",
    )
    parser.add_argument("--cache-dir", default="kv_chunks", help="Where cached KV chunks are stored.")
    parser.add_argument(
        "--index-encoder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model used for semantic search.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to vLLM.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum number of generated tokens for each prompt.",
    )
    parser.add_argument(
        "--report-limit",
        type=int,
        default=5,
        help="How many sample predictions to print in detail after the run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = None if args.max_samples is None or args.max_samples < 0 else args.max_samples
    dataset = load_gqa_dataset(args.dataset_config, args.split, limit, args.shuffle_seed)
    prompts = build_prompts(dataset, args.prompt_template, args.chunk_source)
    print(f"Loaded {len(prompts)} prompts from lmms-lab/GQA ({args.dataset_config}, split={args.split}).")
    summarize_chunk_texts(prompts)

    llm = LLM(model=args.model, trust_remote_code=args.trust_remote_code)
    layer_configs = [parse_embedding_layer_spec(spec) for spec in args.embedding_layer]
    cache_config = SemanticCacheConfig(
        similarity_threshold=args.similarity_threshold,
        max_cached_blocks=args.max_cached_blocks,
        cache_dir=args.cache_dir,
        index_encoder=args.index_encoder,
        embedding_layers=layer_configs,
    )
    cache = SemanticCache(llm, config=cache_config)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    embedding_hook: EmbeddingHook | None
    if layer_configs:
        embedding_hook = load_embedding_hook(args.embedding_hook)
    else:
        embedding_hook = NullEmbeddingHook()

    results = run_samples(llm, cache, prompts, sampling_params, embedding_hook=embedding_hook)
    summarize_results(results)

    print("\n=== Sample outputs ===")
    for result in results[: args.report_limit]:
        source = result.hit.source if result.hit else "none"
        print(
            f"ID={result.sample.dataset_id} | img={result.sample.image_id} | "
            f"{'hit' if result.hit else 'miss'} ({source}) | latency={result.latency:.3f}s"
        )
        print(f"Q: {result.sample.question}")
        print(f"Model: {result.response}")
        if result.sample.reference:
            print(f"Reference: {result.sample.reference}")
        print("---")


if __name__ == "__main__":
    main()
