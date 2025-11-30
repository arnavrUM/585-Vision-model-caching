from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from vllm import LLM, SamplingParams

from experiment.semantic_cache import SemanticCache, SemanticCacheConfig
from experiment.semantic_cache.embedding_hooks import (
    EmbeddingHook,
    NullEmbeddingHook,
    load_embedding_hook,
)
from experiment.test_vllm import (
    PromptResult,
    build_prompts,
    load_gqa_dataset,
    parse_embedding_layer_spec,
    run_samples,
)

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


DEFAULT_PROMPT_TEMPLATE = (
    "You are assisting with the GQA benchmark. "
    "Answer the question based on the referenced image.\n"
    "Image ID: {image_id}\n"
    "Question: {question}\n"
    "Answer:"
)


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
    cache_dir: str = "kv_chunks"
    index_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    temperature: float = 0.0
    max_tokens: int = 64
    trust_remote_code: bool = False
    notes: str | None = None
    enable_fusion_cache: bool = False
    fusion_cache_dir: str = "fusion_chunks"

    @classmethod
    def from_dict(cls, row: dict[str, Any], defaults: dict[str, Any] | None = None) -> "ExperimentSpec":
        merged: dict[str, Any] = {}
        if defaults:
            merged.update(defaults)
        merged.update(row)
        layers = merged.get("embedding_layers")
        if isinstance(layers, str):
            merged["embedding_layers"] = [layers]
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
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "trust_remote_code": self.trust_remote_code,
            "notes": self.notes or "",
            "enable_fusion_cache": self.enable_fusion_cache,
            "fusion_cache_dir": self.fusion_cache_dir,
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


def aggregate_results(results: Sequence[PromptResult]) -> dict[str, Any]:
    total = len(results)
    hits = [result for result in results if result.hit]
    misses = [result for result in results if not result.hit]
    latencies = [result.latency for result in results]
    matchable = [result for result in results if result.is_correct is not None]

    def _mean(values: Iterable[float]) -> float | None:
        return float(statistics.mean(values)) if values else None

    accuracy = None
    if matchable:
        accuracy = sum(1 for result in matchable if result.is_correct) / len(matchable)

    breakdown: dict[str, int] = {}
    for result in hits:
        source = result.hit.source if result.hit else "unknown"
        breakdown[source] = breakdown.get(source, 0) + 1

    return {
        "total_prompts": total,
        "cache_hits": len(hits),
        "cache_hit_rate": (len(hits) / total) if total else 0.0,
        "avg_latency": _mean(latencies),
        "avg_latency_hit": _mean([result.latency for result in hits]),
        "avg_latency_miss": _mean([result.latency for result in misses]),
        "answer_accuracy": accuracy,
        "hit_breakdown": breakdown,
    }


def write_samples(samples_dir: Path, spec: ExperimentSpec, results: Sequence[PromptResult]) -> None:
    samples_dir.mkdir(parents=True, exist_ok=True)
    path = samples_dir / f"{spec.name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            payload = {
                "dataset_id": result.sample.dataset_id,
                "image_id": result.sample.image_id,
                "question": result.sample.question,
                "chunk_text": result.sample.chunk_text,
                "prompt": result.sample.prompt,
                "hit": bool(result.hit),
                "hit_source": result.hit.source if result.hit else None,
                "latency": result.latency,
                "response": result.response,
                "reference": result.sample.reference,
                "is_correct": result.is_correct,
            }
            handle.write(json.dumps(payload) + "\n")


def log_row(path: Path, row: dict[str, Any]) -> None:
    fieldnames = list(row.keys())
    file_exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_spec(spec: ExperimentSpec) -> tuple[dict[str, Any], list[PromptResult]]:
    limit = spec.limit()
    dataset = load_gqa_dataset(spec.dataset_config, spec.split, limit, spec.shuffle_seed)
    prompts = build_prompts(dataset, spec.prompt_template, spec.chunk_source)
    llm = LLM(model=spec.model, trust_remote_code=spec.trust_remote_code)
    layer_configs = [parse_embedding_layer_spec(layer) for layer in spec.embedding_layers]
    cache_config = SemanticCacheConfig(
        similarity_threshold=spec.similarity_threshold,
        max_cached_blocks=spec.max_cached_blocks,
        cache_dir=spec.cache_dir,
        index_encoder=spec.index_encoder,
        embedding_layers=layer_configs,
        enable_fusion_cache=spec.enable_fusion_cache,
        fusion_cache_dir=spec.fusion_cache_dir,
    )
    cache = SemanticCache(llm, config=cache_config)
    sampling_params = SamplingParams(temperature=spec.temperature, max_tokens=spec.max_tokens)
    embedding_hook: EmbeddingHook | None
    if layer_configs:
        embedding_hook = load_embedding_hook(spec.embedding_hook)
    else:
        embedding_hook = NullEmbeddingHook()
    start = time.perf_counter()
    results = run_samples(llm, cache, prompts, sampling_params, embedding_hook=embedding_hook)
    duration = time.perf_counter() - start
    aggregates = aggregate_results(results)
    aggregates["wall_time"] = duration
    return aggregates, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch semantic cache sweeps with logging.")
    parser.add_argument("--specs", required=True, help="Path to JSON spec file describing experiments.")
    parser.add_argument(
        "--log-file",
        default="experiment_logs.csv",
        help="CSV file where per-experiment summaries are appended.",
    )
    parser.add_argument(
        "--samples-dir",
        default=None,
        help="Optional directory for dumping per-sample JSONL logs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiment names that already appear in the log file.",
    )
    return parser.parse_args()


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row.get("experiment", "") for row in reader if row.get("experiment")}


def main() -> None:
    args = parse_args()
    specs = load_spec_file(args.specs)
    if not specs:
        print("No experiments defined in spec file.")
        return
    log_path = Path(args.log_file)
    completed = load_completed(log_path) if args.resume else set()
    samples_dir = Path(args.samples_dir) if args.samples_dir else None

    for spec in specs:
        if spec.name in completed:
            print(f"[skip] Experiment '{spec.name}' already logged.")
            continue
        print(f"\n=== Running experiment: {spec.name} ===")
        aggregates, results = run_spec(spec)
        timestamp = datetime.utcnow().isoformat()
        row = {
            "timestamp": timestamp,
            **spec.to_row(),
            **{
                "total_prompts": aggregates["total_prompts"],
                "cache_hits": aggregates["cache_hits"],
                "cache_hit_rate": aggregates["cache_hit_rate"],
                "avg_latency": aggregates["avg_latency"],
                "avg_latency_hit": aggregates["avg_latency_hit"],
                "avg_latency_miss": aggregates["avg_latency_miss"],
                "answer_accuracy": aggregates["answer_accuracy"],
                "wall_time": aggregates["wall_time"],
                "hit_breakdown": json.dumps(aggregates["hit_breakdown"]),
            },
        }
        log_row(log_path, row)
        print(
            f"[done] {spec.name}: prompts={aggregates['total_prompts']} "
            f"hits={aggregates['cache_hits']} ({aggregates['cache_hit_rate']:.1%}) "
            f"accuracy={(aggregates['answer_accuracy'] if aggregates['answer_accuracy'] is not None else 'n/a')}"
        )
        if samples_dir is not None:
            write_samples(samples_dir, spec, results)


if __name__ == "__main__":
    main()
