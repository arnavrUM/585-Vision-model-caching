from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence, TYPE_CHECKING

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiment2.command_builder import build_test_vllm_command
from experiment2.specs import ExperimentSpec, load_spec_file

if TYPE_CHECKING:  # pragma: no cover - import-time type checking only
    from vllm import LLM, SamplingParams

    from experiment2.semantic_cache import SemanticCache, SemanticCacheConfig
    from experiment2.semantic_cache.embedding_hooks import EmbeddingHook
    from experiment2.test_vllm import PromptResult
else:  # placeholders populated at runtime
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]
    SemanticCache = None  # type: ignore[assignment]
    SemanticCacheConfig = None  # type: ignore[assignment]
    EmbeddingHook = None  # type: ignore[assignment]
    NullEmbeddingHook = None  # type: ignore[assignment]
    load_embedding_hook = None
    PromptResult = None  # type: ignore[assignment]
    build_prompts = None
    load_gqa_dataset = None
    parse_embedding_layer_spec = None
    run_samples = None

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


_RUNTIME_IMPORTED = False


def _import_runtime() -> None:
    global _RUNTIME_IMPORTED
    if _RUNTIME_IMPORTED:
        return
    from vllm import LLM as _LLM, SamplingParams as _SamplingParams

    from experiment2.semantic_cache import SemanticCache as _SemanticCache, SemanticCacheConfig as _SemanticCacheConfig
    from experiment2.semantic_cache.embedding_hooks import (
        EmbeddingHook as _EmbeddingHook,
        NullEmbeddingHook as _NullEmbeddingHook,
        load_embedding_hook as _load_embedding_hook,
    )
    from experiment2.test_vllm import (
        PromptResult as _PromptResult,
        build_prompts as _build_prompts,
        load_gqa_dataset as _load_gqa_dataset,
        parse_embedding_layer_spec as _parse_embedding_layer_spec,
        run_samples as _run_samples,
    )

    globals().update(
        {
            "LLM": _LLM,
            "SamplingParams": _SamplingParams,
            "SemanticCache": _SemanticCache,
            "SemanticCacheConfig": _SemanticCacheConfig,
            "EmbeddingHook": _EmbeddingHook,
            "NullEmbeddingHook": _NullEmbeddingHook,
            "load_embedding_hook": _load_embedding_hook,
            "PromptResult": _PromptResult,
            "build_prompts": _build_prompts,
            "load_gqa_dataset": _load_gqa_dataset,
            "parse_embedding_layer_spec": _parse_embedding_layer_spec,
            "run_samples": _run_samples,
        }
    )
    _RUNTIME_IMPORTED = True
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

    # Calculate separate hit rates for each cache type
    text_exact_hits = sum(1 for result in results if result.techniques.get("exact_text") == "hit")
    text_semantic_hits = sum(1 for result in results if result.techniques.get("semantic_text") == "hit")
    embedding_prompt_hits = sum(1 for result in results if result.techniques.get("embedding:prompt") == "hit")
    embedding_vision_hits = sum(1 for result in results if result.techniques.get("embedding:vision") == "hit")
    
    text_exact_hit_rate = (text_exact_hits / total) if total else 0.0
    text_semantic_hit_rate = (text_semantic_hits / total) if total else 0.0
    embedding_prompt_hit_rate = (embedding_prompt_hits / total) if total else 0.0
    embedding_vision_hit_rate = (embedding_vision_hits / total) if total else 0.0

    return {
        "total_prompts": total,
        "cache_hits": len(hits),
        "cache_hit_rate": (len(hits) / total) if total else 0.0,
        "text_exact_hit_rate": text_exact_hit_rate,
        "text_semantic_hit_rate": text_semantic_hit_rate,
        "embedding_prompt_hit_rate": embedding_prompt_hit_rate,
        "embedding_vision_hit_rate": embedding_vision_hit_rate,
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


def _shutdown_llm(llm: LLM | None) -> None:
    if llm is None:
        return
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is None:
            return
        engine_core = getattr(engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if callable(shutdown):
            shutdown()
        if hasattr(engine, "engine_core"):
            try:
                engine.engine_core = None  # type: ignore[attr-defined]
            except Exception:
                pass
        llm.llm_engine = None  # type: ignore[attr-defined]
    except Exception:
        pass


def _release_llm(llm: LLM | None) -> None:
    if llm is None:
        return
    try:
        _shutdown_llm(llm)
        import gc

        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass


def run_spec(spec: ExperimentSpec, *, cache_mode: str) -> tuple[dict[str, Any], list[PromptResult]]:
    _import_runtime()
    limit = spec.limit()
    dataset = load_gqa_dataset(spec.dataset_config, spec.split, limit, spec.shuffle_seed)
    prompts = build_prompts(dataset, spec.prompt_template, spec.chunk_source)
    llm: LLM | None = None
    cache: SemanticCache | None = None
    try:
        llm = LLM(
            model=spec.model,
            trust_remote_code=spec.trust_remote_code,
            tensor_parallel_size=getattr(spec, "tensor_parallel_size", 1),
            gpu_memory_utilization=getattr(spec, "gpu_memory_utilization", 0.9),
        )
        layer_configs = [parse_embedding_layer_spec(layer) for layer in spec.embedding_layers]
        cache_config = SemanticCacheConfig(
            similarity_threshold=spec.similarity_threshold,
            max_cached_blocks=spec.max_cached_blocks,
            cache_dir=spec.cache_dir,
            index_encoder=spec.index_encoder,
            index_encoder_device=spec.index_encoder_device,
            embedding_layers=layer_configs,
            enable_fusion_cache=spec.enable_fusion_cache,
            fusion_cache_dir=spec.fusion_cache_dir,
            enable_semantic_text_cache=spec.enable_semantic_text_cache,
            enable_exact_text_cache=spec.enable_exact_text_cache,
            dry_run=(cache_mode != "live"),
        )
        cache = SemanticCache(llm, config=cache_config)
        sampling_params = SamplingParams(temperature=spec.temperature, max_tokens=spec.max_tokens)
        embedding_hook: EmbeddingHook | None
        if layer_configs:
            embedding_hook = load_embedding_hook(spec.embedding_hook)
        else:
            embedding_hook = NullEmbeddingHook()
        start = time.perf_counter()
        results = run_samples(
            llm,
            cache,
            prompts,
            sampling_params,
            embedding_hook=embedding_hook,
            model_name=getattr(spec, "model", None),
        )
        duration = time.perf_counter() - start
        aggregates = aggregate_results(results)
        aggregates["wall_time"] = duration
        return aggregates, results
    finally:
        if cache is not None:
            cache.close()
        _release_llm(llm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch semantic cache sweeps with logging.")
    parser.add_argument("--specs", required=True, help="Path to JSON spec file describing experiments.")
    parser.add_argument(
        "--log-file",
        default="experiment2/experiment_logs.csv",
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
    parser.add_argument(
        "--cache-mode",
        choices=["dry-run", "live"],
        default="dry-run",
        help="Use 'dry-run' to simulate cache capture/injection without touching KV tensors.",
    )
    parser.add_argument(
        "--purge-cache-between-runs",
        action="store_true",
        help="Delete cache/fusion directories before each experiment to avoid cross-run reuse.",
    )
    parser.add_argument(
        "--emit-commands",
        action="store_true",
        help="Print per-experiment commands for experiment2/test_vllm.py and exit without running.",
    )
    return parser.parse_args()


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row.get("experiment", "") for row in reader if row.get("experiment")}


def purge_cache_directories(spec: ExperimentSpec) -> None:
    """Delete cache directories so experiments remain isolated."""

    def _purge(path_str: str | None) -> None:
        if not path_str:
            return
        path = Path(path_str).expanduser()
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            resolved = path
        if not resolved.exists():
            return
        if resolved == resolved.anchor:
            raise RuntimeError(f"Refusing to purge root directory: {resolved}")
        shutil.rmtree(resolved, ignore_errors=False)

    _purge(spec.cache_dir)
    if spec.enable_fusion_cache:
        _purge(spec.fusion_cache_dir)


def main() -> None:
    args = parse_args()
    specs = load_spec_file(args.specs)
    if not specs:
        print("No experiments defined in spec file.")
        return

    if args.emit_commands:
        for idx, spec in enumerate(specs):
            print(f"# {spec.name}")
            print(
                build_test_vllm_command(
                    spec,
                    cache_mode=args.cache_mode,
                    log_file=args.log_file,
                    samples_dir=args.samples_dir,
                )
            )
            if idx != len(specs) - 1:
                print()
        return

    log_path = Path(args.log_file)
    completed = load_completed(log_path) if args.resume else set()
    samples_dir = Path(args.samples_dir) if args.samples_dir else None

    for spec in specs:
        if spec.name in completed:
            print(f"[skip] Experiment '{spec.name}' already logged.")
            continue
        if args.purge_cache_between_runs:
            purge_cache_directories(spec)
        print(f"\n=== Running experiment: {spec.name} ===")
        aggregates, results = run_spec(spec, cache_mode=args.cache_mode)
        timestamp = datetime.utcnow().isoformat()
        row = {
            "timestamp": timestamp,
            **spec.to_row(),
            **{
                "total_prompts": aggregates["total_prompts"],
                "cache_hits": aggregates["cache_hits"],
                "cache_hit_rate": aggregates["cache_hit_rate"],
                "text_exact_hit_rate": aggregates.get("text_exact_hit_rate", 0.0),
                "text_semantic_hit_rate": aggregates.get("text_semantic_hit_rate", 0.0),
                "embedding_prompt_hit_rate": aggregates.get("embedding_prompt_hit_rate", 0.0),
                "embedding_vision_hit_rate": aggregates.get("embedding_vision_hit_rate", 0.0),
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
