from __future__ import annotations

import argparse
import csv
import json
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
from urllib.parse import urlparse

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
    try:  # pragma: no cover - optional dependency
        import importlib

        importlib.import_module("hf_transfer")
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import numpy as np
import requests
from datasets import Dataset, load_dataset
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import (
    Counter as MetricCounter,
    Gauge as MetricGauge,
    Histogram as MetricHistogram,
    Metric as VLLMMetric,
    Vector as MetricVector,
)

from experiment.semantic_cache import SemanticCache, SemanticCacheConfig
from experiment.semantic_cache.techniques import EmbeddingLayerConfig
from experiment.semantic_cache.embedding_hooks import (
    EmbeddingHook,
    NullEmbeddingHook,
    load_embedding_hook,
)
from experiment.semantic_cache.experiment_driver import drain_request
from experiment.semantic_cache.semantic_cache import CacheHit, ReuseReport

LLAVA_DATA_URL = (
    "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/"
    "resolve/main/llava_instruct_150k.json?download=1"
)
LLAVA_CACHE_DIR = Path("dataset_cache")


def _ensure_llava_file(data_url: str) -> Path:
    parsed = urlparse(data_url)
    filename = Path(parsed.path).name or "llava_instruct_150k.json"
    cache_dir = LLAVA_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / filename
    if local_path.exists():
        return local_path
    tmp_path = local_path.with_suffix(".tmp")
    print(f"Downloading LLaVA-Instruct-150K JSON to {local_path} ...")
    with requests.get(data_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    tmp_path.replace(local_path)
    return local_path


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


def _raise_model_load_help(model: str, exc: Exception) -> None:
    token_present = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    hints = [
        "1. Verify the model identifier is spelled correctly and publicly available.",
        "2. If it is private or gated, authenticate with Hugging Face via `huggingface-cli login`",
        "   or set the HF_TOKEN/HUGGINGFACEHUB_API_TOKEN environment variable before running.",
        "3. After authenticating, rerun this script so vLLM can download the weights/config.",
    ]
    if token_present:
        hints.append("4. A token is already configured; double-check it has access to the repo.")
    hint_text = "\n".join(hints)
    raise SystemExit(
        f"Failed to load model '{model}' via Hugging Face/vLLM:\n{exc}\n\nTroubleshooting:\n{hint_text}"
    )


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
    techniques: dict[str, str]


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


def load_llava_dataset(data_url: str, limit: int | None, seed: int | None) -> Dataset:
    local_path = _ensure_llava_file(data_url)
    with local_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, list):
        raise ValueError("Expected LLaVA JSON to contain a list of samples.")
    dataset = Dataset.from_list(payload)
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    if limit is not None:
        limit = min(limit, len(dataset))
        dataset = dataset.select(range(limit))
    return dataset


def _extract_llava_qa(conversations: list[dict]) -> tuple[str, str] | None:
    if not conversations:
        return None
    question = ""
    answer = ""
    for turn in conversations:
        speaker = (turn.get("from") or "").lower()
        text = turn.get("value") or ""
        if speaker == "human" and not question:
            question = text.replace("<image>", "").strip()
        elif speaker == "gpt":
            answer = text.strip()
            if question:
                break
    if question and answer:
        return question, answer
    return None


def build_llava_records(dataset: Dataset) -> list[dict]:
    records: list[dict] = []
    for sample in dataset:
        qa = _extract_llava_qa(sample.get("conversations") or [])
        if qa is None:
            continue
        question, answer = qa
        record = {
            "id": str(sample.get("id", "")),
            "imageId": sample.get("image", ""),
            "question": question,
            "answer": answer,
            "fullAnswer": answer,
            "groups": {},
            "semanticStr": "",
            "semantic": [],
        }
        metadata = {
            "source": "llava_instruct_150k",
            "image": sample.get("image", ""),
            "conversations": sample.get("conversations", []),
        }
        record["_metadata"] = metadata
        records.append(record)
    return records


def build_synthetic_samples() -> list[dict]:
    """Deterministic prompts that exercise each cache layer."""

    def record(
        sample_id: str,
        question: str,
        answer: str,
        chunk_text: str,
        *,
        image_id: str = "synthetic-image",
        metadata: dict | None = None,
    ) -> dict:
        payload = {
            "id": sample_id,
            "imageId": image_id,
            "question": question,
            "answer": answer,
            "fullAnswer": answer,
            "groups": {},
            "semanticStr": "",
            "semantic": [],
            "_chunk_override": chunk_text,
            "_metadata": metadata or {},
        }
        return payload

    samples: list[dict] = []

    # Exact-text pair (identical chunk text).
    exact_q = "What color is the test apple?"
    samples.append(
        record(
            "exact-1",
            exact_q,
            "red",
            "exact-match-key",
            metadata={"expected_hit": "exact"},
        )
    )
    samples.append(
        record(
            "exact-2",
            exact_q,
            "red",
            "exact-match-key",
            metadata={"expected_hit": "exact"},
        )
    )

    # Semantic-text pair (punctuation difference avoids exact match).
    sem_q1 = "Describe the colorful bus."
    sem_q2 = "Describe the colorful bus driving by."
    samples.append(
        record(
            "semantic-1",
            sem_q1,
            "It is bright red.",
            "A bright red bus is parked beside the station.",
            metadata={"expected_hit": "semantic"},
        )
    )
    samples.append(
        record(
            "semantic-2",
            sem_q2,
            "It is bright red.",
            "A bright red bus is parked beside the station!!!",
            metadata={"expected_hit": "semantic"},
        )
    )

    # Embedding pair (different chunk text but identical prompt/questions).
    emb_q = "Is the synthetic triangle pointing up?"
    samples.append(
        record(
            "embedding-1",
            emb_q,
            "yes",
            "embedding-control-key-1",
            image_id="embed-image",
            metadata={"expected_hit": "embedding"},
        )
    )
    samples.append(
        record(
            "embedding-2",
            emb_q,
            "yes",
            "embedding-control-key-2",
            image_id="embed-image",
            metadata={"expected_hit": "embedding"},
        )
    )

    return samples


def build_prompts(
    dataset: Iterable[dict],
    prompt_template: str,
    chunk_source: str,
) -> list[GQAPrompt]:
    prompts: list[GQAPrompt] = []
    for sample in dataset:
        prompt = format_prompt(prompt_template, sample)
        chunk_text = sample.get("_chunk_override")
        if not chunk_text:
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
                    "extra": sample.get("_metadata", {}),
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
    total = len(prompts)
    start_run = time.perf_counter()
    for idx, sample in enumerate(prompts, 1):
        request_id = uuid.uuid4().hex
        embeddings: dict[str, np.ndarray] = {}
        if embedding_hook is not None:
            try:
                embeddings = embedding_hook(llm=llm, sample=sample) or {}
            except Exception as exc:
                print(f"[warn] embedding hook failed for {sample.dataset_id}: {exc}")
                embeddings = {}
        engine.add_request(request_id, sample.prompt, sampling_params)
        reuse = cache.try_reuse(request_id, sample.chunk_text, embeddings=embeddings)
        hit = reuse.hit
        if hit is None:
            cache.add_observation(request_id, sample.chunk_text, embeddings=embeddings)
        start = time.perf_counter()
        response = drain_request(engine, request_id)
        latency = time.perf_counter() - start
        is_correct = _answer_matches(sample.reference, response)
        results.append(
            PromptResult(
                sample=sample,
                hit=hit,
                latency=latency,
                response=response.strip(),
                is_correct=is_correct,
                techniques=reuse.statuses,
            )
        )
        source = hit.source if hit else "none"
        technique_str = ", ".join(f"{name}={status}" for name, status in sorted(reuse.statuses.items()))
        elapsed = time.perf_counter() - start_run
        remaining = total - idx
        eta = (elapsed / idx) * remaining if idx else 0.0
        progress = f"{idx}/{total}"
        print(
            f"[{sample.dataset_id}] {progress} | "
            f"{'hit' if hit else 'miss'} ({source}) | latency={latency:.3f}s | "
            f"elapsed={elapsed:.1f}s | eta={eta:.1f}s | answer match={is_correct} | "
            f"{technique_str}"
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

    hits_by_source: dict[str, list[PromptResult]] = {}
    for result in hits:
        source = result.hit.source if result.hit else "unknown"
        hits_by_source.setdefault(source, []).append(result)

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
    if hits_by_source:
        print("Hit rate by cache:")
        for source, source_hits in sorted(hits_by_source.items(), key=lambda item: item[0]):
            rate = len(source_hits) / len(results)
            print(
                f"  - {source}: {len(source_hits)} hits ({rate:.1%} of prompts) | "
                f"avg latency={_mean([r.latency for r in source_hits]):.3f}s"
            )
    technique_summary: dict[str, Counter] = {}
    for result in results:
        for name, status in result.techniques.items():
            technique_summary.setdefault(name, Counter())[status] += 1
    if technique_summary:
        print("Technique breakdown:")
        for name in sorted(technique_summary):
            counts = technique_summary[name]
            hit_count = counts.get("hit", 0)
            miss_count = counts.get("miss", 0)
            skip_count = counts.get("skip", 0)
            print(
                f"  - {name}: hit={hit_count}, miss={miss_count}, skip={skip_count}"
            )


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


def log_results_csv(path: str | Path | None, results: Sequence[PromptResult]) -> None:
    if not path:
        return
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_id",
        "image_id",
        "question",
        "prompt",
        "chunk_text",
        "response",
        "reference",
        "is_correct",
        "latency_seconds",
        "cache_hit",
        "cache_source",
        "techniques_json",
        "metadata_json",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            sample = result.sample
            writer.writerow(
                {
                    "dataset_id": sample.dataset_id,
                    "image_id": sample.image_id,
                    "question": sample.question,
                    "prompt": sample.prompt,
                    "chunk_text": sample.chunk_text,
                    "response": result.response,
                    "reference": sample.reference,
                    "is_correct": "" if result.is_correct is None else result.is_correct,
                    "latency_seconds": result.latency,
                    "cache_hit": bool(result.hit),
                    "cache_source": result.hit.source if result.hit else "",
                    "techniques_json": json.dumps(result.techniques, sort_keys=True),
                    "metadata_json": json.dumps(sample.metadata, sort_keys=True),
                }
            )


def _slug(value: str) -> str:
    value = value.strip().replace("/", "-")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "run"


def _default_log_path(args: argparse.Namespace) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dataset = _slug(getattr(args, "dataset", "run"))
    chunk = _slug(getattr(args, "chunk_source", "chunk"))
    model = _slug(getattr(args, "model", "model"))
    filename = f"{timestamp}-{dataset}-{chunk}-{model}.csv"
    return Path("experiment_logs") / filename


def _format_metric_value(metric: VLLMMetric) -> object:
    if isinstance(metric, (MetricGauge, MetricCounter)):
        return metric.value
    if isinstance(metric, MetricVector):
        return metric.values
    if isinstance(metric, MetricHistogram):
        return {"count": metric.count, "sum": metric.sum, "buckets": metric.buckets}
    return ""


def summarize_vllm_metrics(llm: LLM) -> None:
    try:
        metrics = llm.get_metrics()
    except AssertionError:
        print("\nvLLM metrics unavailable (log stats disabled).")
        return
    except Exception as exc:
        print(f"\n[vLLM metrics] Unable to read metrics: {exc}")
        return
    if not metrics:
        print("\nNo vLLM metrics were reported.")
        return
    interesting = [
        metric
        for metric in metrics
        if any(token in metric.name for token in {"prefix", "kv_cache"})
    ]
    if not interesting:
        interesting = metrics
        print("\n=== vLLM metrics (all) ===")
    else:
        print("\n=== vLLM prefix/KV cache metrics ===")
    for metric in interesting:
        labels = ", ".join(f"{key}={value}" for key, value in sorted(metric.labels.items()))
        value = _format_metric_value(metric)
        label_str = f"[{labels}]" if labels else ""
        print(f"- {metric.name} {label_str}: {value}")


def parse_args() -> argparse.Namespace:
    default_template = (
        "You are assisting with the GQA benchmark. "
        "Answer the question based on the referenced image.\n"
        "Image ID: {image_id}\n"
        "Question: {question}\n"
        "Answer:"
    )
    parser = argparse.ArgumentParser(
        description="Benchmark semantic KV caching on structured VQA datasets using vLLM."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct", help="Model path or name for vLLM.")
    parser.add_argument(
        "--dataset",
        choices=["gqa", "llava150k", "synthetic"],
        default="gqa",
        help="Dataset to evaluate (GQA, LLaVA-Instruct-150K, or the synthetic cache demo).",
    )
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
    parser.add_argument(
        "--log-csv",
        default=None,
        help="Path to the CSV log file. Defaults to experiment_logs/run-<timestamp>.csv.",
    )
    parser.add_argument(
        "--llava-data-url",
        default=LLAVA_DATA_URL,
        help="Override the default download URL for LLaVA-Instruct-150K JSON.",
    )
    parser.add_argument(
        "--disable-log-stats",
        action="store_true",
        help="Disable vLLM internal metrics/logging (prefix cache stats, etc.).",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["dry-run", "live"],
        default="dry-run",
        help="Use 'dry-run' to avoid touching KV tensors (safe default); "
        "set to 'live' to capture/inject real blocks (experimental).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = None if args.max_samples is None or args.max_samples < 0 else args.max_samples
    if args.dataset == "gqa":
        dataset = load_gqa_dataset(args.dataset_config, args.split, limit, args.shuffle_seed)
        base_samples: Sequence[dict] = dataset
        dataset_label = f"lmms-lab/GQA ({args.dataset_config}, split={args.split})"
    elif args.dataset == "llava150k":
        dataset = load_llava_dataset(args.llava_data_url, limit, args.shuffle_seed)
        base_samples = build_llava_records(dataset)
        dataset_label = "LLaVA-Instruct-150K"
    else:
        dataset = build_synthetic_samples()
        base_samples = dataset if limit is None else dataset[:limit]
        dataset_label = "synthetic-cache-validation"
    prompts = build_prompts(base_samples, args.prompt_template, args.chunk_source)
    print(f"Loaded {len(prompts)} prompts from {dataset_label}.")
    summarize_chunk_texts(prompts)

    try:
        llm = LLM(
            model=args.model,
            trust_remote_code=args.trust_remote_code,
            disable_log_stats=args.disable_log_stats,
        )
    except (OSError, RepositoryNotFoundError, GatedRepoError, HfHubHTTPError) as exc:
        _raise_model_load_help(args.model, exc)
    layer_configs = [parse_embedding_layer_spec(spec) for spec in args.embedding_layer]
    cache_config = SemanticCacheConfig(
        similarity_threshold=args.similarity_threshold,
        max_cached_blocks=args.max_cached_blocks,
        cache_dir=args.cache_dir,
        index_encoder=args.index_encoder,
        embedding_layers=layer_configs,
        dry_run=(args.cache_mode != "live"),
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
    log_path = args.log_csv or _default_log_path(args)
    log_results_csv(log_path, results)
    print(f"Detailed results saved to {log_path}")
    summarize_vllm_metrics(llm)

    print("\n=== Sample outputs ===")
    for result in results[: args.report_limit]:
        source = result.hit.source if result.hit else "none"
        print(
            f"ID={result.sample.dataset_id} | img={result.sample.image_id} | "
            f"{'hit' if result.hit else 'miss'} ({source}) | latency={result.latency:.3f}s"
        )
        technique_str = ", ".join(
            f"{name}={status}" for name, status in sorted(result.techniques.items())
        )
        print(f"Techniques: {technique_str}")
        print(f"Q: {result.sample.question}")
        print(f"Model: {result.response}")
        if result.sample.reference:
            print(f"Reference: {result.sample.reference}")
        print("---")


if __name__ == "__main__":
    main()
