from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
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
import torch
from datasets import Dataset, load_dataset
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import (
    Counter as MetricCounter,
    Gauge as MetricGauge,
    Histogram as MetricHistogram,
    Metric as VLLMMetric,
    Vector as MetricVector,
)

from experiment2.model_presets import BASE_PROMPT_TEMPLATE, apply_preset_to_args, list_model_presets
from experiment2.model_router import ModelRouter, ModelRouterConfig
from experiment2.semantic_cache import SemanticCache, SemanticCacheConfig
from experiment2.semantic_cache.embedding_hooks import (
    EmbeddingHook,
    NullEmbeddingHook,
    load_embedding_hook,
)
from experiment2.semantic_cache.embedding_store import EmbeddingStore
from experiment2.semantic_cache.experiment_driver import drain_request
from experiment2.semantic_cache.semantic_cache import CacheHit, ReuseReport
from experiment2.semantic_cache.techniques import EmbeddingLayerConfig

LLAVA_DATA_URL = (
    "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/"
    "resolve/main/llava_instruct_150k.json?download=1"
)
DEFAULT_GQA_IMAGE_DIR = "experiment2/gqa_images"
_IMAGE_PLACEHOLDERS = (
    "<image>",
    "<|image_0|>",
    "<|vision_start|>",
    "<img>",
    "<|vision_start|><|image_pad|><|vision_end|>",
)


@lru_cache(maxsize=1)
def _scratch_root() -> Path | None:
    candidates: tuple[str | None, ...] = (
        os.environ.get("VMC_CACHE_ROOT"),
        os.environ.get("SCRATCH_DIR"),
        "/workspace",
    )
    for candidate in candidates:
        if not candidate:
            continue
        root_path = Path(candidate).expanduser()
        try:
            root_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(root_path, os.W_OK):
            return root_path
    return None


def _resolve_storage_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    scratch_root = _scratch_root()
    if scratch_root is not None:
        return scratch_root / path
    return Path.cwd() / path


def _maybe_log_relocated(label: str, requested: str, resolved: Path) -> None:
    requested_path = Path(requested).expanduser()
    if requested_path.is_absolute():
        return
    default = Path.cwd() / requested_path
    if resolved != default:
        print(f"[info] {label} resolved to {resolved}")


def _gb_to_bytes(limit_gb: float | None) -> int | None:
    if limit_gb is None:
        return None
    if limit_gb <= 0:
        return 0
    return int(limit_gb * (1024**3))


LLAVA_CACHE_DIR = _resolve_storage_dir("experiment2/dataset_cache")


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
    image_path: str | None
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


class PromptImageLoader:
    """Opens cached dataset images once per prompt for multimodal requests."""

    def __init__(self, placeholder: str = "<image>") -> None:
        self._warned_paths: set[str] = set()
        self.placeholder = placeholder

    def load(self, sample: GQAPrompt) -> Image.Image | None:
        path = sample.image_path
        if not path:
            return None
        try:
            with Image.open(path) as pil_image:
                return pil_image.convert("RGB")
        except Exception as exc:  # pragma: no cover - defensive
            if path not in self._warned_paths:
                print(
                    f"[warn] Unable to load image for sample {sample.dataset_id} "
                    f"(image_id={sample.image_id}, path={path}): {exc}"
                )
                self._warned_paths.add(path)
            return None


def _ensure_image_placeholder(prompt: str) -> str:
    lowered = prompt.lower()
    if any(token.lower() in lowered for token in _IMAGE_PLACEHOLDERS):
        return prompt
    if prompt.strip():
        return f"<image>\n{prompt}"
    return "<image>"


def _ensure_image_placeholder_custom(prompt: str, placeholder: str) -> str:
    lowered = prompt.lower()
    if any(token.lower() in lowered for token in _IMAGE_PLACEHOLDERS):
        return prompt
    if prompt.strip():
        return f"{placeholder}\n{prompt}"
    return placeholder


def _default_image_placeholder(model_name: str) -> str:
    lowered = (model_name or "").lower()
    if "qwen" in lowered:
        return "<|vision_start|><|image_pad|><|vision_end|>"
    return "<image>"


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


def _gqa_image_config_name(config: str) -> str:
    if config.endswith("_instructions"):
        return config[: -len("_instructions")] + "_images"
    if config.endswith("_images"):
        return config
    return f"{config}_images"


def _gqa_image_cache_path(root: Path, image_id: str) -> Path:
    slug = str(image_id)
    if len(slug) > 3:
        return root / slug[:3] / f"{slug}.jpg"
    return root / f"{slug}.jpg"


def _serialize_image(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, dict) and "bytes" in image_obj:
        return Image.open(BytesIO(image_obj["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported GQA image payload type: {type(image_obj)}")


def ensure_gqa_images(
    image_ids: Iterable[str],
    dataset_config: str,
    split: str,
    cache_dir: Path,
) -> dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, str] = {}
    pending: set[str] = set()
    for raw_id in image_ids:
        image_id = str(raw_id or "").strip()
        if not image_id:
            continue
        path = _gqa_image_cache_path(cache_dir, image_id)
        if path.is_file():
            resolved[image_id] = str(path)
        else:
            pending.add(image_id)
    if not pending:
        return resolved
    config_name = _gqa_image_config_name(dataset_config)
    print(f"Downloading {len(pending)} GQA images to {cache_dir} ...")
    dataset = load_dataset(
        "lmms-lab/GQA",
        config_name,
        split=split,
        streaming=True,
    )
    for sample in dataset:
        sample_id = str(sample.get("id", "")).strip()
        if sample_id not in pending:
            continue
        image = sample.get("image")
        if image is None:
            continue
        image_path = _gqa_image_cache_path(cache_dir, sample_id)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            converted = _serialize_image(image)
            converted.save(image_path, format="JPEG")
        except Exception as exc:
            print(f"[warn] Failed to save image {sample_id} to {image_path}: {exc}")
            continue
        resolved[sample_id] = str(image_path)
        pending.remove(sample_id)
        if not pending:
            break
    if pending:
        print(
            f"[warn] Unable to locate {len(pending)} GQA images in split '{split}' "
            f"(config={config_name}). Missing examples include: {sorted(list(pending))[:5]}"
        )
    return resolved


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
    image_paths: Mapping[str, str] | None = None,
) -> list[GQAPrompt]:
    prompts: list[GQAPrompt] = []
    for sample in dataset:
        prompt = format_prompt(prompt_template, sample)
        chunk_text = sample.get("_chunk_override")
        if not chunk_text:
            chunk_text = render_chunk_text(sample, chunk_source)
        image_id = sample.get("imageId", "")
        image_path = image_paths.get(str(image_id)) if image_paths and image_id else None
        metadata_extra = dict(sample.get("_metadata", {}) or {})
        if image_path:
            metadata_extra["image_path"] = str(image_path)
        prompts.append(
            GQAPrompt(
                dataset_id=sample.get("id", ""),
                image_id=sample.get("imageId", ""),
                image_path=image_path,
                question=sample.get("question", ""),
                answer=sample.get("answer", ""),
                full_answer=sample.get("fullAnswer", ""),
                prompt=prompt,
                chunk_text=chunk_text,
                metadata={
                    "groups": sample.get("groups", {}),
                    "semantic": sample.get("semanticStr", ""),
                    "extra": metadata_extra,
                },
            )
        )
    return prompts


def _run_samples_hf(
    llm: Any,  # HFModelWrapper
    cache: Any,
    prompts: Sequence[GQAPrompt],
    sampling_params: SamplingParams,
    *,
    embedding_hook: EmbeddingHook | None = None,
    model_name: str | None = None,
    image_loader: PromptImageLoader | None = None,
    embedding_store: EmbeddingStore | None = None,
) -> list[PromptResult]:
    """Run samples using Hugging Face transformers (simpler, no vLLM engine)."""
    results: list[PromptResult] = []
    total = len(prompts)
    start_run = time.perf_counter()
    requires_llm_request = getattr(cache, "requires_llm_request", True)
    
    for idx, sample in enumerate(prompts, 1):
        request_id = uuid.uuid4().hex
        embeddings: dict[str, np.ndarray] = {}
        
        # Try to get cached embeddings first to avoid expensive extraction
        if embedding_store and embedding_hook is not None:
            cached_embeddings = embedding_store.get(sample.chunk_text, sample.image_id)
            if cached_embeddings:
                embeddings = cached_embeddings
        
        # Extract embeddings if not cached
        if not embeddings and embedding_hook is not None:
            try:
                embeddings = embedding_hook(llm=llm, sample=sample) or {}
                # Cache extracted embeddings for future use
                if embeddings and embedding_store:
                    embedding_store.put(sample.chunk_text, embeddings, sample.image_id)
            except Exception as exc:
                print(f"[warn] embedding hook failed for {sample.dataset_id}: {exc}")
        
        # Load image if needed
        image = None
        if image_loader is not None:
            image = image_loader.load(sample)
        
        # Try cache first (if supported by backend)
        cache_start = time.perf_counter()
        reuse = cache.try_reuse(request_id, sample.chunk_text, embeddings=embeddings)
        hit = reuse.hit
        sample_metadata = {"dataset_id": sample.dataset_id, "image_id": sample.image_id}
        
        # For vision models with embedding cache, verify image_id matches (prevent wrong image reuse)
        if hit and hit.source.startswith("embedding"):
            # Check if cached chunk has same image_id
            stored_chunk = cache.store.load(hit.chunk_id)
            if stored_chunk and stored_chunk.metadata.get("image_id") != sample.image_id:
                # Image mismatch - invalidate the hit
                hit_source = hit.source
                hit = None
                reuse.statuses[hit_source] = "miss:image_mismatch"
                reuse = ReuseReport(hit=None, statuses=reuse.statuses, response=None)
        
        response_text: str
        latency: float
        
        if reuse.response is not None:
            # Cache hit - measure cache lookup time
            response_text = reuse.response.strip()
            latency = time.perf_counter() - cache_start
        else:
            # Cache miss - need to run model
            chunk_id = None
            if hit is None:
                result_chunk = cache.add_observation(
                    request_id,
                    sample.chunk_text,
                    embeddings=embeddings,
                    metadata=sample_metadata,
                )
                if result_chunk:
                    chunk_id = result_chunk.chunk_id
            
            start = time.perf_counter()
            
            # Create cache key for encoder reuse
            encoder_cache_key = f"{sample.dataset_id}_{sample.image_id}"
            
            # Generate with HF model (will reuse cached encoder outputs if available)
            outputs = llm.generate(
                prompts=[sample.prompt],
                sampling_params=sampling_params,
                images=[image] if image else None,
                encoder_cache_key=encoder_cache_key if embeddings else None
            )
            response_text = outputs[0].outputs[0].text.strip()
            
            # Clear cached encoder outputs for this sample to save memory
            if hasattr(llm, '_cached_encoder_outputs') and encoder_cache_key in llm._cached_encoder_outputs:
                del llm._cached_encoder_outputs[encoder_cache_key]
            
            latency = time.perf_counter() - start
            
            finalize_metadata = dict(sample_metadata)
            if chunk_id:
                finalize_metadata["chunk_id"] = chunk_id
            cache.finalize_observation(
                request_id,
                response=response_text,
                model_name=model_name,
                metadata=finalize_metadata,
            )
        
        is_correct = _answer_matches(sample.reference, response_text)
        results.append(
            PromptResult(
                sample=sample,
                hit=hit,
                latency=latency,
                response=response_text,
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
            f"{'hit' if hit else 'miss'} ({source}) | latency={latency:.6f}s | "
            f"elapsed={elapsed:.1f}s | eta={eta:.1f}s | answer match={is_correct} | "
            f"{technique_str}"
        )
    
    return results


def run_samples(
    llm: Any,  # LLM or HFModelWrapper
    cache: Any,
    prompts: Sequence[GQAPrompt],
    sampling_params: SamplingParams,
    *,
    embedding_hook: EmbeddingHook | None = None,
    model_name: str | None = None,
    image_loader: PromptImageLoader | None = None,
    embedding_store: EmbeddingStore | None = None,
) -> list[PromptResult]:
    # Check if using HuggingFace wrapper
    is_hf = not hasattr(llm, 'llm_engine')
    
    if is_hf:
        return _run_samples_hf(
            llm, cache, prompts, sampling_params,
            embedding_hook=embedding_hook,
            model_name=model_name,
            image_loader=image_loader,
            embedding_store=embedding_store,
        )
    
    engine = llm.llm_engine
    results: list[PromptResult] = []
    total = len(prompts)
    start_run = time.perf_counter()
    requires_llm_request = getattr(cache, "requires_llm_request", True)
    for idx, sample in enumerate(prompts, 1):
        request_id = uuid.uuid4().hex
        embeddings: dict[str, np.ndarray] = {}
        if embedding_hook is not None:
            try:
                embeddings = embedding_hook(llm=llm, sample=sample) or {}
            except Exception as exc:
                print(f"[warn] embedding hook failed for {sample.dataset_id}: {exc}")
                embeddings = {}
        prompt_payload: Any
        request_prompt_text = sample.prompt
        if image_loader is not None:
            image = image_loader.load(sample)
            if image is not None:
                request_prompt_text = _ensure_image_placeholder_custom(
                    request_prompt_text, image_loader.placeholder
                )
                prompt_payload = {"prompt": request_prompt_text, "multi_modal_data": {"image": image}}
            else:
                prompt_payload = request_prompt_text
        else:
            prompt_payload = request_prompt_text
        engine_prompt = prompt_payload
        tokenization_kwargs = None
        process_inputs = getattr(llm, "_process_inputs", None)
        if process_inputs is not None:
            engine_prompt, tokenization_kwargs = process_inputs(
                request_id,
                prompt_payload,
                sampling_params,
                lora_request=None,
                priority=0,
            )
        elif image_loader is not None:
            # Multi-modal inputs require vLLM v1 processor support.
            image_loader = None
            print(
                "[warn] vLLM build does not expose _process_inputs; "
                "falling back to text-only prompts without images."
            )
            engine_prompt = request_prompt_text
            prompt_payload = request_prompt_text
        engine_request_added = False

        print(request_prompt_text)
        def _enqueue_request() -> None:
            nonlocal engine_request_added
            if engine_request_added:
                return
            if process_inputs is not None:
                engine.add_request(
                    request_id,
                    engine_prompt,
                    sampling_params,
                    tokenization_kwargs=tokenization_kwargs,
                    prompt_text=request_prompt_text,
                )
            else:
                engine.add_request(request_id, request_prompt_text, sampling_params)
            engine_request_added = True

        cache_start = time.perf_counter()
        if requires_llm_request:
            _enqueue_request()
            reuse = cache.try_reuse(request_id, sample.chunk_text, embeddings=embeddings)
        else:
            reuse = cache.try_reuse(request_id, sample.chunk_text, embeddings=embeddings)
            if reuse.response is None:
                _enqueue_request()
        hit = reuse.hit
        sample_metadata = {"dataset_id": sample.dataset_id, "image_id": sample.image_id}
        response_text: str
        latency: float
        if reuse.response is not None:
            response_text = reuse.response.strip()
            latency = time.perf_counter() - cache_start
        else:
            if not engine_request_added:
                _enqueue_request()
            if hit is None:
                cache.add_observation(
                    request_id,
                    sample.chunk_text,
                    embeddings=embeddings,
                    metadata=sample_metadata,
                )
            start = time.perf_counter()
            response = drain_request(engine, request_id)
            latency = time.perf_counter() - start
            response_text = response.strip()
            cache.finalize_observation(
                request_id,
                response=response_text,
                model_name=model_name,
                metadata=sample_metadata,
            )
        is_correct = _answer_matches(sample.reference, response_text)
        results.append(
            PromptResult(
                sample=sample,
                hit=hit,
                latency=latency,
                response=response_text,
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
            f"{'hit' if hit else 'miss'} ({source}) | latency={latency:.6f}s | "
            f"elapsed={elapsed:.1f}s | eta={eta:.1f}s | answer match={is_correct} | "
            f"{technique_str}"
        )
    return results


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
    print(f"Average latency: {_mean(latencies):.6f}s")
    print(f"Average latency (hit): {_mean([r.latency for r in hits]):.6f}s")
    print(f"Average latency (miss): {_mean([r.latency for r in misses]):.6f}s")
    print(f"Answer match rate: {match_rate:.1%} (n={len(matchable)})")
    if hits_by_source:
        print("Hit rate by cache:")
        for source, source_hits in sorted(hits_by_source.items(), key=lambda item: item[0]):
            rate = len(source_hits) / len(results)
            print(
                f"  - {source}: {len(source_hits)} hits ({rate:.1%} of prompts) | "
                f"avg latency={_mean([r.latency for r in source_hits]):.6f}s"
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


def write_samples_jsonl(path: str | Path, results: Sequence[PromptResult]) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
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


def log_summary_row(path: str | Path, args: argparse.Namespace, aggregates: dict[str, Any]) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    max_samples = args.max_samples if args.max_samples is not None else -1
    shuffle_seed: str | int | None = args.shuffle_seed
    if isinstance(shuffle_seed, int) and shuffle_seed < 0:
        shuffle_seed = ""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment_name or "",
        "model": args.model,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "max_samples": max_samples,
        "shuffle_seed": shuffle_seed if shuffle_seed is not None else "",
        "chunk_source": args.chunk_source,
        "prompt_template": args.prompt_template.replace("\n", "\\n"),
        "similarity_threshold": args.similarity_threshold,
        "embedding_layers": ",".join(args.embedding_layer),
        "embedding_hook": args.embedding_hook,
        "max_cached_blocks": args.max_cached_blocks if args.max_cached_blocks is not None else "",
        "cache_dir": args.cache_dir,
        "index_encoder": args.index_encoder,
        "index_encoder_device": args.index_encoder_device,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "trust_remote_code": args.trust_remote_code,
        "notes": "",
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_fusion_cache": args.enable_fusion_cache,
        "fusion_cache_dir": args.fusion_cache_dir,
        "enable_semantic_text_cache": not args.disable_semantic_cache,
        "enable_exact_text_cache": not args.disable_exact_cache,
        "preset": args.preset or "",
        "total_prompts": aggregates["total_prompts"],
        "cache_hits": aggregates["cache_hits"],
        "cache_hit_rate": f"{aggregates['cache_hit_rate']:.6f}",
        "avg_latency": f"{aggregates['avg_latency']:.6f}" if aggregates['avg_latency'] is not None else "",
        "avg_latency_hit": f"{aggregates['avg_latency_hit']:.6f}" if aggregates['avg_latency_hit'] is not None else "",
        "avg_latency_miss": f"{aggregates['avg_latency_miss']:.6f}" if aggregates['avg_latency_miss'] is not None else "",
        "answer_accuracy": f"{aggregates['answer_accuracy']:.6f}" if aggregates['answer_accuracy'] is not None else "",
        "wall_time": aggregates.get("wall_time"),
        "hit_breakdown": json.dumps(aggregates["hit_breakdown"]),
    }
    fieldnames = list(row.keys())
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _purge_cache_dir(path_str: str | None) -> None:
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
    shutil.rmtree(resolved, ignore_errors=True)


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
    return Path("experiment2/experiment_logs") / filename


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_template = BASE_PROMPT_TEMPLATE
    parser = argparse.ArgumentParser(
        description="Benchmark semantic KV caching on structured VQA datasets using vLLM."
    )
    parser.add_argument(
        "--preset",
        choices=list_model_presets(),
        default=None,
        help="Optional preset that pre-populates recommended arguments for a model family.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct", help="Model path or name for vLLM.")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="How many GPUs to use for tensor parallelism.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of each GPU memory vLLM is allowed to use (0-1).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional cap on vLLM max model length to curb KV cache allocation.",
    )
    parser.add_argument(
        "--dataset",
        choices=["gqa", "llava150k", "synthetic"],
        default="gqa",
        help="Dataset to evaluate (GQA, LLaVA-Instruct-150K, or the synthetic cache demo).",
    )
    parser.add_argument(
        "--gqa-image-dir",
        default=DEFAULT_GQA_IMAGE_DIR,
        help="Directory where GQA images downloaded from Hugging Face are cached.",
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
    parser.add_argument(
        "--cache-dir",
        default="experiment2/kv_chunks",
        help="Where cached KV chunks are stored.",
    )
    parser.add_argument(
        "--cache-max-size-gb",
        type=float,
        default=None,
        help="Optional disk budget (in GB) for KV cache; oldest chunks are pruned when exceeded.",
    )
    parser.add_argument(
        "--cache-backend",
        choices=["semantic-kv", "model-router"],
        default="semantic-kv",
        help="Selects the caching backend. 'semantic-kv' reuses KV blocks via vLLM; "
        "'model-router' performs response-level shortcutting based on semantic matches.",
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Use Hugging Face transformers instead of vLLM (avoids multimodal cache bugs).",
    )
    parser.add_argument(
        "--index-encoder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model used for semantic search.",
    )
    parser.add_argument(
        "--index-encoder-device",
        default="cuda",
        help="Device to run the semantic text encoder on (e.g., 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--fusion-cache-dir",
        default="experiment2/fusion_chunks",
        help="Directory for persisted fusion cache states (if enabled).",
    )
    parser.add_argument(
        "--enable-fusion-cache",
        action="store_true",
        help="Capture and inject fusion tensors in addition to KV cache blocks.",
    )
    parser.add_argument(
        "--keep-cache-dirs",
        action="store_true",
        help="Skip purging cache/fusion directories before running (default purges).",
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
        help="Path to the CSV log file. Defaults to experiment2/experiment_logs/run-<timestamp>.csv.",
    )
    parser.add_argument(
        "--summary-log",
        default=None,
        help="Optional CSV file where aggregate metrics are appended (matches run_experiments format).",
    )
    parser.add_argument(
        "--samples-jsonl",
        default=None,
        help="Optional JSONL path for per-sample dumps (mirrors run_experiments).",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional label recorded in summary logs (e.g., qwen-exact-only).",
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
    parser.add_argument(
        "--disable-semantic-cache",
        action="store_true",
        help="Completely bypass the semantic text cache layer.",
    )
    parser.add_argument(
        "--disable-exact-cache",
        action="store_true",
        help="Disable the normalized exact-text cache layer.",
    )
    defaults = parser.parse_args(args=[])
    args = parser.parse_args(args=argv)
    apply_preset_to_args(args, defaults)
    return args


def main() -> None:
    args = parse_args()
    if args.cache_dir:
        resolved_cache_dir = _resolve_storage_dir(args.cache_dir)
        _maybe_log_relocated("cache_dir", args.cache_dir, resolved_cache_dir)
        args.cache_dir = str(resolved_cache_dir)
    if args.fusion_cache_dir:
        resolved_fusion_dir = _resolve_storage_dir(args.fusion_cache_dir)
        _maybe_log_relocated("fusion_cache_dir", args.fusion_cache_dir, resolved_fusion_dir)
        args.fusion_cache_dir = str(resolved_fusion_dir)
    if getattr(args, "gqa_image_dir", None):
        resolved_gqa_dir = _resolve_storage_dir(args.gqa_image_dir)
        _maybe_log_relocated("gqa_image_dir", args.gqa_image_dir, resolved_gqa_dir)
        args.gqa_image_dir = str(resolved_gqa_dir)
    limit = None if args.max_samples is None or args.max_samples < 0 else args.max_samples
    cache_size_limit_bytes = _gb_to_bytes(getattr(args, "cache_max_size_gb", None))
    shuffle_seed = None if args.shuffle_seed is None or args.shuffle_seed < 0 else args.shuffle_seed
    if not args.keep_cache_dirs:
        _purge_cache_dir(args.cache_dir)
        if args.enable_fusion_cache:
            _purge_cache_dir(args.fusion_cache_dir)
    image_path_map: dict[str, str] | None = None
    if args.dataset == "gqa":
        dataset = load_gqa_dataset(args.dataset_config, args.split, limit, shuffle_seed)
        base_samples: Sequence[dict] = dataset
        dataset_label = f"lmms-lab/GQA ({args.dataset_config}, split={args.split})"
        image_ids: Iterable[str]
        if isinstance(dataset, Dataset):
            image_ids = dataset["imageId"]
        else:
            image_ids = [sample.get("imageId", "") for sample in base_samples]
        cache_root = Path(args.gqa_image_dir)
        image_path_map = ensure_gqa_images(image_ids, args.dataset_config, args.split, cache_root)
    elif args.dataset == "llava150k":
        dataset = load_llava_dataset(args.llava_data_url, limit, shuffle_seed)
        base_samples = build_llava_records(dataset)
        dataset_label = "LLaVA-Instruct-150K"
    else:
        dataset = build_synthetic_samples()
        base_samples = dataset if limit is None else dataset[:limit]
        dataset_label = "synthetic-cache-validation"
    prompts = build_prompts(
        base_samples,
        args.prompt_template,
        args.chunk_source,
        image_paths=image_path_map or {},
    )
    print(f"Loaded {len(prompts)} prompts from {dataset_label}.")
    summarize_chunk_texts(prompts)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. vLLM requires a CUDA-enabled GPU.")
        print("Please ensure:")
        print("  1. You have a CUDA-capable GPU installed")
        print("  2. CUDA drivers are properly installed")
        print("  3. PyTorch can detect the GPU")
        print("  4. CUDA_VISIBLE_DEVICES is set correctly if you have multiple GPUs")
        sys.exit(1)
    
    print(f"CUDA device count: {torch.cuda.device_count()}")
    try:
        torch.cuda.init()  # Explicitly initialize CUDA
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    except RuntimeError as e:
        print(f"ERROR: Failed to initialize CUDA device: {e}")
        print("CUDA is detected but cannot be initialized.")
        print("This may indicate:")
        print("  1. CUDA driver version mismatch with PyTorch")
        print("  2. GPU is already in use or in a bad state")
        print("  3. Insufficient permissions to access the GPU")
        print("  4. Multiple GPU initialization issue - try setting CUDA_VISIBLE_DEVICES=0")
        print("Try running: nvidia-smi")
        print("\nTo fix, run your script with: CUDA_VISIBLE_DEVICES=0 python ...")
        sys.exit(1)
    
    # Initialize model - either vLLM or Hugging Face transformers
    if args.use_hf:
        print("[info] Using Hugging Face transformers (no vLLM multimodal cache)")
        from experiment2.hf_model_wrapper import HFModelWrapper
        
        llm = HFModelWrapper(
            model=args.model,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    else:
        # Workaround for vLLM 0.12.0 multimodal cache assertion bug in v1 engine
        # The v1 engine has a bug where multimodal features get evicted from cache
        # but later requests still expect them to be there (race condition with prefix caching)
        import os
        os.environ['VLLM_USE_V1'] = '0'
        print("[info] Using vLLM (attempting v0 engine to avoid multimodal cache bug)")
        
        try:
            llm = LLM(
                model=args.model,
                trust_remote_code=args.trust_remote_code,
                disable_log_stats=args.disable_log_stats,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                enforce_eager=True,
            )
        except (OSError, RepositoryNotFoundError, GatedRepoError, HfHubHTTPError) as exc:
            _raise_model_load_help(args.model, exc)
    layer_configs = [parse_embedding_layer_spec(spec) for spec in args.embedding_layer]
    if args.cache_backend == "semantic-kv":
        cache_config = SemanticCacheConfig(
            similarity_threshold=args.similarity_threshold,
            max_cached_blocks=args.max_cached_blocks,
            cache_dir=args.cache_dir,
            index_encoder=args.index_encoder,
            index_encoder_device=args.index_encoder_device,
            fusion_cache_dir=args.fusion_cache_dir,
            enable_fusion_cache=args.enable_fusion_cache,
            embedding_layers=layer_configs,
            dry_run=(args.cache_mode != "live"),
            enable_semantic_text_cache=not args.disable_semantic_cache,
            enable_exact_text_cache=not args.disable_exact_cache,
            cache_size_limit_bytes=cache_size_limit_bytes,
        )
        cache: Any = SemanticCache(llm, config=cache_config)
    else:
        router_config = ModelRouterConfig(
            similarity_threshold=args.similarity_threshold,
            response_cache_dir=args.cache_dir,
            index_encoder=args.index_encoder,
            index_encoder_device=args.index_encoder_device,
            embedding_layers=layer_configs,
            enable_semantic_text_cache=not args.disable_semantic_cache,
            enable_exact_text_cache=not args.disable_exact_cache,
        )
        cache = ModelRouter(llm, config=router_config)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    prompt_image_loader: PromptImageLoader | None = None
    if image_path_map:
        if hasattr(llm, "_process_inputs"):
            placeholder_text = _default_image_placeholder(args.model)
            prompt_image_loader = PromptImageLoader(placeholder=placeholder_text)
        else:
            print(
                "[warn] Installed vLLM version does not expose _process_inputs; "
                "continuing without attaching images to prompts."
            )
    embedding_hook: EmbeddingHook | None
    if layer_configs:
        embedding_hook = load_embedding_hook(args.embedding_hook)
    else:
        embedding_hook = NullEmbeddingHook()

    # Create embedding store for caching extracted embeddings (avoids re-extraction overhead)
    embedding_store: EmbeddingStore | None = None
    if layer_configs and args.embedding_hook.startswith("native"):
        embedding_store_dir = Path(args.cache_dir) / "embedding_cache"
        embedding_store = EmbeddingStore(embedding_store_dir)
        print(f"[info] Embedding caching enabled at {embedding_store_dir}")

    start_overall = time.perf_counter()
    results = run_samples(
        llm,
        cache,
        prompts,
        sampling_params,
        embedding_hook=embedding_hook,
        model_name=args.model,
        image_loader=prompt_image_loader,
        embedding_store=embedding_store,
    )
    wall_time = time.perf_counter() - start_overall
    aggregates = aggregate_results(results)
    aggregates["wall_time"] = wall_time
    summarize_results(results)
    log_path = args.log_csv or _default_log_path(args)
    log_results_csv(log_path, results)
    print(f"Detailed results saved to {log_path}")
    if args.summary_log:
        log_summary_row(args.summary_log, args, aggregates)
    if args.samples_jsonl:
        write_samples_jsonl(args.samples_jsonl, results)
    summarize_vllm_metrics(llm)

    print("\n=== Sample outputs ===")
    for result in results[: args.report_limit]:
        source = result.hit.source if result.hit else "none"
        print(
            f"ID={result.sample.dataset_id} | img={result.sample.image_id} | "
            f"{'hit' if result.hit else 'miss'} ({source}) | latency={result.latency:.6f}s"
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
