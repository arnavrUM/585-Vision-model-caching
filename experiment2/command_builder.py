from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from experiment2.specs import DEFAULT_PROMPT_TEMPLATE, ExperimentSpec


def build_benchmark_command(
    spec: ExperimentSpec,
    *,
    cache_mode: str,
    script_path: str = "experiment2/run_benchmark.py",
    log_file: str | None = None,
    samples_dir: str | None = None,
) -> str:
    args: list[str] = ["python", script_path]

    def _add(flag: str, value: Any) -> None:
        args.append(flag)
        args.append(str(value))

    _add("--experiment-name", spec.name)
    _add("--model", spec.model)
    _add("--dataset", "gqa")
    _add("--dataset-config", spec.dataset_config)
    _add("--split", spec.split)
    max_samples = spec.max_samples if spec.max_samples is not None else -1
    _add("--max-samples", max_samples)
    shuffle_seed = spec.shuffle_seed if spec.shuffle_seed is not None else -1
    _add("--shuffle-seed", shuffle_seed)
    _add("--chunk-source", spec.chunk_source)
    if spec.prompt_template != DEFAULT_PROMPT_TEMPLATE:
        _add("--prompt-template", spec.prompt_template)
    _add("--similarity-threshold", spec.similarity_threshold)
    for layer in spec.embedding_layers:
        args.extend(["--embedding-layer", layer])
    _add("--embedding-hook", spec.embedding_hook)
    if spec.max_cached_blocks is not None:
        _add("--max-cached-blocks", spec.max_cached_blocks)
    _add("--cache-dir", spec.cache_dir)
    if getattr(spec, "cache_max_size_gb", None) is not None:
        _add("--cache-max-size-gb", spec.cache_max_size_gb)
    _add("--index-encoder", spec.index_encoder)
    _add("--index-encoder-device", spec.index_encoder_device)
    _add("--fusion-cache-dir", spec.fusion_cache_dir)
    if spec.enable_fusion_cache:
        args.append("--enable-fusion-cache")
    _add("--temperature", spec.temperature)
    _add("--max-tokens", spec.max_tokens)
    _add("--tensor-parallel-size", spec.tensor_parallel_size)
    _add("--gpu-memory-utilization", spec.gpu_memory_utilization)
    _add("--cache-mode", cache_mode)
    if not spec.enable_semantic_text_cache:
        args.append("--disable-semantic-cache")
    if not spec.enable_exact_text_cache:
        args.append("--disable-exact-cache")
    if spec.trust_remote_code:
        args.append("--trust-remote-code")
    log_path = None
    if log_file:
        _add("--summary-log", log_file)
        log_path = Path(log_file).expanduser().parent / f"{spec.name}.csv"
    else:
        log_path = Path("experiment2/experiment_logs") / f"{spec.name}.csv"
    _add("--log-csv", log_path)
    if samples_dir:
        sample_path = Path(samples_dir) / f"{spec.name}.jsonl"
        _add("--samples-jsonl", sample_path)
    command = " ".join(shlex.quote(part) for part in args)
    echo = f'echo "=== {spec.name} ==="'
    cleanup_targets = {spec.cache_dir, spec.fusion_cache_dir}
    cleanup = " && ".join(f"rm -rf {shlex.quote(path)}" for path in cleanup_targets if path)
    prefix = cleanup if cleanup else ""
    pieces = [piece for piece in [prefix, echo, f"CUDA_LAUNCH_BLOCKING=1 {command}"] if piece]
    return " && ".join(pieces)


def build_test_vllm_command(**kwargs) -> str:
    """Backward-compatible alias for the old helper name."""
    return build_benchmark_command(**kwargs)
