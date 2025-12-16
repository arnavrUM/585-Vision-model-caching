#!/usr/bin/env python3
"""Quick test to verify model generation works"""
import sys
sys.path.insert(0, '/home/henrw/585-Vision-model-caching')

from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(
    model="Qwen/Qwen2-VL-2B-Instruct",
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
    mm_processor_kwargs={"use_fast": False},
    max_model_len=2048,  # Reduce for faster init
)

# Simple text-only test
prompts = [
    "What is 2+2? Answer:",
    "The capital of France is"
]

sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated!r}")
    print(f"Generated length: {len(generated)}")
    print()
