#!/usr/bin/env python3
"""
Async, cost-aware GQA answer re-grader using OpenAI o3-mini.

Reads question/reference/response triples, sends batched async grading requests,
and writes a JSONL with the added field `grader_label` in {"correct","incorrect"}.
Includes token-based cost estimation so you can gauge spend before running.
"""

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # allow import for tests that don't call the client

from tqdm.asyncio import tqdm_asyncio

try:
    import tiktoken
except ImportError:  # optional; cost estimation will degrade gracefully
    tiktoken = None

# Pricing for o3-mini as of 2025-02: $0.0004 / 1k input, $0.0016 / 1k output
INPUT_PRICE_PER_1K = 0.0004
OUTPUT_PRICE_PER_1K = 0.0016

PROMPT_TEMPLATE = (
    "You are grading a visual question answering task. "
    "Decide if the model answer matches the reference answer in the context of "
    "the question. Respond EXACTLY with one word: correct or incorrect.\n"
    "Q: {question}\n"
    "Reference: {reference}\n"
    "Response: {response}\n"
    "Answer:"
)


def load_records(path: str) -> List[Dict]:
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        raise ValueError("Expected a list of records; got a dict.")
    return list(data)


def validate_records(records: Iterable[Dict]) -> None:
    for i, rec in enumerate(records):
        for key in ("question", "reference", "response"):
            if key not in rec:
                raise KeyError(f"Record {i} missing required key: {key}")


def chunk(seq: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def extract_label_from_choice(choice) -> str:
    """Normalize OpenAI chat choice content across client versions."""
    # choice may be a namespace or dict
    message = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, str):
        return content.strip().lower()

    if isinstance(content, list) and content:
        part = content[0]
        if hasattr(part, "text"):
            text = part.text
        elif isinstance(part, dict):
            text = part.get("text", "")
        else:
            text = ""
        return text.strip().lower()

    # Some SDKs put text directly on the message
    if hasattr(message, "text"):
        return str(message.text).strip().lower()
    if isinstance(message, dict) and "text" in message:
        return str(message["text"]).strip().lower()

    return ""


def extract_label_from_response(resp: Any) -> str:
    """Pull a text label from either Responses API or Chat Completions output."""
    # Responses API: prefer output_text if present
    try:
        if hasattr(resp, "output_text") and resp.output_text:
            return str(resp.output_text).strip().lower()

        output = getattr(resp, "output", None) or (resp.get("output") if isinstance(resp, dict) else None)
        if output:
            # output[0].content is a list of content parts; we want the first text part
            content = getattr(output[0], "content", None)
            if content is None and isinstance(output[0], dict):
                content = output[0].get("content")

            if isinstance(content, list) and content:
                first = content[0]
                text = getattr(first, "text", None)
                if text is None and isinstance(first, dict):
                    text = first.get("text")
                if text:
                    return str(text).strip().lower()
    except Exception:
        pass

    # Chat completions fallback
    try:
        choices = getattr(resp, "choices", None) or (resp.get("choices") if isinstance(resp, dict) else None)
        if choices:
            return extract_label_from_choice(choices[0])
    except Exception:
        pass

    return ""


def normalize_label(label: str) -> str:
    """Map any free-form grader output to canonical 'correct'/'incorrect'."""
    text = (label or "").strip().lower()
    if "incorrect" in text:
        return "incorrect"
    if "correct" in text:
        return "correct"
    return "incorrect"


def is_insufficient_quota(exc: Exception) -> bool:
    """Detect unrecoverable quota errors so we don't mark them incorrect."""
    msg = str(exc).lower()
    return "insufficient_quota" in msg or "exceeded your current quota" in msg


async def grade_one(
    client: AsyncOpenAI,
    item: Dict,
    model: str,
    batch_id: int,
    item_id: int,
    max_retries: int = 5,
    backoff: float = 1.0,
    timeout: float = 60.0,
    chat_fallback_model: str = "gpt-4o-mini",
) -> str:
    """
    Grade a single (question, reference, response) triple.

    Uses the Responses API with `model` (e.g., o3-mini).
    If we fail to extract a label but the call succeeded, we fall back to Chat Completions
    using a *chat-capable* model (default gpt-4o-mini).
    """
    user_text = PROMPT_TEMPLATE.format(
        question=item["question"],
        reference=item["reference"],
        response=item["response"],
    )

    delay = backoff
    for attempt in range(1, max_retries + 1):
        try:
            # ✅ Correct Responses API usage for o3-mini:
            #   - just pass a string to `input`
            #   - no weird content-type structures required
            resp = await asyncio.wait_for(
                client.responses.create(
                    model=model,
                    input=user_text,
                    max_output_tokens=16,
                ),
                timeout=timeout
            )
            label = extract_label_from_response(resp)

            # If the call worked but extraction failed, try a cheap chat fallback
            if not label:
                chat_resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=chat_fallback_model,
                        messages=[{"role": "user", "content": user_text}],
                        max_tokens=8,
                        temperature=0,
                    ),
                    timeout=timeout
                )
                label = extract_label_from_response(chat_resp)

            if not label:
                print(
                    f"[Batch {batch_id} item {item_id}] Empty label; treating as incorrect."
                )
                return "incorrect"

            return normalize_label(label)

        except Exception as exc:
            if is_insufficient_quota(exc):
                # Quota issues are not recoverable inside this run; bubble up.
                raise
            if attempt == max_retries:
                print(
                    f"[Batch {batch_id} item {item_id}] Failed after {attempt} attempts: {exc}"
                )
                return "incorrect"
            print(
                f"[Batch {batch_id} item {item_id}] Error ({exc}); retrying in {delay:.1f}s "
                f"(attempt {attempt}/{max_retries})"
            )
            await asyncio.sleep(delay)
            delay *= 1.8

    # Should never reach here
    return "incorrect"


async def grade_batch(
    client: AsyncOpenAI,
    batch: Sequence[Dict],
    model: str,
    batch_id: int,
    max_retries: int = 5,
    timeout: float = 60.0,
) -> List[str]:
    """Grade each item in the batch concurrently (one API call per sample)."""
    tasks = [
        grade_one(
            client=client,
            item=item,
            model=model,
            batch_id=batch_id,
            item_id=j,
            max_retries=max_retries,
            timeout=timeout,
        )
        for j, item in enumerate(batch)
    ]
    # No tqdm here (outer layer already has a progress bar)
    return await asyncio.gather(*tasks)


async def run_async_grading(
    records: List[Dict],
    model: str = "o3-mini",
    batch_size: int = 64,
    workers: int = 10,
    max_retries: int = 5,
    timeout: float = 60.0,
) -> List[str]:
    if AsyncOpenAI is None:
        raise ImportError("openai package is required to run grading.")
    batches = list(chunk(records, batch_size))
    results: List[Optional[List[str]]] = [None] * len(batches)
    sem = asyncio.Semaphore(workers)
    client = AsyncOpenAI(timeout=timeout)  # uses OPENAI_API_KEY from env

    async def worker(batch_id: int, batch: Sequence[Dict]) -> None:
        async with sem:
            outputs = await grade_batch(
                client=client,
                batch=batch,
                model=model,
                batch_id=batch_id,
                max_retries=max_retries,
                timeout=timeout,
            )
            results[batch_id] = outputs

    await tqdm_asyncio.gather(
        *(worker(i, b) for i, b in enumerate(batches)),
        total=len(batches),
        desc="Grading batches",
    )

    # Close client before leaving the running event loop
    await client.close()

    flat: List[str] = []
    for outs in results:
        if outs is None:
            raise RuntimeError("A batch returned no results; check error logs.")
        flat.extend(outs)
    if len(flat) != len(records):
        raise RuntimeError(
            f"Label count mismatch: got {len(flat)} labels for {len(records)} records."
        )
    return flat


async def grade_first_record_only(
    record: Dict,
    model: str = "o3-mini",
    max_retries: int = 5,
    backoff: float = 1.0,
) -> str:
    """Debug helper: call the API once for a single record and return its label."""
    labels = await run_async_grading(
        records=[record],
        model=model,
        batch_size=1,
        workers=1,
        max_retries=max_retries,
    )
    return labels[0]


def write_jsonl(path: str, records: Iterable[Dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def estimate_tokens_and_cost(records: Sequence[Dict]) -> Tuple[int, int, float]:
    if tiktoken is None:
        print("tiktoken not installed; skipping token estimation.")
        return 0, 0, 0.0

    try:
        enc = tiktoken.get_encoding("o200k_base")
    except Exception:
        print("Tokenizer 'o200k_base' not available; skipping token estimation.")
        return 0, 0, 0.0

    prompt_tokens = 0
    for rec in records:
        text = PROMPT_TEMPLATE.format(
            question=rec["question"],
            reference=rec["reference"],
            response=rec["response"],
        )
        prompt_tokens += len(enc.encode(text))

    # We force a very short output
    output_tokens = len(records)
    input_cost = (prompt_tokens / 1000) * INPUT_PRICE_PER_1K
    output_cost = (output_tokens / 1000) * OUTPUT_PRICE_PER_1K
    total = input_cost + output_cost
    return prompt_tokens, output_tokens, total


def load_ablation_logs(logs_dir: str) -> Tuple[List[Dict], List[str]]:
    """
    Load all CSV files from ablation_logs directory.
    Returns a list of all rows and a list of corresponding file paths.
    """
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        raise FileNotFoundError(f"Directory not found: {logs_dir}")
    
    csv_files = sorted(logs_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {logs_dir}")
    
    all_rows = []
    file_paths = []
    
    for csv_file in csv_files:
        print(f"Loading {csv_file.name}...")
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
                file_paths.append(str(csv_file))
    
    return all_rows, file_paths


def extract_unique_grading_tasks(rows: List[Dict]) -> Tuple[List[Dict], Dict[Tuple, str]]:
    """
    Extract unique (question, reference, response) combinations.
    Returns:
        - List of unique records for grading
        - Mapping from (question, reference, response) -> grader_label (to be filled)
    """
    unique_tasks = {}
    for row in rows:
        question = row.get('question', '').strip()
        reference = row.get('reference', '').strip()
        response = row.get('response', '').strip()
        
        key = (question, reference, response)
        if key not in unique_tasks:
            unique_tasks[key] = {
                'question': question,
                'reference': reference,
                'response': response
            }
    
    print(f"Found {len(unique_tasks)} unique (question, reference, response) combinations out of {len(rows)} total rows")
    return list(unique_tasks.values()), {}


def assign_grades_to_rows(
    rows: List[Dict],
    grades: List[str],
    unique_tasks: List[Dict]
) -> List[Dict]:
    """
    Assign grader labels to all rows based on graded unique tasks.
    """
    # Build a lookup from (question, reference, response) -> grader_label
    grade_lookup = {}
    for task, grade in zip(unique_tasks, grades):
        key = (task['question'], task['reference'], task['response'])
        grade_lookup[key] = grade
    
    # Assign grades to all rows
    graded_rows = []
    for row in rows:
        question = row.get('question', '').strip()
        reference = row.get('reference', '').strip()
        response = row.get('response', '').strip()
        
        key = (question, reference, response)
        grader_label = grade_lookup.get(key, 'incorrect')
        
        # Create a new row with grader_label
        new_row = row.copy()
        new_row['grader_label'] = grader_label
        new_row['match_score'] = 1 if grader_label == 'correct' else 0
        graded_rows.append(new_row)
    
    return graded_rows


def write_graded_csvs(
    graded_rows: List[Dict],
    file_paths: List[str],
    output_dir: str
) -> None:
    """
    Write graded rows back to separate CSV files in output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group rows by original file
    file_to_rows = {}
    for row, file_path in zip(graded_rows, file_paths):
        if file_path not in file_to_rows:
            file_to_rows[file_path] = []
        file_to_rows[file_path].append(row)
    
    # Write each file
    for file_path, rows in file_to_rows.items():
        original_name = Path(file_path).name
        output_file = output_path / original_name
        
        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {len(rows)} rows to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async o3-mini grader for GQA-style QA pairs from ablation logs."
    )
    parser.add_argument(
        "--logs-dir",
        default="experiment_logs/ablation_logs",
        help="Directory containing ablation log CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiment_logs/ablation_logs_graded",
        help="Where to write graded CSV files.",
    )
    parser.add_argument("--model", default="o3-mini", help="OpenAI reasoning model name.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Max concurrent batches (semaphore). Reduced default to prevent timeouts.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=5, help="Retries per sample on failure."
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Timeout in seconds for each API call."
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only print token/cost estimate; do not call the API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load all CSV files from ablation_logs
    print(f"Loading CSV files from {args.logs_dir}...")
    all_rows, file_paths = load_ablation_logs(args.logs_dir)
    print(f"Loaded {len(all_rows)} total rows from {len(set(file_paths))} files")
    
    # Extract unique grading tasks
    unique_tasks, grade_lookup = extract_unique_grading_tasks(all_rows)
    validate_records(unique_tasks)
    
    # Estimate cost
    prompt_tokens, output_tokens, est_cost = estimate_tokens_and_cost(unique_tasks)
    if prompt_tokens and output_tokens:
        print(
            f"Estimated tokens — input: {prompt_tokens:,}, "
            f"output: {output_tokens:,}"
        )
        print(f"Estimated cost at o3-mini rates: ${est_cost:.4f}")
    else:
        print("Token estimation unavailable (install tiktoken for estimates).")

    if args.estimate_only:
        print("Estimate-only flag set; exiting without API calls.")
        return

    # Grade unique tasks
    print(f"\nGrading {len(unique_tasks)} unique tasks...")
    print(f"Using {args.workers} workers, batch size {args.batch_size}, timeout {args.timeout}s")
    start = time.time()
    labels = asyncio.run(
        run_async_grading(
            records=unique_tasks,
            model=args.model,
            batch_size=args.batch_size,
            workers=args.workers,
            max_retries=args.max_retries,
            timeout=args.timeout,
        )
    )
    elapsed = time.time() - start
    print(f"Graded {len(unique_tasks)} unique samples in {elapsed:.1f}s.")
    
    # Count results
    correct_count = sum(1 for label in labels if label == 'correct')
    incorrect_count = len(labels) - correct_count
    print(f"Results: {correct_count} correct, {incorrect_count} incorrect")
    
    # Assign grades to all rows
    print(f"\nAssigning grades to all {len(all_rows)} rows...")
    graded_rows = assign_grades_to_rows(all_rows, labels, unique_tasks)
    
    # Write results
    print(f"\nWriting graded CSV files to {args.output_dir}...")
    write_graded_csvs(graded_rows, file_paths, args.output_dir)
    print(f"Done! Graded files written to {args.output_dir}")


if __name__ == "__main__":
    main()
