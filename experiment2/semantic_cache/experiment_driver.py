from __future__ import annotations

from typing import Any

try:  # vLLM >=0.5
    from vllm.v1.outputs import RequestOutput  # type: ignore
except Exception:  # pragma: no cover - compatibility fallback
    try:
        from vllm.outputs import RequestOutput  # type: ignore
    except Exception:
        RequestOutput = Any  # type: ignore

try:  # vLLM >=0.5
    from vllm.v1.request import Request  # type: ignore
except Exception:  # pragma: no cover - compatibility fallback
    try:
        from vllm.request import Request  # type: ignore
    except Exception:
        Request = Any  # type: ignore


def drain_request(engine: Any, request_id: str) -> str:
    """Run a request to completion and return the generated text."""

    def _pop_request() -> RequestOutput | None:
        iterator = getattr(engine, "step", None)
        if iterator is None:
            raise RuntimeError("vLLM engine missing .step() iterator.")
        try:
            return next(iterator)
        except StopIteration as exc:
            return exc.value

    finished: RequestOutput | None = None
    while finished is None:
        step_output = _pop_request()
        for request_output in step_output.outputs:
            if request_output.request_id != request_id:
                continue
            finished = request_output
            break
    if not finished:
        raise RuntimeError(f"Request {request_id} never completed.")
    text = finished.outputs[0].text
    if hasattr(engine, "abort_request"):
        try:
            engine.abort_request(request_id)
        except Exception:
            pass
    return text


def force_wait_for_engine(engine: Any) -> None:
    """Block until all active requests are drained."""
    iterator = getattr(engine, "step", None)
    if iterator is None:
        return
    try:
        while True:
            next(iterator)
    except StopIteration:
        return
