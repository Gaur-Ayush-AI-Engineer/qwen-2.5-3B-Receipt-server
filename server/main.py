# server/main.py — FastAPI application
#
# Startup sequence:
#   1. lifespan() context manager runs before the server accepts requests.
#   2. create_model_wrapper() either loads MLX weights or instantiates Mock.
#   3. The wrapper is stored on app.state so all route handlers share it.
#
# Concurrency note (expanded in model.py):
#   All /extract and /extract/batch calls funnel through wrapper.infer(),
#   which holds an asyncio.Lock. Requests queue behind the lock.
#   /health and /metrics never touch the lock — they always respond fast.

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from config import MAX_BATCH_SIZE, MODEL_ID, PORT, HOST
from server.metrics import tracker
from server.model import create_model_wrapper
from server.schemas import (
    BatchExtractionRequest,
    BatchExtractionResponse,
    ExtractionRequest,
    ExtractionResponse,
    ExtractedFields,
    HealthResponse,
    MetricsResponse,
)

_start_time: float = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = create_model_wrapper()
    yield
    # Shutdown: nothing to clean up for MLX; executor threads exit naturally.


app = FastAPI(
    title="qwen-receipt-server",
    description="Async receipt extraction server backed by Qwen2.5-3B fine-tuned on MLX.",
    version="0.1.0",
    lifespan=lifespan,
)


def _parse_extracted(raw: str) -> tuple[ExtractedFields, bool]:
    """
    Attempt to parse the model's raw output as JSON.
    Returns (ExtractedFields, parse_success).
    On failure, returns empty fields and parse_success=False rather than
    crashing — the caller decides whether to surface an error.
    """
    try:
        data = json.loads(raw.strip())
        return (
            ExtractedFields(
                company=str(data.get("company", "")),
                date=str(data.get("date", "")),
                address=str(data.get("address", "")),
                total=str(data.get("total", "")),
            ),
            True,
        )
    except (json.JSONDecodeError, TypeError):
        return ExtractedFields(), False


@app.post("/extract", response_model=ExtractionResponse)
async def extract(req: ExtractionRequest):
    """
    Run inference on a single receipt OCR text.
    Blocks until the asyncio.Lock is acquired (queues behind any in-flight request).
    """
    t0 = time.monotonic()
    raw, tokens = await app.state.model.infer(req.ocr_text)
    latency_ms = (time.monotonic() - t0) * 1000

    extracted, success = _parse_extracted(raw)
    tracker.record(latency_ms=latency_ms, tokens_generated=tokens, parse_success=success)

    return ExtractionResponse(
        request_id=req.request_id,
        extracted=extracted,
        raw_output=raw,
        parse_success=success,
        latency_ms=round(latency_ms, 2),
        tokens_generated=tokens,
    )


@app.post("/extract/batch", response_model=BatchExtractionResponse)
async def extract_batch(batch: BatchExtractionRequest):
    """
    Process a list of extraction requests SEQUENTIALLY.

    Why sequential and not parallel?
    The asyncio.Lock in model.infer() serialises all inference calls anyway.
    Launching coroutines in parallel would just mean they all queue at the
    lock and run one-by-one in arbitrary order — same throughput, more
    complexity. Sequential processing is explicit and predictable.

    True batch inference (processing N prompts in a single forward pass)
    requires direct access to the decode loop, which mlx_lm.generate()
    does not expose. See the continuous batching note in model.py.
    """
    if len(batch.requests) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(batch.requests)} exceeds MAX_BATCH_SIZE={MAX_BATCH_SIZE}",
        )

    t_batch_start = time.monotonic()
    results: list[ExtractionResponse] = []
    total_tokens = 0

    for req in batch.requests:
        t0 = time.monotonic()
        raw, tokens = await app.state.model.infer(req.ocr_text)
        latency_ms = (time.monotonic() - t0) * 1000

        extracted, success = _parse_extracted(raw)
        tracker.record(
            latency_ms=latency_ms, tokens_generated=tokens, parse_success=success
        )

        results.append(
            ExtractionResponse(
                request_id=req.request_id,
                extracted=extracted,
                raw_output=raw,
                parse_success=success,
                latency_ms=round(latency_ms, 2),
                tokens_generated=tokens,
            )
        )
        total_tokens += tokens

    total_latency_ms = (time.monotonic() - t_batch_start) * 1000
    tps = (total_tokens / total_latency_ms * 1000) if total_latency_ms > 0 else 0.0

    return BatchExtractionResponse(
        results=results,
        batch_size=len(results),
        total_latency_ms=round(total_latency_ms, 2),
        tokens_per_second=round(tps, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=app.state.model.model_loaded,
        model_id=MODEL_ID,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    return MetricsResponse(**tracker.summary())


@app.post("/metrics/reset")
async def metrics_reset():
    tracker.reset()
    return {"reset": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=False)
