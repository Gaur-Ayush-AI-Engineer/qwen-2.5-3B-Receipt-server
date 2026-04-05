# server/schemas.py — Pydantic v2 request/response models
# Pydantic v2 uses model_config = ConfigDict(...) instead of class Config.

import uuid
from pydantic import BaseModel, Field, model_validator
from pydantic import ConfigDict
from typing import Optional


class ExtractionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    ocr_text: str = Field(
        ...,
        min_length=10,
        max_length=4096,
        description="Raw OCR text from a receipt",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional caller-supplied idempotency key; auto-generated if omitted",
    )

    @model_validator(mode="after")
    def assign_request_id(self) -> "ExtractionRequest":
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
        return self


class ExtractedFields(BaseModel):
    company: str = ""
    date: str = ""       # YYYY-MM-DD
    address: str = ""
    total: str = ""      # numeric string, no currency symbol


class ExtractionResponse(BaseModel):
    request_id: str
    extracted: ExtractedFields
    raw_output: str
    parse_success: bool
    latency_ms: float      # end-to-end wall time including queue wait
    ttft_ms: float         # time to first token (prefill latency)
    tpot_ms: float         # time per output token (decode speed)
    tokens_generated: int


class BatchExtractionRequest(BaseModel):
    model_config = ConfigDict()

    requests: list[ExtractionRequest] = Field(
        ...,
        min_length=1,
        max_length=16,  # mirrors MAX_BATCH_SIZE in config.py
        description="1–16 extraction requests to process sequentially",
    )


class BatchExtractionResponse(BaseModel):
    results: list[ExtractionResponse]
    batch_size: int
    total_latency_ms: float
    tokens_per_second: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_requests: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_ttft_ms: float           # mean prefill latency
    p95_ttft_ms: float            # tail prefill latency
    mean_tpot_ms: float           # true decode speed (hardware)
    mean_tokens_per_second: float # 1000 / mean_tpot_ms — hardware tok/s
    system_throughput_tps: float  # total_tokens / wall_clock (system-level)
    parse_success_rate: float
