# server/model.py — MLX model wrapper with async-safe inference
#
# ─── CONCURRENCY MODEL ────────────────────────────────────────────────────────
#
# asyncio.Lock — why it's here:
#   MLX model state (weights, KV cache buffer) is NOT safe for concurrent
#   access. Running two generate() calls simultaneously would corrupt the
#   internal tensor state. The asyncio.Lock ensures only one inference
#   executes at a time.
#
#   What this means at runtime: concurrency = QUEUING, not parallelism.
#   If 10 requests arrive simultaneously, 9 will await the lock while 1
#   runs. Each request still gets a correct answer; they just wait in line.
#   This is correct behavior — not a code limitation. The model is the
#   bottleneck, and a lock makes that bottleneck explicit and safe.
#
# run_in_executor — why it's here:
#   mlx_lm.generate() is a synchronous, blocking call. It runs for the
#   entire duration of inference (prefill + every decode step) without
#   yielding. If called directly on FastAPI's event loop, the loop would
#   freeze for all other requests — health checks would time out, /metrics
#   would hang, and the OS would see the process as unresponsive.
#
#   loop.run_in_executor(None, ...) moves the blocking call into Python's
#   default ThreadPoolExecutor. The event loop remains live and can serve
#   other endpoints while inference runs in a background thread. The
#   asyncio.Lock is acquired BEFORE the executor call so that only one
#   thread at a time enters generate().
#
# ─── KV CACHE ─────────────────────────────────────────────────────────────────
#
#   What it is: during attention computation, each token attends to all
#   previous tokens. The key (K) and value (V) projections for past tokens
#   don't change — so we cache them instead of recomputing on every step.
#
#   Prefill phase: the full prompt (system + user OCR text) is processed
#   in one forward pass. K/V pairs for every prompt token are computed
#   and stored in the KV cache.
#
#   Decode phase: we generate one token at a time. Each step only needs
#   to compute K/V for the new token — all prior K/V come from the cache.
#   This is what makes autoregressive generation tractable.
#
#   KV cache in MLX vs vLLM:
#   mlx_lm.generate() builds the KV cache from scratch on every call.
#   When the call returns, the cache is discarded. There is no cross-request
#   KV cache reuse in this server.
#
#   Prefix caching (what we're missing): if many requests share the same
#   system prompt prefix, a smarter runtime could cache those K/V pairs
#   once and reuse them across all requests, skipping the prefill cost for
#   the shared prefix. vLLM implements this. We don't.
#
# ─── WHY THIS IS NOT CONTINUOUS BATCHING ──────────────────────────────────────
#
#   Continuous batching (vLLM's key innovation): the decode loop runs
#   continuously. As soon as one sequence finishes, a waiting request is
#   slotted into the freed batch position. The batch is always full.
#   This maximises GPU utilisation because compute is never wasted waiting
#   for a slow sequence to finish.
#
#   Why we can't do it here: mlx_lm.generate() is a black box — it takes
#   a prompt, runs the full decode loop internally, and returns a string.
#   We have no access to the per-step decode loop to insert or remove
#   sequences mid-flight. Batching here means "run N prompts one after
#   another", which is sequential processing under a batch API shape.
#
#   Throughput we're leaving on the table: with continuous batching and a
#   batch size of 16, a vLLM-style server could serve ~10-16x more tokens
#   per second on the same hardware by keeping the compute unit saturated.
#   Our server's throughput is strictly bounded by one request at a time.
#
# ─── WHAT vLLM WOULD GIVE US ──────────────────────────────────────────────────
#
#   1. Continuous batching — described above.
#   2. PagedAttention: KV cache is managed in fixed-size "pages" rather
#      than one contiguous allocation per sequence. This eliminates memory
#      fragmentation, allows more sequences in flight, and enables KV cache
#      sharing across requests with common prefixes.
#   3. Much higher throughput: benchmarks show 10-24x over naive serving
#      on GPU hardware with concurrent users.
#   4. Why we can't use it: vLLM requires CUDA. It has no Apple Silicon
#      (MPS/MLX) backend. This server exists because MLX is the only
#      production-quality inference runtime for M-series Macs.

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor

from config import MAX_TOKENS, MOCK_LATENCY_MS, MODEL_ID, SYSTEM_PROMPT

# Attempt to import mlx_lm. If unavailable (Linux, CI, non-Apple hardware),
# fall back to MockModelWrapper so the server still starts and responds.
try:
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load as mlx_load

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class MockModelWrapper:
    """
    Stands in for MLXModelWrapper when mlx_lm is not installed.
    Returns plausible JSON with a simulated MOCK_LATENCY_MS delay.
    Lets the full server stack (endpoints, metrics, schemas) run on
    Linux / CI without any Apple Silicon hardware.
    """

    model_loaded: bool = True

    def infer_sync(self, prompt: str) -> tuple[str, int, float, float]:
        """Return (json_string, token_count, ttft_ms, tpot_ms). Blocks for MOCK_LATENCY_MS."""
        time.sleep(MOCK_LATENCY_MS / 1000)
        fake = json.dumps(
            {
                "company": "MOCK STORE",
                "date": "2024-03-15",
                "address": "123 Mock Street",
                "total": "99.99",
            }
        )
        # Mock splits latency 30% prefill / 70% decode across 42 tokens
        ttft_ms = MOCK_LATENCY_MS * 0.3
        tpot_ms = (MOCK_LATENCY_MS * 0.7) / 41  # time per decode step
        return fake, 42, ttft_ms, tpot_ms


class MLXModelWrapper:
    """
    Wraps mlx_lm load/generate with an asyncio.Lock and ThreadPoolExecutor
    so inference is both concurrency-safe and non-blocking on the event loop.
    """

    def __init__(self) -> None:
        self.model_loaded: bool = False
        self._model = None
        self._tokenizer = None
        # One thread is enough — the lock already serialises calls.
        # A pool size > 1 would not increase throughput here.
        self._executor = ThreadPoolExecutor(max_workers=1)
        # asyncio.Lock: only one coroutine enters the critical section
        # at a time. Others await here, forming an orderly queue.
        self._lock: asyncio.Lock | None = None  # created in load()

    def load(self) -> None:
        """Download and initialise the model from HuggingFace hub."""
        self._model, self._tokenizer = mlx_load(MODEL_ID)
        self.model_loaded = True
        # Lock must be created inside the running event loop.
        # We defer creation to the first async call to be safe.

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def infer_sync(self, prompt: str) -> tuple[str, int, float, float]:
        """
        Synchronous inference — called from a ThreadPoolExecutor thread,
        never directly from the event loop.
        Returns (raw_output_string, token_count, ttft_ms, tpot_ms).

        Uses stream_generate() to get real per-token timing:
          TTFT  = time from call start → first token emitted (prefill latency)
          TPOT  = average time between tokens during decode phase
                = (last_token_time - first_token_time) / (output_tokens - 1)

        Both are true hardware measurements, not estimates.
        """
        from mlx_lm import stream_generate as mlx_stream_generate

        output_parts: list[str] = []
        first_token_time: float | None = None
        last_token_time: float = 0.0
        output_tokens: int = 0

        t_start = time.monotonic()

        for response in mlx_stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
        ):
            now = time.monotonic()
            if first_token_time is None:
                first_token_time = now
            last_token_time = now
            output_parts.append(response.text)
            output_tokens += 1

        output = "".join(output_parts)

        if first_token_time is None:
            return output, 0, 0.0, 0.0

        ttft_ms = (first_token_time - t_start) * 1000
        tpot_ms = (
            (last_token_time - first_token_time) * 1000 / (output_tokens - 1)
            if output_tokens > 1
            else 0.0
        )

        return output, output_tokens, ttft_ms, tpot_ms

    async def infer(self, ocr_text: str) -> tuple[str, int, float, float]:
        """
        Async entry point called by FastAPI route handlers.

        1. Acquire asyncio.Lock → ensures serial inference.
        2. run_in_executor → moves blocking generate() off the event loop.
        3. Release lock when done (context manager handles this).
        """
        prompt = _build_prompt(ocr_text)
        loop = asyncio.get_event_loop()
        async with self._get_lock():
            raw, tokens, ttft_ms, tpot_ms = await loop.run_in_executor(
                self._executor, self.infer_sync, prompt
            )
        return raw, tokens, ttft_ms, tpot_ms


class MockAsyncWrapper:
    """Async wrapper around MockModelWrapper for use in route handlers."""

    model_loaded: bool = True
    _mock = MockModelWrapper()
    _executor = ThreadPoolExecutor(max_workers=1)

    async def infer(self, ocr_text: str) -> tuple[str, int, float, float]:
        prompt = _build_prompt(ocr_text)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._mock.infer_sync, prompt
        )


def _build_prompt(ocr_text: str) -> str:
    """
    Format the chat prompt using Qwen2.5's ChatML template.
    The tokenizer's apply_chat_template would be cleaner, but constructing
    the string directly avoids a tokenizer dependency in non-MLX paths.
    """
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{ocr_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def create_model_wrapper():
    """
    Factory used by main.py lifespan.
    Returns MLXModelWrapper (loads model) or MockAsyncWrapper on non-Mac.
    """
    if MLX_AVAILABLE:
        wrapper = MLXModelWrapper()
        wrapper.load()
        return wrapper
    return MockAsyncWrapper()
