# qwen-receipt-server

A production-concept LLM serving project demonstrating async request handling, latency measurement, and honest trade-off analysis for single-GPU-class hardware (Apple M-series).

---

## What this project demonstrates

Not "a FastAPI wrapper around a model." Specifically:

- **Async request queuing**: how `asyncio.Lock` serialises inference correctly, and why that's different from being "slow" — it's the right model for a single stateful accelerator.
- **Latency vs. throughput trade-off**: p50/p95/p99 measurement reveals whether your system is predictable or has hidden failure modes under load.
- **KV cache concepts**: what prefill and decode phases are, why per-request KV cache recompute happens in MLX, and what prefix caching would save.
- **Continuous batching (what we don't have, and why)**: a direct explanation of vLLM's core innovation and why `mlx_lm.generate()` can't replicate it.
- **Honest comparison with vLLM**: what a CUDA-based runtime would give us in throughput, and why it's unavailable on Apple Silicon.

---

## Architecture

```
Client → POST /extract
           │
           ▼
     FastAPI route handler
           │
           ▼
     wrapper.infer()  ← await asyncio.Lock
           │               (one at a time — others queue here)
           ▼
     loop.run_in_executor()
           │               (moves blocking mlx_generate off event loop)
           ▼
     mlx_lm.generate()      (prefill + decode, ~0.5–3s on M3 Pro)
           │
           ▼
     JSON parse → MetricsTracker.record()
           │
           ▼
     ExtractionResponse → Client
```

**Concurrency model**: there is one model instance. The `asyncio.Lock` in `server/model.py` ensures exactly one inference runs at any moment. All other requests queue at the lock. This is correct — not a code flaw. The model is the bottleneck; making that bottleneck explicit and safe is the point.

`/health` and `/metrics` never touch the lock. They always respond immediately regardless of how many inference requests are queued.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/largetrader/qwen-receipt-server
cd qwen-receipt-server
pip install -r requirements.txt   # mlx-lm only installs on Apple Silicon

# Start the server (downloads ~6GB model on first run)
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000

# Sanity check (in another terminal)
python scripts/smoke_test.py
```

**On non-Apple hardware**: `mlx-lm` won't install, but the server still starts using `MockModelWrapper`, which returns fake JSON after a 200ms simulated delay. All endpoints, metrics, and benchmarking work normally.

---

## Example Request

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "ocr_text": "SUPERMART\n123 MG Road, Bengaluru\nDate: 12/03/2024\nTotal: Rs.847.00"
  }'
```

```json
{
  "request_id": "f3a2...",
  "extracted": {
    "company": "SUPERMART",
    "date": "2024-03-12",
    "address": "123 MG Road, Bengaluru",
    "total": "847.00"
  },
  "raw_output": "{\"company\": \"SUPERMART\", ...}",
  "parse_success": true,
  "latency_ms": 847.3,
  "ttft_ms": 610.2,
  "tpot_ms": 49.1,
  "tokens_generated": 38
}
```

---

## Benchmark Results

Hardware: Apple M3 Pro 32GB · Model: `largetrader/qwen2.5-3b-receipt-extraction-fused` (3B params, MLX)  
Run with Locust against `/extract` (75%) and `/extract/batch` (25%) endpoints.

```bash
./benchmark/run_all.sh   # runs all concurrency levels and logs results
```

| Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | Req/sec | TTFT (ms) | TPOT (ms) | HW tok/s | Sys tok/s |
|-------------|----------|----------|----------|---------|-----------|-----------|----------|-----------|
| 1           | 3800     | 11000    | 11000    | 0.19    | 610       | 49.1      | 20.4     | 11.4      |
| 5           | 13000    | 45000    | 66000    | 0.30    | 443       | 49.2      | 20.4     | 17.8      |
| 10          | 35000    | 98000    | 112000   | 0.18    | 470       | 49.9      | 20.1     | 17.4      |
| 20          | 48000    | 86000    | 89000    | 0.21    | 433       | 49.6      | 20.2     | 17.6      |

**Column definitions**:
- **TTFT** — Time To First Token: prefill latency, how long the model takes to process the input prompt before generating any output.
- **TPOT** — Time Per Output Token: true decode speed (~49ms/token = ~20 tok/s). This is a hardware measurement — notice it stays flat across all concurrency levels.
- **HW tok/s** — `1000 / TPOT`: the model's actual generation rate. Constant at ~20 tok/s regardless of queue depth, proving the GPU runs at full speed.
- **Sys tok/s** — `total_output_tokens / wall_clock_time`: system-level throughput. This is how vLLM, TGI, and SGLang report throughput.

**Reading the table**: E2E latency (p50→p99) grows sharply with concurrency — that's queue wait time, not slower inference. The flat HW tok/s column confirms the model itself isn't slowing down; requests are just waiting longer for the `asyncio.Lock`.

**Req/sec is not a meaningful scaling metric for this server.** The `asyncio.Lock` bounds throughput to ~1 request per inference duration regardless of concurrency. Use **HW tok/s** and **Sys tok/s** to evaluate throughput.

---

## What vLLM Would Give Us

vLLM is the dominant open-source inference runtime for GPU-based serving. On this workload (short-output JSON extraction, many concurrent users) it would deliver:

**Continuous batching**: as soon as one sequence finishes generating, the next waiting request is slotted into the freed batch position. The GPU is always working on a full batch. Our server's throughput is bounded by one request at a time; vLLM's is bounded by GPU memory / batch size.

**PagedAttention**: instead of allocating a contiguous KV cache block per sequence, vLLM manages KV memory in fixed-size pages. This eliminates fragmentation, allows more sequences in flight simultaneously, and enables KV cache sharing between requests with identical prefixes (our system prompt is shared — vLLM would cache it once).

**Throughput estimate**: for a short-output task like receipt extraction (≈30–60 tokens output), vLLM on an A100 would process 100–400 requests/second at p95 < 500ms. Our M3 Pro server processes 1 request at a time at ~0.5–3s each.

**Why we can't use it**: vLLM requires CUDA. There is no Apple Silicon (MPS/MLX) backend. MLX is the only production-quality inference runtime for M-series Macs, and `mlx_lm.generate()` doesn't expose the decode loop internals needed for continuous batching.

---

## Model

**HuggingFace**: [largetrader/qwen2.5-3b-receipt-extraction-fused](https://huggingface.co/largetrader/qwen2.5-3b-receipt-extraction-fused)

Base: `Qwen/Qwen2.5-3B-Instruct` with LoRA adapters trained on Indian retail receipts, then fused into the base weights. The fused model requires no PEFT runtime — `mlx_lm.load()` treats it as a standard model.

**Fine-tuning repo**: see [largetrader/LLM-serving](https://github.com/largetrader/LLM-serving) for training code, dataset pipeline, and LoRA configuration.
