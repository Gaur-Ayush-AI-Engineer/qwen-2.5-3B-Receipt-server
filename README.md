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
  "tokens_generated": 38
}
```

---

## Benchmark Results

Run with Locust at each concurrency level:

```bash
locust -f benchmark/locustfile.py --host http://localhost:8000 \
       --headless -u 1 -r 1 --run-time 90s --csv benchmark/results_u1
python benchmark/analyze.py --csv benchmark/results_u1_stats.csv \
                             --server http://localhost:8000
```

| Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | Req/sec | Tokens/sec |
|-------------|----------|----------|----------|---------|------------|
| 1           | TBD      | TBD      | TBD      | TBD     | TBD        |
| 5           | TBD      | TBD      | TBD      | TBD     | TBD        |
| 10          | TBD      | TBD      | TBD      | TBD     | TBD        |
| 20          | TBD      | TBD      | TBD      | TBD     | TBD        |

**Reading the table**: p50→p99 gap tells you about variance. A gap of 2-10x under high concurrency is normal (queue buildup). A gap > 10x suggests a systemic problem — memory pressure, GC pause, or a pathologically long input hitting the decode length limit.

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
