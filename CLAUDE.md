# CLAUDE.md — living state file for qwen-receipt-server

This file documents the actual state of the project for future Claude Code sessions.
Update it whenever you make a meaningful structural change.

---

## Project purpose

Portfolio project demonstrating production LLM serving concepts on Apple Silicon.
**Not a production system** — correctness and clarity of architecture matter more than throughput.

Model: `largetrader/qwen2.5-3b-receipt-extraction-fused`  
Hardware target: M3 Pro 32GB, MLX only.

---

## File map

| File | Role |
|------|------|
| `config.py` | Single source of truth — all constants. Edit here, nowhere else. |
| `server/schemas.py` | Pydantic v2 request/response models. Uses `model_config = ConfigDict(...)`. |
| `server/metrics.py` | In-process p50/p95/p99 tracker. No external dependencies. Thread-safe. |
| `server/model.py` | MLX wrapper + MockModelWrapper. All concurrency/KV-cache comments live here. |
| `server/main.py` | FastAPI app. Five endpoints: /extract, /extract/batch, /health, /metrics, /metrics/reset. |
| `scripts/smoke_test.py` | End-to-end sanity check. Run after any server change. |
| `benchmark/locustfile.py` | Locust load test. Four OCR samples, /extract and /extract/batch tasks. |
| `benchmark/analyze.py` | Parses locust CSV → Markdown table. Optionally fetches live tokens/sec from /metrics. |
| `requirements.txt` | fastapi, uvicorn, pydantic>=2.6, mlx-lm>=0.13, locust |

---

## Current status

- [x] All files written and syntax-verified
- [x] Smoke test passes (15/15 checks) with MockModelWrapper
- [x] Smoke test passes with real MLX model (`largetrader/qwen2.5-3b-receipt-extraction-fused` weights uploaded and verified)
- [ ] Benchmark table in README.md has TBD placeholders — fill after running locust

---

## How to run

```bash
# Install deps (mlx-lm only installs on Apple Silicon)
pip install -r requirements.txt

# Start server (downloads ~6GB model weights on first run)
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000

# Sanity check
python scripts/smoke_test.py

# Load test at concurrency=1
locust -f benchmark/locustfile.py --host http://localhost:8000 \
       --headless -u 1 -r 1 --run-time 90s --csv benchmark/results_u1
python benchmark/analyze.py --csv benchmark/results_u1_stats.csv \
                             --server http://localhost:8000
```

---

## Key design decisions

**Why asyncio.Lock + run_in_executor**: explained in detail in `server/model.py`.
Short version: Lock = serial inference safety; run_in_executor = keep event loop alive.

**Why no Redis/Celery**: this is a single-process server. The asyncio queue is the queue.
Adding external dependencies would obscure the serving concepts without adding value.

**MockModelWrapper**: activates automatically when `mlx_lm` is not importable.
Lets the full server stack run on Linux/CI. All endpoints, schemas, and metrics work normally.
Mock latency is controlled by `MOCK_LATENCY_MS` in `config.py`.

**Pydantic v2**: uses `model_config = ConfigDict(...)` throughout — no deprecated `class Config`.

---

## What to add next (if expanding)

- Streaming responses via SSE (`/extract/stream`) — would require exposing mlx_lm's token iterator
- Request timeout: cancel inference if waiting > N seconds (needs `asyncio.wait_for`)
- Prometheus metrics endpoint (currently metrics are in-process only)
- Docker image with mlx-lm pre-installed (macOS arm64 base only)
