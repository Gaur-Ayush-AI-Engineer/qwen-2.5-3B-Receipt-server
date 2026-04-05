# server/metrics.py — in-process latency and token throughput tracker
#
# Percentile explainer (non-negotiable for a portfolio project):
#
#   p50  = median — half of requests complete faster than this.
#          This is the "typical user experience" number.
#
#   p95  = 95% of requests finish faster than this.
#          Think of it as: 1 in 20 requests is slower.
#
#   p99  = tail latency — only 1 in 100 requests is slower.
#          This is what your slowest users experience.
#
#   Why the gap between p50 and p99 matters:
#     A narrow gap (p50=200ms, p99=250ms) means predictable, stable service.
#     A wide gap (p50=200ms, p99=2000ms) means SOMETHING occasionally goes
#     badly wrong — memory pressure forcing a cache evict, GC pause, queue
#     backup building up behind the asyncio.Lock, or a long OCR input
#     triggering a much longer decode sequence.
#     The gap is the diagnostic signal; ignoring it means missing systemic
#     problems that only manifest under load.
#
# Thread safety: uvicorn runs request handlers across multiple threads.
# threading.Lock (not asyncio.Lock) protects the shared lists here because
# these writes happen from synchronous contexts called by executor threads.

import threading
import time


class MetricsTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latencies_ms: list[float] = []
        self._ttft_ms: list[float] = []
        self._tpot_ms: list[float] = []
        self._parse_successes: list[bool] = []
        self._total_output_tokens: int = 0
        self._start_time: float = time.monotonic()

    def record(
        self,
        latency_ms: float,
        tokens_generated: int,
        ttft_ms: float,
        tpot_ms: float,
        parse_success: bool,
    ) -> None:
        with self._lock:
            self._latencies_ms.append(latency_ms)
            self._ttft_ms.append(ttft_ms)
            self._tpot_ms.append(tpot_ms)
            self._parse_successes.append(parse_success)
            self._total_output_tokens += tokens_generated

    def reset(self) -> None:
        with self._lock:
            self._latencies_ms.clear()
            self._ttft_ms.clear()
            self._tpot_ms.clear()
            self._parse_successes.clear()
            self._total_output_tokens = 0
            self._start_time = time.monotonic()

    def summary(self) -> dict:
        with self._lock:
            n = len(self._latencies_ms)
            uptime = time.monotonic() - self._start_time
            if n == 0:
                return {
                    "total_requests": 0,
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                    "mean_ttft_ms": 0.0,
                    "p95_ttft_ms": 0.0,
                    "mean_tpot_ms": 0.0,
                    "mean_tokens_per_second": 0.0,
                    "system_throughput_tps": 0.0,
                    "parse_success_rate": 0.0,
                }
            sorted_lat = sorted(self._latencies_ms)
            sorted_ttft = sorted(self._ttft_ms)

            # TPOT → tokens/sec: exclude zero values (requests with 0 or 1 tokens)
            valid_tpot = [t for t in self._tpot_ms if t > 0]
            mean_tpot = sum(valid_tpot) / len(valid_tpot) if valid_tpot else 0.0
            # tokens/sec = 1000ms / ms-per-token
            mean_tps = (1000.0 / mean_tpot) if mean_tpot > 0 else 0.0

            # System throughput: total tokens produced / total wall-clock time
            system_tps = self._total_output_tokens / uptime if uptime > 0 else 0.0

            return {
                "total_requests": n,
                "p50_latency_ms": self._percentile(sorted_lat, 50),
                "p95_latency_ms": self._percentile(sorted_lat, 95),
                "p99_latency_ms": self._percentile(sorted_lat, 99),
                "mean_ttft_ms": sum(self._ttft_ms) / n,
                "p95_ttft_ms": self._percentile(sorted_ttft, 95),
                "mean_tpot_ms": mean_tpot,
                "mean_tokens_per_second": round(mean_tps, 2),
                "system_throughput_tps": round(system_tps, 2),
                "parse_success_rate": sum(self._parse_successes) / n,
            }

    @staticmethod
    def _percentile(sorted_data: list[float], p: int) -> float:
        # Nearest-rank method — no statistics library needed.
        # rank = ceil(p/100 * n), then clamp to valid index range.
        # Example: 100 samples, p95 → rank = ceil(95) = 95 → index 94.
        n = len(sorted_data)
        if n == 1:
            return sorted_data[0]
        # Use ceiling of the rank formula, then convert to 0-based index.
        rank = int((p / 100) * n + 0.9999)  # ceil without math.ceil
        rank = max(1, min(rank, n))
        return sorted_data[rank - 1]

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time


# Module-level singleton — imported by main.py
tracker = MetricsTracker()
