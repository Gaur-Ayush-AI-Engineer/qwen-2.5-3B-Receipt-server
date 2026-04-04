#!/usr/bin/env python3
# scripts/smoke_test.py — single-request sanity check
#
# Usage:
#   1. Start the server: python -m uvicorn server.main:app --port 8000
#   2. In another terminal: python scripts/smoke_test.py
#
# Exits 0 on success, 1 on any failure.

import json
import sys
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"

SAMPLE_OCR = (
    "NILGIRIS SUPERMARKET & DEPARTMENTAL STORE\n"
    "No. 171, Brigade Road, Shivajinagar, Bengaluru, Karnataka - 560001\n"
    "Date: 15-03-2024\n"
    "Tata Salt 1kg            Rs. 22.00\n"
    "Amul Butter 500g         Rs. 275.00\n"
    "TOTAL                    Rs. 297.00"
)


def post_json(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def get_json(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=10) as resp:
        return json.loads(resp.read())


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    if not condition:
        sys.exit(1)


def main():
    print("=== qwen-receipt-server smoke test ===\n")

    # ── /health ──────────────────────────────────────────────────────────────
    print("GET /health")
    h = get_json("/health")
    check("status == ok", h["status"] == "ok", h.get("status"))
    check("model_loaded is true", h["model_loaded"] is True)
    check("uptime_seconds > 0", h["uptime_seconds"] > 0)
    print()

    # ── /extract ─────────────────────────────────────────────────────────────
    print("POST /extract")
    r = post_json("/extract", {"ocr_text": SAMPLE_OCR})
    check("has request_id", bool(r.get("request_id")))
    check("latency_ms > 0", r["latency_ms"] > 0)
    check("tokens_generated > 0", r["tokens_generated"] > 0)
    check("parse_success", r["parse_success"], r.get("raw_output", "")[:80])
    extracted = r["extracted"]
    check(
        "extracted has 4 keys",
        set(extracted.keys()) == {"company", "date", "address", "total"},
    )
    print(f"  extracted: {json.dumps(extracted, indent=2)}")
    print()

    # ── /extract/batch ────────────────────────────────────────────────────────
    print("POST /extract/batch")
    batch_payload = {
        "requests": [
            {"ocr_text": SAMPLE_OCR},
            {"ocr_text": "SUPERMART\n123 MG Road, Bengaluru\nDate: 12/03/2024\nTotal: Rs.847.00"},
        ]
    }
    b = post_json("/extract/batch", batch_payload)
    check("batch_size == 2", b["batch_size"] == 2)
    check("results length == 2", len(b["results"]) == 2)
    check("total_latency_ms > 0", b["total_latency_ms"] > 0)
    print()

    # ── /metrics ──────────────────────────────────────────────────────────────
    print("GET /metrics")
    m = get_json("/metrics")
    check("total_requests == 3", m["total_requests"] == 3, str(m["total_requests"]))
    check("p50_latency_ms > 0", m["p50_latency_ms"] > 0)
    print()

    # ── /metrics/reset ────────────────────────────────────────────────────────
    print("POST /metrics/reset")
    rr = post_json("/metrics/reset", {})
    check("reset returns true", rr.get("reset") is True)
    m2 = get_json("/metrics")
    check("total_requests reset to 0", m2["total_requests"] == 0)
    print()

    print("=== All checks passed ===")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"\nConnection error: {e}")
        print("Is the server running?  python -m uvicorn server.main:app --port 8000")
        sys.exit(1)
