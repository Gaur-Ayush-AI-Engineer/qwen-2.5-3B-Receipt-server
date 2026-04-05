#!/usr/bin/env python3
# benchmark/analyze.py — parse locust CSV output into a results table
#
# Usage:
#   python benchmark/analyze.py --csv benchmark/results_stats.csv \
#                                --server http://localhost:8000
#
# The --server flag is optional. When provided, it fetches live
# mean_tokens_per_second from GET /server/metrics.
#
# Output: prints a formatted table and saves benchmark_results.md

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def read_locust_csv(csv_path: str) -> list[dict]:
    """
    Parse locust's _stats.csv file.
    Each row is one endpoint aggregate; we want the aggregated row
    that covers all requests (Name == "Aggregated").
    Returns a list of all rows as dicts.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def fetch_server_metrics(server_url: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{server_url}/metrics", timeout=5) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError):
        return None


def safe_float(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def build_table(rows: list[dict], server_metrics: dict | None, concurrency_label: str = "?") -> str:
    """
    Build a Markdown table from the locust stats rows.

    Locust CSV columns we use:
      - "50%"  → p50 response time (ms)
      - "95%"  → p95 response time (ms)
      - "99%"  → p99 response time (ms)
      - "Requests/s" → throughput
      - "User count" → concurrency level (not always in stats CSV;
                        we reconstruct from the run metadata if available)

    Because a single locust run produces one CSV per concurrency level,
    this script expects to be run once per CSV and accumulates rows into
    benchmark_results.md in append mode.
    """
    lines = []

    # Find the Aggregated row for /extract endpoint
    extract_rows = [
        r for r in rows if r.get("Name", "") in ("/extract", "Aggregated")
    ]
    if not extract_rows:
        print("Warning: no /extract or Aggregated row found in CSV.")
        extract_rows = rows  # fallback: use all rows

    # Use the Aggregated row if present, otherwise first /extract row
    agg = next((r for r in extract_rows if r.get("Name") == "Aggregated"), extract_rows[0])

    p50 = safe_float(agg.get("50%", 0))
    p95 = safe_float(agg.get("95%", 0))
    p99 = safe_float(agg.get("99%", 0))
    rps = safe_float(agg.get("Requests/s", 0))

    ttft_str = "N/A"
    tpot_str = "N/A"
    hw_tps_str = "N/A"
    sys_tps_str = "N/A"
    if server_metrics:
        ttft_str = f"{server_metrics.get('mean_ttft_ms', 0):.0f}"
        tpot_str = f"{server_metrics.get('mean_tpot_ms', 0):.1f}"
        hw_tps_str = f"{server_metrics.get('mean_tokens_per_second', 0):.1f}"
        sys_tps_str = f"{server_metrics.get('system_throughput_tps', 0):.1f}"

    concurrency = concurrency_label

    header = "| Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | Req/sec | TTFT (ms) | TPOT (ms) | HW tok/s | Sys tok/s |"
    sep    = "|-------------|----------|----------|----------|---------|-----------|-----------|----------|-----------|"
    row    = (
        f"| {concurrency:^11} | {p50:^8.0f} | {p95:^8.0f} | {p99:^8.0f} | {rps:^7.2f} |"
        f" {ttft_str:^9} | {tpot_str:^9} | {hw_tps_str:^8} | {sys_tps_str:^9} |"
    )

    lines.append(header)
    lines.append(sep)
    lines.append(row)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze locust CSV benchmark output")
    parser.add_argument("--csv", required=True, help="Path to locust _stats.csv file")
    parser.add_argument("--server", default=None, help="Server URL to fetch live metrics")
    parser.add_argument("--output", default="benchmark_results.md", help="Output markdown file")
    parser.add_argument("--concurrency", default="?", help="Concurrency level label for the table")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"Error: CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    rows = read_locust_csv(args.csv)
    if not rows:
        print("Error: CSV file is empty or malformed.", file=sys.stderr)
        sys.exit(1)

    concurrency_label = args.concurrency

    server_metrics = None
    if args.server:
        server_metrics = fetch_server_metrics(args.server)
        if server_metrics is None:
            print(f"Warning: could not reach {args.server}/metrics — tokens/sec will be N/A")

    print("\n=== Benchmark Results ===\n")
    table = build_table(rows, server_metrics, concurrency_label=concurrency_label)
    print(table)

    # Append to benchmark_results.md so multiple runs accumulate
    out_path = Path(args.output)
    mode = "a" if out_path.exists() else "w"
    with open(out_path, mode) as f:
        if mode == "w":
            f.write("# Benchmark Results\n\n")
            f.write("Generated by `benchmark/analyze.py`.\n\n")
        f.write(f"\n## Run: {args.csv}\n\n")
        f.write(table + "\n")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
