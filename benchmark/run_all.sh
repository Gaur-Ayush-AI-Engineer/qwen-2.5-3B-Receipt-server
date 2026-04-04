#!/usr/bin/env bash
# benchmark/run_all.sh — run full benchmark ramp and log results
#
# Usage:
#   chmod +x benchmark/run_all.sh
#   ./benchmark/run_all.sh
#
# Requires server running at http://localhost:8000

set -euo pipefail

SERVER="http://localhost:8000"
LOG="benchmark/benchmark_run.log"
OUTPUT="benchmark_results.md"

LEVELS=(
  "1 1 90s"
  "5 1 2m"
  "10 2 2m"
  "20 4 2m"
)

# Clear old results file so we get a clean table
rm -f "$OUTPUT"

echo "=== Benchmark run started at $(date) ===" | tee "$LOG"

for level in "${LEVELS[@]}"; do
  read -r users spawn_rate run_time <<< "$level"
  csv_prefix="benchmark/results_u${users}"

  echo "" | tee -a "$LOG"
  echo "--- Concurrency=$users | spawn_rate=$spawn_rate | run_time=$run_time ---" | tee -a "$LOG"

  # Reset server metrics before each run
  curl -s -X POST "${SERVER}/metrics/reset" >> "$LOG"
  echo "" >> "$LOG"

  # Run locust
  locust -f benchmark/locustfile.py \
    --host "$SERVER" \
    --headless \
    -u "$users" \
    -r "$spawn_rate" \
    --run-time "$run_time" \
    --csv "$csv_prefix" 2>&1 | tee -a "$LOG"

  # Analyze and append to results file
  python benchmark/analyze.py \
    --csv "${csv_prefix}_stats.csv" \
    --server "$SERVER" \
    --output "$OUTPUT" \
    --concurrency "$users" 2>&1 | tee -a "$LOG"

done

echo "" | tee -a "$LOG"
echo "=== Benchmark run completed at $(date) ===" | tee -a "$LOG"
echo ""
echo "Results saved to: $OUTPUT"
echo "Full log:         $LOG"
