# config.py — single source of truth for all constants
# All tunable values live here; nothing is hardcoded elsewhere.

MODEL_ID = "largetrader/qwen2.5-3b-receipt-extraction-fused"

HOST = "0.0.0.0"
PORT = 8000

MAX_TOKENS = 256
MAX_BATCH_SIZE = 16

# MockModelWrapper uses this to simulate realistic inference latency
# when mlx_lm is unavailable (Linux, CI, non-Apple hardware).
MOCK_LATENCY_MS = 200

# This prompt must match exactly what was used during fine-tuning.
# The model was trained to respond to this system message with JSON only.
SYSTEM_PROMPT = (
    "You are a structured data extraction assistant. "
    "Given raw OCR text from a receipt, extract the fields as a JSON object "
    "with exactly these keys: company, date, address, total. "
    "date must be YYYY-MM-DD format. total must be a numeric string only "
    "(no currency symbol). If a field is missing, use empty string. "
    "Respond with ONLY valid JSON. No explanation, no markdown, no extra text."
)
