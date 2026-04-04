# benchmark/locustfile.py — load test scenarios for qwen-receipt-server
#
# Usage:
#   locust -f benchmark/locustfile.py --host http://localhost:8000
#   # Then open http://localhost:8089 in a browser, or use headless mode:
#   locust -f benchmark/locustfile.py --host http://localhost:8000 \
#          --headless -u 20 -r 2 --run-time 2m \
#          --csv benchmark/results
#
# Ramp schedule (run these in sequence to fill the benchmark table):
#   -u 1  -r 1   --run-time 90s
#   -u 5  -r 1   --run-time 2m
#   -u 10 -r 2   --run-time 2m
#   -u 20 -r 4   --run-time 2m
#
# The CSV output is consumed by benchmark/analyze.py to build the results table.

import random

from locust import HttpUser, between, task

# ── Receipt OCR samples ───────────────────────────────────────────────────────
# Four samples with varying length and OCR noise level.
# Using real-world-style Bengaluru receipts so extracted fields are realistic.

SAMPLE_1 = "SUPERMART\n123 MG Road, Bengaluru\nDate: 12/03/2024\nTotal: Rs.847.00"

SAMPLE_2 = (
    "MEDPLUS PHARMACY\nShop 4, Koramangala Blk 5\nBengaluru - 560095\n"
    "GSTIN: 29AABCM1596Q1ZX\nBill No: MP/2024/8834\nDt: 15-03-2024\n"
    "Paracetamol 500mg x2  Rs.48\nVitamin C x1           Rs.120\n"
    "Total Amt: Rs.168.00"
)

SAMPLE_3 = (
    "SR1 ELECTR0NICS\n4/2 BrigadeRd Banaalore\nGSTlN29SRIEL1234Q\n"
    "lnv#SE2O24O315\nDt:l5/O3/2O24\n"
    "USB Cabl3 x2 Rs 349\nPh0ne Case x1 Rs 599\nTOTAL Rs.948/-"
)

SAMPLE_4 = (
    "NILGIRIS SUPERMARKET & DEPARTMENTAL STORE\n"
    "No. 171, Brigade Road, Shivajinagar, Bengaluru, Karnataka - 560001\n"
    "GSTIN: 29AABCN2241R1ZF | Tel: 080-22863636\n"
    "Bill No: NGR/2024/031547 | Cashier: Kumar | Counter: 3\n"
    "Date: 15-03-2024 | Time: 14:32:07\n"
    "Tata Salt 1kg            Rs. 22.00\n"
    "Amul Butter 500g         Rs. 275.00\n"
    "Fortune Sunflower Oil 1L Rs. 145.00\n"
    "Parle-G Biscuits x3      Rs. 30.00\n"
    "Horlicks 500g            Rs. 320.00\n"
    "Colgate MaxFresh 150g    Rs. 89.00\n"
    "SUBTOTAL                 Rs. 881.00\n"
    "SGST @9%                 Rs. 79.29\n"
    "CGST @9%                 Rs. 79.29\n"
    "TOTAL                    Rs. 1039.58"
)

SAMPLES = [SAMPLE_1, SAMPLE_2, SAMPLE_3, SAMPLE_4]


class ReceiptExtractionUser(HttpUser):
    """
    Simulates a user sending receipt OCR text for extraction.
    wait_time adds realistic think-time between requests.
    """

    # Between each task, wait 1–3 seconds to simulate realistic request spacing.
    wait_time = between(1, 3)

    @task(3)
    def extract_single(self):
        """POST /extract with a randomly selected sample."""
        ocr_text = random.choice(SAMPLES)
        self.client.post(
            "/extract",
            json={"ocr_text": ocr_text},
            name="/extract",
        )

    @task(1)
    def extract_batch(self):
        """
        POST /extract/batch with 2–4 randomly selected samples.
        Weight is 1 vs 3 for single — batch is less common in practice.
        """
        n = random.randint(2, 4)
        requests = [{"ocr_text": random.choice(SAMPLES)} for _ in range(n)]
        self.client.post(
            "/extract/batch",
            json={"requests": requests},
            name="/extract/batch",
        )

    @task(1)
    def health_check(self):
        """GET /health — lightweight probe; should always be fast."""
        self.client.get("/health", name="/health")
