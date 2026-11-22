# Live Cham OCR Quality Benchmark — 2026-11-20

Endpoint: https://ocr.cham.asia | mode: dbnet | reading order: rows
UTC report time: 2026-11-20T04:36:46.203018+00:00
POST /ocr calls recorded: 120

## Overall (excluding blank images)

| Model | Valid cases | Micro CER | Micro WER | Exact line count | Median backend | Errors |
|---|---:|---:|---:|---:|---:|---:|
| v24_1_r4 | 29 | 7.63% | 27.85% | 96.55% | 1.5 s | 0 |
| v24_1_r3_1 | 29 | 7.41% | 28.29% | 96.55% | 1.57 s | 0 |
| v24_1_r3 | 29 | 8.18% | 31.59% | 96.55% | 1.62 s | 0 |
| v24_1_r2_1_6 | 29 | 7.95% | 28.36% | 96.55% | 1.52 s | 0 |

## Difficulty tiers

| Model | Tier | Cases | CER | WER | Exact line count |
|---|---|---:|---:|---:|---:|
| v24_1_r4 | 0: Blank negative | 1 | 0.0% | 0.0% | 100.0% |
| v24_1_r4 | 1: Standard | 4 | 6.8% | 29.95% | 100.0% |
| v24_1_r4 | 2: Aged paper | 6 | 6.86% | 29.38% | 100.0% |
| v24_1_r4 | 3: Narrow gap | 6 | 6.3% | 27.11% | 100.0% |
| v24_1_r4 | 4: Wavy | 6 | 11.99% | 34.65% | 83.33% |
| v24_1_r4 | 5: Extreme overlap | 6 | 7.95% | 30.0% | 100.0% |
| v24_1_r4 | 6: Real camera | 1 | 3.55% | 11.23% | 100.0% |
| v24_1_r3_1 | 0: Blank negative | 1 | 0.0% | 0.0% | 100.0% |
| v24_1_r3_1 | 1: Standard | 4 | 6.8% | 30.92% | 100.0% |
| v24_1_r3_1 | 2: Aged paper | 6 | 5.11% | 23.71% | 100.0% |
| v24_1_r3_1 | 3: Narrow gap | 6 | 5.69% | 25.7% | 100.0% |
| v24_1_r3_1 | 4: Wavy | 6 | 11.3% | 34.35% | 83.33% |
| v24_1_r3_1 | 5: Extreme overlap | 6 | 8.67% | 33.12% | 100.0% |
| v24_1_r3_1 | 6: Real camera | 1 | 5.2% | 19.25% | 100.0% |
| v24_1_r3 | 0: Blank negative | 1 | 0.0% | 0.0% | 100.0% |
| v24_1_r3 | 1: Standard | 4 | 7.02% | 31.88% | 100.0% |
| v24_1_r3 | 2: Aged paper | 6 | 5.61% | 25.26% | 100.0% |
| v24_1_r3 | 3: Narrow gap | 6 | 6.38% | 30.99% | 100.0% |
| v24_1_r3 | 4: Wavy | 6 | 12.06% | 35.87% | 83.33% |
| v24_1_r3 | 5: Extreme overlap | 6 | 9.97% | 38.75% | 100.0% |
| v24_1_r3 | 6: Real camera | 1 | 6.5% | 25.13% | 100.0% |
| v24_1_r2_1_6 | 0: Blank negative | 1 | 0.0% | 0.0% | 100.0% |
| v24_1_r2_1_6 | 1: Standard | 4 | 6.34% | 26.09% | 100.0% |
| v24_1_r2_1_6 | 2: Aged paper | 6 | 5.61% | 22.16% | 100.0% |
| v24_1_r2_1_6 | 3: Narrow gap | 6 | 6.53% | 28.17% | 100.0% |
| v24_1_r2_1_6 | 4: Wavy | 6 | 12.13% | 35.56% | 83.33% |
| v24_1_r2_1_6 | 5: Extreme overlap | 6 | 9.39% | 33.75% | 100.0% |
| v24_1_r2_1_6 | 6: Real camera | 1 | 5.67% | 20.32% | 100.0% |

## Method and limitations

- Micro CER: summed Levenshtein character edits over summed ground-truth characters, with NFC normalization, preserving line breaks.
- Micro WER: tokenized by whitespace; interpretation is weak for Cham scriptio continua.
- Exact line count is not a true line-detection recall metric; split and merge errors can cancel.
- Symbol hits in CSV are upper bounds based on occurrence counts, not position-aligned true positives.
- Synthetic cases are not proven independent from model training distribution. Treat real-camera results separately.
- A valid row requires actual_model == requested model, to avoid silent model fallback.
- Wall latency includes network; backend does not. Initial model calls may include cold start.
- Raw ground truth and predictions are retained per request in results.jsonl.

## Unexpected errors or model mismatch

None recorded.
