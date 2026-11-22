# Live symbol and hyphen crop test

Endpoint: /ocr_crop (recognition-only, not line detection)
Report UTC: 2026-11-20T04:42:53.458491+00:00
Requests collected: 168

| Model | Valid crops | Character CER | Exact full lines | Request failures |
|---|---:|---:|---:|---:|
| v24_1_r4 | 42 | 0.69% | 92.86% | 0 |
| v24_1_r3_1 | 42 | 0.69% | 92.86% | 0 |
| v24_1_r3 | 42 | 1.84% | 85.71% | 0 |
| v24_1_r2_1_6 | 42 | 1.84% | 80.95% | 0 |

## Each required symbol (minimum-edit alignment)

| Symbol | v24_1_r4 | v24_1_r3_1 | v24_1_r3 | v24_1_r2_1_6 |
|---||---:||---:||---:||---:|
| ! | 2/2 | 2/2 | 1/2 | 2/2 |
| # | 2/2 | 2/2 | 2/2 | 2/2 |
| & | 2/2 | 2/2 | 2/2 | 2/2 |
| * | 2/2 | 2/2 | 2/2 | 2/2 |
| / | 1/2 | 1/2 | 1/2 | 1/2 |
| < | 2/2 | 2/2 | 2/2 | 2/2 |
| > | 2/2 | 2/2 | 2/2 | 2/2 |
| ? | 2/2 | 2/2 | 2/2 | 2/2 |
| [ | 2/2 | 2/2 | 2/2 | 2/2 |
| BACKSLASH | 2/2 | 2/2 | 2/2 | 2/2 |
| ] | 2/2 | 2/2 | 2/2 | 2/2 |
| _ | 1/2 | 1/2 | 1/2 | 2/2 |
| { | 2/2 | 2/2 | 2/2 | 2/2 |
| } | 2/2 | 2/2 | 1/2 | 2/2 |

## Hyphen contexts (correct hyphen count)

| Context | v24_1_r4 | v24_1_r3_1 | v24_1_r3 | v24_1_r2_1_6 |
|---|---:|---:|---:|---:|
| double hyphen | 2/2 | 2/2 | 2/2 | 2/2 |
| line beginning | 2/2 | 2/2 | 2/2 | 2/2 |
| line ending | 2/2 | 2/2 | 2/2 | 2/2 |
| number-number | 2/2 | 2/2 | 1/2 | 1/2 |
| word - word | 2/2 | 2/2 | 2/2 | 2/2 |
| word-word | 2/2 | 2/2 | 2/2 | 2/2 |
| year-year | 2/2 | 2/2 | 2/2 | 2/2 |

**Limitations:** Each crop is a synthetic pre-segmented 48px textline.
Symbol correctness is checked by minimum-edit character alignment against ground truth.
Hyphen count correctness does not prove correct character placement.
Results are not interchangeable with full-page end-to-end CER.
