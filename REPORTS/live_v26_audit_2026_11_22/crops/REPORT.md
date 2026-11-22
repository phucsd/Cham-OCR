# Live symbol and hyphen crop test

Endpoint: /ocr_crop (recognition-only, not line detection)
Report UTC: 2026-11-22T04:35:24.139358+00:00
Requests collected: 421

| Model | Valid crops | Character CER | Exact full lines | Request failures |
|---|---:|---:|---:|---:|
| v24_1_r4 | 63 | 0.75% | 93.65% | 0 |
| v25 | 63 | 43.18% | 0.0% | 0 |
| auto | 63 | 0.6% | 93.65% | 0 |
| v24_1_r3_1 | 63 | 1.2% | 87.3% | 0 |
| v24_1_r2_1_6 | 43 | 1.41% | 83.72% | 0 |
| v24 | 42 | 18.89% | 0.0% | 0 |
| v23 | 42 | 26.9% | 0.0% | 0 |
| vi | 42 | 118.69% | 0.0% | 0 |

## Each required symbol (minimum-edit alignment)

| Symbol | v24_1_r4 | v25 | auto | v24_1_r3_1 | v24_1_r2_1_6 | v24 | v23 | vi |
|---||---:||---:||---:||---:||---:||---:||---:||---:|
| ! | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 1/3 |
| # | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| & | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 1/3 |
| * | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| / | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 2/3 |
| < | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| > | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| ? | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 3/3 |
| [ | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 |
| BACKSLASH | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| ] | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| _ | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| { | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| } | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |

## Hyphen contexts (correct hyphen count)

| Context | v24_1_r4 | v25 | auto | v24_1_r3_1 | v24_1_r2_1_6 | v24 | v23 | vi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| double hyphen | 3/3 | 0/3 | 3/3 | 2/3 | 1/1 | 0/0 | 0/0 | 0/0 |
| line beginning | 3/3 | 0/3 | 3/3 | 2/3 | 0/0 | 0/0 | 0/0 | 0/0 |
| line ending | 3/3 | 0/3 | 3/3 | 3/3 | 0/0 | 0/0 | 0/0 | 0/0 |
| number-number | 3/3 | 0/3 | 3/3 | 3/3 | 0/0 | 0/0 | 0/0 | 0/0 |
| word - word | 3/3 | 0/3 | 3/3 | 3/3 | 0/0 | 0/0 | 0/0 | 0/0 |
| word-word | 3/3 | 0/3 | 3/3 | 3/3 | 0/0 | 0/0 | 0/0 | 0/0 |
| year-year | 3/3 | 0/3 | 3/3 | 3/3 | 0/0 | 0/0 | 0/0 | 0/0 |

**Limitations:** Each crop is a synthetic pre-segmented 48px textline.
Symbol correctness is checked by minimum-edit character alignment against ground truth.
Hyphen count correctness does not prove correct character placement.
Results are not interchangeable with full-page end-to-end CER.
