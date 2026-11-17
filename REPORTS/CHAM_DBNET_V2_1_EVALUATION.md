# Đánh giá `Cham-DBNet v2.1` (Unified Sinusoidal Wave `Level-5` + HarfBuzz Multi-Layout)
- **Thời điểm**: `2026-11-17T10:13:01.337972+00:00`
- **Epoch tốt nhất**: Epoch `10` (`SHA256: 5a2cb125ed99edfb...`)
- **Train Loop / Overhead / Total Wall-clock**: `3106.0s` / `525.3s` / `3631.3s` (60.52 phút)
- **Quyết định Promotion Gate**: `kept_v1_no_promotion` (`promote_v2_1=False`)

## 1. Held-Out Validation (`Val-120` Multi-Layout & `Val-160` Hybrid Wave+Multi-Layout)
| Detector | `Val-120` H-Mean (`R / P`) | `Val-160` Wave+Multi H-Mean (`R / P`) | `Val-160` Usable Crop (`>=85%`) |
| :--- | :---: | :---: | :---: |
| `Cham-DBNet v1` | **94.30%** (97.08% / 91.67%) | **96.58%** (98.61% / 94.63%) | 34.38% |
| `Cham-DBNet v2.1` | **100.00%** (100.0% / 100.0%) | **99.77%** (99.54% / 100.0%) | 26.73% |

## 2. End-to-End Page Benchmarks (`DET_MODE='hybrid'`, cố định Recognizer `v24_1_r4`)
| Detector | `harum_palei_01` E2E CER | `short:5` Doc CER | `a4:5` Doc CER | `audit_18` Doc CER |
| :--- | :---: | :---: | :---: | :---: |
| `Cham-DBNet v1` | 3.19% (25/25) | 16.47% (62/64) | 12.78% (240/241) | **11.32%** (134/134) |
| `Cham-DBNet v2.1` | 3.19% (25/25) | 67.34% (36/64) | 63.23% (125/241) | **21.03%** (124/134) |
