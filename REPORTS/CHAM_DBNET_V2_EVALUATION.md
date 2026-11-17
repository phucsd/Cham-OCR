# Đánh giá `Cham-DBNet v2` (`ch_PP-OCRv4_det_cham_v2` — HarfBuzz Multi-Layout)
- **Thời điểm**: `2026-11-17T05:32:39.242418+00:00`
- **Epoch tốt nhất**: Epoch `latest` (`SHA256: 0d3d104009eefdd5...`)
- **Train Loop Duration**: `1728.0s` (28.8 phút)
- **Pipeline / Container Overhead**: `564.37s` (9.41 phút)
- **Total Kernel Wall-clock Time**: `2292.37s` (38.21 phút)
- **Quyết định Promotion Gate**: `kept_v1_no_promotion` (`promote_v2=False`)

## 1. Held-Out `detector_v3_hb` Val-120 (1,473 dòng HarfBuzz đa bố cục)
| Detector | Recall | Precision | H-Mean | Crop Usability (`>=90%` GT) | Mean IoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Cham-DBNet v1` | 97.08% | 91.67% | **94.30%** | 14.39% | 0.7705 |
| `Cham-DBNet v2` | 100.00% | 100.00% | **100.00%** | 5.50% | 0.7469 |

## 2. End-to-End Page Benchmarks (cố định Recognizer `v24_1_r4`)
| Detector | `harum_palei_01` Lines | `harum_palei_01` E2E Doc CER | `short:5` Doc CER | `a4:5` Doc CER | `audit_18` Doc CER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Cham-DBNet v1` | 25/25 | 3.19% | 16.47% | 12.78% | 11.32% |
| `Cham-DBNet v2` | 25/25 | 3.19% | 60.9% | 53.33% | 20.9% |
