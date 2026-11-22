# Đánh giá `Cham-DBNet v3` (Full-Cell Curved Polygons + Border OHEM Hard-Negatives + Short-Line Recall)
- **Thời điểm**: `2026-11-21T17:29:30.523547+00:00`
- **Kế toán huấn luyện Kaggle Dual T4x2**: Train Loop = `3537.0s`, Overhead = `378.5s`, Wall-clock = `3915.5s`

## 1. Đối chuẩn End-to-End & Non-Regression Guards (`DET_MODE='hybrid'`) 
| Config | `Val-160` poly-IoU | `Val-v3` poly-IoU | `short:5` (lines) | `a4:5` (lines) | `audit_18` (lines) | `a4:1-4` Guard (lines) | `FullPage-50` E2E (Recall) | Qualifies |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `v1_quad` | 96.63% | 91.18% | 16.4% (62/64) | 12.95% (240/241) | 11.25% (134/134) | 6.45% (1153/1153) | 14.21% (95.4%) | `baseline` |
| `v1_poly_u1.8` | 95.1% | 88.7% | 7.37% (64/64) | 6.96% (241/241) | 10.14% (134/134) | 6.31% (1153/1153) | 7.82% (99.58%) | `baseline` |
| `v3_e16_poly_u3.0_vpad0.15` | 100.0% | 100.0% | 7.06% (64/64) | 8.73% (254/241) | 9.91% (134/134) | 6.2% (1153/1153) | 7.59% (99.58%) | `False` |
| `v3_e16_poly_u3.2_vpad0.15` | 100.0% | 100.0% | 6.75% (64/64) | 8.38% (252/241) | 9.91% (134/134) | 6.15% (1153/1153) | 7.39% (99.58%) | `False` |
| `v3_e20_poly_u3.0_vpad0.15` | 100.0% | 99.95% | 6.89% (64/64) | 7.51% (249/241) | 9.86% (134/134) | 6.17% (1153/1153) | 7.7% (99.58%) | `False` |
| `v3_e20_poly_u3.2_vpad0.15` | 100.0% | 99.9% | 6.68% (64/64) | 7.54% (248/241) | 9.75% (134/134) | 6.13% (1153/1153) | 7.59% (99.58%) | `False` |
| `v3_soup_16_18_20_poly_u3.0_vpad0.15` | 100.0% | 99.97% | 6.96% (64/64) | 8.3% (252/241) | 9.99% (134/134) | 6.22% (1153/1153) | 7.63% (99.58%) | `False` |
| `v3_soup_16_18_20_poly_u3.2_vpad0.15` | 100.0% | 99.97% | 7.27% (64/64) | 8.04% (251/241) | 10.06% (134/134) | 6.14% (1153/1153) | 7.42% (99.58%) | `False` |

## 2. Quyết định Promotion
- **Winner**: `None`; **promoted**: `False`; active `ch_PP-OCRv4_det_cham_infer` SHA256_16: `cfc65a686ecaa5fd`
- `v3_e16_poly_u3.0_vpad0.15` (`sum=25.7%` vs `v1_poly=24.47%`): qualifies=`False` (trượt: `['a4_5_exact_241_lines', 'beats_v1_poly_on_l5_audit_sum']`)
- `v3_e16_poly_u3.2_vpad0.15` (`sum=25.04%` vs `v1_poly=24.47%`): qualifies=`False` (trượt: `['a4_5_exact_241_lines', 'beats_v1_poly_on_l5_audit_sum']`)
- `v3_e20_poly_u3.0_vpad0.15` (`sum=24.26%` vs `v1_poly=24.47%`): qualifies=`False` (trượt: `['a4_5_exact_241_lines']`)
- `v3_e20_poly_u3.2_vpad0.15` (`sum=23.97%` vs `v1_poly=24.47%`): qualifies=`False` (trượt: `['a4_5_exact_241_lines']`)
- `v3_soup_16_18_20_poly_u3.0_vpad0.15` (`sum=25.25%` vs `v1_poly=24.47%`): qualifies=`False` (trượt: `['a4_5_exact_241_lines', 'beats_v1_poly_on_l5_audit_sum']`)
- `v3_soup_16_18_20_poly_u3.2_vpad0.15` (`sum=25.37%` vs `v1_poly=24.47%`): qualifies=`False` (trượt: `['a4_5_exact_241_lines', 'beats_v1_poly_on_l5_audit_sum']`)
