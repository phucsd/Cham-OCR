# Đánh giá V26.1 Masked-SVTR (16 Epochs, 2 Chặng, Early-Backbone Frozen + Paleographic Augmentation)
- **Thời điểm**: `2026-11-22T01:15:20.866193+00:00`
- **Production trước**: `v24_1_r4` SHA256_16 `85525b0ca091acaf` (`det_sig=72d1bd072e4f5c2a`)
- **Tổng kế toán 2 chặng**: Train Loop = `12450.0s` (`207.5m`), Overhead = `490.44s`, Wall-clock = `12940.44s`

## 1. Kế toán theo từng chặng
| Chặng | Train Loop (s) | Overhead (s) | Wall-clock (s) |
| :--- | :---: | :---: | :---: |
| stage_1 | 6194.0 | 238.63 | 6432.63 |
| stage_2 | 6256.0 | 251.81 | 6507.81 |

## 2. Kết quả Benchmark Frozen-838 & Bộ Phụ (`W=480`)
| Nguồn | Clean | Char | Symbols | Hyphen | FullPage | Stress-82 | Probe | Harum | Gates |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Production trước | 5.28% | 2.64% | 99.29% | 95.71% | 7.39% | 6.93% | 4.2% | 2.55% | 5/6 |
| S1 E4 | 5.11% | 1.94% | 100.0% | 97.14% | 6.72% | 3.6% | 3.48% | 2.31% | 6/6 |
| S1 E6 | 4.92% | 1.94% | 100.0% | 98.57% | 6.49% | 3.19% | 3.48% | 2.31% | 6/6 |
| S1 E8 | 4.97% | 1.85% | 100.0% | 95.71% | 6.72% | 3.14% | 3.02% | 2.55% | 6/6 |
| S2 E12 | 4.92% | 1.81% | 100.0% | 98.57% | 6.5% | 3.04% | 2.87% | 2.31% | 6/6 |
| S2 E14 | 4.8% | 1.75% | 100.0% | 98.57% | 6.48% | 2.85% | 3.12% | 2.19% | 6/6 |
| S2 E15 | 4.86% | 1.77% | 100.0% | 98.57% | 6.46% | 2.71% | 2.92% | 2.19% | 6/6 |
| S2 E16 | 4.88% | 1.77% | 99.64% | 98.57% | 6.47% | 2.72% | 2.97% | 2.31% | 5/6 |
| soup_top2_e14_e15 | 4.81% | 1.75% | 100.0% | 98.57% | 6.42% | 2.8% | 3.07% | 2.19% | 6/6 |
| soup_top3_e14_e15_e12 | 4.81% | 1.75% | 100.0% | 98.57% | 6.44% | 2.78% | 2.87% | 2.31% | 6/6 |
| soup_last3_e14_e15_e16 | 4.85% | 1.75% | 100.0% | 98.57% | 6.46% | 2.75% | 2.92% | 2.31% | 6/6 |
| wise_inbasin_a35 | 5.01% | 2.18% | 100.0% | 98.57% | 6.62% | 3.52% | 3.64% | 2.31% | 6/6 |
| wise_inbasin_a50 | 4.9% | 2.08% | 100.0% | 97.14% | 6.55% | 3.36% | 3.69% | 2.55% | 6/6 |
| wise_s2_a35 | 5.04% | 2.08% | 100.0% | 97.14% | 6.5% | 3.24% | 3.48% | 2.19% | 6/6 |
| wise_s2_a50 | 5.04% | 1.96% | 100.0% | 98.57% | 6.4% | 3.03% | 3.23% | 2.19% | 6/6 |
| wise_s2_a65 | 5.1% | 1.92% | 100.0% | 98.57% | 6.32% | 2.73% | 3.28% | 2.07% | 6/6 |
| wise_s2_a80 | 5.02% | 1.85% | 100.0% | 98.57% | 6.32% | 2.63% | 3.07% | 2.19% | 6/6 |

## 3. RealScan-Gold v2 (`9/9` Promotion Checks)
| Model | harum crops | Wikimedia | OOD-180 crops | OOD-12 E2E Doc CER (dòng) | Qualifies |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Production trước | 2.55% | 35.24% | 26.35% | 25.58% (180/180) | — |
| V26.1 (epoch_4) | 2.31% | 34.29% | 27.7% | 25.33% (180/180) | `False` |
| V26.1 (epoch_6) | 2.31% | 40.0% | 28.06% | 25.08% (180/180) | `False` |
| V26.1 (epoch_8) | 2.55% | 37.14% | 28.4% | 25.01% (180/180) | `False` |
| V26.1 (epoch_12) | 2.31% | 37.14% | 28.13% | 25.08% (180/180) | `False` |
| V26.1 (epoch_14) | 2.19% | 37.14% | 28.4% | 25.44% (180/180) | `False` |
| V26.1 (epoch_15) | 2.19% | 37.14% | 28.3% | 25.05% (180/180) | `False` |
| V26.1 (soup_top2_e14_e15) | 2.19% | 37.14% | 28.32% | 25.19% (180/180) | `False` |
| V26.1 (soup_top3_e14_e15_e12) | 2.31% | 37.14% | 28.16% | 25.12% (180/180) | `False` |
| V26.1 (soup_last3_e14_e15_e16) | 2.31% | 37.14% | 28.35% | 25.14% (180/180) | `False` |
| V26.1 (wise_inbasin_a35) | 2.31% | 37.14% | 25.87% | 23.34% (180/180) | `True` |
| V26.1 (wise_inbasin_a50) | 2.55% | 36.19% | 25.8% | 23.2% (180/180) | `True` |
| V26.1 (wise_s2_a35) | 2.19% | 32.38% | 25.51% | 23.18% (180/180) | `True` |
| V26.1 (wise_s2_a50) | 2.19% | 34.29% | 25.44% | 22.72% (180/180) | `True` |
| V26.1 (wise_s2_a65) | 2.07% | 34.29% | 25.2% | 22.65% (180/180) | `True` |
| V26.1 (wise_s2_a80) | 2.19% | 35.24% | 25.92% | 22.95% (180/180) | `True` |

## 4. Quyết định thăng cấp tự động
- **epoch_4**: qualifies = `False` (trượt: `['ood180_crop_non_regression']`)
- **epoch_6**: qualifies = `False` (trượt: `['ood180_crop_non_regression', 'wikimedia_non_regression']`)
- **epoch_8**: qualifies = `False` (trượt: `['ood180_crop_non_regression']`)
- **epoch_12**: qualifies = `False` (trượt: `['ood180_crop_non_regression']`)
- **epoch_14**: qualifies = `False` (trượt: `['ood180_crop_non_regression']`)
- **epoch_15**: qualifies = `False` (trượt: `['ood180_crop_non_regression']`)
- **soup_top2_e14_e15**: qualifies = `False` (trượt: `['ood180_crop_non_regression']`)
- **soup_top3_e14_e15_e12**: qualifies = `False` (trượt: `['ood180_crop_non_regression']`)
- **soup_last3_e14_e15_e16**: qualifies = `False` (trượt: `['ood180_crop_non_regression']`)
- **wise_inbasin_a35**: qualifies = `True` (**PASS `9/9`**)
- **wise_inbasin_a50**: qualifies = `True` (**PASS `9/9`**)
- **wise_s2_a35**: qualifies = `True` (**PASS `9/9`**)
- **wise_s2_a50**: qualifies = `True` (**PASS `9/9`**)
- **wise_s2_a65**: qualifies = `True` (**PASS `9/9`**)
- **wise_s2_a80**: qualifies = `True` (**PASS `9/9`**)
- **Winner**: `wise_s2_a65`; **promoted**: `True`; production SHA256_16 sau: `1deee5f5ff351070`
