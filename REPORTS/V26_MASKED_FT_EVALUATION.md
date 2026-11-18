# Đánh giá V26 Masked-SVTR Fine-Tune (mask attention + SE theo padding) và nhánh đối chứng
- **Thời điểm**: `2026-11-18T05:43:04.234302+00:00`
- **Production trước**: `v24_1_r4` SHA256_16 `85525b0ca091acaf` (R4.1 WiSE-FT, `det_sig=72d1bd072e4f5c2a`)
- **masked**: epoch tốt nhất `3`; Train Loop / Overhead / Wall-clock = `3188.0s` / `1326.35s` / `4514.35s`
- **control**: epoch tốt nhất `2`; Train Loop / Overhead / Wall-clock = `3235.0s` / `1070.16s` / `4305.16s`

## 1. Frozen-838 và các bộ phụ (cùng cấu hình detector hiện hành)
| Metric | Production R4.1 WiSE | V26 masked | V26 control | V26 masked soup | V26 masked WiSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Clean CER (%) | 5.28 | 5.07 | 5.56 | 5.11 | 4.98 |
| Character CER (%) | 2.64 | 2.37 | 2.6 | 2.33 | 2.33 |
| 14 Symbols (%) | 99.29 | 98.93 | 100.0 | 100.0 | 100.0 |
| Hyphen (%) | 95.71 | 95.71 | 97.14 | 94.29 | 94.29 |
| FullPage Matched CER (%) | 7.39 | 6.76 | 7.15 | 6.72 | 6.67 |
| Stress-82 CER (%) | 6.93 | 4.64 | 5.43 | 4.28 | 4.56 |
| Probe-100 W=480 CER (%) | 4.2 | 4.1 | 4.2 | 4.0 | 4.25 |
| harum crops CER (%) | 2.55 | 2.68 | 2.07 | 2.31 | 2.31 |

## 2. Theo từng epoch
| Nhánh | Epoch | Clean | Char | Symbols | Hyphen | FullPage | Stress-82 | Probe | Harum | Gates |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| masked | 1 | 5.46% | 2.91% | 98.21% | 90.0% | 7.25% | 5.46% | 4.35% | 2.55% | 4/6 |
| masked | 2 | 5.33% | 2.43% | 98.93% | 94.29% | 7.09% | 4.61% | 3.79% | 2.55% | 4/6 |
| masked | 3 | 5.07% | 2.37% | 98.93% | 95.71% | 6.76% | 4.64% | 4.1% | 2.68% | 5/6 |
| masked | 4 | 5.15% | 2.39% | 100.0% | 94.29% | 6.86% | 3.99% | 3.94% | 1.95% | 5/6 |
| masked | 5 | 5.19% | 2.35% | 99.64% | 94.29% | 6.83% | 4.27% | 3.94% | 2.43% | 4/6 |
| masked | 6 | 5.21% | 2.39% | 99.64% | 94.29% | 6.8% | 4.14% | 3.69% | 2.31% | 4/6 |
| control | 1 | 5.72% | 2.47% | 98.93% | 94.29% | 7.36% | 6.41% | 4.35% | 2.43% | 4/6 |
| control | 2 | 5.56% | 2.6% | 100.0% | 97.14% | 7.15% | 5.43% | 4.2% | 2.07% | 6/6 |
| control | 3 | 5.26% | 2.47% | 99.64% | 97.14% | 6.84% | 5.21% | 4.2% | 2.31% | 5/6 |
| control | 4 | 5.24% | 2.49% | 99.64% | 94.29% | 6.88% | 4.56% | 4.25% | 2.19% | 4/6 |
| control | 5 | 5.19% | 2.47% | 99.64% | 95.71% | 6.83% | 4.75% | 4.1% | 2.07% | 5/6 |
| control | 6 | 5.07% | 2.41% | 99.64% | 95.71% | 6.87% | 4.64% | 3.48% | 2.43% | 5/6 |
| masked_soup | soup_50e3_50e4 | 5.11% | 2.33% | 100.0% | 94.29% | 6.72% | 4.28% | 4.0% | 2.31% | 5/6 |
| masked_soup | soup_40e3_40e4_20e6 | 5.1% | 2.37% | 100.0% | 94.29% | 6.72% | 4.12% | 3.94% | 2.31% | 5/6 |
| masked_soup | soup_60e3_40e4 | 5.14% | 2.35% | 99.64% | 94.29% | 6.73% | 4.33% | 4.0% | 2.31% | 4/6 |
| masked_wise | a=0.15 | 4.98% | 2.33% | 100.0% | 94.29% | 6.67% | 4.56% | 4.25% | 2.31% | 5/6 |
| masked_wise | a=0.25 | 4.94% | 2.39% | 100.0% | 90.0% | 6.71% | 4.51% | 4.2% | 2.43% | 5/6 |
| masked_wise | a=0.35 | 4.94% | 2.41% | 98.57% | 87.14% | 6.72% | 4.7% | 4.25% | 2.31% | 4/6 |

## 3. RealScan-Gold v2
| Model | harum crops | Wikimedia | OOD-180 crops | OOD-12 E2E Doc CER (dòng) |
| :--- | :---: | :---: | :---: | :---: |
| Production | 2.55% | 35.24% | 26.35% | 25.58% (180/180) |
| V26 masked | 2.68% | 35.24% | 26.59% | 25.12% (180/180) |
| V26 control | 2.07% | 35.24% | 26.5% | 24.77% (180/180) |
| V26 masked_soup | 2.31% | 36.19% | 26.33% | 24.75% (180/180) |
| V26 masked_wise | 2.31% | 35.24% | 26.26% | 24.89% (180/180) |

## 4. Quyết định thăng cấp tự động
- **masked**: qualifies = `False` (trượt: `['valid_for_release_6_of_6']`)
- **control**: qualifies = `False` (trượt: `['clean_non_regression']`)
- **masked_soup**: qualifies = `False` (trượt: `['valid_for_release_6_of_6']`)
- **masked_wise**: qualifies = `False` (trượt: `['valid_for_release_6_of_6']`)
- **Winner**: `None`; **promoted**: `False`; production SHA256_16 sau: `85525b0ca091acaf`
