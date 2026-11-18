# Đánh giá `RealScan-Gold v2` & `OOD-Archival-Gold v2` (212 dòng hoàn toàn độc lập tập huấn luyện)
- **Thời điểm**: `2026-11-18T05:43:03.741761+00:00`
- **Quy mô**: `1` trang chụp thực (`harum_palei_01`, 25 dòng) + `7` mẫu công khai Wikimedia Commons + `12` trang cổ thư/sách OOD (`180` dòng từ 12 phông EFEO + 6 phông Held-Out chưa từng dùng huấn luyện).

## 1. Tổng hợp Xuyên Miền (`harum_palei_01` + Wikimedia + 180 dòng OOD Archival)
| Model | `harum_palei_01` Crop CER | Wikimedia Specimens CER | OOD-180 Crop CER | OOD-180 WER | OOD-12 Pages E2E Lines | OOD-12 Pages E2E Doc CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `prod_r41wise` | **2.55%** | 35.24% (2/7) | **26.35%** | 57.1% | 180/180 | **25.58%** |
| `v26_masked` | **2.68%** | 35.24% (2/7) | **26.59%** | 57.57% | 180/180 | **25.12%** |
| `v26_control` | **2.07%** | 35.24% (2/7) | **26.5%** | 58.04% | 180/180 | **24.77%** |
| `v26_masked_soup` | **2.31%** | 36.19% (2/7) | **26.33%** | 57.57% | 180/180 | **24.75%** |
| `v26_masked_wise` | **2.31%** | 35.24% (2/7) | **26.26%** | 58.23% | 180/180 | **24.89%** |

## 2. Phân rã Theo 3 Miền Tư liệu Cổ thư OOD (Crop CER / E2E Doc CER)
| Model | `efeo_palm_leaf_agal` (Lá buông) | `efeo_regional_folio` (Sách cổ 1-2 cột) | `heldout_camera_book` (Sách in chụp camera) |
| :--- | :---: | :---: | :---: |
| `prod_r41wise` | 30.04% / 28.86% (58/58) | 43.0% / 43.77% (62/62) | 9.37% / 7.57% (60/60) |
| `v26_masked` | 30.68% / 27.56% (58/58) | 43.4% / 43.54% (62/62) | 9.1% / 7.7% (60/60) |
| `v26_control` | 30.97% / 26.95% (58/58) | 42.83% / 43.31% (62/62) | 9.04% / 7.51% (60/60) |
| `v26_masked_soup` | 30.47% / 27.29% (58/58) | 42.59% / 42.77% (62/62) | 9.23% / 7.57% (60/60) |
| `v26_masked_wise` | 30.54% / 27.43% (58/58) | 42.67% / 42.69% (62/62) | 8.91% / 7.89% (60/60) |
