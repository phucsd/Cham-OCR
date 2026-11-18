# Đánh giá `RealScan-Gold v2` & `OOD-Archival-Gold v2` (212 dòng hoàn toàn độc lập tập huấn luyện)
- **Thời điểm**: `2026-11-18T18:15:50.364205+00:00`
- **Quy mô**: `1` trang chụp thực (`harum_palei_01`, 25 dòng) + `7` mẫu công khai Wikimedia Commons + `12` trang cổ thư/sách OOD (`180` dòng từ 12 phông EFEO + 6 phông Held-Out chưa từng dùng huấn luyện).

## 1. Tổng hợp Xuyên Miền (`harum_palei_01` + Wikimedia + 180 dòng OOD Archival)
| Model | `harum_palei_01` Crop CER | Wikimedia Specimens CER | OOD-180 Crop CER | OOD-180 WER | OOD-12 Pages E2E Lines | OOD-12 Pages E2E Doc CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `prod_active` | **2.55%** | 35.24% (2/7) | **26.35%** | 57.1% | 180/180 | **25.58%** |
| `v26_full_best` | **1.58%** | 36.19% (2/7) | **27.92%** | 56.44% | 180/180 | **24.96%** |
| `v26_full_soup` | **1.58%** | 36.19% (2/7) | **27.89%** | 56.26% | 180/180 | **24.98%** |

## 2. Phân rã Theo 3 Miền Tư liệu Cổ thư OOD (Crop CER / E2E Doc CER)
| Model | `efeo_palm_leaf_agal` (Lá buông) | `efeo_regional_folio` (Sách cổ 1-2 cột) | `heldout_camera_book` (Sách in chụp camera) |
| :--- | :---: | :---: | :---: |
| `prod_active` | 30.04% / 28.86% (58/58) | 43.0% / 43.77% (62/62) | 9.37% / 7.57% (60/60) |
| `v26_full_best` | 31.96% / 26.2% (58/58) | 47.77% / 44.47% (62/62) | 7.98% / 7.76% (60/60) |
| `v26_full_soup` | 31.39% / 26.4% (58/58) | 48.26% / 44.47% (62/62) | 8.05% / 7.63% (60/60) |
