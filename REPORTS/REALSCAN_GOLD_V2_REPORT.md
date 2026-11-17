# Đánh giá `RealScan-Gold v2` & `OOD-Archival-Gold v2` (212 dòng hoàn toàn độc lập tập huấn luyện)
- **Thời điểm**: `2026-11-17T11:30:01.548673+00:00`
- **Quy mô**: `1` trang chụp thực (`harum_palei_01`, 25 dòng) + `7` mẫu công khai Wikimedia Commons + `12` trang cổ thư/sách OOD (`180` dòng từ 12 phông EFEO + 6 phông Held-Out chưa từng dùng huấn luyện).

## 1. Tổng hợp Xuyên Miền (`harum_palei_01` + Wikimedia + 180 dòng OOD Archival)
| Model | `harum_palei_01` Crop CER | Wikimedia Specimens CER | OOD-180 Crop CER | OOD-180 WER | OOD-12 Pages E2E Lines | OOD-12 Pages E2E Doc CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `v24` | **6.33%** | 35.24% (2/7) | **32.6%** | 70.56% | 180/180 | **32.1%** |
| `v24_1_r3` | **6.33%** | 26.67% (2/7) | **27.05%** | 59.55% | 180/180 | **29.26%** |
| `v24_1_r3_1` | **4.62%** | 27.62% (2/7) | **25.15%** | 55.79% | 180/180 | **24.24%** |
| `v24_1_r4` | **2.55%** | 35.24% (2/7) | **26.35%** | 57.1% | 180/180 | **25.58%** |
| `auto` | **2.55%** | 35.24% (2/7) | **27.15%** | 57.38% | 180/180 | **26.12%** |

## 2. Phân rã Theo 3 Miền Tư liệu Cổ thư OOD (Crop CER / E2E Doc CER)
| Model | `efeo_palm_leaf_agal` (Lá buông) | `efeo_regional_folio` (Sách cổ 1-2 cột) | `heldout_camera_book` (Sách in chụp camera) |
| :--- | :---: | :---: | :---: |
| `v24` | 38.64% / 35.29% (58/58) | 46.23% / 49.11% (62/62) | 15.9% / 15.14% (60/60) |
| `v24_1_r3` | 31.53% / 33.65% (58/58) | 43.97% / 48.96% (62/62) | 9.1% / 8.97% (60/60) |
| `v24_1_r3_1` | 28.55% / 25.79% (58/58) | 40.81% / 43.23% (62/62) | 9.23% / 7.19% (60/60) |
| `v24_1_r4` | 30.04% / 28.86% (58/58) | 43.0% / 43.77% (62/62) | 9.37% / 7.57% (60/60) |
| `auto` | 30.04% / 28.86% (58/58) | 45.67% / 45.55% (62/62) | 9.37% / 7.57% (60/60) |
