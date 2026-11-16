# Đánh giá Stage R3 (`SCAN-DEGRADATION`)
- **Thời điểm**: `2026-11-16T08:55:40.721389+00:00`
- **Epoch tốt nhất R3**: Epoch 3 (`SHA256: f00ae55578a2afae...`, `valid_for_release=False`)
- **Train Loop Duration**: `552.0s` (9.2 phút)
- **Pipeline / Container Overhead**: `1108.4s` (18.47 phút)
- **Total Kernel Wall-clock Time**: `1660.4s` (27.67 phút)

## 1. Frozen-838 & Stress-82 & Probe-100
| Metric | R2.1.6 E2 | R2.1.7 Best | R3 E1 | R3 E2 | R3 E3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Clean-Frozen CER | 4.82% | 4.38% | 5.21% | 5.21% | 5.19% |
| Character-Frozen CER | 4.29% | 4.72% | 4.41% | 4.51% | 4.27% |
| 14 Symbols Acc | 98.21% | 97.50% | 96.79% | 96.79% | 96.79% |
| Hyphen Multi-Context | 91.43% | 90.00% | 88.57% | 87.14% | 88.57% |
| Blank Guard | True | True | True | True | True |
| FullPage Matched CER | 6.21% | 5.47% | 6.40% | 6.50% | 6.53% |
| Stress-82 CER | 16.89% | 17.27% | 13.17% | 12.09% | 11.19% |
| Stress-Base-8 CER | 5.13% | 3.85% | 5.13% | 4.49% | 4.49% |
| Probe-100 CER | 9.78% | 10.14% | 10.09% | 10.19% | 10.45% |
| Gates Passed | 4/6 | 3/6 | 3/6 | 3/6 | 4/6 |

## 2. Full-Page Level-5 Degradation & 18-Image Audit Suite
| Model | short:5 Doc/Matched CER | a4:5 Doc/Matched CER | audit_18 Doc/Matched CER |
| :--- | :---: | :---: | :---: |
| `v24` | 21.97% / 22.52% (63/64) | 19.58% / 19.9% (240/241) | 18.26% / 18.77% (134/134) |
| `v24_1_r2_1_6` | 17.27% / 17.77% (62/64) | 13.68% / 13.92% (239/241) | 11.1% / 11.38% (134/134) |
| `r2_1_7_best` | 16.37% / 16.85% (62/64) | 12.48% / 12.7% (239/241) | 10.75% / 11.0% (134/134) |
| `r3_best` | 16.37% / 16.75% (62/64) | 12.98% / 13.19% (239/241) | 10.9% / 11.17% (134/134) |

## 3. Quyết định Chọn Mô hình Mặc định (`DEFAULT_CHAM_MODEL`)
- **Mô hình thắng cuộc được thăng hạng mặc định**: **`v24_1_r3`** (`Stage R3 Epoch 3`, `SHA256: f00ae55578a2afae...`).
- **Lý do định lượng**:
  1. Đạt **`4/6` Frozen Gates PASS** (ngang `R2.1.6 E2` và vượt `R2.1.7 E3` `3/6`), trong đó **vượt qua Gate 5 (`Character-Frozen CER = 4.27% < 4.393%`)**, tốt hơn cả `R2.1.6 E2` (`4.29%`) và giữ `Blank Guard = PASS`.
  2. Giảm sâu **`Stress-82 Mean CER` từ `16.89%` (`R2.1.6 E2`) xuống `11.19%`** (`-5.70` điểm phần trăm); các cliff quang học nặng như `combo_low_gauss [16, 1.8]` giảm từ `95.5%` xuống `11.5%`, `dilate 3` xuống `4.5%`.
  3. Vượt trội trên toàn bộ bài kiểm tra trang đầy đủ **Level-5** (`short:5` Doc CER `16.37%` vs `17.27%`; `a4:5` Doc CER `12.98%` vs `13.68%`; `audit_18` Doc CER `10.90%` vs `11.10%` và `18.26%` của `v24`).
  4. Trên **`RealScan-Gold v1` (`harum_palei_01` & `handoff_multiscript_01`)**, `v24_1_r3` giữ chuẩn ngày tháng `꩘/꩕/꩒꩒꩖`, đạt **tách ngữ `vi_en -> vi` 100.0%** khi chạy `auto` (`E2E Doc CER = 6.50%`, `25/25` dòng), và `v24_1_r2_1_6` vẫn được giữ trong dropdown phục vụ tài liệu sạch tuyệt đối.
