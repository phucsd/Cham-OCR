# Báo cáo Kiểm thử Trực tiếp trên Production (`https://ocr.cham.asia`)
- **Thời điểm nghiệm thu**: `2026-11-23T03:35:22.130432+00:00`
- **HF Space**: `phucsd/cham-ocr-studio` (Commit `a67d1cc08eaf0e40b49ebd70e53af9848856056f`)
- **Thời gian rollout container**: `45s`
- **Blank-Page Guard**: `0` dòng, `status = no_text` (**PASS**)

## 1. Kết quả Live E2E trên Ảnh Chụp Thực tế `harum_palei_01` (25 dòng)
| Chế độ Model | Trọng số (`SHA256_16`) | Input Shape | Số dòng phát hiện | Live E2E Doc CER | Live Matched CER | Backend Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `default_v24_1_r4` | `1deee5f5ff351070` | `3, 48, 480` | **25/25** | **2.6%** | **2.68%** | `16.1902s` |
| `v24_1_r3_1` | `5fb0d8e4d512b4c0` | `3, 48, 320` | **25/25** | **5.32%** | **5.47%** | `15.6026s` |
| `auto` | `1deee5f5ff351070` | `3, 48, 480` | **25/25** | **2.6%** | **2.68%** | `18.3715s` |

## 2. Kết quả Live Replay Bộ 18 Ảnh Audit (`ocr-audit-2026-11-04`)
- **Tổng số dòng phát hiện**: `134/134` (`abs_line_err = 0`, `blank_zero_lines = True`)
- **Live Document CER**: **`8.97%`**
- **Live Matched Line CER**: **`9.15%`**
