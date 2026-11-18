# Báo cáo Kiểm thử Trực tiếp trên Production (`https://ocr.cham.asia`)
- **Thời điểm nghiệm thu**: `2026-11-18T05:46:48.675196+00:00`
- **HF Space**: `phucsd/cham-ocr-studio` (Commit `e46e8328f4618347ac165b841847f5393f5fbcec`)
- **Thời gian rollout container**: `45s`
- **Blank-Page Guard**: `0` dòng, `status = no_text` (**PASS**)

## 1. Kết quả Live E2E trên Ảnh Chụp Thực tế `harum_palei_01` (25 dòng)
| Chế độ Model | Trọng số (`SHA256_16`) | Input Shape | Số dòng phát hiện | Live E2E Doc CER | Live Matched CER | Backend Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `default_v24_1_r4` | `85525b0ca091acaf` | `3, 48, 480` | **25/25** | **3.43%** | **3.53%** | `6.9857s` |
| `v24_1_r3_1` | `5fb0d8e4d512b4c0` | `3, 48, 320` | **25/25** | **5.08%** | **5.23%** | `7.0285s` |
| `auto` | `85525b0ca091acaf` | `3, 48, 480` | **25/25** | **3.43%** | **3.53%** | `7.2906s` |

## 2. Kết quả Live Replay Bộ 18 Ảnh Audit (`ocr-audit-2026-11-04`)
- **Tổng số dòng phát hiện**: `134/134` (`abs_line_err = 0`, `blank_zero_lines = True`)
- **Live Document CER**: **`10.16%`**
- **Live Matched Line CER**: **`10.53%`**
