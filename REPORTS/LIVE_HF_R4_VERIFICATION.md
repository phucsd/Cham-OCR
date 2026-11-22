# Báo cáo Kiểm thử Trực tiếp trên Production (`https://ocr.cham.asia`)
- **Thời điểm nghiệm thu**: `2026-11-22T06:50:50.760227+00:00`
- **HF Space**: `phucsd/cham-ocr-studio` (Commit `9a4e559e61e980f4d5cb59a86dfcdf1985dfe239`)
- **Thời gian rollout container**: `61s`
- **Blank-Page Guard**: `0` dòng, `status = no_text` (**PASS**)

## 1. Kết quả Live E2E trên Ảnh Chụp Thực tế `harum_palei_01` (25 dòng)
| Chế độ Model | Trọng số (`SHA256_16`) | Input Shape | Số dòng phát hiện | Live E2E Doc CER | Live Matched CER | Backend Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `default_v24_1_r4` | `1deee5f5ff351070` | `3, 48, 480` | **25/25** | **2.96%** | **3.04%** | `12.2717s` |
| `v24_1_r3_1` | `5fb0d8e4d512b4c0` | `3, 48, 320` | **25/25** | **5.08%** | **5.23%** | `11.7376s` |
| `auto` | `1deee5f5ff351070` | `3, 48, 480` | **25/25** | **2.96%** | **3.04%** | `12.8329s` |

## 2. Kết quả Live Replay Bộ 18 Ảnh Audit (`ocr-audit-2026-11-04`)
- **Tổng số dòng phát hiện**: `134/134` (`abs_line_err = 0`, `blank_zero_lines = True`)
- **Live Document CER**: **`9.4%`**
- **Live Matched Line CER**: **`9.6%`**
