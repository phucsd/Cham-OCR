# Nghiệm thu Sau Bản Vá Bước 1 trên Production (`https://ocr.cham.asia`)
- **Thời điểm (UTC)**: `2026-11-22T06:55:22Z`
- **HF Space Commit**: `9a4e559e61e980f4d5cb59a86dfcdf1985dfe239`

## 1. Vá lỗi P0/P1 API Contract & Model Provenance (`6/6` trả HTTP 400 JSON)
| Trường hợp | Trước bản vá | Sau bản vá | Thông báo lỗi trả về |
| :--- | :---: | :---: | :--- |
| `missing_image` | HTTP `502` | **HTTP `400`** | `Missing or empty 'image' field (expected base64-encoded image)` |
| `invalid_model` (`nonexistent_model_2026`) | HTTP `200` (V23 giả danh) | **HTTP `400`** | `Unknown model 'nonexistent_model_2026'. Allowed models: v24_1_r4, auto, v24_1_r3_1, v24_1_r2_1_6, vi, v25, v24, v23` |
| `invalid_segmentation` (`not_a_method`) | HTTP `200` (rơi vào adaptive) | **HTTP `400`** | `Invalid segmentation method 'not_a_method'. Allowed methods: dbnet, valley, adaptive` |
| `invalid_reading_order` (`invalid`) | HTTP `200` | **HTTP `400`** | `Invalid reading_order 'invalid'. Allowed values: auto, rows` |
| `invalid_bbox_zero` (`[0,0,0,0]`) | HTTP `200` (crop `1x1`) | **HTTP `400`** | `Invalid 'bbox' [0, 0, 0, 0]: must satisfy x2 > x1 and y2 > y1` |
| `invalid_bbox_negative` (`[-200,-200,-1,-1]`) | HTTP `200` (crop `1x1`) | **HTTP `400`** | `Invalid 'bbox' [-200, -200, -1, -1]: region lies outside image bounds (850x249)` |

## 2. `Auto Router v2.2` (Vá dòng tiếng Việt nhiều số/ký hiệu + Xuất `selected_recognizer`)
- **`vietnamese_00`**: CER giảm mạnh xuống **`4.88%`**, dòng 3 (`Số trang: 0123456789, dấu ! ? : ; ( ) / -`) định tuyến chính xác sang `selected_recognizer = "vi"` (`Số trang: 0123456789, dấu !?:; () /-`).
- **`mixed_vi_cham_00`**: CER **`0.0%`**, dòng 1 → `"vi"`, dòng 2 → `"v24_1_r4"`.

## 3. Chữ số Chăm `U+AA50..U+AA59` sau khi sửa lỗi cắt khung mực (`ImageOps.invert`)
- **Trước khi sửa script test** (canvas trắng `1100x100` chưa cắt, tắt Padding Mask): CER = `14.0%`, Đúng toàn dòng = `33.33%` (`4/12`).
- **Sau khi cắt sát khung mực thực tế** (`12` crops held-out Light/Medium/SemiBold): CER = **`6.0%`**, Đúng toàn dòng = **`75.0%`** (`9/12`).
