# Benchmark & Hardware Limit Analysis: NVIDIA L4 (24GB)
**Cluster / GPU**: `NVIDIA L4 (24GB)` | **Pricing**: `$0.79/h` | **Primary Bottleneck**: `GPU Compute Bound & VRAM Limited`

## 1. Kết Quả Đo Lường Chi Tiết (Full Measurement Trials)
| Batch | Workers | Status | IPS (mẫu/s) | GPU Util % | VRAM (MB) | VRAM % | Reader Cost | Batch Cost | Reader Ratio | Est 40 Epoch | Est Cost | IPS/$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 128 💥 *(OOM)* | 4 | OOM | - | - | 22,556 | 100% | - | - | - | - | - | - |
| 96 | 4 | SUCCESS | 74.0 | 98% | 18,240 | 79.2% | 0.0001s | 1.2964s | 0.00 | 37.53h | $29.65 | 93.7 |
| 96 | 8 | SUCCESS | 74.0 | 98% | 18,240 | 79.2% | 0.0001s | 1.2965s | 0.00 | 37.53h | $29.65 | 93.7 |
| 96 | 12 | SUCCESS | 74.0 | 98% | 18,240 | 79.2% | 0.0001s | 1.2979s | 0.00 | 37.53h | $29.65 | 93.7 |
| 96 🌟 *(Best)* | 16 | SUCCESS | 74.3 | 98% | 18,240 | 79.2% | 0.0001s | 1.2927s | 0.00 | 37.38h | $29.53 | 94.1 |
| 96 🛡️ *(Max Safe)* | 24 | SUCCESS | 74.3 | 98% | 17,146 | 74.4% | 0.0001s | 1.2913s | 0.00 | 37.38h | $29.53 | 94.1 |

## 2. Kết Luận & Đề Xuất Cấu Hình
- **Max Batch Chạy Được (Hardware Limit)**: `96` (ngưỡng tối đa trước khi tràn bộ nhớ 24GB VRAM).
- **Best Batch Nên Dùng (Sweet Spot)**: `96` (đạt throughput `74.3` mẫu/s, khai thác `79.2%` VRAM và `98%` GPU Compute).
- **Workers Tối Ưu**: `16` hoặc `24` worker (giữ `reader_ratio = 0.00`, không bị nghẽn nạp dữ liệu).
- **Điểm Nghẽn Chính (Bottleneck)**: `GPU Compute Bound & VRAM Limited` (nhân Ada Lovelace L4 xử lý tốn 1.29s/step, chậm gấp 2.5 lần A100 SXM4).
- **Thời Gian Huấn Luyện Full 40 Epoch (10M samples)**: `37.38 giờ` (~`2,243 phút` = 1.56 ngày).
- **Chi Phí Ước Tính Full 40 Epoch**: `$29.53`.
- **Hiệu Suất Kinh Tế (Throughput / $)**: `94.1 samples / $`.