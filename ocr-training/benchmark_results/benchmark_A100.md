# Benchmark & Hardware Limit Analysis: A100
**Cluster / GPU**: `A100` | **Pricing**: `$2.19/h` | **Primary Bottleneck**: `GPU Compute Saturated (Optimal)`

## 1. Kết Quả Đo Lường Chi Tiết (Full Measurement Trials)
| Batch | Workers | Status | IPS (mẫu/s) | GPU Util % | VRAM (MB) | VRAM % | Reader Cost | Batch Cost | Reader Ratio | Est 40 Epoch | Est Cost | IPS/$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 128 | 4 | SUCCESS | 246.5 | 93% | 24457 | 59.7% | 0.0002s | 0.5194s | 0.00 | 11.27h | $24.68 | 112.5 |
| 128 | 8 | SUCCESS | 246.2 | 92% | 24457 | 59.7% | 0.0002s | 0.5198s | 0.00 | 11.28h | $24.71 | 112.4 |
| 128 | 12 | SUCCESS | 246.3 | 92% | 24457 | 59.7% | 0.0002s | 0.5197s | 0.00 | 11.28h | $24.70 | 112.5 |
| 128 | 16 | SUCCESS | 246.9 | 93% | 24457 | 59.7% | 0.0002s | 0.5185s | 0.00 | 11.25h | $24.64 | 112.7 |
| 128 | 24 | SUCCESS | 246.1 | 92% | 24457 | 59.7% | 0.0002s | 0.5200s | 0.00 | 11.29h | $24.71 | 112.4 |
| 256 💥 *(OOM)* | 16 | OOM | - | - | - | - | - | - | - | - | - | - |
| 192 🌟 *(Best)* | 16 | SUCCESS | 255.9 | 95% | 36233 | 88.5% | 0.0002s | 0.7503s | 0.00 | 10.85h | $23.77 | 116.8 |

## 2. Kết Luận & Đề Xuất Cấu Hình
- **Max Batch Chạy Được (Hardware Limit)**: `192` (ngưỡng tối đa trước khi tràn VRAM).
- **Best Batch Nên Dùng (Sweet Spot)**: `192` (đạt throughput `255.9` mẫu/s, khai thác `88.5%` VRAM).
- **Workers Tối Ưu**: `16` (giữ `reader_ratio = 0.00`, đảm bảo GPU không bị đói dữ liệu).
- **Điểm Nghẽn Chính (Bottleneck)**: `GPU Compute Saturated (Optimal)`.
- **Thời Gian Huấn Luyện Full 40 Epoch (10M samples)**: `10.85 giờ` (~`651 phút`).
- **Chi Phí Ước Tính Full 40 Epoch**: `$23.77`.
- **Hiệu Suất Kinh Tế (Throughput / $)**: `116.8 samples / $`.