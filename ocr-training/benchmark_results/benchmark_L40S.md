# Benchmark & Hardware Limit Analysis: L40S
**Cluster / GPU**: `L40S` | **Pricing**: `$2.14/h` | **Primary Bottleneck**: `GPU Compute Saturated (Optimal)`

## 1. Kết Quả Đo Lường Chi Tiết (Full Measurement Trials)
| Batch | Workers | Status | IPS (mẫu/s) | GPU Util % | VRAM (MB) | VRAM % | Reader Cost | Batch Cost | Reader Ratio | Est 40 Epoch | Est Cost | IPS/$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 128 | 4 | SUCCESS | 182.3 | 56% | 24193 | 52.5% | 0.0871s | 0.7022s | 0.12 | 15.24h | $32.61 | 85.2 |
| 128 | 8 | SUCCESS | 209.3 | 93% | 24193 | 52.5% | 0.0001s | 0.6114s | 0.00 | 13.27h | $28.39 | 97.8 |
| 128 | 12 | SUCCESS | 209.5 | 93% | 24193 | 52.5% | 0.0001s | 0.6111s | 0.00 | 13.26h | $28.38 | 97.9 |
| 128 🌟 *(Best)* | 16 | SUCCESS | 209.9 | 94% | 24193 | 52.5% | 0.0001s | 0.6098s | 0.00 | 13.23h | $28.32 | 98.1 |
| 128 | 24 | SUCCESS | 209.8 | 1% | 22771 | 49.4% | 0.0001s | 0.6102s | 0.00 | 13.24h | $28.34 | 98.0 |
| 160 | 16 | SUCCESS | 206.8 | 94% | 30149 | 65.4% | 0.0001s | 0.7738s | 0.00 | 13.43h | $28.75 | 96.6 |
| 192 | 16 | SUCCESS | 204.3 | 96% | 35901 | 77.9% | 0.0001s | 0.9398s | 0.00 | 13.60h | $29.10 | 95.5 |
| 224 🛡️ *(Max Safe)* | 16 | SUCCESS | 203.9 | 96% | 41617 | 90.3% | 0.0002s | 1.0986s | 0.00 | 13.62h | $29.16 | 95.3 |

## 2. Kết Luận & Đề Xuất Cấu Hình
- **Max Batch Chạy Được (Hardware Limit)**: `224` (ngưỡng tối đa trước khi tràn VRAM).
- **Best Batch Nên Dùng (Sweet Spot)**: `128` (đạt throughput `209.9` mẫu/s, khai thác `52.5%` VRAM).
- **Workers Tối Ưu**: `16` (giữ `reader_ratio = 0.00`, đảm bảo GPU không bị đói dữ liệu).
- **Điểm Nghẽn Chính (Bottleneck)**: `GPU Compute Saturated (Optimal)`.
- **Thời Gian Huấn Luyện Full 40 Epoch (10M samples)**: `13.23 giờ` (~`794 phút`).
- **Chi Phí Ước Tính Full 40 Epoch**: `$28.32`.
- **Hiệu Suất Kinh Tế (Throughput / $)**: `98.1 samples / $`.