# Báo Cáo Đối Soánh Hiệu Năng & Chi Phí Huấn Luyện Cham-OCR Đa GPU
## (PP-OCRv4 SVTR_LCNet - Pipeline Huấn Luyện Thật)

### 1. Bảng Tổng Hợp So Sánh Các Cấu Hình Tối Ưu (Best & Max Batch)
| GPU | Phân Loại | Batch | Workers | IPS (mẫu/s) | GPU% | VRAM% | Reader Time | Batch Time | $/h | Est 40 Epoch | Est Cost | IPS/$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A100** | 🌟 **Best** | 192 | 16 | **255.9** | 95% | 88.5% | 0.0002s | 0.7503s | $2.19 | 10.8h | **$23.77** | **116.8** |
| **NVIDIA L4 (24GB)** | 🌟 **Best** | 96 | 16 | **74.3** | 98% | 79.2% | 0.0001s | 1.2927s | $0.79 | 37.4h | **$29.53** | **94.0** |
| **L40S** | 🌟 **Best** | 128 | 16 | **209.9** | 94% | 52.5% | 0.0001s | 0.6098s | $2.14 | 13.2h | **$28.32** | **98.1** |
| L40S | 🛡️ Max Safe | 224 | 16 | 203.9 | 96% | 90.3% | 0.0002s | 1.0986s | $2.14 | 13.6h | $29.16 | 95.3 |

### 2. Kết Luận Chi Tiết Cho Từng Machine

#### 🔹 Machine: **A100** ($2.19/h)
- **Max Batch Chạy Được**: `192` (ngưỡng giới hạn phần cứng trước khi OOM).
- **Best Batch Nên Dùng**: `192` (điểm ngọt cho throughput cao nhất trước khi bão hòa).
- **Num Workers Tối Ưu**: `16` worker.
- **Điểm Nghẽn Chính (Bottleneck)**: `GPU Compute Saturated (Optimal)`.
- **Thời Gian Huấn Luyện Full 40 Epoch (10,000,000 mẫu)**: `10.85 giờ` (~`651 phút`).
- **Chi Phí Ước Tính Full 40 Epoch**: `$23.77`.
- **Hiệu Năng / Chi Phí (P/P)**: `116.8 mẫu / $`.

#### 🔹 Machine: **NVIDIA L4 (24GB)** ($0.79/h)
- **Max Batch Chạy Được**: `96` (ngưỡng giới hạn phần cứng trước khi OOM).
- **Best Batch Nên Dùng**: `96` (điểm ngọt cho throughput cao nhất trước khi bão hòa).
- **Num Workers Tối Ưu**: `16` worker.
- **Điểm Nghẽn Chính (Bottleneck)**: `GPU Compute Bound & VRAM Limited`.
- **Thời Gian Huấn Luyện Full 40 Epoch (10,000,000 mẫu)**: `37.38 giờ` (~`2243 phút`).
- **Chi Phí Ước Tính Full 40 Epoch**: `$29.53`.
- **Hiệu Năng / Chi Phí (P/P)**: `94.0 mẫu / $`.

#### 🔹 Machine: **L40S** ($2.14/h)
- **Max Batch Chạy Được**: `224` (ngưỡng giới hạn phần cứng trước khi OOM).
- **Best Batch Nên Dùng**: `128` (điểm ngọt cho throughput cao nhất trước khi bão hòa).
- **Num Workers Tối Ưu**: `16` worker.
- **Điểm Nghẽn Chính (Bottleneck)**: `GPU Compute Saturated (Optimal)`.
- **Thời Gian Huấn Luyện Full 40 Epoch (10,000,000 mẫu)**: `13.23 giờ` (~`794 phút`).
- **Chi Phí Ước Tính Full 40 Epoch**: `$28.32`.
- **Hiệu Năng / Chi Phí (P/P)**: `98.1 mẫu / $`.

### 3. Xếp Hạng Hiệu Quả Đầu Tư (Performance / Price Ranking)
| Hạng | GPU | Best Batch | Tốc Độ (IPS) | Thời Gian Full 40 Epoch | Tổng Chi Phí ($) | Hiệu Suất Kinh Tế (IPS/$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **A100** | 192 | 255.9 mẫu/s | 10.8h | **$23.77** | **116.8** |
| 🥈 | **L40S** | 128 | 209.9 mẫu/s | 13.2h | **$28.32** | **98.1** |
| 🥉 | **NVIDIA L4 (24GB)** | 96 | 74.3 mẫu/s | 37.4h | **$29.53** | **94.0** |

> 🏆 **LỰA CHỌN TỐI ƯU NHẤT (P/P)**: **A100** là cỗ máy kinh tế nhất với **116.8 mẫu / $**, tổng chi phí train full 40 epoch chỉ **$23.77** trong **10.8 giờ**.