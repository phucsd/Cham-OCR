# Báo Cáo Benchmark V2 Chuẩn Xác Cao Trên Kaggle Dual GPU Tesla T4x2 (Cham-OCR V24)

| Thí nghiệm | BS/GPU | Tổng BS | Workers | Chế độ | Median Dual IPS (mẫu/s) | Tăng tốc (%) | Batch Cost (s) | GPU Compute (s) | Peak VRAM | Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| T1 (Baseline FP32 BS64) | 64 | 128 | 2 | FP32 | **96.6** | **+0.0%** | 1.3245s | 1.3185s | 11891 MB (77.4%) | SUCCESS |
| T2 (Tensor Cores BS64) | 64 | 128 | 2 | AMP-O1 | **119.1** | **+23.3%** | 1.0734s | 1.0669s | 11494 MB (74.8%) | SUCCESS |
| T3 (AMP BS48) | 48 | 96 | 2 | AMP-O1 | **116.6** | **+20.7%** | 0.8140s | 0.8086s | 8681 MB (56.5%) | SUCCESS |
| T4 (AMP BS72) | 72 | 144 | 2 | AMP-O1 | **120.4** | **+24.6%** | 1.1214s | 1.1153s | 12878 MB (83.8%) | SUCCESS |
| T5 (AMP BS80) | 80 | 160 | 2 | AMP-O1 | **116.6** | **+20.7%** | 1.3380s | 1.3316s | 14300 MB (93.1%) | SUCCESS |
| T6 (AMP BS96) | 96 | 192 | 2 | AMP-O1 | **N/A** | **N/A** | N/A | N/A | N/A | OOM |
| T7 (NCCL Tuned BS64) | 64 | 128 | 2 | AMP-O1+NCCL | **119.2** | **+23.4%** | 1.0733s | 1.0672s | 11485 MB (74.8%) | SUCCESS |
| T8 (Workers 3 BS64) | 64 | 128 | 3 | AMP-O1 | **119.0** | **+23.2%** | 1.0742s | 1.0685s | 11485 MB (74.8%) | SUCCESS |

## 🏆 Kết luận Cấu hình Tối ưu Tuyệt đối trên Dual GPU T4x2
- **Tên cấu hình**: `T4 (AMP BS72)`
- **Batch Size mỗi card**: `72` (Tổng Global Batch Size: `144`)
- **Số Workers**: `2`
- **Chế độ tính toán**: `AMP O1`
- **Throughput Steady-State Dual IPS**: **`120.4 mẫu/s`** (Tăng **`+24.6%`**)
- **Thời gian hoàn thành 40 epochs (10M mẫu)**: **`23.07 giờ`** (Tiết kiệm **`5.69 giờ`**)
- **Mức tiêu thụ VRAM đỉnh**: **`12878 MB`** (**`83.8%`** VRAM)
- **Kiểm định Loss**: `156.72` -> `146.15` (`HEALTHY`)
