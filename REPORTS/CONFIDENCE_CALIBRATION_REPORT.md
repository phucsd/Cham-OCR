# Báo cáo Hiệu chỉnh Độ tự tin Confidence Calibration (Temperature Scaling)
- **Thời điểm**: `2026-11-16T08:59:50.262036+00:00`
- **Tập fit (in-sample)**: `held_out_diagnostic_probe_100.json` (100 dòng, tách biệt hoàn toàn khỏi RealScan-Gold v1)
- **Tập đánh giá Out-of-Sample**: `RealScan-Gold v1 / harum_palei_01` (25 dòng chụp thực tế)

| Model | T* (Fitted) | T Deployed | OOS ECE (T=1 -> T*) | OOS Brier (T=1 -> T*) | OOS NLL (T=1 -> T*) | Mean Conf vs Empirical Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `v24_1_r2_1_6` | `0.9345` | `0.9345` | 0.0028 → 0.0066 | 0.0023 → 0.0023 | 0.2141 → 0.2147 | 0.9403 → 0.9497 (acc=0.9431) |
| `v24_1_r3` | `0.9280` | `0.9280` | 0.0072 → 0.0169 | 0.0028 → 0.0029 | 0.2400 → 0.2428 | 0.9400 → 0.9502 (acc=0.9368) |
