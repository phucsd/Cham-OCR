# Đánh giá Stage R4.1 (`W=480` Surgical Gate-Closure Micro-Polish)
- **Thời điểm**: `2026-11-17T09:27:45.713733+00:00`
- **Epoch tốt nhất**: Epoch `1` (`SHA256: 09bdbedc275f26c3...`, **`valid_for_release = False`**, `5/6` Release Gates)
- **Train Loop / Overhead / Total Wall-clock**: `689.0s` / `751.9s` / `1440.9s` (24.02 phút)
- **Nâng cấp trực tiếp vào `v24_1_r4`**: `False`

| Metric | R4.0 E3 (`W=480`) | R4.1 E1 (`W=480`) | R4.1 E2 (`W=480`) | R4.1 E3 (`W=480`) |
| :--- | :---: | :---: | :---: | :---: |
| Clean-Frozen CER | 5.06% | 5.42% | 5.58% | 5.43% |
| Character-Frozen CER | 2.76% | 2.70% | 2.53% | 2.62% |
| 14 Mandatory Symbols Acc | 98.93% | 98.93% | 97.14% | 98.93% |
| Failing Symbols | `['_', '/']` | `['*', '_', '/']` | `['*', '_', '/']` | `['*', '_', '/']` |
| Hyphen Multi-Context | 94.29% | 95.71% | 94.29% | 94.29% |
| Blank Guard | True | True | True | True |
| FullPage Matched CER | 7.41% | 7.53% | 7.47% | 7.48% |
| Stress-82 CER | 6.45% | 6.99% | 6.97% | 7.06% |
| Probe-100 (`W=480`) | 4.20% | 4.20% | 4.41% | 4.56% |
| **Release Gates Passed** | `4/6` | **`5/6`** | **`4/6`** | **`4/6`** |
