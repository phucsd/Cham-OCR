# Đánh giá Stage R4.2 (`W=480` Ink-Tight Geometry & 6/6 Gate-Closure)
- **Thời điểm**: `2026-11-17T10:45:03.981679+00:00`
- **Epoch tốt nhất**: Epoch `2` (`SHA256: 37ccf1f8a570fdba...`, **`valid_for_release = False`**, `4/6` Release Gates)
- **Train Loop / Overhead / Total Wall-clock**: `333.0s` / `1144.53s` / `1477.53s` (24.63 phút)
- **Nâng cấp trực tiếp vào `v24_1_r4`**: `False`

| Metric | R4.0 E3 | R4.1 E1 | R4.2 E1 (`W=480`) | R4.2 E2 (`W=480`) |
| :--- | :---: | :---: | :---: | :---: |
| Clean-Frozen CER | 5.06% | 5.42% | 5.54% | 5.42% |
| Character-Frozen CER | 2.76% | 2.70% | 2.80% | 2.84% |
| 14 Mandatory Symbols Acc | 98.93% | 98.93% | 98.57% | 98.57% |
| Failing Symbols | `['_', '/']` | `['*', '_', '/']` | `['*', '_', '/']` | `['*', '_', '/']` |
| Hyphen Multi-Context | 94.29% | 95.71% | 90.00% | 91.43% |
| Blank Guard | True | True | True | True |
| FullPage Matched CER | 7.41% | 7.53% | 7.40% | 7.30% |
| `harum_palei_01` Crop CER | 2.68% | 2.80% | **2.80%** | **2.55%** |
| **Release Gates Passed** | `4/6` | `5/6` | **`4/6`** | **`4/6`** |
