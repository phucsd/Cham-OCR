# Đánh giá Stage R3.1 (`GATE-CLOSURE + SESQUISYLLABIC + EROSION`)
- **Thời điểm**: `2026-11-17T03:30:28.556090+00:00`
- **Epoch tốt nhất R3.1**: Epoch 3 (`SHA256: ae8631e38ba7057c...`, `valid_for_release=False`)
- **Train Loop Duration**: `599.0s` (9.98 phút)
- **Pipeline / Container Overhead**: `796.2s` (13.27 phút)
- **Total Kernel Wall-clock Time**: `1395.2s` (23.25 phút)

## 1. Frozen-838 & Stress-82 & Probe-100
| Metric | R2.1.6 E2 | R3 E3 | R3.1 E1 | R3.1 E2 | R3.1 E3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Clean-Frozen CER | 4.82% | 5.19% | 5.08% | 5.32% | 4.98% |
| Character-Frozen CER | 4.29% | 4.27% | 3.65% | 3.56% | 3.67% |
| Character Insertions | 47 | 50 | 34 | 31 | 34 |
| 14 Symbols Acc | 98.21% | 96.79% | 97.14% | 96.79% | 97.14% |
| Failing Symbols | N/A | ['!', '}', '_', '/'] | ['?', '}', '_', '/'] | ['?', '}', '_', ']', '/'] | ['?', '}', '_', ']', '/'] |
| Hyphen Multi-Context | 91.43% | 88.57% | 91.43% | 91.43% | 92.86% |
| Blank Guard | True | True | True | True | True |
| FullPage Matched CER | 6.21% | 6.53% | 6.77% | 6.98% | 6.74% |
| Stress-82 CER | 16.89% | 11.19% | 7.41% | 7.07% | 6.46% |
| Stress-Base-8 CER | 5.13% | 4.49% | 1.92% | 1.28% | 0.64% |
| Probe-100 CER | 9.78% | 10.45% | 9.84% | 9.94% | 10.04% |
| Gates Passed | 4/6 | 4/6 | 4/6 | 4/6 | 4/6 |

## 2. Full-Page Level-5 Degradation & 18-Image Audit Suite
| Model | short:5 Doc/Matched CER | a4:5 Doc/Matched CER | audit_18 Doc/Matched CER |
| :--- | :---: | :---: | :---: |
| `v24` | 21.97% / 22.52% (63/64) | 19.58% / 19.9% (240/241) | 18.26% / 18.77% (134/134) |
| `v24_1_r2_1_6` | 17.27% / 17.77% (62/64) | 13.68% / 13.92% (239/241) | 11.1% / 11.38% (134/134) |
| `v24_1_r3` | 16.37% / 16.75% (62/64) | 12.98% / 13.19% (239/241) | 10.9% / 11.17% (134/134) |
| `r3_1_best` | 15.67% / 16.05% (62/64) | 12.05% / 12.24% (240/241) | 10.82% / 11.1% (134/134) |
