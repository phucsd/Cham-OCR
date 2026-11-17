# Đánh giá Stage R4 (`WIDE-CANVAS W=480, T=60 CTC FRAMES + CORPUS & OPTICAL`)
- **Thời điểm**: `2026-11-17T05:19:23.493362+00:00`
- **Epoch tốt nhất R4**: Epoch 3 (`SHA256: b98e0c3400d520d4...`, `valid_for_release=False`)
- **Train Loop Duration**: `868.0s` (14.47 phút)
- **Pipeline / Container Overhead**: `357.89s` (5.96 phút)
- **Total Kernel Wall-clock Time**: `1225.89s` (20.43 phút)
- **CPU Full-Page Latency (`harum_palei_01`, 25 dòng)**: `6.065s` (`< 15s SLA`: `True`)
- **OOD Font Uncontaminated Check**: `{'n_samples': 150, 'r3_1_w320': {'cer_pct': 4.01, 'edits': 126, 'chars': 3140}, 'r4_w480': {'cer_pct': 4.55, 'edits': 143, 'chars': 3140}}`

## 1. Frozen-838 & Stress-82 & Probe-100 (`W=320` vs `W=480`)
| Metric | R3.1 Best (`W=320`) | R4 E1 (`W=480`) | R4 E2 (`W=480`) | R4 E3 (`W=480`) |
| :--- | :---: | :---: | :---: | :---: |
| Clean-Frozen CER | 4.98% | 5.01% | 5.01% | 5.06% |
| Character-Frozen CER | 3.67% | 3.13% | 2.80% | 2.76% |
| 14 Symbols Acc | 97.14% | 92.14% | 97.86% | 98.93% |
| Failing Symbols | ['?', '}', '_', ']', '/'] | ['*', '{', '_', '/'] | ['{', '_', '/'] | ['_', '/'] |
| Hyphen Multi-Context | 92.86% | 91.43% | 91.43% | 94.29% |
| Blank Guard | True | True | True | True |
| FullPage Matched CER | 6.74% | 7.03% | 7.08% | 7.41% |
| Stress-82 CER | 6.46% | 7.09% | 6.21% | 6.45% |
| Probe-100 CER (`W=320`) | 10.04% | 11.48% | 10.91% | 10.81% |
| Probe-100 CER (`W=480`) | N/A | 4.00% | 4.15% | 4.20% |
| Gates Passed | 4/6 | 4/6 | 4/6 | 4/6 |

## 2. Full-Page Level-5 Degradation & 18-Image Audit Suite
| Model | short:5 Doc/Matched CER | a4:5 Doc/Matched CER | audit_18 Doc/Matched CER |
| :--- | :---: | :---: | :---: |
| `v24` | 21.97% / 22.52% (63/64) | 19.58% / 19.9% (240/241) | 18.26% / 18.77% (134/134) |
| `v24_1_r3` | 16.37% / 16.75% (62/64) | 12.98% / 13.19% (239/241) | 10.9% / 11.17% (134/134) |
| `r3_1_w320` | 15.67% / 16.05% (62/64) | 12.05% / 12.24% (240/241) | 10.82% / 11.1% (134/134) |
| `r4_w480` | 16.47% / 16.89% (62/64) | 12.78% / 12.98% (240/241) | 11.32% / 11.72% (134/134) |
