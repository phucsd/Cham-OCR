# Đánh giá Stage R2.1.7 (`SURGICAL-GATE-CLOSURE`)
- **Thời điểm**: `2026-11-16T07:49:28.221154+00:00`
- **Epoch tốt nhất**: Epoch 3 (`SHA256: 070e44a383681831...`, `valid_for_release=False`)
- **Train Loop Duration**: `569.0s` (9.48 phút)
- **Pipeline / Container Overhead**: `1171.99s` (19.53 phút)
- **Total Kernel Wall-clock Time**: `1740.99s` (29.02 phút)

| Metric | R2.1.6 E2 | R2.1.7 E1 | R2.1.7 E2 | R2.1.7 E3 |
| :--- | :---: | :---: | :---: | :---: |
| Clean-Frozen CER | 4.82% | 5.14% | 4.58% | 4.38% |
| Character-Frozen CER | 4.29% | 4.90% | 4.72% | 4.72% |
| 14 Symbols Acc | 98.21% | 96.07% | 97.50% | 97.50% |
| Hyphen Multi-Context | 91.43% | 87.14% | 90.00% | 90.00% |
| Blank Guard | True | True | True | True |
| FullPage Matched CER | 6.21% | 6.57% | 6.03% | 5.47% |
| Stress-82 CER | 16.89% | 18.21% | 17.68% | 17.27% |
| Stress-Base-8 CER | 5.13% | 7.05% | 5.77% | 3.85% |
| Probe-100 CER | 9.78% | 10.91% | 10.25% | 10.14% |
| Composite Score | 0.6535 | 0.6307 | 0.6461 | 0.6463 |
| Gates Passed | 4/6 | 3/6 | 3/6 | 3/6 |
