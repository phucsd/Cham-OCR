---
title: Cham OCR Studio — Paleographic Transcription Workbench
emoji: 📜
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

**English** | [Tiếng Việt](README_VI.md)

# Cham OCR Monorepo: Deep Learning Pipeline & Paleographic Transcription Workbench

[![Live Demo](https://img.shields.io/badge/Live_Demo-ocr.cham.asia-C96442?style=for-the-badge&logo=google-chrome&logoColor=white)](https://ocr.cham.asia)
[![Technical Report](https://img.shields.io/badge/Technical-Report-8C533E?style=for-the-badge&logo=googlescholar&logoColor=white)](RESEARCH.md)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-F59E0B?style=for-the-badge)](https://huggingface.co/spaces/phucsd/cham-ocr-studio)
[![GitHub Repository](https://img.shields.io/badge/GitHub-phucsd%2FCham--OCR-24292e?style=for-the-badge&logo=github)](https://github.com/phucsd/Cham-OCR)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end deep learning research platform and paleographic transcription system specifically engineered for the digitized preservation, line segmentation, and optical character recognition of historical **Cham manuscripts** (encompassing traditional *Akhar Thrah* of Eastern Cham and *Cam Srak* of Western Cham).

> 📄 **Technical Report & Academic Documentation**: Read our comprehensive technical report at **[RESEARCH.md](RESEARCH.md)** or online at **[https://ocr.cham.asia/research](https://ocr.cham.asia/research)** detailing Brahmic paleography, DBNet line segmentation, SVTR recognition, 150K synthetic data generation, and empirical benchmarks.
> 
> 🌟 **Live Interactive Web Demo**: Experience the full-featured transcription and diagnostic studio online at **[https://ocr.cham.asia](https://ocr.cham.asia)** (hosted via [Hugging Face Spaces](https://huggingface.co/spaces/phucsd/cham-ocr-studio)).

---

## 📑 Core Technical Innovations

1. **Dual-Stage Indic Line Segmentation**:
   - Primary: Fine-tuned **PaddleOCR DBNet** (Differentiable Binarization Network with an adaptive unclip ratio of $1.8$) predicting boundary polygons that tightly conform to undulating text baselines.
   - Secondary Fallback: **Indic Valley-Cut Heuristic Engine** combining Connected Component Analysis (CCA) with horizontal projection profiles.
   - **Boundary Stroke Protection**: Employs strict exclusion masks (`difference_update`) to prevent erasing touching glyph strokes between adjacent lines, enforced with a $0.40 \times \text{median line height}$ vertical padding safeguard and a *Legacy-First / No-Change Gate* on clean documents.
2. **Neural Recognition Network (PP-OCRv4 SVTR-LCNet)**:
   - Backbone featuring Depthwise Separable Convolutions and Squeeze-and-Excitation (SE) channel attention modules.
   - Receptive tensor dimensions extended to `[3, 48, 480]` to prevent character compression across lengthy multilingual phrases.
   - Dual-head multi-task learning with **Connectionist Temporal Classification (CTC)** sequence alignment and **NRTR Multi-Head Cross-Attention** auxiliary decoding.
3. **Canonical Unicode & Logical Order Normalization**:
   - Automatically parses recognized grapheme clusters via `normalize_unicode` to reconstruct standard **Brahmic Logical Order** (`Base Consonant + Medials + Pre-Vowels + Dependent Vowels + Finals / Signs`), resolving local visual permutation artifacts.
4. **150,000 Textline Synthetic Pipeline**:
   - Font map validation via `fontTools` cmap parsing, eliminating tofu/missing glyph artifacts.
   - Dynamic In-RAM augmentation applying directional motion blur (kernels $7\times 7$ to $13\times 13$, $\theta \in [0^\circ, 180^\circ]$ at $35\%$ probability) on clean disk renders to prevent double-blur degradation.
   - Four stratified training buckets: Classical Literature ($50\%$), Multilingual Code-Switching ($18\%$), Classical Verse Numerals 1–99 ($18\%$), and Adversarial Minimal Pairs ($14\%$).
5. **Distributed Multi-GPU Cloud Training**:
   - Adaptive data-parallel execution (`paddle.distributed.launch --gpus '0,1'`) on Kaggle Nvidia Tesla T4x2 and Lightning Cloud A100 SXM4.
   - 3-Stage Checkpoint & Resume protocol ensuring uninterrupted progress across Kaggle's 12-hour session timeout limits.
   - Character-Mapped Weight Surgery (`surgery_v25_weights.py`) for seamless transfer learning across expanding lexicons.

---

## 📁 Monorepo Architecture

```
Cham-OCR/
├── ocr-studio/                         # 1. Interactive Transcription & Review Studio
│   ├── app.py                          # Backend HTTP server & multi-crop orchestrator
│   ├── index.html                      # Studio UI (Academic English, Claude Warm Light Theme)
│   ├── research.html                   # Interactive Academic Technical Report & Whitepaper
│   ├── start_studio.py                 # Quick-launch utility
│   ├── data/                           # Dictionaries & exported inference models (v24, v23, v22)
│   ├── PaddleOCR/                      # Standalone PaddleOCR inference runtime
│   └── README_STUDIO.md                # Comprehensive Studio documentation
│
├── ocr-training/                       # 2. Model Training & Cloud Pipeline (Kaggle & Lightning)
│   ├── paddleocr_cham_finetune.ipynb  # Primary training notebook for Kaggle GPU T4x2
│   ├── configs/                        # YAML training configurations (v24, v25)
│   ├── scripts/                        # Synthetic data generators, weight surgery & watchdog tools
│   ├── data/                           # Training fonts & normalized Cham literary corpora
│   ├── tests/                          # Automated benchmark & validation test suites
│   └── README_TRAINING.md              # Detailed training & fine-tuning documentation
│
├── Dockerfile                          # Multi-stage Docker container for Hugging Face Spaces
├── .dockerignore                       # Build exclusion filter
├── .gitignore                          # Git tracking filter
├── DESIGN.md                           # Formal Design System tokens (linted with designmd)
├── LICENSE                             # MIT License (Software code and scripts)
├── DATA_PROVENANCE.md                  # Linguistic corpora, font licensing & dataset provenance
├── RESEARCH.md                         # Full Academic Technical Report & Paleographic Study
├── ACADEMIC_AUDIT.md                   # Internal Documentation Audit & Reconciliation Log
├── kernel-metadata.json                # Kaggle Kernel Metadata configuration
└── requirements.txt                    # Unified dependency specification
```

---

## 🚀 Quickstart: Local Studio Development

### 1. Environment Setup
Clone the repository and install all required dependencies:
```bash
git clone https://github.com/phucsd/Cham-OCR.git
cd Cham-OCR
pip install -r requirements.txt
```
*(Note: Python 3.10+ is recommended. Runtime compatibility monkeypatching for NumPy 2.x is automatically applied).*

### 2. Launching Cham OCR Studio
Execute the backend server:
```bash
python ocr-studio/app.py
```
Or via the quick-launch helper:
```bash
python ocr-studio/start_studio.py
```
The application will automatically bind to an available port (`7860`, `8080`, etc.) and launch:
```
======================================================================
🚀 Cham OCR Diagnostic Studio is running at: http://localhost:7860
📁 Corrections will be saved to: ocr-studio/data/ocr_corrections.txt
======================================================================
```
Open your browser and navigate to `http://localhost:7860`.

---

## 🏋️ Model Training & Cloud Pipelines

For detailed instructions on preparing synthetic data, font validation, and fine-tuning on Kaggle GPU T4x2 or Lightning Cloud A100, refer to **[ocr-training/README_TRAINING.md](ocr-training/README_TRAINING.md)**.

### Kaggle Multi-GPU Distributed Command:
```bash
python3 -m paddle.distributed.launch --gpus '0,1' tools/train.py -c configs/rec_cham_v25.yml
```

---

## 📊 Empirical Benchmarks & Quantitative Evaluations

All reported metrics are strictly grounded in empirical evaluation logs located in `ocr-studio/data/` and `ocr-benchmark/results/`.

### 1. Controlled 50-Test Stratified Benchmark (Model V23 vs V24 Validated Baseline)

| Evaluation Category | Test Count | V23 CER (%) | V24 CER (%) | CER Gain | V23 Pass (%) | V24 Pass (%) | V24 Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cat 1: Clean Baseline** | 5 | 14.84% | **6.20%** | -8.64% | 20.0% | **80.0%** | 0.958 |
| **Cat 2: Single-Digit Stanzas** | 5 | 18.42% | **11.71%** | -6.71% | 0.0% | **40.0%** | 0.944 |
| **Cat 3: Two-Digit Stanzas** | 5 | 17.59% | **7.65%** | -9.94% | 20.0% | **60.0%** | 0.956 |
| **Cat 4: Low Height / Low-Res** | 5 | 27.65% | **17.96%** | -9.69% | 0.0% | **20.0%** | 0.848 |
| **Cat 5: Blur Degradation** | 5 | 41.78% | **35.56%** | -6.22% | 0.0% | 0.0% | 0.839 |
| **Cat 6: Noise & Grain** | 5 | 50.59% | **8.82%** | -41.77% | 0.0% | **80.0%** | 0.944 |
| **Cat 7: Paper Texture & Contrast** | 5 | 33.33% | **21.33%** | -12.00% | 0.0% | 0.0% | 0.921 |
| **Cat 8: Tilt & Perspective** | 5 | 22.14% | **0.71%** | -21.43% | 0.0% | **100.0%** | 0.982 |
| **Cat 9: Stroke Degradation** | 5 | 26.67% | **6.67%** | -20.00% | 0.0% | **80.0%** | 0.928 |
| **Cat 10: Diacritics & Punctuation** | 5 | 28.72% | **22.19%** | -6.53% | 0.0% | **40.0%** | 0.913 |
| **Overall Benchmark Average** | **50** | **28.17%** | **13.88%** | **-14.29%** | **4.0%** | **50.0%** | **0.923** |

### 2. Controlled 200-Page Synthetic Document Stress-Test (903 Textlines)

Evaluated end-to-end (Cham-DBNet + Cham-SVTR V24) on 200 synthetic document images across 5 calibrated difficulty tiers generated by `ocr-benchmark/scripts/generate_benchmark_200.py`:

| Difficulty Tier | Stress & Degradation Profile | Samples | GT Lines | Detection Rate | Mean CER | Mean WER |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Level 1: Standard** | Line gap 25–35px, rectilinear alignment | 40 | 201 | **100.0%** | 14.43% | 45.50% |
| **Level 2: Aged Paper** | Yellowed paper texture, mild grain, skew $\le 2^\circ$ | 40 | 189 | **100.0%** | 14.44% | 45.86% |
| **Level 3: Narrow Gap** | Tight gap 8–14px, subtle waviness 1–2px | 40 | 170 | **100.0%** | 12.77% | 43.52% |
| **Level 4: Wavy Sinusoid** | Sine wave amp 3–5.5px, perspective tilt 0.03–0.05 | 40 | 177 | **99.44%** | 14.53% | 45.64% |
| **Level 5: Extreme Overlap** | Gap 2–6px < wave amp 5.5–8px (Breaking Point) | 40 | 166 | **61.45%** | 60.02% | 77.63% |
| **Overall Suite Summary** | **Total across all 5 tiers** | **200** | **903** | **92.80%** | **23.24%** | **51.63%** |

### 3. Model Classification: V24 Validated Baseline vs V25 Experimental Checkpoint

Logged in `ocr-studio/data/benchmark_v24_vs_v25_results.json`:
- **Version 24 (Validated Baseline)**: Demonstrates proven convergence with **16.81% CER** and **44.0% pass rate**, serving as our default production model.
- **Version 25 (Experimental Checkpoint)**: Ingests lexicon expansions and neural weight surgery (`surgery_v25_weights.py`). Current checkpoints yield **51.73% CER** and **10.0% pass rate** due to early alignment shifts, remaining under active training and calibration.

---

## 📑 Academic Citation

If you utilize Cham-OCR, our synthetic dataset pipeline, or the OCR Studio in your research, please cite:

```bibtex
@software{cham_ocr_studio,
  author = {Nguyen, Phuc},
  title = {Cham OCR Studio: Deep Learning Pipeline and Paleographic Transcription Workbench for Historical Cham Manuscripts},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/phucsd/Cham-OCR},
  note = {Live web service: https://ocr.cham.asia}
}
```

---

## 📄 License & Attribution

- **Software**: Released under the [MIT License](LICENSE). Copyright © 2026 Phuc H. Nguyen.
- **Fonts & Linguistic Corpora**: Details on third-party digital typefaces (*Noto Sans Cham*, *EFEO Cham*), classical literary text sources, and synthetic dataset licensing are documented in **[DATA_PROVENANCE.md](DATA_PROVENANCE.md)**.
- **Audit & Verification**: See **[ACADEMIC_AUDIT.md](ACADEMIC_AUDIT.md)** for our ground-truth verification log and academic reconciliation ledger.
