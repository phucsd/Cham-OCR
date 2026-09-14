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

# Cham-OCR: A Deep-Learning OCR Pipeline for Unicode Eastern Cham, toward Historical Manuscript and Epigraphic Recognition

[![Live Demo](https://img.shields.io/badge/Live_Demo-ocr.cham.asia-C96442?style=for-the-badge&logo=google-chrome&logoColor=white)](https://ocr.cham.asia)
[![Technical Report](https://img.shields.io/badge/Technical-Report-8C533E?style=for-the-badge&logo=googlescholar&logoColor=white)](RESEARCH.md)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-F59E0B?style=for-the-badge)](https://huggingface.co/spaces/phucsd/cham-ocr-studio)
[![GitHub Repository](https://img.shields.io/badge/GitHub-phucsd%2FCham--OCR-24292e?style=for-the-badge&logo=github)](https://github.com/phucsd/Cham-OCR)
[![Contact Email](https://img.shields.io/badge/Contact-phucsd%40gmail.com-blue?style=for-the-badge&logo=gmail&logoColor=white)](mailto:phucsd@gmail.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end deep learning research platform and paleographic transcription system engineered for the digitized preservation, line segmentation, and optical character recognition of **Cham documents**, with validated performance on Unicode Eastern Cham (*Akhar Thrah*) and ongoing extensions toward Western Cham (*Cam Srak*), physical palm-leaf/kertas manuscripts, and stone epigraphy.

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
   - Automatically parses recognized grapheme clusters via `normalize_unicode` to reconstruct standard **Brahmic Logical Order** (`Base Consonant + Medials + Pre-Vowels + Dependent Vowels + Vowel Lengthener AA + Finals / Signs / Final VA`), resolving local visual permutation artifacts.
4. **150,000 Textline Synthetic Pipelines**:
   - Font map validation via `fontTools` cmap parsing, eliminating tofu/missing glyph artifacts.
   - Dynamic In-RAM augmentation applying directional motion blur on clean disk renders to prevent double-blur degradation.
   - **Dual-Generator Architecture**:
     - *Pipeline A (Baseline: `scripts/generate_data.py`)*: 150,000 textlines partitioned into 15% evaluation/locked splits and 85% training across 6 categories: clean short (25%), clean medium (20%), clean long (15%), noisy short (15%), noisy long (15%), and hard adversarial examples (10%).
     - *Pipeline B (Experimental V25: `scripts/generate_data_v25.py`)*: 150,000 textlines (140,000 train + 10,000 val) synchronized with `configs/v25_dataset_manifest.json` across 5 core pillars: canonical Cham literature (43.3%), anti-blur base (20.0%), inline bilingual (16.7%), stanza numerals & punctuation (12.0%), and minimal pairs (8.0%).
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

### 3. Model Progression: V23 vs V24 Baseline vs V25 SOTA

- **Version 25 (Production SOTA — Recommended)**: Completed full 40-epoch multi-stage distributed GPU training on Dual Tesla T4x2 accelerators (87,480 cumulative steps). Reaches **90.96% Sequence Accuracy** (exact match) and **99.28% Normalized Edit Distance (CER 0.72%)** on the standardized 10,000 frozen validation dataset (`cham_v25_val_freeze.zip`). Resolves CTC blank collapse on Double Danda (`꩝꩝`), eliminates numeral-to-consonant misclassifications (`꩔` vs `ꨤ`, `꩕` vs `ꨅ`), and sharply disambiguates diacritic minimal pairs (`ꨲ` vs `ꨶ`).
- **Version 24 (Logical Baseline)**: First logical order model achieving **88.94% validation accuracy** and 13.88% CER across the 50-test stress suite.
- **Version 23 (Visual Order Legacy)**: Maintained for historical visual rendering order comparison (84.12% validation accuracy).

---

## 📑 Academic Citation

If you utilize Cham-OCR, our synthetic dataset pipeline, or the OCR Studio in your research, please cite:

```bibtex
@software{cham_ocr_pipeline,
  author = {Nguyen, Phuc H.},
  title = {Cham-OCR: A Deep-Learning OCR Pipeline for Unicode Eastern Cham, toward Historical Manuscript and Epigraphic Recognition},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/phucsd/Cham-OCR},
  note = {Production web service: https://ocr.cham.asia}
}
```

---

## 📄 License & Attribution

- **Software**: Released under the [MIT License](LICENSE). Copyright © 2026 Phuc H. Nguyen.
- **Fonts & Linguistic Corpora**: Details on digital typefaces (bundled *Noto Sans Cham*, external reference typefaces), classical literary text sources, and synthetic dataset licensing are documented in **[DATA_PROVENANCE.md](DATA_PROVENANCE.md)**.
- **Audit & Verification**: See **[ACADEMIC_AUDIT.md](ACADEMIC_AUDIT.md)** for our ground-truth verification log and academic reconciliation ledger.

---

## 📬 Contact & Inquiries

For academic collaborations, questions, bug reports, or contributions to digitized historical Cham archives, please reach out to:
- **Author & Maintainer**: Phuc H. Nguyen
- **Email**: [phucsd@gmail.com](mailto:phucsd@gmail.com)
- **GitHub**: [@phucsd](https://github.com/phucsd)
- **Live Web Service**: [https://ocr.cham.asia](https://ocr.cham.asia)

