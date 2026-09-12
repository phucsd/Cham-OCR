---
title: Cham OCR Studio — Paleographic Transcription Workbench
emoji: 📜
colorFrom: amber
colorTo: stone
sdk: docker
app_port: 7860
pinned: false
---

**English** | [Tiếng Việt](README_VI.md)

# Cham OCR Monorepo: Deep Learning Pipeline & Paleographic Transcription Workbench

[![Live Demo](https://img.shields.io/badge/Live_Demo-ocr.cham.asia-C96442?style=for-the-badge&logo=google-chrome&logoColor=white)](https://ocr.cham.asia)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-F59E0B?style=for-the-badge)](https://huggingface.co/spaces/phucsd/cham-ocr-studio)
[![GitHub Repository](https://img.shields.io/badge/GitHub-phucsd1%2FCham--OCR-24292e?style=for-the-badge&logo=github)](https://github.com/phucsd1/Cham-OCR)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end deep learning research platform and paleographic transcription system specifically engineered for the digitized preservation, line segmentation, and optical character recognition of historical **Cham manuscripts** (encompassing traditional *Akhar Thrah* of Eastern Cham and *Cam Srak* of Western Cham).

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
   - Automatically parses recognized grapheme clusters via `normalize_unicode` to reconstruct standard **Brahmic Logical Order** (`Base Consonant + Medials + Pre-Ra + Dependent Vowels + Final Consonants / Punctuation`), resolving local visual permutation artifacts.
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
├── kernel-metadata.json                # Kaggle Kernel Metadata configuration
└── requirements.txt                    # Unified dependency specification
```

---

## 🚀 Quickstart: Local Studio Development

### 1. Environment Setup
Clone the repository and install all required dependencies:
```bash
git clone https://github.com/phucsd1/Cham-OCR.git
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

## 📊 Empirical Benchmarks

Evaluated across a stratified test suite of **490 benchmark scenarios** (comprising 200 deformed textline crops, 200 multi-paragraph A4 folios, and 90 mobile field photography captures):

| Metric | Version 22 (Baseline) | Version 23 (Golden Visual) | Version 24 (Noise Resilient) | Version 25 (Unified SOTA) |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Textline CER** | 7.82% | 5.14% | 3.62% | **3.18%** |
| **Motion Blur CER (Handshake)** | 42.10% | 38.74% | 17.80% | **14.20%** |
| **Inline Multilingual CER** | 34.50% | 27.84% | 13.20% | **10.85%** |
| **Verse Numeral Accuracy** | 82.4% | 87.1% | 96.8% | **97.4%** |
| **Paragraph Flow F1 Score** | 64.2% | 70.7% | 89.5% | **91.4%** |

---

## 📑 Academic Citation

If you utilize Cham-OCR, our synthetic dataset pipeline, or the OCR Studio in your research, please cite:

```bibtex
@software{cham_ocr_studio,
  author = {Nguyen, Phuc and Contributors},
  title = {Cham OCR Studio: Deep Learning Pipeline and Paleographic Transcription Workbench for Historical Cham Manuscripts},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/phucsd1/Cham-OCR},
  note = {Live web service: https://ocr.cham.asia}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
