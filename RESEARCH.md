# Cham-OCR: A Deep-Learning OCR Pipeline for Unicode Eastern Cham, toward Historical Manuscript and Epigraphic Recognition

> **Technical Report / Research Preprint — Not Peer Reviewed**  
> **Author**: Phuc H. Nguyen  
> **Email / Contact**: [phucsd@gmail.com](mailto:phucsd@gmail.com)  
> **Affiliation**: Independent Researcher and Software Developer, Vietnam — Cham-OCR Project  
> **Repository**: [https://github.com/phucsd/Cham-OCR](https://github.com/phucsd/Cham-OCR)  
> **Production Web Service**: [https://ocr.cham.asia](https://ocr.cham.asia)  
> **Interactive Space**: [https://huggingface.co/spaces/phucsd/cham-ocr-studio](https://huggingface.co/spaces/phucsd/cham-ocr-studio)

[![Live Demo](https://img.shields.io/badge/Live_Demo-ocr.cham.asia-C96442?style=for-the-badge&logo=google-chrome&logoColor=white)](https://ocr.cham.asia)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-F59E0B?style=for-the-badge)](https://huggingface.co/spaces/phucsd/cham-ocr-studio)
[![GitHub Repository](https://img.shields.io/badge/GitHub-phucsd%2FCham--OCR-24292e?style=for-the-badge&logo=github)](https://github.com/phucsd/Cham-OCR)
[![Contact Email](https://img.shields.io/badge/Contact-phucsd%40gmail.com-blue?style=for-the-badge&logo=gmail&logoColor=white)](mailto:phucsd@gmail.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

---

## Abstract

Historical manuscripts of the Cham civilization—primarily preserved in Eastern Cham (*Akhar Thrah*) alongside Western Cham (*Cam Srak*) traditions—represent a fragile, millennia-old Brahmic epigraphic and literary heritage in Southeast Asia. While this project validates an end-to-end deep learning OCR pipeline specifically on Unicode Eastern Cham (*Akhar Thrah*), Western Cham (*Cam Srak*), real physical manuscripts, and epigraphic stone carvings represent active and ongoing future extensions. Automatic transcription via Optical Character Recognition (OCR) has historically faced steep barriers due to complex Brahmic abugida orthography, multi-directional dependent vowel diacritics, predominantly continuous script (*scriptio continua*) in classical literary manuscripts, physical carrier degradation, and tight interlinear diacritic collisions.

In this technical report, we describe the design, implementation, and empirical evaluation of an end-to-end deep learning pipeline and diagnostic transcription workbench engineered for Cham documents:
1. **Hybrid Text Detection & Line Segmentation**: Combines a fine-tuned **Mobile DBNet (PPLCNetV3)** with an **Indic Valley-Cut Heuristic Engine** that preserves interlinear boundary strokes via mathematical set-difference masking (`difference_update`) and a $0.40 \times \text{median line height}$ vertical safety cushion.
2. **Neural Textline Recognition**: Employs **PP-OCRv4 SVTR-LCNet** with expanded input dimensions $[3, 48, 480]$, joint Connectionist Temporal Classification (CTC) and sequence attention loss, and post-OCR deterministic Unicode cluster normalization (`normalize_unicode`).
3. **High-Fidelity Synthetic Dataset Pipeline**: Synthesizes 150,000 textlines featuring dynamic in-RAM motion blur, TrueType `cmap` tofu filtering, and hard-example adversarial mining for verse numerals ($1$–$99$) and confusing diacritic minimal pairs.

Evaluations across a **controlled 200-page synthetic document stress-test** (903 textlines across 5 difficulty levels) and a **50-test stratified failure-mode benchmark** demonstrate robust line detection with a 100.0% line detection rate across standard, aged paper, and narrow-gap folios (Levels 1–3) and graceful degradation up to Level 4 (99.44% detection), before encountering an empirical breaking point at Level 5 (61.45% detection) where interlinear wave amplitude exceeds the interlinear gap. For textline recognition, our state-of-the-art model (**V25**) achieves **90.96% Sequence Accuracy** and **99.28% Normalized Edit Distance (CER 0.72%)** across 10,000 frozen validation samples after 40 epochs of multi-stage training on Dual Tesla T4x2 GPUs, significantly outperforming the V24 logical order baseline (88.94% accuracy) and V23 visual order legacy model. V25 successfully resolves Connectionist Temporal Classification (CTC) blank collapse on boundary punctuation (`꩝꩝` vs `꩝`), eliminates numeral-to-character misclassifications (`꩔` vs `ꨤ`, `꩕` vs `ꨅ`), and robustly disambiguates subtle diacritic minimal pairs (`ꨲ` vs `ꨶ`). Multi-GPU distributed scaling benchmarks across Kaggle Tesla T4x2, Cloud A100, L40S, and L4 characterize hardware efficiency for non-profit cultural preservation. Finally, we discuss critical limitations regarding synthetic-to-real domain gaps and provide a production reference deployed at `https://ocr.cham.asia`.

**Keywords**: Optical Character Recognition (OCR), Cham Script, Akhar Thrah, Cam Srak, Brahmic Paleography, Differentiable Binarization (DBNet), SVTR-LCNet, Connectionist Temporal Classification (CTC), Indic Line Segmentation.

---

## Table of Contents
- [1. Introduction & Paleographic Foundations](#1-introduction--paleographic-foundations)
  - [1.1 Historical Script Typology](#11-historical-script-typology)
  - [1.2 Abugida Orthography & Multi-Directional Glyphs](#12-abugida-orthography--multi-directional-glyphs)
  - [1.3 Logical Storage Order vs Visual Rendering Order](#13-logical-storage-order-vs-visual-rendering-order)
- [2. Related Work on Cham Document Analysis & Epigraphy](#2-related-work-on-cham-document-analysis--epigraphy)
  - [2.1 Early Epigraphic & Glyph Recognition](#21-early-epigraphic--glyph-recognition)
  - [2.2 Deep Learning for Cham Epigraphy & Transliteration](#22-deep-learning-for-cham-epigraphy--transliteration)
  - [2.3 Manuscript Preservation & Digitization Initiatives](#23-manuscript-preservation--digitization-initiatives)
  - [2.4 Positioning of the Present System](#24-positioning-of-the-present-system)
- [3. Text Detection & Indic Line Segmentation (Cham-DBNet)](#3-text-detection--indic-line-segmentation-cham-dbnet)
  - [3.1 The Interlinear Diacritic Collision Problem](#31-the-interlinear-diacritic-collision-problem)
  - [3.2 Specialized Cham-DBNet: Architecture & Training](#32-specialized-cham-dbnet-architecture--training)
  - [3.3 Hybrid Indic Valley-Cut Heuristic & Masking Safeguards](#33-hybrid-indic-valley-cut-heuristic--masking-safeguards)
- [4. Neural Text Recognition Architecture (Cham-SVTR)](#4-neural-text-recognition-architecture-cham-svtr)
  - [4.1 PP-OCRv4 SVTR-LCNet Backbone](#41-pp-ocrv4-svtr-lcnet-backbone)
  - [4.2 Joint CTC & Sequence Attention Loss](#42-joint-ctc--sequence-attention-loss)
  - [4.3 Post-OCR Deterministic Canonical Normalization](#43-post-ocr-deterministic-canonical-normalization)
- [5. Synthetic Dataset Generation & Hard-Example Mining](#5-synthetic-dataset-generation--hard-example-mining)
  - [5.1 The Data Scarcity Bottleneck](#51-the-data-scarcity-bottleneck)
  - [5.2 Calibrated Synthetic Dataset Partitioning & Dual-Generator Architecture](#52-calibrated-synthetic-dataset-partitioning--dual-generator-architecture)
  - [5.3 TrueType/OpenType cmap Tofu Glyph Safeguards](#53-truetypeopentype-cmap-tofu-glyph-safeguards)
  - [5.4 On-the-Fly Dynamic Augmentation in RAM](#54-on-the-fly-dynamic-augmentation-in-ram)
- [6. Empirical Benchmarks & Quantitative Evaluations](#6-empirical-benchmarks--quantitative-evaluations)
  - [6.1 Controlled 50-Test Stratified Benchmark (V23 vs V24)](#61-controlled-50-test-stratified-benchmark-v23-vs-v24)
  - [6.2 Controlled 200-Page Synthetic Document Stress-Test (903 Textlines)](#62-controlled-200-page-synthetic-document-stress-test-903-textlines)
  - [6.3 Model Progression & Empirical Validation: V23 vs V24 vs V25 SOTA](#63-model-progression--empirical-validation-v23-vs-v24-vs-v25-sota)
  - [6.4 Multi-GPU Distributed Hardware Scaling & Cost Efficiency](#64-multi-gpu-distributed-hardware-scaling--cost-efficiency)
- [7. Production Serving & Deployment Infrastructure](#7-production-serving--deployment-infrastructure)
  - [7.1 Containerized Microservice Architecture](#71-containerized-microservice-architecture)
  - [7.2 Concurrency Safeguards & NumPy 2.x Forward Compatibility](#72-concurrency-safeguards--numpy-2x-forward-compatibility)
- [8. Limitations & Open Research Challenges](#8-limitations--open-research-challenges)
  - [8.5 Post-V25 Future Iterations (Planned V25.1 / V26 Improvements)](#85-post-v25-future-iterations-planned-v251--v26-improvements)
- [9. BibTeX Citation & Academic References](#9-bibtex-citation--academic-references)

---

## 1. Introduction & Paleographic Foundations

### 1.1 Historical Script Typology
The Cham language belongs to the Austronesian language family (Malayo-Polynesian branch) and was the primary medium of state administration, sacred liturgy, and classical literature across the Champa polities along coastal central and southern Vietnam from the 2nd to early 19th centuries. The script descends from Southern Brahmic Grantha/Pallava lineages, diverging over centuries into two principal extant traditions:
* **Eastern Cham (*Akhar Thrah*)**: Inscribed and preserved primarily in Ninh Thuận and Bình Thuận provinces (Vietnam). Historical manuscripts are executed on traditional bark-paper (*kertas*) or palm-leaf folios using calligraphic bamboo pens.
* **Western Cham (*Cam Srak*)**: Preserved by Cham communities in the Mekong Delta (An Giang, Tây Ninh) and Cambodia. It features distinctive stylistic glyph contours, regional ligature forms, and specific phonetic loan signs.

Due to tropical climate conditions, ink bleed, microbiological decay, and physical fragmentation, remaining manuscript archives (such as the *Akayet Inra Patra*, *Ariya Po Pareng*, and divination compendia) face significant preservation risks. Standard off-the-shelf OCR engines designed primarily for Latin or CJK scripts struggle significantly on historical Cham manuscripts due to fundamental typographical divergences.

### 1.2 Abugida Orthography & Multi-Directional Glyphs
Unlike alphabetic writing systems where letters follow a single horizontal axis, Cham is an **abugida** governed by complex spatial attachment rules:
1. **Independent Vowels (`U+AA00`–`U+AA05`)**: Six standalone vowel glyphs represent syllable-initial vowel phonemes without a preceding consonant base:
   - `ꨀ` (U+AA00: `CHAM LETTER A`)
   - `ꨁ` (U+AA01: `CHAM LETTER I`)
   - `ꨂ` (U+AA02: `CHAM LETTER U`)
   - `ꨃ` (U+AA03: `CHAM LETTER E`)
   - `ꨄ` (U+AA04: `CHAM LETTER AI`)
   - `ꨅ` (U+AA05: `CHAM LETTER O`)
2. **Consonantal Inventory (`U+AA06`–`U+AA28`)**: Thirty-five consonant glyphs carry an inherent vowel (/a/ or /ɔ/), beginning with `ꨆ` (U+AA06: `CHAM LETTER KA`) through `ꨨ` (U+AA28: `CHAM LETTER HA`), including aspiration classes, nasals, and liquids.
3. **Subjoined Medials (`U+AA33`–`U+AA36`)**: Four medial consonants attach underneath or encircle the consonant base:
   - `ꨳ` (U+AA33: `CHAM CONSONANT SIGN YA` / Medial Ya)
   - `ꨴ` (U+AA34: `CHAM CONSONANT SIGN RA` / Medial Ra)
   - `ꨵ` (U+AA35: `CHAM CONSONANT SIGN LA` / Medial La)
   - `ꨶ` (U+AA36: `CHAM CONSONANT SIGN WA` / Medial Wa)
4. **Multi-Directional Dependent Vowels (`U+AA29`–`U+AA32`)**: Dependent vowel signs attach across all four geometric quadrants around the base:
   - *Above base*: Vowel Sign I (`ꨪ` U+AA2A), II (`ꨫ` U+AA2B), EI (`ꨬ` U+AA2C).
   - *Below base*: Vowel Sign U (`ꨭ` U+AA2D), UE (`ꨲ` U+AA32).
    - *Post-base / Dependent*: Vowel Sign AA (`ꨩ` U+AA29), OE (`ꨮ` U+AA2E), AU (`ꨱ` U+AA31).
    - *Pre-base (Left / Visual Ahead)*: Vowel Sign O (`ꨯ` U+AA2F), AI (`ꨰ` U+AA30).
5. **Final Consonants & Signs ([`U+AA40`–`U+AA4D`, `U+AA25`])**: Explicit syllable codas including separate base glyphs with extended terminal strokes (`U+AA40`–`U+AA4B`), combining coda marks (Final Ng `ꩃ` U+AA43, Final M / Anusvara `ꩌ` U+AA4C, Final H / Visarga `ꩍ` U+AA4D), and the base consonant `ꨥ` (`U+AA25` CHAM LETTER VA) which, per Unicode Standard Chapter 16 Table 16-16, functions without alteration in both initial and syllable-final positions (e.g., in syllables such as `ꨀꨍꨯꨱꨥ` and `ꨝꨗꨴꨭꨥ`).
6. **Script Spacing Conventions**: Many historical Cham manuscripts exhibit limited or inconsistent inter-word spacing, while modern printed Cham commonly uses spaces. Classical literary texts and verse compendia frequently exhibit continuous running script (*scriptio continua*), though section divisions, poetic hemistichs, or phrasing pauses are demarcated by punctuation marks (e.g., Danda `꩝`, Double Danda `꩞`, Triple Danda `꩟`) and occasional whitespace gaps.
7. **Punctuation & Verse Markers**: Traditional texts structure discourse via Danda (`꩝` U+AA5D), Double Danda (`꩞` U+AA5E, also serving as stanza/section mark), and Triple Danda (`꩟` U+AA5F). Poetic stanzas frequently begin with Cham numerals followed by a section mark (e.g., `꩑꩞` for Stanza 1, `꩒꩞` for Stanza 2).

### 1.3 Logical Storage Order vs Visual Rendering Order
In conformance with the Unicode Standard Chapter 16 (Section 16.3 Cham, Table 16-16) and Brahmic script architecture, the Unicode Standard mandates **Logical Order** encoding:

$$\text{Canonical Cluster} = \text{Base Consonant} + [\text{Medials RA/LA}] + [\text{Medials YA/WA}] + [\text{Pre-Vowels}] + [\text{Dependent Vowels}] + [\text{Lengthener AA}] + [\text{Finals / Signs}]$$

However, pre-vowels such as `ꨯ` (Vowel Sign O, U+AA2F) and `ꨰ` (Vowel Sign AI, U+AA30) are rendered *visually to the left* of the base consonant. When an optical sequence decoder scans an image crop strictly left-to-right, its visual perception encounters the pre-vowel ahead of the consonant glyph (e.g. perceiving `ꨯ` before base `ꨕ`). Standard CTC decoders trained without canonical sequence constraints naturally emit the non-standard sequence `ꨯꨕ` instead of standard `ꨕꨯ`.

> [!WARNING]
> **Paleographic Pitfall**: Attempting to resolve visual ordering with naive string replacements or visual-to-unicode cluster splitters damages multi-glyph compounds containing subjoined medials (e.g., corrupting `ꨙꨳꨯꨮ`). Our architecture mandates a deterministic post-OCR **Logical Order Normalizer** (Section 4.3) that sorts local visual permutations into canonical Unicode sequences without mangling base-medial complexes.

---

## 2. Related Work on Cham Document Analysis & Epigraphy

### 2.1 Early Epigraphic & Glyph Recognition
Automated processing of Cham script originated in the analysis of stone inscriptions from ancient Champa sanctuaries (Mỹ Sơn, Po Nagar, Đồng Dương). Nguyen, Schweyer, Le, Tran, & Vu (2019a) presented preliminary results on recognizing ancient Cham glyphs extracted from stone rubbings and photographs using Histogram of Oriented Gradients (HOG), Non-Parametric Weighted (NPW) features, and CNN-derived feature representations coupled with $k$-NN and linear SVM classifiers at the IEEE MAPR 2019 conference. Due to severe stone weathering, erosion, and irregular surface texture, isolated glyph classification across noisy background substrates was evaluated.

To mitigate extreme sample scarcity in epigraphic corpora, Nguyen, Schweyer, Le, Tran, & Vu (2019b) evaluated data augmentation strategies and transfer learning from scripts of similar or the same language family to ancient stone inscription glyphs at the ICIAP 2019 Workshops (PatReCH), demonstrating quantitative gains on script-to-inscription transfer across weathered stone carvings.

### 2.2 Deep Learning for Cham Epigraphy & Transliteration
In his doctoral dissertation, Nguyen (2023) developed comprehensive document image processing and understanding methodologies dedicated to ancient Cham documents. His work investigated stone inscription image restoration, automatic glyph segmentation under rough stone substrate conditions, and deep convolutional representations for historical Brahmic scripts.

Complementing glyph recognition, Nguyen, Burie, Le, & Schweyer (2022) proposed an effective multi-stage method for text line segmentation in historical document images presented at the 26th International Conference on Pattern Recognition (ICPR 2022), addressing line curvature and touching diacritics in historical archives. Expanding to linguistic representation, Nguyen, Burie, Le, & Schweyer (2023) introduced a two-step sequence transformer model for Cham-to-Latin script transliteration presented at the HIP@ICDAR 2023 workshop. Their system demonstrated the efficacy of sequence-to-sequence transformers in mapping recognized Cham text sequences to standardized Latin phonological romanizations (*Cam-Latin*). Most recently, Nguyen, Burie, Le, & Schweyer (2025) published a comprehensive study entitled *Text line segmentation approach combining deep learning model and traditional image processing techniques - application to transliteration of Cham manuscripts* in *Multimedia Tools and Applications*, evaluating text-line segmentation and automatic transliteration across 627 manuscript images and approximately 8,300 textlines (focusing on line segmentation and transliteration rather than end-to-end character OCR benchmarking).

### 2.3 Manuscript Preservation & Digitization Initiatives
Beyond epigraphy on stone, historical Cham manuscripts on paper (*kertas*) and palm-leaf have been documented through preservation missions. The Historic Cham Manuscripts of Vietnam collection in the Southeast Asia Digital Library (SEADL) contains 977 digitized manuscripts comprising more than 57,800 page scans. In addition, field surveys estimate approximately 3,000 physical manuscripts preserved within Cham community families and religious dignitaries across Vietnam and Cambodia. The French National Research Agency funded the CHAMDOC project (*Cham Documentation*, ANR-19-CE27-0018, 2019–2024) to support preservation, cataloging, and linguistic documentation.

However, despite the preservation of tens of thousands of scanned folios, almost none of these archives possess open, machine-readable character-level or line-level bounding box ground-truth annotations suitable for training supervised deep learning models. The bottleneck for Cham OCR is therefore not an absolute absence of physical manuscripts, but rather the severe scarcity of annotated, verified training ground truth.

### 2.4 Positioning of the Present System
Prior research has focused predominantly on either **isolated glyph recognition on stone inscriptions**, **downstream transliteration of cleaned text**, or academic research prototypes evaluated on private manuscript subsets. In contrast, the system presented in this report addresses the complete **end-to-end continuous document pipeline** for historical manuscripts:
* Full-page line detection and segmentation robust to undulating baselines and tight interlinear diacritics.
* End-to-end continuous textline recognition without manual character pre-segmentation.
* Deterministic canonical reordering adhering strictly to the official Unicode standard.
* Production-grade serving designed for cultural heritage researchers and native community access.

---

## 3. Text Detection & Indic Line Segmentation (Cham-DBNet)

### 3.1 The Interlinear Diacritic Collision Problem
In digital modeling and synthetic text rendering at nominal glyph heights (32–36px), interlinear clearance often narrows to 3–12 pixels between adjacent lines (with ascender excursions extending up to $+18\text{px}$ and descender excursions down to $-22\text{px}$ in normalized coordinate space). Because upper dependent vowels reach high above the core text band and subjoined medials plunge below, the ascenders of line $n+1$ routinely touch or intertwine with the descenders of line $n$ under dense formatting. Standard off-the-shelf text detection models (e.g., baseline DBNet trained on natural scene Latin or CJK text) frequently encounter two severe failure modes when applied to dense historical manuscripts:
1. **Line Merging**: Grouping adjacent lines into a single bounding box when diacritics touch.
2. **Diacritic Amputation**: Slicing off upper vowel signs or subjoined medials during rectangular cropping, which causes fatal downstream recognition errors.

### 3.2 Specialized Cham-DBNet: Architecture & Training
To overcome these limitations, we fine-tuned a specialized text detector based on **PP-OCRv4 Mobile DBNet** featuring a lightweight **PPLCNetV3** backbone:

| Hyperparameter / Component | Specification | Engineering Rationale |
| :--- | :--- | :--- |
| Backbone Architecture | `PPLCNetV3` (scale 0.75) | Depthwise separable convolutions + SE attention; low-latency CPU execution. |
| Loss Function | Probability Loss ($L_s$) + Binary Loss ($L_b$) + Threshold Loss ($L_{th}$) | Differentiable binarization with adaptive threshold learning (Dice Loss). |
| Unclip Ratio ($\alpha$) | `1.8` | Increased from Latin default (1.5) to encompass high ascenders and deep medials. |
| Binarization Threshold | `0.25` | Calibrated for faint, faded ink on aged manuscript paper. |
| Polygon Box Threshold | `0.50` | Suppresses stray ink specks, paper grain, and background bleed-through artifacts. |
| Max Candidates | `1500` | Accommodates densely packed multi-line folios without candidate truncation. |
| Training Dataset | 2,000 document pages | 50% synthesized with narrow interline spacing (3–10px) to penalize box merging (as configured in `launch_h100_full_pipeline.py`). |
| Optimizer & Schedule | Adam (Cosine decay, initial lr 0.001, 5 warmup epochs, L2 factor 5e-5) | Stable gradient convergence without aggressive parameter divergence. |

**Empirical Verification**: Evaluated within the complete document processing pipeline on our 200-page synthetic stress-test suite (`evaluation_report.json`), Cham-DBNet delivers a **100.0% textline detection rate** across standard folios, aged paper textures, and narrow interlinear gaps (Levels 1–3, 560/560 lines detected) and **99.44%** on undulating sinusoidal baselines (Level 4, 176/177 lines detected). Performance declines only at Level 5 (**61.45% detection rate**, 102/166 lines) where interlinear wave amplitude mathematically exceeds line spacing ($\text{Amplitude} > \text{Gap}$), creating physical stroke collisions.

### 3.3 Hybrid Indic Valley-Cut Heuristic & Masking Safeguards
For CPU-bound execution in low-resource environments (or when DBNet probability maps exhibit spatial ambiguity), we implemented an **Indic Valley-Cut Heuristic Engine** operating as a secondary verification layer:
1. **Connected Component Analysis (CCA)**: Identifies all connected ink components on the binarized page.
2. **Core Text Band Determination**: Locates the median baseline and core bounding rectangle of each line using horizontal projection histograms.
3. **Boundary Stroke Masking via `difference_update`**: When constructing an erasing mask to occlude adjacent interfering lines, ink components belonging to the target line are strictly excluded using mathematical set difference ($S_{\text{mask}} = S_{\text{adjacent}} \setminus S_{\text{target}}$). This prevents erasing touching diacritics.
4. **Vertical Padding Safeguards**: Crop boundaries are padded by $0.40 \times \text{median line height}$ above and below, bounded at a minimum safety distance of $2\text{px}$ from adjacent core bands.
5. **Legacy-First / No-Change Gate**: For clean documents where interline gap $\ge 0.35 \times \text{height}$ and initial recognition confidence $\ge 0.75$, multi-crop splitting is bypassed entirely to guarantee zero regression on pristine pages.

---

## 4. Neural Text Recognition Architecture (Cham-SVTR)

### 4.1 PP-OCRv4 SVTR-LCNet Backbone
Textline recognition is modeled as continuous sequence labeling. We adapted the **SVTR-LCNet** architecture from PP-OCRv4 with custom input and output configurations:
* **Input Tensor Dimensions**: Expanded to $[3, 48, 480]$ (channels $\times$ height $\times$ width). Widening the horizontal dimension to 480 prevents spatial compression of long Cham verse lines containing stacked diacritics.
* **SVTR Blocks**: Combines local convolutional feature extractors with patch-wise self-attention mechanisms to capture contextual dependencies across unspaced text.
* **Squeeze-and-Excitation (SE) Attention**: Adaptively recalibrates channel-wise feature responses, emphasizing subtle diacritic marks over paper texture noise.

### 4.2 Joint CTC & Sequence Attention Loss
Training utilizes joint multi-task supervision combining Connectionist Temporal Classification (CTC) with an Attention sequence decoder:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CTC}} + \lambda \mathcal{L}_{\text{attention}}$$

Where $\mathcal{L}_{\text{CTC}}$ maximizes the marginal log-likelihood of all valid sequence alignments over transcription $\mathbf{y}$:

$$\mathcal{L}_{\text{CTC}} = -\ln P(\mathbf{y} \mid \mathbf{x}) = -\ln \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})} P(\pi \mid \mathbf{x})$$

During runtime deployment, greedy CTC decoding is executed to provide lightweight, low-latency single-thread CPU execution without costly beam search.

### 4.3 Post-OCR Deterministic Canonical Normalization
To reconcile visual ordering with standard Unicode Logical Order, our post-processing pipeline executes `normalize_unicode` (built upon `parse_unicode_clusters`). The normalizer parses raw CTC strings into discrete grapheme clusters and reorders tokens into standard Brahmic sequence:

```python
def normalize_unicode(text: str) -> str:
    """
    Enforces Canonical Brahmic Logical Order on CTC decoded Cham text:
    Base Consonant -> Medial RA/LA -> Medial YA/WA -> Pre-Vowels (O/AI) ->
    Other Dependent Vowels -> Vowel Lengthener AA -> Final Consonants / Signs (incl. Final VA)
    Example: Corrects visual sequence 'ꨙꨯꨳꨮ' into canonical 'ꨙꨳꨯꨮ'.
    """
    clusters = parse_unicode_clusters(text)
    uni_clusters = [visual_to_unicode_cluster(c) for c in clusters]
    return "".join("".join(c) for c in uni_clusters)
```

---

## 5. Synthetic Dataset Generation & Hard-Example Mining

### 5.1 The Data Scarcity Bottleneck
Deep neural recognition networks require hundreds of thousands of diverse labeled textline instances. However, surviving physical Cham manuscripts lack character-level bounding box ground truth. To overcome this limitation, we engineered a synthetic generation pipeline (`scripts/generate_data.py`) capable of synthesizing 150,000 diverse textline crops from authentic literary corpora.

### 5.2 Calibrated Synthetic Dataset Partitioning & Dual-Generator Architecture
To support both baseline reproducibility and ongoing experimental modeling, the repository provides two distinct synthetic dataset generation pipelines:

#### Pipeline A: Baseline General Generator (`scripts/generate_data.py`)
The baseline generator produces a 150,000 textline corpus partitioned into **15% evaluation/locked splits** (22,500 lines: `val_clean_short` 3%, `val_clean_long` 3%, `val_noisy_short` 3%, `val_noisy_long` 3%, `locked_test` 3%) and an **85% training split** (127,500 lines) structured across 6 calibrated sub-categories:

| Training Sub-Category | Proportion | Line Count | Target Composition & Protocol | Target Failure Mode Addressed |
| :--- | :---: | :---: | :--- | :--- |
| `train_clean_short` | **25%** | 31,875 | Short clean literary phrases (1–4 tokens). | Fast convergence on high-frequency base consonants and common syllables. |
| `train_clean_medium` | **20%** | 25,500 | Medium-length lines (5–8 tokens) from classical texts (*Akayet Inra Patra*, *Ariya Po Pareng*). | Teaches authentic Cham phonotactic and n-gram transitions. |
| `train_clean_long` | **15%** | 19,125 | Long full verse hemistichs and prose sentences (9+ tokens). | Robustness to full-line receptive fields and CTC sequence alignment without truncation. |
| `train_noisy_short` | **15%** | 19,125 | Short lines rendered with Point Spread Function (PSF) blur, noise, and lighting gradients. | Invariance to archival handling artifacts, tremor, and fading ink. |
| `train_noisy_long` | **15%** | 19,125 | Long verse lines with severe photometric degradation and spatial elastic distortion. | Prevents alignment collapse on degraded multi-word lines. |
| `train_hard_examples` | **10%** | 12,750 | Adversarial minimal pairs (`ꨲ` vs `ꨶ`, `꩝꩝` vs `꩝`), verse numerals (1–99) with section marks (`꩑꩞`), and stacked subjoined clusters. | Eliminates CTC misclassification of numerals as letters (`꩔` vs `ꨤ`, `꩕` vs `ꨅ`) and diacritic collapsing. |

#### Pipeline B: Experimental V25 Generator (`scripts/generate_data_v25.py`)
For the V25 experimental iteration, a dedicated multi-threaded generator (`generate_data_v25.py`) synthesizes 150,000 textline images (140,000 train + 10,000 validation) directly synchronized with `configs/v25_dataset_manifest.json` across 5 core pillars:
1. **Canonical Cham Literature (43.3% / 65,000 lines)**: Classical literary vocabulary (*Po Klong Garai*, 57-stanza verse).
2. **Anti-Motion Blur Base (20.0% / 30,000 lines)**: Clean base images paired with dynamic in-RAM directional motion blur ($7\times 7$ to $13\times 13$) and defocus.
3. **Inline Bilingual Code-Switching (16.7% / 25,000 lines)**: Intra-line mixed phrases with Vietnamese diacritics and Latin annotations.
4. **Stanza Numerals & Boundary Punctuation (12.0% / 18,000 lines)**: Verse numbers $1$–$99$ (`꩑꩞`..`꩙꩙꩞`) and boundary marks (`꩞`, `:`, `–`).
5. **Adversarial Minimal Pairs (8.0% / 12,000 lines)**: Micro-stroke pairs (`ꨲ` U+AA32 Vowel Sign UE vs `ꨶ` U+AA36 Medial WA, variable Double Danda `꩝꩝` spacing 2–8px, 3-tier diacritic stacks).

> [!NOTE]
> **Completed Multi-Stage Training (V25)**: Model Version 25 has completed its full 40-epoch multi-stage distributed GPU training on Kaggle Dual Tesla T4x2 accelerators (Stage 1: Epochs 1–20 warmup and baseline adaptation; Stage 2: Epochs 21–30 hard-example fine-tuning; Stage 3: Epochs 31–40 final convergence with cosine learning rate decay, completing 87,480 cumulative training steps). The final checkpoint (`best_accuracy.pdparams`, 58.84 MB) achieved **90.96% Sequence Accuracy** on the 10,000 frozen validation suite.

### 5.3 Digital Typefaces & TrueType cmap Tofu Glyph Safeguards
When rendering synthetic text, digital Cham typefaces must be verified for glyph coverage. The primary bundled and reproducible typeface distributed directly within this repository is **Noto Sans Cham** (`Regular`, `Bold`, `Black`; SIL Open Font License 1.1, located in `ocr-training/data/fonts/`). External reference typefaces evaluated during research—including *Cham Roman*, *EFEO Cham*, and community fonts—are not redistributed in this repository and must be acquired independently if researchers wish to reproduce specific historical font variations.

To prevent missing Unicode codepoints from producing blank rectangles ("tofu" glyphs), our generator incorporates pre-render verification via `fontTools.ttLib` to extract each font's `cmap` table. Candidate strings are inspected character-by-character; any string containing unmapped glyphs for the active font is rejected prior to rasterization.

### 5.4 On-the-Fly Dynamic Augmentation in RAM
To avoid disk I/O bottlenecks during generation, photometric distortions are synthesized dynamically in memory:
* **Point Spread Function (PSF) Motion Blur**: Simulates hand tremors during mobile photography in archival field collections (kernel size 3–9px, angle $[0, 180^\circ]$).
* **Gaussian & Salt-and-Pepper Noise**: Simulates coarse parchment fibers, dust grain, and ink spatter.
* **Non-Linear Illumination Gradients**: Models uneven ambient lighting across warped manuscript folios.

---

## 6. Empirical Benchmarks & Quantitative Evaluations

### 6.1 Controlled 50-Test Stratified Benchmark (V23 vs V24)
To measure progression across model iterations, we evaluated 50 stratified test cases categorized across 10 distinct failure modes. The results reflect exact empirical data logged in `benchmark_50_tests_results.json` and `benchmark_50_tests_results_v24.json`:

| Evaluation Category | Count | V23 CER (%) | V24 CER (%) | CER Gain | V23 Pass (%) | V24 Pass (%) | V24 Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Cat 1: Clean Baseline | 5 | 14.84% | **6.20%** | **-8.64%** | 20.0% | **80.0%** | 0.958 |
| Cat 2: Single-Digit Stanzas | 5 | 18.42% | **11.71%** | **-6.71%** | 0.0% | **40.0%** | 0.944 |
| Cat 3: Two-Digit Stanzas | 5 | 17.59% | **7.65%** | **-9.94%** | 20.0% | **60.0%** | 0.956 |
| Cat 4: Low Height / Low-Res | 5 | 27.65% | **17.96%** | **-9.69%** | 0.0% | **20.0%** | 0.848 |
| Cat 5: Blur Degradation | 5 | 41.78% | **35.56%** | **-6.22%** | 0.0% | 0.0% | 0.839 |
| Cat 6: Noise & Grain | 5 | 50.59% | **8.82%** | **-41.77%** | 0.0% | **80.0%** | 0.944 |
| Cat 7: Paper Texture & Contrast | 5 | 33.33% | **21.33%** | **-12.00%** | 0.0% | 0.0% | 0.921 |
| Cat 8: Tilt & Perspective | 5 | 22.14% | **0.71%** | **-21.43%** | 0.0% | **100.0%** | 0.982 |
| Cat 9: Stroke Degradation | 5 | 26.67% | **6.67%** | **-20.00%** | 0.0% | **80.0%** | 0.928 |
| Cat 10: Diacritics & Punctuation | 5 | 28.72% | **22.19%** | **-6.53%** | 0.0% | **40.0%** | 0.913 |
| **Overall Benchmark Average** | **50** | **28.17%** | **13.88%** | **-14.29%** | **4.0%** | **50.0%** | **0.923** |

### 6.2 Controlled 200-Page Synthetic Document Stress-Test (903 Textlines)
To establish the operational limits of the complete end-to-end pipeline (Cham-DBNet + Cham-SVTR V24), we evaluated a **controlled synthetic stress-test suite** of 200 generated page images comprising 903 ground-truth textlines. Generated by `ocr-benchmark/scripts/generate_benchmark_200.py`, this suite renders authentic literary passages under calibrated geometric and photometric stress across 5 difficulty tiers:

| Difficulty Tier | Stress & Degradation Profile | Samples | GT Lines | Detection Rate | Mean CER | Mean WER | CPU Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Level 1: Standard | Line gap 25–35px, rectilinear alignment, white background. | 40 | 201 | **100.0%** | 14.43% | 45.50% | 1.68s |
| Level 2: Aged Paper | Yellowed paper texture, mild dust grain, slight skew ($\le 2^\circ$). | 40 | 189 | **100.0%** | 14.44% | 45.86% | 1.50s |
| Level 3: Narrow Gap | Tight interline gap 8–14px, micro-waviness (1–2px). | 40 | 170 | **100.0%** | 12.77% | 43.52% | 1.26s |
| Level 4: Wavy Sinusoid | Sine wave warping (amp 3–5.5px), perspective tilt (0.03–0.05), gap 5–10px. | 40 | 177 | **99.44%** | 14.53% | 45.64% | 1.24s |
| Level 5: Extreme Overlap | Gap 2–6px < wave amplitude 5.5–8.0px. Intersecting boundary strokes. | 40 | 166 | **61.45%** | 60.02% | 77.63% | 0.99s |
| **Overall Suite Summary** | **200 synthetic document images, 903 textlines total.** | **200** | **903** | **92.80%** | **23.24%** | **51.63%** | **1.33s/page** |

> [!NOTE]
> **The Empirical Breaking Point**: Results from Level 3 and Level 4 establish that Cham-DBNet handles narrow line spacing (8–14px) and moderate wavy distortions with near-perfect reliability (99.44%–100% detection rate). The sharp decline at Level 5 (61.45% detection rate, 60.02% CER) marks the mathematical breaking point where sinusoidal distortion amplitude exceeds the interlinear gap ($\text{Amplitude} > \text{Gap}$), causing strokes from adjacent lines to physically intersect on the binarized image plane.

### 6.3 Model Progression & Empirical Validation: V23 vs V24 vs V25 SOTA

Across iterative development cycles, we evaluated the trajectory of the Cham-SVTR architecture across three foundational milestones: **Version 23** (visual ordering legacy), **Version 24** (first logical ordering baseline), and **Version 25** (current production State-Of-The-Art). Quantitative validation on the standardized 10,000 frozen validation split (`cham_v25_val_freeze.zip`) and controlled failure-mode test suites documents the compounding gains achieved through lexicon expansion, resolution widening, and targeted hard-example synthesis:

| Metric / Dimension | Version 23 (Visual Legacy) | Version 24 (Logical Baseline) | Version 25 (SOTA Production) |
| :--- | :---: | :---: | :---: |
| **Orthographic Representation** | Visual Rendering Order | Logical Brahmic Order | **Logical Brahmic Order** |
| **Lexicon Dictionary Size** | 153 tokens | 156 tokens | **162 tokens** (+Numerals 1–99, Section Mark `꩞`) |
| **Input Shape $(C, H, W)$** | $[3, 48, 320]$ | $[3, 48, 320]$ | **$[3, 48, 480]$** (Accommodates long stanzas) |
| **Frozen Val Sequence Accuracy** | 84.12% | 88.94% | **90.96%** (+2.02% over V24, +6.84% over V23) |
| **Normalized Edit Distance (Norm-ED)** | 96.85% | 98.61% | **99.28%** |
| **Character Error Rate (CER)** | 3.15% | 1.39% | **0.72%** (Halved vs V24, $-77.1\%$ vs V23) |
| **Inference Throughput (Dual T4)** | 158.2 FPS | 154.6 FPS | **143.57 FPS** |
| **Training Steps / Epochs** | 40 epochs (1 stage) | 40 epochs (2 stages) | **40 epochs (3 stages, 87,480 steps)** |
| **Model Parameter Footprint** | 58.82 MB | 58.83 MB | **58.84 MB** (`best_accuracy.pdparams`) |

#### Detailed Error Mode Resolution & Paleographic Breakthroughs in V25

1. **Stanza Numerals vs Consonant Disambiguation**:
   Prior iterations suffered from catastrophic CTC collapse where Western Cham and Eastern Cham digit ligatures were forcefully decoded as visually similar consonants: `꩔` (digit 4) was frequently misclassified as `ꨤ` (*la*), `꩕` (digit 5) as `ꨅ` (*e*) or `ꨂ` (*u*), and verse markers like `꩑꩞` were collapsed into `ꨩꩌ`. By incorporating 18,000 dedicated adversarial samples covering all numerals $1$–$99$ with boundary markers (`꩑꩞` through `꩙꩙꩞`), V25 achieves $>98.5\%$ exact match on stanza headers.

2. **Boundary Punctuation & Double Danda (`꩝꩝`) Collapse**:
   CTC decoders with standard temporal stride tend to merge closely adjacent vertical strokes into a single timestep ("blank collapse"). V25 synthesized variable Double Danda spacings ($2$px to $8$px) under varying Gaussian blur kernels, successfully training the network to distinguish single `꩝` (U+AA5D) from double `꩝꩝` (U+AA5D U+AA5D).

3. **Diacritic Minimal Pair Disambiguation (`ꨲ` vs `ꨶ`)**:
   Under degraded ink conditions, Vowel Sign UE (`ꨲ`, U+AA32) and Medial Consonant Sign WA (`ꨶ`, U+AA36) exhibit extreme visual proximity under base glyphs `ꨀ`, `ꨓ`, `ꨚ`, `ꨆ`. Through 12,000 targeted adversarial minimal pairs, V25 forces the convolutional feature extractor to attend to subtle sub-stroke curvature differences, dropping the pair substitution rate from $18.4\%$ in V24 to $<1.2\%$ in V25.

4. **Integration with Deterministic Unicode Cluster Normalization**:
   Operating strictly in Logical Storage Order, V25 pairs directly with `normalize_unicode` (via `scripts/generate_data.py`). Any localized CTC sequence permutations are deterministically regularized to the canonical Brahmic order:
   $$\text{Canonical Order} = \text{Base Consonant} + \text{Medials} + \text{Pre-Ra} + \text{Pre-Vowels} + \text{Post-Vowels} + \text{Finals}$$
   Eliminating visual ordering inversion errors while maintaining $100\%$ compliance with modern Unicode display engines.

### 6.4 Multi-GPU Distributed Hardware Scaling & Cost Efficiency
To support reproducible training across constrained computational budgets, we conducted multi-GPU benchmarks across four hardware environments:

| Accelerator Hardware | Batch Size | Throughput (IPS) | GPU Util | VRAM Usage | Full 40 Epochs | Cost ($) | Cost Efficiency (IPS/$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA A100 (80GB SXM4)** | 192 | **255.9** | 95% | 88.5% | **10.85 hrs** | **$23.77** | **116.8** |
| **NVIDIA L40S (48GB Ada)** | 128 | **209.9** | 94% | 52.5% | 13.23 hrs | $28.32 | 98.1 |
| **Kaggle Dual Tesla T4x2 (DDP)** | 64 (32x2) | **120.4** | 92% | 83.8% | 23.07 hrs | **Free Quota** | **Infinite** |
| **NVIDIA L4 (24GB)** | 96 | **74.3** | 98% | 79.2% | 37.38 hrs | $29.53 | 94.0 |

---

## 7. Production Serving & Deployment Infrastructure

### 7.1 Containerized Microservice Architecture
The production transcription environment is packaged as a multi-stage Docker container deployed on **Hugging Face Spaces** (`phucsd/cham-ocr-studio`) and routed through a Cloudflare edge proxy to the custom apex domain:

$$\text{Production URL: } \mathbf{\text{https://ocr.cham.asia}}$$

### 7.2 Concurrency Safeguards & NumPy 2.x Forward Compatibility
To ensure stability on free-tier container instances (2 vCPUs, 16GB RAM):
* **Thread Contention Elimination**: OpenMP and MKL thread pools are constrained via `OMP_NUM_THREADS=1` and `CPU_THREADS=1` to prevent CPU context switching overhead during concurrent HTTP requests.
* **NumPy 2.x Monkeypatching**: Python 3.12+ environments encounter breaking removals in NumPy 2.x (`np.sctypes`, `np.bool`, `np.typeDict` utilized by legacy PaddleOCR and `imgaug` modules). An automated compatibility shim is executed at module startup:

```python
import numpy as np
if not hasattr(np, 'sctypes'):
    np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64],
                  'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
                  'float': [np.float16, np.float32, np.float64],
                  'complex': [np.complex64, np.complex128],
                  'others': [bool, object, bytes, str]}
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'typeDict'): np.typeDict = {}
```

---

## 8. Limitations & Open Research Challenges

While our pipeline demonstrates substantial gains on synthetic and semi-controlled documents, rigorous academic honesty requires acknowledging the following boundaries:

1. **Synthetic-to-Real Domain Gap**: The neural recognizer is trained predominantly on synthetic textlines rendered with bundled *Noto Sans Cham* (SIL OFL 1.1). Historical manuscripts feature idiosyncratic scribal hands, variable ink viscosity, and non-standard ligature variations that may exhibit lower recognition confidence in the field.
2. **Absence of Large-Scale Real Manuscript Ground Truth**: Due to the acute scarcity of digitized historical Cham archives with line-level annotations, current benchmarks rely on controlled synthetic stress-tests and curated test slices. Establishing an open, expert-verified historical palm-leaf manuscript benchmark represents an urgent necessity requiring future collaborative scholarship with native Cham elders and linguists.
3. **Severe Biological Degradation & Epigraphy**: Inscription stone rubbings and severely mold-damaged palm leaves (where character ink has substantially eroded) remain beyond the capabilities of pure vision-based CTC sequence models. Integrating masked language models (MLMs) trained on historical Cham corpora is an active subject of future study.
4. **Extreme Interlinear Overlap (Level 5)**: When severe paper wrinkling causes interlinear wave amplitude to exceed line spacing ($\text{Amplitude} > \text{Gap}$), 1D projection and horizontal bounding boxes collapse, indicating the need for 2D polygonal baseline tracking models in future iterations.

### 8.5 Post-V25 Future Iterations (Planned V25.1 / V26 Improvements)
Following the successful training and deployment of Model Version 25 (90.96% Sequence Accuracy), all subsequent pipeline enhancements are tracked systematically for future V25.1 and V26 releases (see [FUTURE_WORK.md](file:///FUTURE_WORK.md)):
1. **Dual-Script Font Fallback & Glyph Validation**: Implementation of a coordinated rendering pipeline for mixed Cham and Vietnamese/Latin text with fontTools cmap validation across both scripts to eliminate tofu artifacts in bilingual annotations.
2. **Dedicated Directional Motion-Blur Augmentation**: Physics-grounded Point Spread Function (PSF) kernels modeling hand tremor motion angles and shutter exposure times.
3. **Manifest-Driven Exact Package Allocation**: Deterministic line count allocation per category instead of probabilistic multinomial sampling.
4. **Deterministic Frozen Validation Generation**: Permanent pre-generated frozen validation splits to guarantee zero-variance benchmark comparisons across epochs.

---

## 9. BibTeX Citation & Academic References

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

### Academic References

1. Everson, M. (2006). *Proposal for encoding the Cham script in the UCS (ISO/IEC JTC1/SC2/WG2 N3120)*. Unicode Consortium. [https://www.unicode.org/L2/L2006/06257-n3120-cham.pdf](https://www.unicode.org/L2/L2006/06257-n3120-cham.pdf)
2. Nguyen, M.-T., Schweyer, A.-V., Le, T.-L., Tran, T.-H., & Vu, H. (2019a). *Preliminary Results on Ancient Cham Glyph Recognition from Cham Inscription Images*. In 2019 International Conference on Multimedia Analysis and Pattern Recognition (MAPR 2019), IEEE, pp. 1–6. DOI: [10.1109/MAPR.2019.8743540](https://doi.org/10.1109/MAPR.2019.8743540).
3. Nguyen, M.-T., Schweyer, A.-V., Le, T.-L., Tran, T.-H., & Vu, H. (2019b). *Improving Ancient Cham Glyph Recognition Using Data Augmentation and Transfer Learning*. In International Conference on Image Analysis and Processing (ICIAP 2019 Workshops: PatReCH), Springer, Cham, LNCS 11808, pp. 115–125. DOI: [10.1007/978-3-030-30754-7_12](https://doi.org/10.1007/978-3-030-30754-7_12).
4. Nguyen, T.-N. (2023). *Segmentation, Recognition and Indexing of Cham characters in Cham documents* (*Segmentation, reconnaissance et indexation des caractères Cham dans les documents Cham*). Ph.D. Dissertation, Université de La Rochelle, France.
5. Nguyen, T.-N., Burie, J.-C., Le, T.-L., & Schweyer, A.-V. (2023). *A Two-Step Sequence Transformer Based Method for Cham to Latin Script Transliteration*. In Proceedings of the 7th International Workshop on Historical Document Imaging and Processing (HIP@ICDAR 2023), ACM, pp. 25–30. DOI: [10.1145/3604951.3605525](https://doi.org/10.1145/3604951.3605525).
6. Nguyen, T.-N., Burie, J.-C., Le, T.-L., & Schweyer, A.-V. (2022). *An effective method for text line segmentation in historical document images*. In 26th International Conference on Pattern Recognition (ICPR 2022), IEEE, pp. 2686–2692. DOI: [10.1109/ICPR56361.2022.9956617](https://doi.org/10.1109/ICPR56361.2022.9956617).
7. Nguyen, T.-N., Burie, J.-C., Le, T.-L., & Schweyer, A.-V. (2025). *Text line segmentation approach combining deep learning model and traditional image processing techniques - application to transliteration of Cham manuscripts*. *Multimedia Tools and Applications*, Springer. DOI: [10.1007/s11042-025-20602-x](https://doi.org/10.1007/s11042-025-20602-x).
8. ANR CHAMDOC Project. (2019–2024). *Cham Documentation (CHAMDOC)*, Project ANR-19-CE27-0018, Agence Nationale de la Recherche, France.
9. Liao, M., Wan, Z., Yao, C., Chen, K., & Bai, X. (2020). *Real-time Scene Text Detection with Differentiable Binarization*. Proceedings of the AAAI Conference on Artificial Intelligence, 34(07), 11474–11481.
10. Du, Y., Chen, Z., Jia, C., Yin, X., Zheng, T., Li, C., ... & Yu, K. (2023). *PP-OCRv4: A Compact, Accurate and Practical Ultra-Lightweight OCR System*. arXiv preprint arXiv:2309.09941.
11. Du, Y., Chen, Z., Jia, C., Yin, X., Zheng, T., Li, C., ... & Yu, K. (2022). *SVTR: Scene Text Recognition with a Single Visual Model*. arXiv preprint arXiv:2205.00159.
12. Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). *Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks*. Proceedings of the 23rd International Conference on Machine Learning (ICML '06), 369–376.

---
© 2026 Phuc H. Nguyen. Released under the MIT License. Contact: [phucsd@gmail.com](mailto:phucsd@gmail.com).
