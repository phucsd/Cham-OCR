# Data Provenance, Linguistic Corpora & Font Licensing

This document provides a transparent, verifiable record of the provenance, rights status, digital typefaces, and synthetic benchmarks utilized in the **Cham-OCR** project.

---

## 1. Primary Text Corpus (`cham_text.txt`)

* **File Location**: `ocr-training/data/corpus/cham_text.txt` (approx. 733 KB, 18,829 textlines)
* **Script Tradition**: Eastern Cham (*Akhar Thrah*, Unicode Block `U+AA00`–`U+AA5F`)
* **Linguistic Scope**: Classical literary verse, liturgical prose, historical chronicles, and lexical compendia.

### 1.1 Text Source Attribution & Provenance Ledger

| Source / Title | Author / Editor / Institution | Web Reference / Archive | Acquisition Method | Rights / License Status | Redistributed in Repo? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| *Akayet Inra Patra* (Classical verse epic) | Traditional Cham scribal transmission; modern editorial transliterations by Cham scholars | Academic & community preservation publications | Curated and transcribed from traditional cultural heritage texts | Traditional cultural heritage / specific edition rights unverified / non-commercial educational & preservation use (line-by-line edition mapping subject to archival verification) | Yes (as normalized Unicode textlines in `cham_text.txt`) |
| *Ariya Po Pareng* (Classical narrative poem) | Traditional Cham scribal transmission | Cham cultural preservation archives | Curated and transcribed from traditional manuscripts | Traditional cultural heritage / specific edition rights unverified / non-commercial educational & preservation use (line-by-line edition mapping subject to archival verification) | Yes (normalized Unicode textlines in `cham_text.txt`) |
| Historical chronicles & folk narratives (*Dalikal*) | Cham traditional oral and written literature | Cham community collections and regional cultural publications | Digitized and normalized into Unicode | Traditional cultural heritage / specific edition rights unverified / non-commercial educational & preservation use (line-by-line edition mapping subject to archival verification) | Yes (normalized Unicode textlines in `cham_text.txt`) |
| Cham-Vietnamese-French Lexical Compendia | Historical missionary and linguistic wordlists (e.g. Aymonier & Cabaton 1906; Moussay 1971; modern pedagogical materials) | Academic library collections & SEADL references | Digital normalization into standard Unicode 16.0 Cham | Historical reference works / specific edition rights unverified / non-commercial educational & preservation use (line-by-line edition mapping subject to archival verification) | Yes (vocabulary tokens merged into `cham_text.txt`) |

> [!NOTE]
> **Archival Verification Caveat**: Exact published print editions, original publication years, critical apparatus variants, and persistent digital repository shelfmarks/URLs for the literary texts transcribed in `cham_text.txt` have not been independently verified by an institutional archivist or manuscript librarian. The corpus represents normalized Unicode transcriptions assembled from diverse educational and community preservation sources for OCR training and evaluation. Downstream researchers must not rely on `cham_text.txt` as an authoritative or critical philological edition.

### 1.2 Corpus Curation & Linguistic Preprocessing
* **Orthographic Normalization**: Conversion into canonical Unicode 16.0 Cham representation (`U+AA00`–`U+AA5F`).
* **Encoding Rectification**: Removal of legacy ASCII transliteration hacks, unmapped symbols, and malformed encoding artifacts.
* **Textline Segmentation**: Partitioning into sentence and clause fragments corresponding to natural textline sequence lengths.

### 1.3 Legal & Cultural Usage Rights Disclaimer
> **Important Notice on Text Rights & Archival Provenance**:  
> The project maintainer claims no proprietary rights or copyright over underlying traditional Cham literature, historical chronicles, or modern scholarly editions. Redistribution rights remain subject to the rights status of each source corpus.  
> The curated file `cham_text.txt` is compiled and provided strictly for non-commercial educational scholarship, cultural preservation, and algorithmic OCR training under fair-use research principles. Because persistent canonical archival URLs and institutional repository licenses cannot be independently verified for all aggregated fragments, any commercial utilization or redistribution outside academic research must seek permission from relevant cultural rights holders and publishing bodies.

---

## 2. Digital Typefaces & Fonts

To train neural textline recognition models without bias toward a single rendering engine, digital Cham typefaces were evaluated. We strictly distinguish between fonts bundled directly in this repository and external candidate/reference fonts:

### A. Bundled Repository Fonts
The following font family is distributed directly within this repository (located in `ocr-training/data/fonts/`):

| Font Name | Foundry / Origin | Bundled Styles | License | Primary Use |
| :--- | :--- | :--- | :--- | :--- |
| **Noto Sans Cham** | Google Fonts / Monotype | `Regular`, `Bold`, `Black` | **SIL Open Font License, Version 1.1** (OFL-1.1) | Primary clean rendering baseline for modern and classical orthography in synthetic line generation. |

All bundled font files retain their upstream copyright notices and OFL-1.1 licensing terms.

### B. External Candidate & Reference Fonts (Not Bundled)
The following typefaces were evaluated as reference designs during research experiments or may be acquired independently by researchers wishing to extend synthetic rendering diversity. **They are not bundled or distributed in this repository**:

| Font Name | Origin / Organization | Access / Licensing Context | Note |
| :--- | :--- | :--- | :--- |
| **EFEO Cham / CamEFEO** | École française d'Extrême-Orient (EFEO) | Academic research and cultural preservation use | Reference typeface modeling classical *Akhar Thrah* manuscript glyph ductus; not bundled in repo. |
| **Cham Roman / Community Fonts** | Northern Illinois University (SeaSite) & Cham community archives | Freely available for non-commercial educational use | Reference typefaces evaluated for stylistic variation; not bundled in repo. |

---

## 3. Synthetic Document & Textline Datasets

Due to the acute scarcity of character-level annotated historical Cham manuscript datasets, training and benchmark datasets were synthesized programmatically from `cham_text.txt`. The repository maintains two distinct synthetic generation pipelines:

### 3.1 Pipeline A: Baseline General Generator (`scripts/generate_data.py`)
The baseline generator produces a 150,000 textline corpus partitioned into **15% evaluation/locked splits** (22,500 lines) and **85% training split** (127,500 lines):
* **Evaluation & Locked Splits (15% total / 22,500 lines)**:
  - `val_clean_short` (3% / 4,500 lines)
  - `val_clean_long` (3% / 4,500 lines)
  - `val_noisy_short` (3% / 4,500 lines)
  - `val_noisy_long` (3% / 4,500 lines)
  - `locked_test` (3% / 4,500 lines)
* **Training Split (85% total / 127,500 lines)** partitioned into 6 sub-categories:
  - `train_clean_short` (**25%** / 31,875 lines): Short clean literary phrases (1–4 tokens).
  - `train_clean_medium` (**20%** / 25,500 lines): Medium-length lines (5–8 tokens) from classical texts (*Akayet Inra Patra*, *Ariya Po Pareng*).
  - `train_clean_long` (**15%** / 19,125 lines): Long full verse hemistichs and prose sentences (9+ tokens).
  - `train_noisy_short` (**15%** / 19,125 lines): Short lines rendered with Point Spread Function (PSF) blur, noise, and lighting gradients.
  - `train_noisy_long` (**15%** / 19,125 lines): Long verse lines with severe photometric degradation and spatial elastic distortion.
  - `train_hard_examples` (**10%** / 12,750 lines): Adversarial minimal pairs (`ꨲ` vs `ꨶ`, `꩝꩝` vs `꩝`), verse numerals (1–99) with section marks (`꩑꩞`), and stacked subjoined clusters.

### 3.2 Pipeline B: Experimental V25 Generator (`scripts/generate_data_v25.py`)
The V25 generator synthesizes 150,000 textline images (140,000 train + 10,000 val) synchronized with `configs/v25_dataset_manifest.json` across 5 core pillars:
1. **Canonical Cham Literature (43.3% / 65,000 lines)**: Classical literary vocabulary (*Po Klong Garai*, 57-stanza verse).
2. **Anti-Motion Blur Base (20.0% / 30,000 lines)**: Clean base images paired with dynamic in-RAM directional motion blur ($7\times 7$ to $13\times 13$) and defocus.
3. **Inline Bilingual Code-Switching (16.7% / 25,000 lines)**: Intra-line mixed phrases with Vietnamese diacritics and Latin annotations.
4. **Stanza Numerals & Boundary Punctuation (12.0% / 18,000 lines)**: Verse numbers $1$–$99$ (`꩑꩞`..`꩙꩙꩞`) and boundary marks (`꩞`, `:`, `–`).
5. **Adversarial Minimal Pairs (8.0% / 12,000 lines)**: Micro-stroke pairs (`ꨲ` U+AA32 Vowel Sign UE vs `ꨶ` U+AA36 Medial WA, variable Double Danda `꩝꩝` spacing 2–8px, 3-tier diacritic stacks).

### 3.3 Controlled 200-Page Synthetic Document Stress-Test (`ocr-benchmark/`)
* 200 full-page synthetic documents comprising 903 ground-truth textlines across 5 difficulty levels (Standard, Aged Paper, Narrow Gap, Wavy Sinusoid, Extreme Overlap).
* Generated by `ocr-benchmark/scripts/generate_benchmark_200.py` to establish the mathematical breaking point of line segmentation and CTC recognition.

### 3.4 50-Test Stratified Diagnostic Suite (`ocr-studio/data/benchmark_50_tests_results.json`)
* 50 targeted synthetic image crops covering 10 distinct failure modes (noise, tilt, blur, diacritic confusion, low resolution).

### 3.5 Rights Status of Synthetic Outputs
Synthetically rendered document images and transcriptions generated by repository scripts inherit the rights and usage status of the underlying linguistic corpora and fonts. The synthetic generator code and annotations produced by the project maintainers are provided under the project's [MIT License](LICENSE), while redistribution of generated textline images remains subject to the rights status of the respective underlying source texts.

---

## 4. Third-Party Code & Software Dependencies

* **PaddleOCR**: Developed by Baidu Inc. Licensed under the [Apache License 2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/LICENSE).
* **fontTools**: Developed by fontTools authors. Licensed under the [MIT License](https://github.com/fonttools/fonttools/blob/main/LICENSE).
* **OpenCV / NumPy / Pillow / Albumentations**: Standard open-source computational imaging libraries licensed under BSD/Apache/MIT licenses.

---

## 5. Contact & Inquiries

For inquiries regarding corpus provenance, paleographic validation, or corrections to linguistic ground-truth:
* **Maintainer**: Phuc H. Nguyen
* **Email**: [phucsd@gmail.com](mailto:phucsd@gmail.com)
* **Project Repository**: [https://github.com/phucsd/Cham-OCR](https://github.com/phucsd/Cham-OCR)
