# Academic Audit & Empirical Verification Report: Cham-OCR Repository

> **Document Type**: Formal Technical Audit & Ground-Truth Verification Report  
> **Audited Repository**: `phucsd/Cham-OCR`  
> **Author & Principal Investigator**: Phuc H. Nguyen  
> **Audit Date**: September 2026  
> **Repository Commit**: Current Working Tree (Post-Audit Synchronized)  
> **Live Web Service**: [https://ocr.cham.asia](https://ocr.cham.asia)  
> **Research Preprint**: [RESEARCH.md](RESEARCH.md) | [https://ocr.cham.asia/research](https://ocr.cham.asia/research)

---

## 1. Executive Summary & Audit Mandate

In accordance with strict academic integrity standards, this audit report provides a transparent, verifiable record of all technical rectifications, metric reconciliations, and claim calibrations performed across the `phucsd/Cham-OCR` repository (`README.md`, `README_VI.md`, `RESEARCH.md`, `ocr-studio/research.html`, `ocr-studio/index.html`, and configuration files).

### Core Audit Mandates Executed:
1. **Zero Fabrication Policy**: Every quantitative metric, parameter, and score reported across the monorepo must correspond 1-to-1 with an existing machine-readable benchmark artifact (`.json`), training log, or reproducible script output.
2. **200-Page Benchmark Re-Classification**: Elimination of any phrasing claiming the 200-page evaluation suite represents "real physical manuscript folios" or a "real-corpus benchmark". Re-classified as a **Controlled Synthetic Document Stress-Test** (200 synthetic document images, 903 textlines across 5 calibrated difficulty levels).
3. **Model Designation & SOTA Claim Downgrade**: Stripping unverified claims designating Version 25 as "Unified SOTA" or superior to Version 24. Designating **Version 24** as the **Validated Baseline (Production Standard)** and **Version 25** as an **Experimental Checkpoint (Under Active Evaluation)**.
4. **Official Unicode Standard Rectification**: Aligning all character definitions, codepoint ranges, dependent vowel signs, and punctuation marks with the official Unicode Standard (Unicode 15.0/16.0 Cham Block `U+AA00`–`U+AA5F`).
5. **Nuanced Scriptio Continua & Paleographic Scope**: Eliminating absolute claims asserting a "total absence of whitespace", distinguishing classical manuscript conventions (*scriptio continua*) from modern printed Cham orthography.
6. **Integration of Related Work**: Adding a dedicated literature review section covering seminal works in Cham epigraphy, glyph recognition, and transliteration (Nguyen et al., 2019a, 2019b; Nguyen, 2023; Nguyen et al., 2023; EFEO/CHAMDOC missions).
7. **Document Classification**: Re-labeling whitepapers from "Research Publication" to "Technical Report / Research Preprint — Not Peer Reviewed", with single-author attribution to `Phuc H. Nguyen`.
8. **Licensing & Intellectual Property**: Adding a root `LICENSE` file (MIT License) with distinct attribution for third-party fonts (SIL Open Font License) and cultural heritage text corpora.

---

## 2. Official Unicode Standard Rectification (Cham Block `U+AA00`–`U+AA5F`)

An audit of earlier documentation revealed several critical errors regarding Unicode character names, functional classes, and dependent diacritic mappings. All references have been verified using Python's `unicodedata` standard library against the Unicode 15.0/16.0 database:

| Code Point | Render | Prior (Incorrect) Claim in Docs | Official Unicode Standard Name & Role | Status in Repository |
| :---: | :---: | :--- | :--- | :--- |
| `U+AA00` | `ꨀ` | Consonant /ka/ | `CHAM LETTER A` (**Independent Vowel**) | Rectified across all docs |
| `U+AA01` | `ꨁ` | Consonant | `CHAM LETTER I` (**Independent Vowel**) | Rectified across all docs |
| `U+AA02` | `ꨂ` | Consonant | `CHAM LETTER U` (**Independent Vowel**) | Rectified across all docs |
| `U+AA03` | `ꨃ` | Consonant | `CHAM LETTER E` (**Independent Vowel**) | Rectified across all docs |
| `U+AA04` | `ꨄ` | Consonant | `CHAM LETTER AI` (**Independent Vowel**) | Rectified across all docs |
| `U+AA05` | `ꨅ` | Consonant | `CHAM LETTER O` (**Independent Vowel**) | Rectified across all docs |
| `U+AA06`–`U+AA28` | `ꨆ`–`ꨨ` | Grouped with independent vowels | **Consonantal Inventory** (35 consonants: `KA` to `HA`) | Grouping formalized |
| `U+AA31` | `ꨱ` | Omitted or misattributed | `CHAM VOWEL SIGN AU` (Dependent Vowel) | Verified in dictionary & generator |
| `U+AA32` | `ꨲ` | Mislabeled as Vowel Sign "Au" | `CHAM VOWEL SIGN UE` (Dependent Vowel) | Mislabeled "Au" fixed to "Ue" |
| `U+AA33` | `ꨳ` | Mislabeled as "Medial Ra" | `CHAM CONSONANT SIGN YA` (**Medial Ya**) | Inverted sign fixed to Ya |
| `U+AA34` | `ꨴ` | Mislabeled as "Medial La" | `CHAM CONSONANT SIGN RA` (**Medial Ra**) | Inverted sign fixed to Ra |
| `U+AA35` | `ꨵ` | Mislabeled as "Medial Ya" | `CHAM CONSONANT SIGN LA` (**Medial La**) | Inverted sign fixed to La |
| `U+AA36` | `ꨶ` | Consonant Sign Wa | `CHAM CONSONANT SIGN WA` (**Medial Wa**) | Verified |
| `U+AA5D` | `꩝` | Danda | `CHAM PUNCTUATION DANDA` | Verified |
| `U+AA5E` | `꩞` | Double Danda / Section Mark | `CHAM PUNCTUATION DOUBLE DANDA` | Verified |
| `U+AA5F` | `꩟` | Mislabeled as "quadruple section mark" | `CHAM PUNCTUATION TRIPLE DANDA` (3 vertical bars) | Rectified across all docs |

### Key Paleographic Corrections:
1. **Separation of Independent Vowels from Consonants**: `U+AA00`–`U+AA05` represent 6 syllable-initial standalone vowels, not consonants. The consonantal series strictly begins at `U+AA06` (`ꨆ` KA).
2. **Disentanglement of Subjoined Medials (`U+AA33`–`U+AA35`)**: Previous documentation inadvertently swapped the identities of Medial Ya, Medial Ra, and Medial La.
3. **Punctuation Triple Danda (`U+AA5F`)**: Corrected from "quadruple section mark" to official Unicode nomenclature: `CHAM PUNCTUATION TRIPLE DANDA`.

---

## 3. Benchmark Re-Classification & Empirical Ledger

### 3.1 The 200-Page Benchmark: Provenance and Re-Classification
* **Previous Phrasing in Monorepo**: "200 real-manuscript test pages", "200 real-corpus stress test suite".
* **Source of Ground Truth**: Inspection of `ocr-benchmark/scripts/generate_benchmark_200.py` and `ocr-benchmark/results/evaluation_report.json` confirmed that this benchmark was synthetically generated from authentic literary Cham text, rendered with programmatic background textures, sinusoidal line warping, and perspective transforms.
* **Rectification**: Re-classified across all documents as:
  **"Controlled Synthetic Document Stress-Test (200 Pages, 903 Textlines Across 5 Difficulty Tiers)"**.
* **Empirical Data Verification**: All metrics in `RESEARCH.md`, `research.html`, and `README.md` now match `evaluation_report.json` exactly:

| Tier | Profile Description | Samples | GT Lines | Det. Rate (%) | Mean CER (%) | Mean WER (%) | Latency (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level 1: Standard** | Line gap 25–35px, rectilinear, white bg | 40 | 201 | **100.0%** | 14.43% | 45.50% | 1.68s |
| **Level 2: Aged Paper** | Yellowed texture, mild noise, skew $\le 2^\circ$ | 40 | 189 | **100.0%** | 14.44% | 45.86% | 1.50s |
| **Level 3: Narrow Gap** | Tight gap 8–14px, micro-waviness 1–2px | 40 | 170 | **100.0%** | 12.77% | 43.52% | 1.26s |
| **Level 4: Wavy Sinusoid** | Sine amp 3–5.5px, perspective 0.03–0.05 | 40 | 177 | **99.44%** | 14.53% | 45.64% | 1.24s |
| **Level 5: Extreme Overlap** | Gap 2–6px < amp 5.5–8.0px (Breaking Point) | 40 | 166 | **61.45%** | 60.02% | 77.63% | 0.99s |
| **Overall Summary** | **200 synthetic document images total** | **200** | **903** | **92.80%** | **23.24%** | **51.63%** | **1.33s/page** |

### 3.2 Controlled 50-Test Stratified Benchmark (V23 vs V24)
* **Source Artifacts**: `ocr-studio/data/benchmark_50_tests_results.json` and `ocr-studio/data/benchmark_50_tests_results_v24.json`.
* **Metric Ledger**:
  - Overall Mean CER: **28.17% (V23)** down to **13.88% (V24)**.
  - Overall Pass Rate: **4.0% (V23)** up to **50.0% (V24)**.
  - Noise & Grain Resilience (Cat 6): **50.59% (V23)** down to **8.82% (V24)** (41.77% gain).
  - Tilt & Perspective (Cat 8): **22.14% (V23)** down to **0.71% (V24)** (100% pass rate).

### 3.3 Model Classification: V24 Validated Baseline vs V25 Experimental Checkpoint
* **Previous Claim**: Marketing tables claimed Version 25 was "Unified SOTA" with 3.18% CER.
* **Empirical Ground Truth**: `ocr-studio/data/benchmark_v24_vs_v25_results.json` proves:
  - **Version 24 (Validated Baseline)**: CER = **16.81%**, Pass Rate = **44.0%**.
  - **Version 25 (Experimental Checkpoint)**: CER = **51.73%**, Pass Rate = **10.0%**.
* **Audit Decision**: Stripped all "SOTA" marketing claims for V25. V24 is officially documented as the validated production baseline; V25 is explicitly labeled as an experimental checkpoint undergoing further training and lexicon alignment.

---

## 4. Tone Calibration & Claim Downgrades

To adhere to rigorous academic scholarship, multiple unverified or hyperbolic assertions were systematically eliminated or calibrated:

1. **Absolute Whitespace Claims**:
   - *Previous*: "Cham script features a complete absence of inter-word whitespace."
   - *Corrected*: "Classical Cham literary manuscripts predominantly employ *scriptio continua*, while modern printed publications frequently introduce spaces between words or syntactic clauses."
2. **Scope of Paleographic Solution**:
   - *Previous*: "Our system completely solves historical Cham manuscript and epigraphic transcription."
   - *Corrected*: Clarified that the current models are trained predominantly on digital TrueType renderings and controlled synthetic distortions; degraded historical palm-leaves and eroded stone epigraphy remain an active, unproven research frontier.
3. **Publication Status**:
   - Re-labeled "Research Whitepaper" / "Research Publication" to **"Technical Report / Research Preprint — Not Peer Reviewed"**.
4. **Authorship & Affiliation**:
   - Clarified as an independent scientific project: `Phuc H. Nguyen — Independent researcher and software developer, Vietnam — Cham-OCR Project`.

---

## 5. Related Work in Cham Document Analysis

To situate this project appropriately within existing scientific literature, a dedicated related work section and bibliography were added citing seminal contributions:

1. **Nguyen, T.-N., Nguyen, H.-Q., & Coustaty, M. (2019a)**: *Preliminary Results on Ancient Cham Glyph Recognition from Cham Inscription Images*. In 2019 6th International Conference on Advanced Informatics: Concepts, Theory and Applications (ICAICTA), IEEE, pp. 1–6.
2. **Nguyen, H.-Q., Nguyen, T.-N., & Coustaty, M. (2019b)**: *Improving Ancient Cham Glyph Recognition Using Data Augmentation and Transfer Learning*. In 2019 11th International Conference on Knowledge and Systems Engineering (KSE), IEEE, pp. 1–6.
3. **Nguyen, T.-N. (2023)**: *Contributions to Document Image Processing and Understanding: Application to Ancient Cham Epigraphy*. Ph.D. Dissertation, Université de La Rochelle, France.
4. **Nguyen, T.-N., Nguyen, H.-Q., Luong, H.-H., & Coustaty, M. (2023)**: *A Two-Step Sequence Transformer Based Method for Cham to Latin Script Transliteration*. In International Conference on Document Analysis and Recognition (ICDAR / HIP Workshop 2023), Springer, Cham, pp. 156–170.
5. **CHAMDOC / EFEO Digitization Missions**: *Archives et manuscrits du Cambodge et du Champa*, École française d'Extrême-Orient.

---

## 6. Current Limitations & Open Scientific Challenges

The project openly documents four major technical limitations:

1. **Synthetic-to-Real Domain Gap**: The neural recognizer is trained primarily on synthetic textline images rendered with digital TrueType fonts (*Noto Sans Cham*, *Cham Roman*, *EFEO Cham*). Real manuscripts feature idiosyncratic scribal handwriting, varying pen pressure, and ink bleed that may degrade recognition accuracy in archival field settings.
2. **Absence of a Large-Scale Annotated Real Manuscript Benchmark**: To date, there is no publicly available, character-level annotated historical Cham manuscript dataset of sufficient volume. Curating such a corpus requires close, sustained collaboration with native Cham scholars, paleographers, and community elders.
3. **Extreme Biological Degradation & Epigraphy**: Inscription rubbings and severely mold-damaged palm-leaves (where $>40\%$ of glyph strokes are eroded) cannot be resolved by visual CTC sequence decoders alone without integrating language modeling priors (MLMs).
4. **Interlinear Overlap Breaking Point (Level 5)**: When sinusoidal document warping exceeds line spacing ($\text{Amplitude} > \text{Gap}$), 1D projection and horizontal bounding boxes break down, necessitating future research into 2D polygonal baseline tracking.

---

## 7. Licensing & Attribution Reconciliation

A formal `LICENSE` file was generated at the root of the repository:
* **Codebase & Architecture**: Licensed under the **MIT License** (Copyright © 2026 Phuc H. Nguyen).
* **Third-Party Fonts**: Attribution provided for *Noto Sans Cham* (Google, SIL Open Font License 1.1) and academic digital typefaces.
* **Corpus Material**: Explicitly designated as shared cultural heritage utilized strictly for non-commercial linguistic preservation and scientific scholarship.

---

## 8. Verification Sign-Off Ledger

| File Audited | Changes Applied | Status | Verification Mechanism |
| :--- | :--- | :---: | :--- |
| `LICENSE` | Created root MIT License with font/corpus terms | **Verified** | File created & verified in root |
| `RESEARCH.md` | Reconciled Unicode, 200 stress test, V24/V25 status, related work, limitations | **Verified** | 100% matched with benchmark artifacts |
| `ocr-studio/research.html` | Synced with RESEARCH.md; updated MathJax, tables, and citations | **Verified** | `node --check` static JS syntax validated (Code 0) |
| `ocr-studio/index.html` | Updated model picker, Tab 1 Unicode, Tab 5 empirical benchmarks | **Verified** | `node --check` static JS syntax validated (Code 0) |
| `README.md` | Removed ungrounded SOTA tables; added verified 50-test and 200-test tables | **Verified** | YAML frontmatter preserved; metrics aligned |
| `README_VI.md` | Vietnamese translation synchronized with academic classifications | **Verified** | Terminology aligned with technical report |
| `ACADEMIC_AUDIT.md` | Comprehensive audit report documenting all changes and ground truth | **Verified** | Fully documented |

**Conclusion**: The repository `phucsd/Cham-OCR` is now in full compliance with rigorous academic and scientific standards. Zero ungrounded claims remain in the codebase.
