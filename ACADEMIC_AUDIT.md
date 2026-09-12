# Internal Documentation Audit & Reconciliation Log: Cham-OCR Repository

> **Document Type**: Internal Documentation Audit & Reconciliation Log  
> **Audited Repository**: `phucsd/Cham-OCR`  
> **Author & Project Maintainer**: Phuc H. Nguyen  
> **Audit Date**: September 2026  
> **Repository Commit**: Current Working Tree (Post-Audit Synchronized)  
> **Live Web Service**: [https://ocr.cham.asia](https://ocr.cham.asia)  
> **Research Preprint**: [RESEARCH.md](RESEARCH.md) | [https://ocr.cham.asia/research](https://ocr.cham.asia/research)

---

## 1. Executive Summary & Audit Mandate

In accordance with strict academic integrity standards, this audit report provides a transparent, verifiable record of all technical rectifications, metric reconciliations, and claim calibrations performed across the `phucsd/Cham-OCR` repository (`README.md`, `README_VI.md`, `RESEARCH.md`, `ocr-studio/research.html`, `ocr-studio/index.html`, configuration files, and Python training scripts).

### Core Audit Mandates Executed:
1. **Zero Fabrication Policy**: Every quantitative metric, parameter, and score reported across the monorepo must correspond 1-to-1 with an existing machine-readable benchmark artifact (`.json`), training log, or reproducible script output.
2. **200-Page Benchmark Re-Classification**: Elimination of any phrasing claiming the 200-page evaluation suite represents "real physical manuscript folios" or a "real-corpus benchmark". Re-classified as a **Controlled Synthetic Document Stress-Test** (200 synthetic document images, 903 textlines across 5 calibrated difficulty levels).
3. **Model Designation & SOTA Claim Downgrade**: Stripping unverified claims designating Version 25 as "Unified SOTA" or superior to Version 24. Designating **Version 24** as the **Validated Baseline (Production Standard)** and **Version 25** as an **Experimental Checkpoint (Under Active Evaluation)**.
4. **Official Unicode Standard Rectification**: Aligning all character definitions, codepoint ranges, dependent vowel signs, and punctuation marks with the official Unicode Standard (Unicode 15.0/16.0 Cham Block `U+AA00`–`U+AA5F`).
5. **Nuanced Scriptio Continua & Paleographic Scope**: Eliminating absolute claims asserting a "total absence of whitespace", distinguishing classical manuscript conventions (*scriptio continua*) from modern printed Cham orthography.
6. **Integration of Related Work**: Adding a dedicated literature review section covering seminal works in Cham epigraphy, glyph recognition, and transliteration (Nguyen et al., 2019a, 2019b; Nguyen, 2023; Nguyen et al., 2023, 2025; EFEO/CHAMDOC missions).
7. **Document Classification**: Re-labeling whitepapers from "Research Publication" to "Technical Report / Research Preprint — Not Peer Reviewed", with single-author attribution to `Phuc H. Nguyen`.
8. **Licensing & Intellectual Property**: Maintaining a clean MIT License for original source code, with a dedicated `DATA_PROVENANCE.md` governing third-party fonts (SIL Open Font License) and cultural heritage text corpora.
9. **Academic Bibliography Accuracy**: Corrected 4 heavily misattributed citations with verified authors, conference/journal venues, page ranges, and DOIs (including IEEE MAPR 2019, Springer ICIAP 2019 Workshops LNCS 11808, La Rochelle PhD Dissertation 2023, ACM HIP@ICDAR 2023, and Springer *Multimedia Tools and Applications* 2025; along with ANR CHAMDOC project description).
10. **Manuscript Preservation Volume**: Corrected manuscript preservation statements to accurately attribute large digitized archives (specifically the Southeast Asia Digital Library / SEADL at Northern Illinois University preserving 977 digitized manuscripts, >57,800 pages, alongside ~3,000 community manuscripts), clarifying that the true scarcity is in expert-annotated line-level OCR ground truth.
11. **Detector Parameter & Metric Alignment**: Aligned DBNet hyperparameters (`thresh: 0.25`, `box_thresh: 0.50`, `unclip_ratio: 1.8`, `Adam` optimizer) with active YAML configurations (`det_cham_h100.yml`) and replaced unverified standalone F1 claims with verified end-to-end evaluation metrics from `evaluation_report.json` (Levels 1–3: 100%, Level 4: 99.44%, Level 5: 61.45%).
12. **Unicode Canonical Implementation & Bug Fixes**: Refactored `generate_data.py`, `validate_unicode_source_labels.py`, and `paddleocr_cham_finetune.ipynb`: removed base consonants `ꨣꨤꨥꨦꨧꨨ` and medials `ꨴꨵ` from `VOWEL_DIACRITIC_SIGNS`; restricted `PRE_SIGNS` to left-side vowels `ꨯꨰ`; included all medials `ꨴꨵꨳꨶ` in `MEDIAL_SIGNS`; and enforced the canonical sequence in `visual_to_unicode_cluster` (`Base -> Medials RA/LA -> Medials YA/WA -> Pre-Vowels -> Dependent Vowels -> AA Lengthener -> Finals`).
13. **Data Provenance**: Created `DATA_PROVENANCE.md` detailing sources, curation, and licensing for `cham_text.txt`, digital typefaces (*Noto Sans Cham*, *EFEO Cham*), and synthetic datasets.

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
| `U+AA25` | `ꨥ` | Treated solely as base consonant | `CHAM LETTER VA` (**Dual Initial / Syllable-Final Consonant per Table 16-16**) | Implemented & verified |
| `U+AA2F` | `ꨯ` | Mislabeled as E-vowel in older draft | `CHAM VOWEL SIGN O` (**Pre-base Dependent Vowel**) | Reconciled across all documentation |
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
4. **Syllable-Final Consonant VA (`U+AA25`)**: Documented and verified per Unicode Standard Chapter 16 Table 16-16 that base consonant `ꨥ` (`CHAM LETTER VA`) functions without alteration in both initial and syllable-final positions (e.g., `ꨀꨍꨯꨱꨥ`, `ꨝꨗꨴꨭꨥ`).

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

1. **Nguyen, M.-T., Schweyer, A.-V., Le, T.-L., Tran, T.-H., & Vu, H. (2019a)**: *Preliminary Results on Ancient Cham Glyph Recognition from Cham Inscription Images*. In 2019 International Conference on Multimedia Analysis and Pattern Recognition (MAPR 2019), IEEE, pp. 1–6. DOI: 10.1109/MAPR.2019.8743540. (Evaluated HOG, NPW, and CNN-derived features with k-NN and linear SVM).
2. **Nguyen, M.-T., Schweyer, A.-V., Le, T.-L., Tran, T.-H., & Vu, H. (2019b)**: *Improving Ancient Cham Glyph Recognition Using Data Augmentation and Transfer Learning*. In International Conference on Image Analysis and Processing (ICIAP 2019 Workshops: PatReCH), Springer, Cham, LNCS 11808, pp. 115–125. DOI: 10.1007/978-3-030-30754-7_12. (Evaluated data augmentation and transfer learning from scripts of similar or the same language family to ancient stone inscription glyphs).
3. **Nguyen, T.-N. (2023)**: *Segmentation, Recognition and Indexing of Cham characters in Cham documents* (*Segmentation, reconnaissance et indexation des caractères Cham dans les documents Cham*). Ph.D. Dissertation, Université de La Rochelle, France.
4. **Nguyen, T.-N., Burie, J.-C., Le, T.-L., & Schweyer, A.-V. (2022)**: *An effective method for text line segmentation in historical document images*. In 26th International Conference on Pattern Recognition (ICPR 2022), IEEE, pp. 2686–2692. DOI: 10.1109/ICPR56361.2022.9956617.
5. **Nguyen, T.-N., Burie, J.-C., Le, T.-L., & Schweyer, A.-V. (2023)**: *A Two-Step Sequence Transformer Based Method for Cham to Latin Script Transliteration*. In Proceedings of the 7th International Workshop on Historical Document Imaging and Processing (HIP@ICDAR 2023), ACM, pp. 25–30. DOI: 10.1145/3604951.3605525.
6. **Nguyen, T.-N., Burie, J.-C., Le, T.-L., & Schweyer, A.-V. (2025)**: *Text line segmentation approach combining deep learning model and traditional image processing techniques - application to transliteration of Cham manuscripts*. *Multimedia Tools and Applications*, Springer. DOI: 10.1007/s11042-025-20602-x. (Evaluates line segmentation and transliteration across 627 images, ~8,300 lines).
7. **ANR CHAMDOC Project (2019–2024)**: *Cham Documentation (CHAMDOC)*, Project ANR-19-CE27-0018, Agence Nationale de la Recherche, France.
8. **Digitization Archives**: The Historic Cham Manuscripts of Vietnam collection in the Southeast Asia Digital Library (SEADL) contains 977 digitized manuscripts comprising more than 57,800 page scans. Cham community collections preserve an estimated 3,000 physical manuscripts across Vietnam and Cambodia.

---

## 6. Current Limitations & Open Scientific Challenges

The project openly documents four major technical limitations:

1. **Synthetic-to-Real Domain Gap**: The neural recognizer is trained predominantly on synthetic textline images rendered with bundled *Noto Sans Cham* (SIL OFL 1.1). Real manuscripts feature idiosyncratic scribal handwriting, varying pen pressure, and ink bleed that may degrade recognition accuracy in archival field settings.
2. **Absence of a Large-Scale Annotated Real Manuscript Benchmark**: To date, there is no publicly available, character-level annotated historical Cham manuscript dataset of sufficient volume. Curating such a corpus requires close, sustained collaboration with native Cham scholars, paleographers, and community elders.
3. **Extreme Biological Degradation & Epigraphy**: Inscription rubbings and severely mold-damaged palm-leaves (where character ink has substantially eroded) cannot be resolved by visual CTC sequence decoders alone without integrating language modeling priors (MLMs).
4. **Interlinear Overlap Breaking Point (Level 5)**: When sinusoidal document warping exceeds line spacing ($\text{Amplitude} > \text{Gap}$), 1D projection and horizontal bounding boxes break down, necessitating future research into 2D polygonal baseline tracking.

---

## 7. Licensing & Attribution Reconciliation

* **Codebase & Architecture**: Licensed under the **MIT License** (Copyright © 2026 Phuc H. Nguyen) covering original source code and scripts.
* **Data Provenance Document**: Created **`DATA_PROVENANCE.md`** governing third-party fonts (SIL Open Font License 1.1), historical text corpora, and synthetic datasets.
* **Corpus Material**: Explicitly designated as shared cultural heritage utilized strictly for non-commercial linguistic preservation and scientific scholarship under fair use / cultural preservation principles.

---

## 8. Verification Sign-Off Ledger

| File Audited | Changes Applied | Status | Verification Mechanism |
| :--- | :--- | :---: | :--- |
| `LICENSE` | Clean MIT License for code; pointers to DATA_PROVENANCE.md | **Verified** | Standard legal wording |
| `DATA_PROVENANCE.md` | Documented text corpora, fonts (*Noto Sans Cham*, external reference fonts), rights disclaimer, and dual-generator architecture | **Verified** | Created & linked across docs |
| `RESEARCH.md` | Reconciled title, scope, Unicode order, citations (DOIs, ICPR 2022, 2025 paper), detector parameters, manuscript counts, dual generators (Pipeline A baseline 25/20/15/15/15/10 and Pipeline B V25 5-pillar) | **Verified** | 100% matched with benchmarks |
| `ocr-studio/research.html` | Verbatim synchronization with RESEARCH.md; updated title, tag pill, abstract grammar, citations, and dual generators | **Verified** | `node --check` static JS syntax validated (Code 0) |
| `ocr-studio/index.html` | Updated model picker, Tab 1 Unicode, Tab 5 empirical benchmarks | **Verified** | `node --check` static JS syntax validated (Code 0) |
| `README.md` & `README_VI.md` | Standardized title, qualified scope to Eastern Cham, updated synthetic pipeline to dual generators | **Verified** | Synchronized across languages |
| `generate_data.py` | Fixed PRE_SIGNS (`ꨯꨰ`), MEDIAL_SIGNS (`ꨴꨵꨳꨶ`), cleaned VOWEL_DIACRITIC_SIGNS, enforced canonical Unicode order | **Verified** | Unit tests passing |
| `generate_data_v25.py` | Fixed docstring (140k/10k), renamed `MINIMAL_PAIRS_UE_WA` (`ꨲ` UE vs `ꨶ` WA), synchronized with manifest | **Verified** | Code inspection & syntax clean |
| `v25_dataset_manifest.json` | Reconciled batch size (32/card, 64 global), stages (1-12, 13-22, 23-40), and diacritic naming with `rec_cham_v25.yml` | **Verified** | JSON syntax validated |
| `TRAINING_ROADMAP_V25.md` | Aligned data distribution table to 5 pillars, updated execution plan stages to 1-12, 13-22, 23-40, fixed diacritic naming | **Verified** | Documented & reconciled |
| `test_vol_roundtrip.py` | Added comprehensive test suites for VA initial/final, medials, pre-vowels, dependent vowels, stanzas | **Verified** | 100% test pass rate |
| `validate_unicode_source_labels.py` | Fixed PRE_SIGNS, MEDIAL_SIGNS, and regex for diacritic validation | **Verified** | Script execution clean |
| `paddleocr_cham_finetune.ipynb` | Synchronized character sets and canonical cluster reordering with generate_data.py | **Verified** | JSON validated |
| `ACADEMIC_AUDIT.md` | Documentation Audit & Reconciliation Log tracking known issues and empirical uncertainties | **Verified** | Fully documented |
| `FUTURE_WORK.md` | Standalone roadmap decoupling post-V25 proposals (font fallback, directional PSF blur, manifest allocation, val freeze) from active training | **Verified** | Decoupled & documented |
| **Active V25 Experiment Freeze** | Strict freeze on `generate_data_v25.py`, `rec_cham_v25.yml`, `v25_dataset_manifest.json`, `TRAINING_ROADMAP_V25.md` during training | **Verified** | Active run frozen & untouched |

---

## 9. Residual Uncertainties & Evidence Still Needed

To maintain full transparency with academic collaborators and external reviewers, this section catalogs remaining empirical uncertainties and areas where additional field evidence is needed:

1. **Exact Text-Line Provenance in `cham_text.txt`**:
   - *Current Status*: `cham_text.txt` contains 18,829 normalized Unicode lines derived from classical epics (*Akayet Inra Patra*, *Ariya Po Pareng*), chronicles, and lexicons.
   - *Evidence Needed*: A granular line-by-line metadata mapping tracing each individual textline index back to its specific source chapter, edition, or field collection.

2. **Real Physical Manuscript Benchmark**:
   - *Current Status*: The 200-page benchmark (`ocr-benchmark/`) is a controlled synthetic stress-test modeling severe document degradations (aging, waviness, tight gaps).
   - *Evidence Needed*: A publicly released benchmark of real historical palm-leaf/kertas folios from archives (e.g. SEADL or EFEO collections) with expert-verified line-level polygon and character ground truth.

3. **Western Cham (*Cam Srak*) Linguistic Validation**:
   - *Current Status*: Recognition models (V23, V24) and synthetic generators have been evaluated primarily on Eastern Cham (*Akhar Thrah*, Unicode Block `U+AA00`–`U+AA5F`).
   - *Evidence Needed*: Formal benchmark suites testing Western Cham (*Cam Srak*) orthographic conventions, regional ligatures, and phonetic adaptations.

4. **Epigraphic Stone Carving Recognition**:
   - *Current Status*: The current DBNet and SVTR models are designed for 2D page documents (paper manuscripts and print).
   - *Evidence Needed*: Field testing and adaptation on 3D weathered stone stelae (such as those from Po Nagar, Đồng Dương, and Mỹ Sơn sanctuaries) featuring rough substrates, erosion, and oblique lighting.

5. **Historical Font Traceability in Early Experimental Models**:
   - *Current Status*: The repository bundles *Noto Sans Cham* (OFL-1.1). Earlier trial checkpoints (V21–V23) evaluated external typefaces (*EFEO Cham*, *Cham Roman*).
   - *Evidence Needed*: A formal manifest recording the exact font weights, render parameters, and seeds used for each specific legacy checkpoint.

---

**Summary of Documentation & Verification Status**: The documentation across `phucsd/Cham-OCR` has been calibrated against actual code artifacts and empirical benchmark logs. Open empirical uncertainties and ongoing data collection needs remain cataloged above to guide future research and collaborative fieldwork.
