# Future Work & Roadmap (Post-V25: Planned V25.1 / V26 Improvements)

> **Document Status**: Architectural Planning & Future Scope  
> **Applicable Targets**: Model Versions 25.1, 26, and subsequent pipeline iterations  
> **Relationship to Active Training**: Standalone tracking document. The active Version 25 training run on Kaggle GPU hardware is strictly frozen and operates under its existing configurations (`rec_cham_v25.yml`, `v25_dataset_manifest.json`, `generate_data_v25.py`).

---

## 1. Context & Motivation

Model Version 25 introduced critical foundational expansions: bilingual Latin/Vietnamese token support, explicit Cham numerals ($1$–$99$), stanza section marks (`꩑꩞`), and neural weight surgery (`surgery_v25_weights.py`). 

To preserve scientific reproducibility and execution determinism, the entire V25 training experiment is frozen during its active multi-stage run. The engineering proposals detailed below represent targeted improvements scheduled for subsequent iterations (V25.1 / V26) based on insights gathered during the V25 design audit.

---

## 2. Planned Pipeline Enhancements

### 2.1 Dual-Script Font Fallback & Combined Glyph Validation
* **Current Limitation in V25**:  
  When synthesizing Pillar 3 (Inline Bilingual Code-Switching), textlines containing mixed Cham and Vietnamese/Latin text may encounter font glyph coverage mismatches if rendered with a single font. While *Noto Sans Cham* contains full Unicode Cham codepoints (`U+AA00`–`U+AA5F`), its coverage of Vietnamese precomposed accented Latin characters (`à`, `ả`, `ã`, `ắ`, `ặ`, `đ`, etc.) depends on the specific font build or system fallback, potentially risking blank boxes ("tofu") during mixed-script rendering.
* **Proposed Architecture for V25.1 / V26**:
  - Implement a coordinated **Dual-Font Composite Renderer**:
    - Primary Cham font: `NotoSansCham-Regular.ttf` / `NotoSansCham-Bold.ttf`.
    - Secondary Latin/Vietnamese fallback font: `NotoSans-Regular.ttf` / `NotoSans-Bold.ttf`.
  - Validate character glyph coverage using `fontTools.ttLib` across both font instances:
    ```python
    def validate_bilingual_string(text, cham_cmap, viet_cmap):
        for char in text:
            cp = ord(char)
            if 0xAA00 <= cp <= 0xAA5F or cp in {0x25CC, 0x0020}:
                if cp not in cham_cmap:
                    return False
            else:
                if cp not in viet_cmap:
                    return False
        return True
    ```
  - Segment textlines into script-homogeneous runs during layout rendering, drawing Cham spans with the Cham font and Vietnamese spans with the Vietnamese font at matched baseline metrics.

---

### 2.2 Dedicated Directional Motion-Blur Augmentation
* **Current Limitation in V25**:  
  Pillar 2 uses dynamic in-RAM blurring with variable kernel sizes ($7\times 7$ to $13\times 13$). In some implementations, isotropic Gaussian blurring or standard box filtering is applied rather than strict directional Point Spread Function (PSF) motion vectors.
* **Proposed Architecture for V25.1 / V26**:
  - Implement a physics-grounded **Directional PSF Motion-Blur Filter**:
    ```python
    def generate_motion_blur_kernel(length, angle_degrees):
        kernel = np.zeros((length, length), dtype=np.float32)
        angle_rad = np.deg2rad(angle_degrees)
        center = length // 2
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        for i in range(-center, center + 1):
            x = int(round(center + i * dx))
            y = int(round(center + i * dy))
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1.0
        kernel_sum = kernel.sum()
        return kernel / kernel_sum if kernel_sum > 0 else np.eye(length, dtype=np.float32) / length
    ```
  - Parameterize trajectory angles $[0^\circ, 180^\circ]$ and displacement lengths (3–15px) to realistically model hand tremors during mobile phone photography in regional archival repositories.

---

### 2.3 Manifest-Driven Exact Package Allocation
* **Current Limitation in V25**:  
  Pillar selection in `generate_data_v25.py` uses probabilistic multinomial sampling based on weights defined in `v25_dataset_manifest.json` ($43.3\%$, $20.0\%$, $16.7\%$, $12.0\%$, $8.0\%$). Under finite sample generation (e.g. 140,000 lines), stochastic sampling yields slight variance around the exact target counts (e.g. $\pm 0.5\%$).
* **Proposed Architecture for V25.1 / V26**:
  - Replace stochastic multinomial draws with an **Exact Deterministic Allocation Queue**:
    - Allocate exact integer line budgets per pillar ($65,000$, $30,000$, $25,000$, $18,000$, $12,000$).
    - Distribute line quotas deterministically across worker threads.
    - Guarantee $100.0\%$ mathematical alignment between manifest specifications and generated disk labels.

---

### 2.4 Deterministic Frozen Validation Generation
* **Current Limitation in V25**:  
  In multi-stage training runs spanning multiple Kaggle sessions, dynamic synthetic validation generators may synthesize slightly different validation batches if random seeds are not strictly locked across external worker sub-processes.
* **Proposed Architecture for V25.1 / V26**:
  - Generate a permanent, pre-rendered **Frozen Validation Archive** (`val_frozen_10k.tar.gz`) stored in an immutable Kaggle Dataset or Git LFS release.
  - Package 10,000 validated images and labels across calibrated stratification tiers:
    - 4,000 Clean Literary Lines
    - 2,000 Motion Blurred Lines
    - 1,500 Bilingual Code-Switched Lines
    - 1,500 Stanza Numerals & Boundary Lines
    - 1,000 Adversarial Minimal Pairs
  - Ensure zero validation variance across all subsequent training stages, epochs, and ablation studies.

---

## 3. Implementation Schedule

| Milestone | Target Model | Key Deliverable | Prerequisite |
| :--- | :---: | :--- | :--- |
| **Stage A** | Post-V25 Analysis | Full benchmark evaluation of completed V25 checkpoint against V24 baseline across 50-test diagnostic suite and 200-page stress test. | Completion of Kaggle 40-epoch training. |
| **Stage B** | V25.1 Maintenance | Integration of Dual-Script Font Fallback and Exact Deterministic Manifest Allocation in `generate_data.py`. | Post-V25 error analysis on bilingual validation crops. |
| **Stage C** | V26 Major | Physics-grounded Directional PSF Blur and permanent immutable frozen validation dataset distribution. | Full architectural review. |

---

## 4. Completed Architectural Milestones (V24.1, Cham-DBNet Poly & V26 Milestones, Nov 2026)

Following the 40-epoch V25 training milestone (`90.96%` Sequence Accuracy, `0.72%` CER on `cham_v25_val_freeze.zip`), a parallel vocabulary-isolated pure-Cham production track (**`V24.1`: `R2.1.6 -> R3 -> R3.1 (W=320) -> R4 / R4.1 WiSE-FT (W=480, T=60)`**, `114` tokens paired with `AutoRoutingOCRWrapper` for Vietnamese code-switching) and next-generation detector/recognizer upgrades were executed:

1. **HarfBuzz OpenType Shaping Parity (`uharfbuzz` + `freetype`)**: Eliminated PIL `ImageDraw.text` pre-vowel / subjoined medial positioning discrepancies (`U+AA2F/AA30`, `U+AA33..AA36`) and verified OpenType contextual glyph substitutions (`hyphen.cham` `gid=10` vs `hyphen` `gid=118`).
2. **`PPLCNetV3` Dynamic Width Pooling Fix (`rec_lcnetv3.py`)**: Replaced hardcoded `F.adaptive_avg_pool2d(x, [1, 40])` with `F.adaptive_avg_pool2d(x, [1, int(x.shape[3]) // 2])`, restoring exact train-eval temporal stride parity ($T = W / 8 = 60$ at $W = 480$) and cutting real camera line CER on `harum_palei_01` by `-44.8%` (`4.62% -> 2.55%`).
3. **`DBNet` `MakeShrinkMap` `seed=1024` Bug Fix & Detached Diacritic Fragment Suppression**: Resolved the silent zero-GT bug where `build_dataloader(..., seed=1024)` overwrote `kwargs['epoch'] = 1024 > total_epoch`, and added geometric micro-diacritic speck suppression in `merge_line_boxes` (`100.0%` line recall across both `harum_palei_01` `25/25` and `RealScan-Gold v2` `180/180` OOD archival lines).
4. **Post-Hoc Logit Temperature Scaling ($T^*$) & Active Learning Pipeline**: Calibrated `confidence_temperature` on held-out `Probe-100` (`calibrated_confidence` in `/ocr`), hardened `/save_correction` (`normalize_unicode` + lossless PNG crops + JSONL ledger), and shipped `export_active_learning_r4.py` and `scripts/ingest_studio_to_realscan_gold.py`.
5. **`RealScan-Gold v1 & v2` (`212` Uncontaminated Lines)**: Established SHA-256-locked cross-domain benchmarks spanning authentic camera poetry (`harum_palei_01`), public-domain Wikimedia specimens, and 12 OOD archival pages rendered exclusively with unseen EFEO and held-out typefaces (`efeo_palm_leaf_agal`, `efeo_regional_folio`, `heldout_camera_book`).
6. **`Cham-DBNet` Polygon Extraction & Centerline Ribbon Rectification (`v1_poly_u1.8`, Deployed Live on `https://ocr.cham.asia`)**: Replaced `box_type='quad'` with native `DBPostProcess(box_type='poly')` (`unclip=1.8`), `merge_line_polygons`, and `rectify_polygon_line` (`vpad=0.15`, neighboring-polygon masking), raising `FullPage-Frozen 50` recall from `95.40% -> 99.58%` (**Level-5 recall `77.27% -> 100.00%`**) and halving E2E CER (`14.43% -> 7.99%`), verified live (`audit_18`: `10.16%` Doc CER, `harum_palei_01`: `3.43%` Doc CER, Commit `e46e8328`).
7. **`V26` Padding-Masked SVTR (`use_valid_mask=True`, Phase A Causal Ablation & Phase B 30-Epoch `V26-Full`)**: Implemented zero-parameter key-padding masks inside `SVTRNet` + valid-width `SELayer` pooling, confirmed causal Stress-82/Harum gain in a 6-epoch controlled ablation, and trained `V26-Full` (`64,000` train / `6,400` val) for 30 unfrozen epochs (`26,514.0s` Train Loop across 3 Kaggle Dual T4x2 stages). All 9 checkpoints + Stage-3 soups passed **`6/6` Frozen-838 Release Gates** (`100.0%` symbols, `97.14%–100.0%` hyphens, `S3 E25` `1.42%` Char CER, `E26` `2.41%` Stress-82 / `1.46%` Harum CER, `24.96%` OOD-12 E2E Doc CER); held in staging (`promoted=False`) under strict zero-relaxation gate integrity because `efeo_regional_folio` crop CER shifted (`43.0% -> 47.77%`), while an in-basin WiSE-FT sweep back toward 6-epoch `V26-Masked` revealed a `4/6` inter-basin barrier across 30k unfrozen steps.
8. **`V26.1` Frozen-Backbone + `BN`-Locked Masked-SVTR (`wise_s2_a65`, Promoted Live as `v24_1_r4` `1deee5f5ff351070`)**: Froze `backbone.conv1..blocks3` and locked their `BatchNorm2D` running statistics in `eval()` mode (`Global.freeze_bn_in_frozen_prefixes: true` in `tools/program.py`) across 16 epochs (2 Kaggle Dual T4x2 stages: `12,450.0s` Train Loop / `490.44s` Overhead / `12,940.44s` Wall-clock) with `5,801` calligraphic/bamboo-pen/elastic augmentation lines (`0` benchmark overlap). Preserved 100% linear mode connectivity with `v26_1_base`, enabling **all 6** in-basin `WiSE-FT` blends to pass **`6/6` `Frozen-838` Gates + `9/9` `RealScan-Gold v2` Promotion Checks**; winner `wise_s2_a65` (`1deee5f5ff351070`) reached `5.10%` Clean, `1.92%` Char, **`100.0%` (`280/280`)** 14 Symbols, `98.57%` Hyphens, `6.32%` FullPage, `2.73%` Stress-82, **`2.07%` Harum Crop (`2.96%` E2E)**, **`25.20%` OOD-180 Crop**, **`22.65%` OOD-12 E2E Doc CER** (`180/180`), and calibrated `T* = 0.8831` (`OOS ECE` `0.0165 -> 0.0039`).
9. **`Cham-DBNet v3` Curved-Polygon Fine-Tuning & Inverse-Shrink Discovery (`REPORTS/CHAM_DBNET_V3_EVALUATION.md`)**: Trained 20 epochs on Kaggle Dual T4x2 (`3,537.0s` Train Loop / `3,915.5s` Wall-clock) on `detector_v3_2_poly` (`1,400` train / `160` val) and derived the exact analytical inverse expansion `unclip = 3.2` for `MakeShrinkMap(shrink_ratio=0.4)` ($h_k \approx 0.16 H$). Passed **`8/9` strict guards** (`Val-160` `95.1% -> 100.0%`, `short:5` `7.37% -> 6.68%`, `audit_18` `10.14% -> 9.75%`, `a4:1-4` `6.31% -> 6.13%`, `FullPage-50` `7.82% -> 7.59%`), while strictly holding `promoted = False` (`v1_poly_u1.8` retained active) due to `shrink_ratio = 0.4` eroding a `28px` single-glyph line on `a4:5` (`248/241`).
10. **Tri-Specialist `Auto` Router v2.1 & `RealScan-Gold v3` (`434` Items)**: Upgraded `AutoRoutingOCRWrapper` (`V26.1 W=480` + `V24.1-R3.1 W=320` + multi-threaded MIT-compatible `PP-OCRv5-vi`) and built `RealScan-Gold v3` (`434` SHA-256-locked items covering **12/12 EFEO fonts**, **6/6 Held-Out weights**, curved *agal*, and bilingual Cham–Vietnamese commentary): `auto` preserves **`6/6` Gates**, cuts `OOD-v2` Crop CER to `24.55%` (`23.91%` / `24.56%` E2E across all 20 OOD pages), slashes Bilingual E2E CER from `64.53% -> 6.58%` (`28/28` lines), and averages **`5.56s/page` on CPU** (`< 15s` budget).
11. **Independent Live Production Audit (`2026-11-22`, `REPORTS/live_v26_audit_2026_11_22/`), Peer-Review Calibration & Priority A Hardening (`Auto Router v2.2`)**: Executed a 947-record black-box E2E + Playwright Chrome production audit across all 8 advertised modes on `https://ocr.cham.asia` (constituting an initial production audit and controlled baseline comparison across shared source images/transforms rather than a full real-world archival certification):
    - **Independent Confirmation of `V26.1` (`1deee5f5ff351070`)**: Achieved **`6.71%` Micro CER** on `frozen` (`-0.92` percentage points vs. `2026-11-20` `85525b0ca091acaf` `7.63%`, beating `V24` `28–0` and `R3.1` `15–6`), **`3.07%` E2E CER** (`25/25` lines) on real camera `harum_palei_01`, **`4.04%` CER** across 9 stress degradations, **`100%` (`42/42`)** on 14 mandatory symbols, **`100%` (`21/21` count; positional alignment pending Priority B)** on 7 hyphen contexts, **`0%` Latin script leakage**, and **`0/4` blank/ruled/speckle false positives**.
    - **Closed Backend API Contract & Exact Per-Line Provenance (`ocr-studio/app.py`, Priority A)**: Enforced `ALLOWED_MODEL_IDS` allowlist validation (eliminating silent HTTP `200` fallback to `v23` on unknown model IDs and deriving `recognizers[].version` strictly from on-disk manifests), guarded `POST /ocr` against missing `image` keys (HTTP `400` JSON instead of `502`), rejected invalid `method` / `reading_order` / degenerate `bbox` (`x2 <= x1` or out-of-bounds) with HTTP `400`, propagated exact `routed_version` through both Pass 1 and Pass 2 (`seen_boxes` + `deferred_copies`) of `segment_lines_advanced()` (`valley`/`adaptive`) while removing all post-OCR Unicode character-based recognizer guessing, and reclassified `v25` under `Historical Baselines` (`V25 Bilingual (Legacy)`). While `v25` matches its exported 162-token dictionary and dynamic `pool2d(stride=[3,2])` PIR graph yet exhibits `49.81%` E2E CER and `280` leaked Latin letters across `21/29` Cham pages on `frozen`, exact causal attribution among domain shift, unmasked `W=480` resize, expanded 162-token CTC space, and crop preprocessing remains open pending single-factor controlled ablations.
    - **`Auto Router v2.2` & Paired Cham Digit Control Check**: Added a low-Cham-ratio (`c_ratio < 0.35`) Latin+digit handoff branch to `AutoRoutingOCRWrapper` (`vietnamese_00` CER cut from `26.02% -> 4.88%` with all 3 lines routed to `vi`, while intra-line Cham–Latin code-switching remains open). On Cham numerals (`U+AA50..U+AA59`), paired re-testing on tight ink crops (`ImageOps.invert(...).getbbox()`, preserving historical `2026-11-22/digits` artifacts untouched) lowered `v24_1_r4` CER from `14.0% -> 6.0%` (`33.33% -> 75.0%` exact), demonstrating that `8.0` percentage points stemmed from horizontally compressing an uncropped `1100x100` white canvas into `W=480` while **`6.0%` CER (`3/12` lines) remains genuine recognition error** on dense Cham numeral strings.
12. **Priority B Controlled 5-Module Diagnostic Suite (`2026-11-22`, `REPORTS/priority_b_diagnostics_2026_11_22/PRIORITY_B_DIAGNOSTIC_REPORT_VI.md`)**: Executed `132` non-destructive live probes isolating detector vs. recognizer error, strict vs. pre-registered philological CER, genuine two-column `auto` reading order, intra-line bilingual & `V25` width isolation, and safe-border deskew:
    - **`synthetic_142` & `uncovered_ink_ratio` Root Cause**: Decomposed `synthetic_142`'s `23.84%` CER (`87` edits) into `74.71%` (`65` edits) from missed `GT[1]` (`y=[73,107]`) and `25.29%` (`22` edits, `7.48%` Matched-Line CER) on the `7` matched lines; identified that `compute_uncovered_ink()`'s `0.25 * median_line_h = 11px` vertical box dilation bridged across the missed `34px` line (`uncovered_ink_ratio = 0.0`). On `Frozen-30`, Strict Matched-Line CER is `5.80%` vs. `5.13%` Philological CER (`9.23%` genuine equivalences).
    - **Two-Column Reading Order, Intra-Line Bilingual & Safe-Border Deskew**: Verified Kendall's $\tau = 1.000$ (`5/5` 2-column pages; clean 2-col CER `69.96% -> 3.70%` under `auto`), isolated `Auto Router v2.2`'s `33.33%` CER on intra-line Cham–Vietnamese lines and `c_conf < 0.85` gate miss on `Năm 2000-2025`, falsified the single-cause `V25` `use_valid_mask` hypothesis (`41.67%` CER even at `0%` right-padding `ar48 ≈ 465`), and demonstrated safe-border deskew cutting `-11°` `synthetic_167` CER `40.0% -> 4.62%` (`harum_palei_01` `-14°` `8.27% -> 2.36%`, gated at `|θ| >= 9°`).
13. **`RealScan-Gold v4` Multi-Source Physical Archival Corpus (`2026-11-22`, `ocr-benchmark/data/realscan_gold_v4/manifest.json`, `REPORTS/REALSCAN_GOLD_V4_COLLECTION_REPORT_VI.md`)**: Harvested **`56` new authentic physical pages (`1,023` pre-segmented draft lines)** across **`25` independent manuscripts & historical printed monographs** (`57` multi-line physical pages + `7` Wikimedia specimens across **`27` independent sources** total, `synthetic_pages_included = 0`) spanning palm-leaf *agal* (`8`), traditional handmade/*Đại*/*Quyến*/Chinese paper (`23`), 20th-century cement-bag/notebook/newspaper/A4/scroll substrates (`13`), and 1889/1906 lithographed & 2-column intra-line bilingual printed books (`12`), under a strict **2-Tier Anti-Circular Ground-Truth Protocol** (`gt: null` on all `staging_review_queue` pages until human sign-off via `scripts/ingest_studio_to_realscan_gold.py --version v4 --promote-verified`) and gitignored binary storage honoring British Library EAP terms.

---

## 5. Evidence-Gated Research & Evaluation Roadmap (Priorities B -> C -> D)

1. **Priority B — Methodological & Diagnostic Isolation Before Model Retraining**:
   - **Separate Fixed-Pipeline vs. Default-Workflow Benchmarks**: Distinguish *Fixed-Pipeline Recognition Comparison* (`reading_order='rows'` on single-column pages) from *Default-User-Workflow Evaluation* (`reading_order='auto'` evaluated on genuine multi-column layouts using explicit column-grouping and line-order metrics), preserving historical benchmark outputs without overwriting.
   - **Decouple Detector vs. Recognizer Errors & Audit Missed-Line Warnings**: Report *Detector Precision / Recall / F1*, *Missed-Line Error*, and *Matched-Line Recognition CER* separately (e.g., decomposing `synthetic_142`'s `7/8` detected lines and auditing whether `uncovered_ink_ratio >= 0.12` / `possible_missed_lines` catches single missed short lines despite `95.76%` recognition confidence).
   - **Strict CER Primary + Pre-Registered Philological Normalization**: Always publish **Strict CER** first; compute **Editorial-Normalized CER** only under pre-registered philological rules (never masking `–` deletions or arbitrary `:` / whitespace errors), score hyphens and numerals via positional alignment, and test Active Learning (`/save_correction`) strictly inside isolated non-production sandboxes.
   - **10-Dimension Acceptance Matrix (`PASS / FAIL / INCONCLUSIVE / NOT TESTED`)**: Replace subjective overall percentages with explicit status tracking across API contract, provenance, frozen non-regression, digits/punctuation, detector P/R/F1, multi-column order, intra/inter-line bilingual routing, multi-source real hold-out, confidence/page-risk calibration, and isolated sequential latency.
2. **Priority C — Multi-Source Document-Split Physical Archive Hold-Out (`RealScan-Gold v4`)**:
   - Collect `>= 50–100` pages partitioned strictly by **independent document, scribe, collection, and capture condition** (preventing same-book or augmented-sibling leakage), with dual-annotator verified Cham Unicode Logical Order ground truth across palm-leaf (*agal*), *kertas* manuscripts, old printed books, and Western Cham (*Cam Srak*). Treat `geometry = 18.47%` CER as an observed benchmark result rather than an intrinsic model ceiling.
3. **Priority D — Controlled Single-Factor Ablations to Gate `V27` & `DBNet` Decisions**:
   - Run controlled crop/width/spacing ablations on residual Cham numeral errors (`6.0%` tight-crop CER), repeated ASCII digits (`1990-2000`, testing whether losses stem from crop clipping, downscaling, training scarcity, or CTC decoding), `V25` crop behavior, intra-line Cham–Latin segments, and angle-sweep deskew A/B experiments (verifying both CER and line recall to guard against clipping border Cham diacritics before enabling any pre-detection deskew or `V27` retraining).

---

© 2026 Phuc H. Nguyen. Cham-OCR Research Project. Released under the MIT License. Contact: [phucsd@gmail.com](mailto:phucsd@gmail.com).
