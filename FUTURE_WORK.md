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

---

## 5. Next-Generation Open Research Roadmap (`Cham-DBNet v3` & `V26.1`)

1. **Jointly Fine-Tuned Curved Polygon Detector (`Cham-DBNet v3`)**: Fine-tune DBNet head representations directly for 16-vertex curved ribbon boundaries to further lower `audit_18` (`10.14%`) beyond inference-time `v1_poly_u1.8`.
2. **OOD Regional Paleographic Regularization (`V26.1` Masked-SVTR)**: Combine `V26-Full`'s `6/6`-Gate padding-masked `W=480` architecture with regional calligraphic variation and early-backbone regularization so `OOD-180` Crop CER drops below `26.35%` without sacrificing `1.42%` Char / `2.41%` Stress / `1.46%` Harum performance.
3. **Field Manuscript Active-Learning Expansion (`RealScan-Gold v3`)**: Partner with Cham scholars and community custodians to ingest verified palm-leaf (*agal*) and bark-paper (*kertas*) folios exported from `https://ocr.cham.asia` via `scripts/ingest_studio_to_realscan_gold.py`, closing the ~43%–44% paleographic domain gap observed on unseen historical regional hands (`efeo_regional_folio`).

---

© 2026 Phuc H. Nguyen. Cham-OCR Research Project. Released under the MIT License. Contact: [phucsd@gmail.com](mailto:phucsd@gmail.com).
