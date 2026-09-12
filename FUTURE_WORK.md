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

© 2026 Phuc H. Nguyen. Cham-OCR Research Project. Released under the MIT License.
