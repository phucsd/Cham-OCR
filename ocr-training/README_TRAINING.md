**English** | [Tiếng Việt](README_TRAINING_VI.md)

# Fine-Tuning PaddleOCR PP-OCRv4 for Eastern Cham Script (Akhar Thrah)

This sub-project provides automated pipelines and tooling to synthesize large-scale labeled Cham datasets, configure training parameters, perform character-mapped weight surgery, and fine-tune **PaddleOCR PP-OCRv4** recognition and detection models on **Kaggle GPU Notebooks** (Dual Nvidia Tesla T4) and **Lightning Cloud** (A100 SXM4). The current validated scope focuses on Unicode Eastern Cham (*Akhar Thrah*), with Western Cham (*Cam Srak*) and epigraphy planned as future extensions.

> 🌟 **Live Interactive Web Demo**: **[https://ocr.cham.asia](https://ocr.cham.asia)** (mirrored on [Hugging Face Spaces](https://huggingface.co/spaces/phucsd/cham-ocr-studio))

---

## 📁 Training Sub-Project Structure

```
ocr-training/
├── README_TRAINING.md                  # This documentation (English)
├── README_TRAINING_VI.md               # Vietnamese documentation
├── paddleocr_cham_finetune.ipynb      # Main execution notebook for Kaggle GPU
├── configs/                            # Training YAML configs (v24, v25, DBNet)
├── scripts/                            # Synthesis, weight surgery & watchdog tools
│   ├── generate_data.py               # Synthetic generator with fontTools cmap validation
│   ├── configure_training.py          # Dynamic hardware detection & YAML adapter
│   ├── surgery_v25_weights.py         # Character-mapped neural weight transfer
│   └── kaggle_auth.py                 # Secure Kaggle credential loader
├── data/                               # Input data assets
│   ├── fonts/                          # TrueType/OpenType fonts for Cham rendering
│   └── corpus/                         # Classical Cham textual corpora
└── tests/                              # Benchmark validation test suites
```

---

## 🚀 Kaggle GPU Quickstart Protocol

To take full advantage of dual GPU acceleration (Nvidia Tesla T4x2) under Kaggle's quota:

### 1. Kaggle Authentication
Ensure Kaggle API authentication is initialized securely using `.env` or the project helper:
```python
from scripts.kaggle_auth import init_kaggle_auth
init_kaggle_auth()  # Loads securely from environment variables
```

### 2. Launch Distributed Multi-GPU Training
Leverage Paddle distributed execution across both GPUs:
```bash
python3 -m paddle.distributed.launch --gpus '0,1' tools/train.py -c configs/rec_cham_v25.yml
```

### 3. 12-Hour Session Limits & 3-Stage Checkpoints
Kaggle enforces a strict 12-hour timeout per kernel execution. Training is partitioned into 3 consecutive safe stages per `stage_end_epoch` in `configs/rec_cham_v25.yml`:
- **Stage 1**: Epochs 1 – 12 (~7.5h) $\to$ Stops at epoch 12 (`stage_end_epoch: 12`) and exports `latest` and `best_accuracy` checkpoints.
- **Stage 2**: Epochs 13 – 22 (~6.5h) $\to$ Stops at epoch 22 (`stage_end_epoch: 22`) and resumes optimizer and LR state seamlessly via `Global.checkpoints`.
- **Stage 3**: Epochs 23 – 40 (~11.0h) $\to$ Completes training and exports inference models.

---

## 🛠️ Key Technical Modules

### 1. Missing/Tofu Glyph Filtering (`fontTools` Validation)
`scripts/generate_data.py` inspects the character map (`cmap`) table of each candidate font. Any line containing codepoints unsupported by the active font is discarded automatically, preventing missing character placeholders (tofu boxes) from contaminating the dataset.

### 2. On-the-Fly Dynamic Motion Blur
Photometric noise, Gaussian defocus, and directional motion blur (kernels $7\times 7$ to $13\times 13$, $\theta \in [0^\circ, 180^\circ]$ at $35\%$ probability) are injected dynamically in RAM (`RecAug`) during batch construction, preserving pristine renders on disk and eliminating double-blur degradation.

### 3. Character-Mapped Weight Surgery
`scripts/surgery_v25_weights.py` transfers converged weights from baseline models to newer architectures by matching Unicode characters directly instead of relying on token indices, preventing tensor dimension mismatch crashes during vocabulary expansion.
