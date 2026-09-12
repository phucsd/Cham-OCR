**English** | [Tiếng Việt](README_STUDIO_VI.md)

# Cham OCR Studio: Paleographic Transcription & Diagnostic Workbench

Cham OCR Studio is an interactive paleographic document analysis and transcription environment designed for digitized historical Cham manuscripts (*Akhar Thrah* and *Cam Srak*). It features deep neural text recognition (PP-OCRv4 SVTR), Indic Line Segmentation (PaddleOCR DBNet + Valley-Cut Heuristics), continuous ground-truth editing, and a built-in virtual paleographic keyboard.

> 🌟 **Live Interactive Web Demo**: **[https://ocr.cham.asia](https://ocr.cham.asia)** (mirrored on [Hugging Face Spaces](https://huggingface.co/spaces/phucsd/cham-ocr-studio))

---

## 📁 Studio Directory Structure

```
ocr-studio/
├── README_STUDIO.md                    # This documentation (English)
├── README_STUDIO_VI.md                 # Vietnamese documentation
├── app.py                              # HTTP Server backend & multi-crop orchestrator
├── index.html                          # Frontend UI (Claude Warm Light Theme, Google Sans Flex)
├── start_studio.py                     # Quick-launch helper script
├── PaddleOCR/                          # Standalone PaddleOCR inference library
└── data/                               # Inference data & model assets
    ├── ocr_corrections.txt             # Verified textline corrections log
    ├── cham_dict_v*.txt                # Character dictionaries for model versions
    └── output/                         # Exported model checkpoints (inference format)
        ├── rec_cham_inference_v24/
        ├── rec_cham_inference_v23/
        ├── rec_cham_inference_v22/
        └── rec_cham_inference_v21/
```

---

## ⚡ System Requirements & Installation

1. Install project dependencies from the repository root:
   ```bash
   pip install -r requirements.txt
   ```
   *Note*: On modern Python runtimes (Python 3.10+ / 3.13+), NumPy 2.x backward-compatibility monkeypatching is automatically applied.

2. Verify that inference models are present in `ocr-studio/data/output/` alongside the corresponding dictionaries in `ocr-studio/data/`.

---

## 🚀 Launching the Studio

Run the application from the repository root or from within the `ocr-studio` directory:

```bash
python ocr-studio/app.py
```

Alternatively, use the quick-launch script:
```bash
python ocr-studio/start_studio.py
```

The server automatically binds to an open port (defaulting to `7860`, `8080`, `8081`...):
```
======================================================================
🚀 Cham OCR Diagnostic Studio is running at: http://localhost:7860
📁 Corrections will be saved to: ocr-studio/data/ocr_corrections.txt
======================================================================
```

Open your browser and visit `http://localhost:7860`.

---

## ⌨️ Reviewer Keybindings

| Key | Action | Description |
| :---: | :--- | :--- |
| **A** | **Approve** | Marks current line as verified ($100\%$ confidence) and moves to the next line. |
| **M** | **Merge Below** | Merges selected line with the subsequent line below into a unified crop. |
| **S** | **Split Line** | Splits selected line into two equal halves. |
| **W** | **Flag Error** | Flags current line as inaccurate ($0\%$ confidence) for retraining. |
| **Alt + K** | **Virtual Keyboard** | Toggles the Cham paleographic virtual keyboard drawer. |
| **Arrow Up / Down** | **Navigation** | Traverses previous and subsequent textlines. |
| **Double Click** | **In-Place Edit** | Activates inline text editing for the target line. |
