---
title: AuraReader Pro
emoji: ⚡
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 6.20.0
python_version: '3.10'
app_file: app.py
pinned: false
license: mit
hardware: zero-gpu
---

# ⚡ AuraReader Pro

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Abhi2280/aurareaderOCR)

AuraReader Pro is a high-performance, local Vision-Language Model (VLM) document parsing and OCR dashboard. Built on top of the **baidu/Unlimited-OCR** model, it is optimized for local CPU and Apple Silicon (MPS) execution and personalized for a seamless, private document-to-text experience.

---

## Key Features

*   **⚡ Local Acceleration**: Dynamic PyTorch shims intercept CUDA calls and redirect them to macOS Metal (MPS) or CPU execution automatically.
*   **📂 Multi-Format Support**: Instantly parse standard image formats (PNG, JPG, WEBP, TIFF) and PDF files.
*   **📑 Safe PDF Splitting**: PDFs are safely processed page-by-page to keep system memory consumption low.
*   **🛠️ Prompt Presets**:
    *   **General OCR**: Standard layout-aware text transcription.
    *   **Table Extraction**: Extract layout tables directly to formatted markdown.
    *   **Math & Formulas**: High-fidelity transcription of mathematical expressions.
    *   **Layout with Bounding Boxes**: Detect structural regions and bounding coordinates.
*   **📦 Robust Cleanups**: Automatically purges temporary image folders and page slices to prevent disk leaks.
*   **🚀 Zero-Config Build**: Run the automated `setup.sh` script to set up virtual environments and launch the server.

---

## Restructured Project Architecture

```
unlimited-ocr/
├── Dockerfile          # Reproducible Docker image definition
├── app.py              # Thin launcher entrypoint
├── setup.sh            # One-click environment bootstrap script
├── core/
│   ├── model.py        # Model loader + CUDA shims/patches
│   ├── streaming.py    # Custom thread-safe QueueStreamer
│   ├── pdf.py          # PyMuPDF rendering and directory cleanups
│   └── validation.py   # File sizes and page count limits
├── api/
│   └── routes.py       # Gradio Blocks UI declaration
├── tests/
│   └── test_pdf.py     # Cleanups & validation unit tests
└── requirements.txt    # Pinned dependencies
```

---

## Setup & Running

### Option 1: One-Click Setup Script (Recommended)
Simply run the setup script in your terminal. It will detect Python 3.10+, configure your virtual environment, download necessary dependencies, and start the app:
```bash
./setup.sh
```

### Option 2: Manual Run
1. Create and source a Python 3.10 virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install standard requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the entrypoint:
   ```bash
   python app.py
   ```
4. Access the dashboard at **`http://localhost:7860/`**.
