# Novel Writer LLM Fine-Tuning Design

## Overview
A project to fine-tune an LLM (Llama-3-8B) on a custom dataset of web novels to write in a specific style. The system supports a Hybrid interaction model (Instruction + Completion).

## Goals
- **Style Mimicry:** Learn specific writing styles from provided PDF/TXT novels.
- **Hybrid Interaction:** Support both "Continue this story" (Completion) and "Write a scene where..." (Instruct).
- **Colab Efficiency:** Optimized for Google Colab (Free/Pro) using Unsloth and QLoRA.

## Architecture

### 1. Data Pipeline
**Inputs:**
- Raw Text files (`.txt`)
- PDF files (`.pdf`)

**Processing (`scripts/clean_data.py`):**
- **Ingestion:** `pdfplumber` for robust PDF text extraction (cropping headers/footers).
- **Cleaning:** Regex-based artifact removal (page numbers, URLs, chapter junk).
- **Segmentation:** Heuristic parsing of "Chapter N" to respect narrative boundaries.

**Formatting (`scripts/format_data.py`):**
- **Format:** JSONL compatible with Unsloth/HuggingFace.
- **Completion Data:** Sliding window chunks (2048-4096 tokens).
- **Instruct Data:** (Optional/Future) Synthetic pairs of Summary -> Text. Initially focusing on Completion format with Instruct templates.

### 2. Fine-Tuning Strategy
**Environment:** Google Colab (T4 GPU).
**Stack:**
- **Library:** `unsloth` (Fast, low memory).
- **Base Model:** `Llama-3-8B-Instruct`.
- **Method:** QLoRA (4-bit quantization, LoRA rank=32).

**Hyperparameters:**
- Context Window: 8192 tokens.
- Epochs: 1-3 (depending on dataset size).
- Learning Rate: Standard QLoRA defaults (2e-4).

### 3. Project Structure
```text
Novel_Writer/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── plans/
├── scripts/
│   ├── clean_data.py
│   └── format_data.py
├── training/
│   └── Train_Llama3_Colab.ipynb
├── requirements.txt
└── README.md
```

## Next Steps
1. Set up directory structure.
2. Implement `clean_data.py`.
3. Implement `format_data.py`.
4. Generate the `Train_Llama3_Colab.ipynb`.
