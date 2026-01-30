# Novel Writer Fine-Tuner

A toolset to fine-tune Llama-3 on your custom web novels using Google Colab and Unsloth.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your source novels (PDF or TXT) in `data/raw/`.

## Usage

### 1. Data Preparation
Clean and format your data:

```bash
# Clean artifacts and extract text
python scripts/clean_data.py

# Format into training chunks (JSONL)
python scripts/format_data.py
```

This will create `data/processed/train.jsonl`.

### 2. Training (Google Colab)
1. Download `training/Train_Llama3_Colab.ipynb`.
2. Upload it to [Google Colab](https://colab.research.google.com/).
3. In Colab, upload your generated `train.jsonl` file to the file browser.
4. Run all cells.

## Output
The fine-tuned LoRA adapters will be saved in the `lora_model` folder in Colab.
