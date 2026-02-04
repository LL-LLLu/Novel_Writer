# Novel Writer Fine-Tuner

A professional toolset to fine-tune Llama-3 on your custom web novels using Google Colab and Unsloth.

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Novel_Writer

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Configuration

Create or modify `config.yaml`:

```yaml
data:
  input_dir: data/raw
  output_dir: data/processed
  chunk_size: 4000
  overlap: 500

training:
  base_model: unsloth/llama-3-8b-instruct-bnb-4bit
  epochs: 3
  learning_rate: 0.0002
```

## Usage

### CLI Commands

```bash
# Clean raw data
novel-writer clean --input data/raw --output data/clean

# Format training data
novel-writer format --input data/clean --output data/train.jsonl

# Run full pipeline
novel-writer pipeline --clean

# Verbose logging
novel-writer -v pipeline
```

### Data Preparation

Place your source novels (PDF or TXT) in `data/raw/`, then:

```bash
novel-writer pipeline --clean
```

This creates `data/processed/train.jsonl`.

### Training (Google Colab)

1. Upload `training/Train_Llama3_Colab.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Upload `data/processed/train.jsonl` to Colab.
3. Run all cells.

## Output

The fine-tuned LoRA adapters are saved in the `lora_model` folder in Colab.
