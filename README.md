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

### Advanced Features

#### Quality Pipeline

Improve model quality with our advanced processing pipeline:

```bash
# Segment novels into chapters
novel-writer segment --input data/processed/temp_cleaned

# Remove duplicates
novel-writer deduplicate --input data/processed/train.jsonl --threshold 0.85

# Filter by quality
novel-writer filter --input data/processed/train.jsonl_dedup.jsonl --keep-ratio 0.8

# Run full quality pipeline
novel-writer pipeline --segment --deduplicate --filter
```

#### Advanced Training

Use `Train_Llama3_Advanced.ipynb` for:
- Multi-epoch training (3 epochs)
- Train/validation split (90/10)
- Checkpointing and resume capability
- Early stopping (patience=3)
- Evaluation metrics
- Weights & Biases integration

#### Features

- **Chapter Segmentation**: Intelligent detection of chapter boundaries (supports English, Chinese, Japanese)
- **Deduplication**: MinHash-based near-duplicate removal
- **Quality Filtering**: Scoring based on dialogue density, vocabulary variety, text structure
- **Multi-Epoch Training**: Proper training loop with evaluation

### Data Preparation

Place your source novels (PDF or TXT) in `data/raw/`, then:

```bash
novel-writer pipeline --clean --segment --deduplicate --filter
```

This creates `data/processed/train.jsonl`.

### Training (Google Colab)

1. Upload `training/Train_Llama3_Advanced.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Upload `data/processed/train.jsonl` to Colab.
3. Run all cells.

## Output

The fine-tuned LoRA adapters are saved in the `lora_model` folder in Colab.
