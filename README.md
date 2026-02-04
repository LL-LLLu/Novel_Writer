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

#### Instruction Tuning

Generate instruction-response pairs from completion data:

```bash
novel-writer instruct --input data/processed/train.jsonl --output data/processed/train_instruct.jsonl
```

#### Inference

Generate text with your trained model:

```bash
# CLI
novel-writer generate --prompt "The hero drew his sword..." --max-tokens 1000

# API server
novel-writer api
# Then POST to http://localhost:8000/generate
```

#### Style Mixing

Blend multiple trained styles:

```bash
novel-writer mix \
  --loras style_fantasy_lora style_romance_lora \
  --weights 0.7 0.3 \
  --output mixed_style_model
```

#### Data Dashboard

Launch interactive dashboard:

```bash
novel-writer dashboard
```

View chunk distribution, dialogue analysis, quality scores, and more.

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

## API Reference

### POST /generate

Generate novel continuation.

**Request:**
```json
{
  "prompt": "The hero entered the dark cave...",
  "max_tokens": 500,
  "temperature": 0.8,
  "top_p": 0.9
}
```

**Response:**
```json
{
  "generated_text": "...",
  "prompt_length": 35,
  "generated_length": 512
}
```
