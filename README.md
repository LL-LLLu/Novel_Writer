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

### Production Features

#### Hyperparameter Tuning

Automatically find optimal training parameters:

```bash
novel-writer tune \
  --base-model unsloth/llama-3-8b-bnb-4bit \
  --dataset data/processed/train.jsonl \
  --trials 20 \
  --output best_params.json
```

#### Model Export

Export to various deployment formats:

```bash
# Full merged model (for serving)
novel-writer export --format full --output merged_model

# GGUF for llama.cpp (CPU inference)
novel-writer export --format gguf --quantization q4_k_m --output model.gguf

# vLLM format (high-throughput serving)
novel-writer export --format vllm --output vllm_model

# ONNX format (edge deployment)
novel-writer export --format onnx --output onnx_model
```

#### Performance Profiling

Profile your pipeline and generation:

```bash
novel-writer profile --pipeline
novel-writer profile --generation --num-calls 10
```

#### Docker Deployment

Build and run with Docker:

```bash
# Build
docker build -t novel-writer:latest .

# Run pipeline
docker run -it \
  -v $(pwd)/data:/app/data \
  novel-writer:latest \
  novel-writer pipeline --clean

# Run API
docker run -d -p 8000:8000 novel-writer:latest python -m novel_writer.api

# Run dashboard
docker run -d -p 8501:8501 novel-writer:latest streamlit run ...
```

#### Multi-Format Ingestion

Ingest novels from EPUB, HTML, Markdown, and MOBI formats:

```bash
# Ingest all supported formats from a directory
novel-writer ingest --input data/raw --output data/clean

# Filter by specific formats
novel-writer ingest --input data/raw -e .epub -e .html

# Include ingestion in the full pipeline
novel-writer pipeline --ingest --clean --segment --deduplicate --filter
```

Supported formats: `.epub`, `.html`, `.htm`, `.md`, `.mobi`

#### Evaluation & Benchmarks

Evaluate generated text quality with automated metrics:

```bash
# Run benchmarks on a trained model
novel-writer evaluate --model lora_model --output benchmark_results.json
```

Metrics include: repetition rate, vocabulary diversity, average sentence length, coherence score (requires sentence-transformers), and an overall quality score.

#### DPO Preference Pairs

Generate preference pairs for Direct Preference Optimization training:

```bash
# Generate preference pairs from a dataset
novel-writer preference --input data/processed/train.jsonl --output data/processed/pairs.jsonl

# Control pair quality with minimum score difference
novel-writer preference --input data/processed/train.jsonl --min-diff 0.15 --max-pairs 1000
```

#### Curriculum Learning

Sort training data by difficulty for curriculum learning (easy-to-hard training):

```bash
# Sort easy-to-hard (default)
novel-writer curriculum --input data/processed/train.jsonl

# Sort hard-to-easy (anti-curriculum)
novel-writer curriculum --input data/processed/train.jsonl --reverse

# Control granularity with difficulty buckets
novel-writer curriculum --input data/processed/train.jsonl --buckets 5
```

### Data Preparation

Place your source novels (PDF, TXT, EPUB, HTML, Markdown, or MOBI) in `data/raw/`, then:

```bash
novel-writer pipeline --ingest --clean --segment --deduplicate --filter
```

This creates `data/processed/train.jsonl`.

### Training (Google Colab)

Several training notebooks are provided in the `training/` directory:

| Notebook | Description |
|----------|-------------|
| `Train_Llama3_Advanced.ipynb` | Standard fine-tuning with validation, checkpointing, and W&B logging |
| `DPO_Training.ipynb` | DPO alignment training using preference pairs |
| `Train_Llama3_Curriculum.ipynb` | Curriculum learning with staged easy-to-hard training |
| `Train_Llama3_MultiGPU.ipynb` | Multi-GPU distributed training with Accelerate |

1. Upload the desired notebook to [Google Colab](https://colab.research.google.com/).
2. Upload your dataset (e.g. `data/processed/train.jsonl`) to Colab.
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
