# Novel Writer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a data processing pipeline to clean and format novels for LLM fine-tuning, and create a Google Colab notebook for training Llama-3 with Unsloth.

**Architecture:** Python scripts for local data processing (cleaning, segmentation, formatting) and a Jupyter Notebook for the cloud training workflow.

**Tech Stack:** Python, pdfplumber, unsloth (in Colab), pandas, jsonl.

### Task 1: Environment & Directory Setup

**Files:**
- Create: `Novel_Writer/requirements.txt`
- Create: `Novel_Writer/.gitignore`
- Create: `Novel_Writer/data/raw/.gitkeep`
- Create: `Novel_Writer/data/processed/.gitkeep`
- Create: `Novel_Writer/scripts/__init__.py`

**Step 1: Create requirements.txt**
Define dependencies for local data processing.
```text
pdfplumber==0.10.3
pandas==2.2.0
tqdm==4.66.1
```

**Step 2: Create .gitignore**
Ignore data files and virtual envs.
```text
venv/
__pycache__/
*.pyc
.DS_Store
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
```

**Step 3: Create directory structure**
Run:
```bash
mkdir -p Novel_Writer/data/raw Novel_Writer/data/processed Novel_Writer/scripts Novel_Writer/training
touch Novel_Writer/data/raw/.gitkeep Novel_Writer/data/processed/.gitkeep Novel_Writer/scripts/__init__.py
```

**Step 4: Commit**
```bash
git add .
git commit -m "chore: setup project structure and dependencies"
```

### Task 2: Data Cleaning Script (`clean_data.py`)

**Files:**
- Create: `Novel_Writer/scripts/clean_data.py`

**Step 1: Implement PDF Extraction & Cleaning**
Create a script that:
1. Iterates through `data/raw`.
2. Uses `pdfplumber` to extract text, cropping headers/footers (top/bottom 10%).
3. Cleans text (removes page numbers, cleans whitespace).
4. Saves cleaned text to `data/processed/temp_cleaned`.

```python
import os
import re
import pdfplumber
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Crop header/footer (approx 10% top/bottom)
            width, height = page.width, page.height
            bbox = (0, height * 0.1, width, height * 0.9)
            cropped_page = page.crop(bbox=bbox)
            page_text = cropped_page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean_text(text):
    # Remove "Page X of Y" patterns
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'Page \d+', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_files(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file_path in input_path.glob('*'):
        print(f"Processing {file_path.name}...")
        content = ""
        if file_path.suffix.lower() == '.pdf':
            content = extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        if content:
            cleaned = clean_text(content)
            out_file = output_path / f"{file_path.stem}_cleaned.txt"
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(cleaned)

if __name__ == "__main__":
    process_files("data/raw", "data/processed/temp_cleaned")
```

**Step 2: Commit**
```bash
git add scripts/clean_data.py
git commit -m "feat: add data cleaning script"
```

### Task 3: Data Formatting Script (`format_data.py`)

**Files:**
- Create: `Novel_Writer/scripts/format_data.py`

**Step 1: Implement JSONL Formatting**
Create a script that:
1. Reads cleaned text files.
2. Chunks them into overlapping windows (e.g., 2048 chars).
3. Formats as Alpaca/Unsloth JSONL style: `{"instruction": "Continue the story.", "input": "", "output": "chunk_text"}`.
4. Saves to `data/processed/train.jsonl`.

```python
import json
from pathlib import Path

def create_chunks(text, chunk_size=4000, overlap=500):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def format_dataset(input_dir, output_file):
    input_path = Path(input_dir)
    data = []
    
    for file_path in input_path.glob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            chunks = create_chunks(text)
            for chunk in chunks:
                entry = {
                    "instruction": "Continue the narrative in the established style.",
                    "input": "",
                    "output": chunk
                }
                data.append(entry)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')
    print(f"Saved {len(data)} entries to {output_file}")

if __name__ == "__main__":
    format_dataset("data/processed/temp_cleaned", "data/processed/train.jsonl")
```

**Step 2: Commit**
```bash
git add scripts/format_data.py
git commit -m "feat: add jsonl formatting script"
```

### Task 4: Colab Training Notebook

**Files:**
- Create: `Novel_Writer/training/Train_Llama3_Colab.ipynb`

**Step 1: Create Notebook Content**
Generate a JSON-format `.ipynb` file. Since writing raw JSON for notebooks is error-prone, we will write a Python script `scripts/generate_notebook.py` that generates it using `nbformat`, run it, then delete the generator.

**Script content (`scripts/generate_notebook.py`):**
```python
import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# Fine-Tune Llama-3 for Novel Writing
This notebook uses Unsloth to fine-tune Llama-3-8B on your custom novel dataset."""

code_install = """%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.26" "trl<0.8.0" peft accelerate bitsandbytes"""

code_setup = """from unsloth import FastLanguageModel
import torch

max_seq_length = 8192
dtype = None # Auto detection
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)"""

code_data = """from datasets import load_dataset
dataset = load_dataset("json", data_files="train.jsonl", split="train")

alpaca_prompt = \"\"\"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}\"\"\"

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)"""

code_train = """from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
trainer.train()"""

code_save = """model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_install),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_code_cell(code_data),
    nbf.v4.new_code_cell(code_train),
    nbf.v4.new_code_cell(code_save)
]

with open('Novel_Writer/training/Train_Llama3_Colab.ipynb', 'w') as f:
    nbf.write(nb, f)
```

**Step 2: Generate and Commit**
Run the script to create the `.ipynb`, then remove the script.
```bash
pip install nbformat
python3 scripts/generate_notebook.py
rm scripts/generate_notebook.py
git add training/Train_Llama3_Colab.ipynb
git commit -m "feat: generate colab training notebook"
```

### Task 5: Documentation

**Files:**
- Modify: `Novel_Writer/README.md`

**Step 1: Write Instructions**
Add clear instructions on how to:
1. Put files in `data/raw`.
2. Run `pip install -r requirements.txt`.
3. Run `python scripts/clean_data.py`.
4. Run `python scripts/format_data.py`.
5. Upload `train.jsonl` to Colab and run the notebook.

**Step 2: Commit**
```bash
git add README.md
git commit -m "docs: add usage instructions"
```
