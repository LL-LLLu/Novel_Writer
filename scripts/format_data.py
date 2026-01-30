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
    # Ensure directories exist (optional improvement, but sticking to provided code logic mostly)
    # The user provided code has specific paths.
    format_dataset("Novel_Writer/data/processed/temp_cleaned", "Novel_Writer/data/processed/train.jsonl")
