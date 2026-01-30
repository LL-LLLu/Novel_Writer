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
