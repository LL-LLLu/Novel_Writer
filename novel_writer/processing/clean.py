import os
import re
from pathlib import Path
from typing import List, Tuple, Optional
import pdfplumber

from loguru import logger
from ..config import Config

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_pdf(pdf_path: Path) -> bool:
    """Validate PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                raise ValidationError(f"Empty PDF: {pdf_path}")
            if len(pdf.pages) > 10000:
                raise ValidationError(f"PDF too large ({len(pdf.pages)} pages): {pdf_path}")
            return True
    except Exception as e:
        logger.error(f"PDF validation failed for {pdf_path}: {e}")
        raise

def validate_txt(txt_path: Path) -> bool:
    """Validate TXT file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if len(content) == 0:
                raise ValidationError(f"Empty TXT: {txt_path}")
            if len(content) < 100:
                raise ValidationError(f"TXT too short ({len(content)} chars): {txt_path}")
            return True
    except UnicodeDecodeError:
        raise ValidationError(f"Invalid encoding in TXT: {txt_path}")
    except Exception as e:
        logger.error(f"TXT validation failed for {txt_path}: {e}")
        raise

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF with header/footer cropping."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                width, height = page.width, page.height
                bbox = (0, height * 0.1, width, height * 0.9)
                cropped_page = page.crop(bbox=bbox)
                page_text = cropped_page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"PDF extraction failed for {pdf_path}: {e}")
        raise
    return text

def clean_text(text: str) -> str:
    """Clean extracted text."""
    # Remove page numbers
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'Page \d+', '', text)
    # Remove artifacts
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return text

def process_file(file_path: Path) -> Tuple[str, str]:
    """Process single file (path, content)."""
    logger.info(f"Processing: {file_path.name}")

    # Validate and extract
    if file_path.suffix.lower() == '.pdf':
        validate_pdf(file_path)
        content = extract_text_from_pdf(file_path)
    elif file_path.suffix.lower() == '.txt':
        validate_txt(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        logger.warning(f"Skipping unsupported file: {file_path}")
        return str(file_path), ""

    # Clean
    cleaned = clean_text(content)
    if len(cleaned) < 500:
        logger.warning(f"Cleaned text too short for {file_path}: {len(cleaned)} chars")
        return str(file_path), ""

    return str(file_path), cleaned

def clean_data(config: Config) -> List[Path]:
    """Clean all files from input directory."""
    input_path = config.data.input_dir
    output_path = config.data.temp_dir
    output_path.mkdir(parents=True, exist_ok=True)

    files = list(input_path.glob('*'))
    logger.info(f"Found {len(files)} files to process")

    from ..utils.progress import process_with_progress

    def process_func(file_path):
        try:
            file_name, cleaned = process_file(file_path)
            if cleaned:
                out_file = output_path / f"{file_path.stem}_cleaned.txt"
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned)
                return out_file
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return None

    results = process_with_progress(
        files,
        process_func,
        description="Cleaning files...",
        total=len(files)
    )

    results = [r for r in results if r is not None]
    logger.success(f"Successfully processed {len(results)} files")

    return results
