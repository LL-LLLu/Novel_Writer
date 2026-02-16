"""
Multi-format file ingestion with registry-based reader dispatch.

Provides an abstract reader base class, concrete readers (e.g. EPUB),
and a registry that auto-selects the appropriate reader by file extension.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from loguru import logger


class IngestReader(ABC):
    """Abstract base class for file readers."""

    @abstractmethod
    def extensions(self) -> list[str]:
        """Return file extensions this reader handles (e.g. ['.epub'])."""
        ...

    @abstractmethod
    def read(self, path: Path) -> str:
        """Read a file and return its extracted text content."""
        ...

    def can_read(self, path: Path) -> bool:
        """Check if this reader can handle the given file path."""
        return path.suffix.lower() in self.extensions()


class EPUBReader(IngestReader):
    """Reader for EPUB ebook files."""

    def extensions(self) -> list[str]:
        return [".epub"]

    def read(self, path: Path) -> str:
        """
        Read an EPUB file and extract text from all HTML spine items.

        Uses ebooklib to parse the EPUB structure and BeautifulSoup
        to extract clean text from the HTML content.
        """
        from ebooklib import epub, ITEM_DOCUMENT
        from bs4 import BeautifulSoup

        logger.info(f"Reading EPUB: {path.name}")

        book = epub.read_epub(str(path))

        texts = []
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            html_content = item.get_body_content()
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text()
            if text.strip():
                texts.append(text.strip())

        result = "\n\n".join(texts)
        logger.info(f"Extracted {len(texts)} sections, {len(result)} chars from {path.name}")
        return result


class HTMLReader(IngestReader):
    """Reader for HTML files."""

    def extensions(self) -> list[str]:
        return [".html", ".htm"]

    def read(self, path: Path) -> str:
        """
        Read an HTML file and extract clean text.

        Uses BeautifulSoup to parse HTML and strip all tags.
        """
        from bs4 import BeautifulSoup

        logger.info(f"Reading HTML: {path.name}")

        html_content = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text()
        result = text.strip()

        logger.info(f"Extracted {len(result)} chars from {path.name}")
        return result


class MarkdownReader(IngestReader):
    """Reader for Markdown files."""

    def extensions(self) -> list[str]:
        return [".md"]

    def read(self, path: Path) -> str:
        """
        Read a Markdown file and extract clean text.

        Converts Markdown to HTML using the markdown library,
        then uses BeautifulSoup to extract plain text.
        """
        import markdown
        from bs4 import BeautifulSoup

        logger.info(f"Reading Markdown: {path.name}")

        md_content = path.read_text(encoding="utf-8")
        html_content = markdown.markdown(md_content)
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text()
        result = text.strip()

        logger.info(f"Extracted {len(result)} chars from {path.name}")
        return result


class MOBIReader(IngestReader):
    """Reader for MOBI ebook files."""

    def extensions(self) -> list[str]:
        return [".mobi"]

    def read(self, path: Path) -> str:
        """
        Read a MOBI file and extract text content.

        Uses the mobi library to extract HTML content from the MOBI
        file, then BeautifulSoup to extract clean text.
        """
        import mobi
        from bs4 import BeautifulSoup

        logger.info(f"Reading MOBI: {path.name}")

        tempdir, filepath = mobi.extract(str(path))
        html_content = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text()
        result = text.strip()

        logger.info(f"Extracted {len(result)} chars from {path.name}")
        return result


class ReaderRegistry:
    """Registry that maps file extensions to reader instances."""

    def __init__(self):
        self._readers: dict[str, IngestReader] = {}

    def register(self, reader: IngestReader) -> None:
        """Register a reader for all its supported extensions."""
        for ext in reader.extensions():
            self._readers[ext] = reader
            logger.debug(f"Registered reader {reader.__class__.__name__} for {ext}")

    def get_reader(self, path: Path) -> Optional[IngestReader]:
        """Return the appropriate reader for a file path, or None."""
        return self._readers.get(path.suffix.lower())

    def supported_extensions(self) -> list[str]:
        """Return all registered file extensions."""
        return list(self._readers.keys())

    def read_file(self, path: Path) -> str:
        """
        Read a file using the appropriate registered reader.

        Raises:
            ValueError: If no reader is registered for the file extension.
        """
        reader = self.get_reader(path)
        if reader is None:
            raise ValueError(
                f"Unsupported file extension '{path.suffix}'. "
                f"Supported: {self.supported_extensions()}"
            )
        return reader.read(path)


# Module-level registry pre-populated with built-in readers
registry = ReaderRegistry()
registry.register(EPUBReader())
registry.register(HTMLReader())
registry.register(MarkdownReader())
registry.register(MOBIReader())


def ingest_file(path: Path) -> str:
    """Convenience function: read a single file using the global registry."""
    return registry.read_file(path)


def ingest_directory(
    input_dir: Path,
    extensions: Optional[list[str]] = None,
) -> list[tuple[Path, str]]:
    """
    Scan a directory and read all files with supported extensions.

    Args:
        input_dir: Directory to scan for files.
        extensions: Optional list of extensions to filter by (e.g. ['.epub', '.txt']).
                    If None, uses all extensions supported by the registry.

    Returns:
        List of (path, content) tuples for successfully read files.
    """
    if extensions is None:
        extensions = registry.supported_extensions()

    results: list[tuple[Path, str]] = []

    # Collect all matching files
    files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )

    logger.info(f"Found {len(files)} files with extensions {extensions} in {input_dir}")

    for file_path in files:
        try:
            content = registry.read_file(file_path)
            results.append((file_path, content))
            logger.debug(f"Ingested: {file_path.name} ({len(content)} chars)")
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")

    logger.info(f"Successfully ingested {len(results)}/{len(files)} files")
    return results
