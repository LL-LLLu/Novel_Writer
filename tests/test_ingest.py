import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from novel_writer.processing.ingest import (
    IngestReader,
    EPUBReader,
    HTMLReader,
    MarkdownReader,
    MOBIReader,
    ReaderRegistry,
    registry,
    ingest_file,
    ingest_directory,
)


# --- Concrete test reader for registry tests ---

class TxtTestReader(IngestReader):
    """A simple test reader that reads .txt files."""

    def extensions(self) -> list[str]:
        return [".txt"]

    def read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")


class MultiExtReader(IngestReader):
    """A test reader that handles multiple extensions."""

    def extensions(self) -> list[str]:
        return [".md", ".rst"]

    def read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")


# --- IngestReader ABC tests ---

class TestIngestReaderCanRead:
    """Tests for IngestReader.can_read() method."""

    def test_can_read_matching_extension(self):
        reader = TxtTestReader()
        assert reader.can_read(Path("novel.txt")) is True

    def test_can_read_non_matching_extension(self):
        reader = TxtTestReader()
        assert reader.can_read(Path("novel.pdf")) is False

    def test_can_read_case_insensitive(self):
        """Suffix comparison should handle uppercase extensions."""
        reader = TxtTestReader()
        assert reader.can_read(Path("novel.TXT")) is True

    def test_can_read_no_extension(self):
        reader = TxtTestReader()
        assert reader.can_read(Path("README")) is False

    def test_can_read_multiple_extensions(self):
        reader = MultiExtReader()
        assert reader.can_read(Path("doc.md")) is True
        assert reader.can_read(Path("doc.rst")) is True
        assert reader.can_read(Path("doc.txt")) is False


# --- ReaderRegistry tests ---

class TestReaderRegistry:
    """Tests for ReaderRegistry class."""

    def test_register_and_get_reader(self):
        reg = ReaderRegistry()
        reader = TxtTestReader()
        reg.register(reader)

        result = reg.get_reader(Path("file.txt"))
        assert result is reader

    def test_get_reader_returns_none_for_unknown(self):
        reg = ReaderRegistry()
        assert reg.get_reader(Path("file.xyz")) is None

    def test_register_multiple_extensions(self):
        reg = ReaderRegistry()
        reader = MultiExtReader()
        reg.register(reader)

        assert reg.get_reader(Path("file.md")) is reader
        assert reg.get_reader(Path("file.rst")) is reader

    def test_supported_extensions(self):
        reg = ReaderRegistry()
        reg.register(TxtTestReader())
        reg.register(MultiExtReader())

        exts = reg.supported_extensions()
        assert sorted(exts) == sorted([".txt", ".md", ".rst"])

    def test_supported_extensions_empty(self):
        reg = ReaderRegistry()
        assert reg.supported_extensions() == []

    def test_read_file_success(self, tmp_path):
        reg = ReaderRegistry()
        reg.register(TxtTestReader())

        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!", encoding="utf-8")

        content = reg.read_file(test_file)
        assert content == "Hello, world!"

    def test_read_file_raises_for_unsupported(self):
        reg = ReaderRegistry()
        with pytest.raises(ValueError, match="Unsupported file extension"):
            reg.read_file(Path("file.xyz"))

    def test_register_overwrites_existing(self):
        """If two readers handle same extension, last one wins."""
        reg = ReaderRegistry()
        reader1 = TxtTestReader()
        reader2 = TxtTestReader()
        reg.register(reader1)
        reg.register(reader2)

        assert reg.get_reader(Path("file.txt")) is reader2


# --- EPUBReader tests ---

class TestEPUBReader:
    """Tests for EPUBReader class."""

    def test_extensions(self):
        reader = EPUBReader()
        assert reader.extensions() == [".epub"]

    def test_can_read_epub(self):
        reader = EPUBReader()
        assert reader.can_read(Path("book.epub")) is True
        assert reader.can_read(Path("book.pdf")) is False

    @patch("ebooklib.epub.read_epub")
    @patch("bs4.BeautifulSoup")
    def test_read_extracts_text_from_spine_items(self, mock_bs_class, mock_read_epub):
        """Test that EPUBReader extracts text from EPUB HTML spine items."""
        reader = EPUBReader()

        # Create mock EPUB book
        mock_book = MagicMock()

        # Create mock spine items
        mock_item1 = MagicMock()
        mock_item1.get_body_content.return_value = b"<p>Chapter 1 text.</p>"

        mock_item2 = MagicMock()
        mock_item2.get_body_content.return_value = b"<p>Chapter 2 text.</p>"

        # Mock read_epub to return mock book
        mock_read_epub.return_value = mock_book

        # Items returned by get_items_of_type(ITEM_DOCUMENT)
        mock_book.get_items_of_type.return_value = [mock_item1, mock_item2]

        # Mock BeautifulSoup to return text extraction
        mock_soup_instance1 = MagicMock()
        mock_soup_instance1.get_text.return_value = "Chapter 1 text."

        mock_soup_instance2 = MagicMock()
        mock_soup_instance2.get_text.return_value = "Chapter 2 text."

        mock_bs_class.side_effect = [mock_soup_instance1, mock_soup_instance2]

        result = reader.read(Path("test.epub"))

        assert "Chapter 1 text." in result
        assert "Chapter 2 text." in result
        mock_read_epub.assert_called_once_with(str(Path("test.epub")))

    @patch("ebooklib.epub.read_epub")
    def test_read_handles_empty_epub(self, mock_read_epub):
        """Test that EPUBReader handles EPUB with no HTML items."""
        reader = EPUBReader()

        mock_book = MagicMock()
        mock_read_epub.return_value = mock_book
        mock_book.get_items_of_type.return_value = []

        result = reader.read(Path("empty.epub"))
        assert result == ""


# --- HTMLReader tests ---

class TestHTMLReader:
    """Tests for HTMLReader class."""

    def test_extensions(self):
        reader = HTMLReader()
        assert reader.extensions() == [".html", ".htm"]

    def test_can_read_html(self):
        reader = HTMLReader()
        assert reader.can_read(Path("page.html")) is True
        assert reader.can_read(Path("page.htm")) is True
        assert reader.can_read(Path("page.txt")) is False

    def test_read_extracts_text_from_html(self, tmp_path):
        """Test that HTMLReader strips tags and extracts text."""
        reader = HTMLReader()

        html_file = tmp_path / "test.html"
        html_file.write_text(
            "<html><body><h1>Title</h1><p>Hello, world!</p></body></html>",
            encoding="utf-8",
        )

        result = reader.read(html_file)
        assert "Title" in result
        assert "Hello, world!" in result
        assert "<p>" not in result
        assert "<h1>" not in result

    def test_read_handles_empty_html(self, tmp_path):
        """Test that HTMLReader handles empty HTML content."""
        reader = HTMLReader()

        html_file = tmp_path / "empty.html"
        html_file.write_text("", encoding="utf-8")

        result = reader.read(html_file)
        assert result == ""

    def test_read_handles_complex_html(self, tmp_path):
        """Test that HTMLReader handles HTML with nested tags and attributes."""
        reader = HTMLReader()

        html_file = tmp_path / "complex.htm"
        html_file.write_text(
            '<html><body><div class="chapter"><h2>Chapter 1</h2>'
            "<p>First paragraph.</p><p>Second paragraph.</p></div></body></html>",
            encoding="utf-8",
        )

        result = reader.read(html_file)
        assert "Chapter 1" in result
        assert "First paragraph." in result
        assert "Second paragraph." in result
        assert "class=" not in result


# --- MarkdownReader tests ---

class TestMarkdownReader:
    """Tests for MarkdownReader class."""

    def test_extensions(self):
        reader = MarkdownReader()
        assert reader.extensions() == [".md"]

    def test_can_read_md(self):
        reader = MarkdownReader()
        assert reader.can_read(Path("notes.md")) is True
        assert reader.can_read(Path("notes.txt")) is False

    def test_read_converts_markdown_to_text(self):
        """Test that MarkdownReader converts markdown via HTML to text."""
        reader = MarkdownReader()

        md_file = MagicMock(spec=Path)
        md_file.name = "test.md"
        md_file.read_text.return_value = "# Hello\n\nSome **bold** text."

        mock_markdown_mod = MagicMock()
        mock_markdown_mod.markdown.return_value = (
            "<h1>Hello</h1><p>Some <strong>bold</strong> text.</p>"
        )

        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "Hello\nSome bold text."
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup

        with patch.dict(sys.modules, {"markdown": mock_markdown_mod, "bs4": mock_bs4}):
            result = reader.read(md_file)

        assert result == "Hello\nSome bold text."
        md_file.read_text.assert_called_once_with(encoding="utf-8")
        mock_markdown_mod.markdown.assert_called_once_with(
            "# Hello\n\nSome **bold** text."
        )

    def test_read_handles_empty_markdown(self):
        """Test that MarkdownReader handles empty markdown content."""
        reader = MarkdownReader()

        md_file = MagicMock(spec=Path)
        md_file.name = "empty.md"
        md_file.read_text.return_value = ""

        mock_markdown_mod = MagicMock()
        mock_markdown_mod.markdown.return_value = ""

        mock_soup = MagicMock()
        mock_soup.get_text.return_value = ""
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup

        with patch.dict(sys.modules, {"markdown": mock_markdown_mod, "bs4": mock_bs4}):
            result = reader.read(md_file)

        assert result == ""

    def test_read_with_actual_markdown(self):
        """Test MarkdownReader with markdown content (mocked libraries)."""
        reader = MarkdownReader()

        md_file = MagicMock(spec=Path)
        md_file.name = "test.md"
        md_file.read_text.return_value = "# Title\n\nA paragraph.\n\n- item 1\n- item 2\n"

        mock_markdown_mod = MagicMock()
        mock_markdown_mod.markdown.return_value = (
            "<h1>Title</h1>\n<p>A paragraph.</p>\n<ul>\n"
            "<li>item 1</li>\n<li>item 2</li>\n</ul>"
        )

        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "Title\nA paragraph.\n\nitem 1\nitem 2"
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup

        with patch.dict(sys.modules, {"markdown": mock_markdown_mod, "bs4": mock_bs4}):
            result = reader.read(md_file)

        assert "Title" in result
        assert "A paragraph." in result
        assert "item 1" in result


# --- MOBIReader tests ---

class TestMOBIReader:
    """Tests for MOBIReader class."""

    def test_extensions(self):
        reader = MOBIReader()
        assert reader.extensions() == [".mobi"]

    def test_can_read_mobi(self):
        reader = MOBIReader()
        assert reader.can_read(Path("book.mobi")) is True
        assert reader.can_read(Path("book.epub")) is False

    def test_read_extracts_text_from_mobi(self, tmp_path):
        """Test that MOBIReader extracts text from a MOBI file."""
        reader = MOBIReader()

        # Create a real extracted HTML file for Path.read_text to work
        extracted_html = tmp_path / "extracted.html"
        extracted_html.write_text(
            "<html><body><p>MOBI content here.</p></body></html>",
            encoding="utf-8",
        )

        mock_mobi_mod = MagicMock()
        mock_mobi_mod.extract.return_value = (str(tmp_path), str(extracted_html))

        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "MOBI content here."
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup

        with patch.dict(sys.modules, {"mobi": mock_mobi_mod, "bs4": mock_bs4}):
            result = reader.read(Path("test.mobi"))

        assert result == "MOBI content here."
        mock_mobi_mod.extract.assert_called_once_with(str(Path("test.mobi")))

    def test_read_handles_extraction_error(self):
        """Test that MOBIReader propagates errors from mobi extraction."""
        reader = MOBIReader()

        mock_mobi_mod = MagicMock()
        mock_mobi_mod.extract.side_effect = Exception("Extraction failed")

        mock_bs4 = MagicMock()

        with patch.dict(sys.modules, {"mobi": mock_mobi_mod, "bs4": mock_bs4}):
            with pytest.raises(Exception, match="Extraction failed"):
                reader.read(Path("broken.mobi"))

    def test_read_handles_empty_mobi(self, tmp_path):
        """Test that MOBIReader handles MOBI with empty content."""
        reader = MOBIReader()

        extracted_html = tmp_path / "empty.html"
        extracted_html.write_text("", encoding="utf-8")

        mock_mobi_mod = MagicMock()
        mock_mobi_mod.extract.return_value = (str(tmp_path), str(extracted_html))

        mock_soup = MagicMock()
        mock_soup.get_text.return_value = ""
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup

        with patch.dict(sys.modules, {"mobi": mock_mobi_mod, "bs4": mock_bs4}):
            result = reader.read(Path("empty.mobi"))

        assert result == ""


# --- Registry with all readers tests ---

class TestRegistryAllReaders:
    """Tests that the module-level registry supports all expected extensions."""

    def test_registry_supports_all_extensions(self):
        expected = {".epub", ".html", ".htm", ".md", ".mobi"}
        actual = set(registry.supported_extensions())
        assert expected.issubset(actual), (
            f"Missing extensions: {expected - actual}"
        )

    def test_registry_html_reader(self):
        reader = registry.get_reader(Path("page.html"))
        assert isinstance(reader, HTMLReader)

    def test_registry_htm_reader(self):
        reader = registry.get_reader(Path("page.htm"))
        assert isinstance(reader, HTMLReader)

    def test_registry_md_reader(self):
        reader = registry.get_reader(Path("notes.md"))
        assert isinstance(reader, MarkdownReader)

    def test_registry_mobi_reader(self):
        reader = registry.get_reader(Path("book.mobi"))
        assert isinstance(reader, MOBIReader)


# --- Module-level registry tests ---

class TestModuleRegistry:
    """Tests for the module-level registry instance."""

    def test_registry_has_epub_reader(self):
        assert registry.get_reader(Path("book.epub")) is not None

    def test_registry_epub_reader_is_epub_reader_type(self):
        reader = registry.get_reader(Path("book.epub"))
        assert isinstance(reader, EPUBReader)

    def test_registry_supports_epub(self):
        assert ".epub" in registry.supported_extensions()


# --- ingest_file convenience function tests ---

class TestIngestFile:
    """Tests for the ingest_file convenience function."""

    def test_ingest_file_delegates_to_registry(self, tmp_path):
        """Test that ingest_file delegates to the registry."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content", encoding="utf-8")

        with patch("novel_writer.processing.ingest.registry") as mock_registry:
            mock_registry.read_file.return_value = "Test content"
            result = ingest_file(test_file)

        assert result == "Test content"
        mock_registry.read_file.assert_called_once_with(test_file)

    def test_ingest_file_raises_for_unsupported(self):
        """Test that ingest_file propagates ValueError for unsupported extensions."""
        with patch("novel_writer.processing.ingest.registry") as mock_registry:
            mock_registry.read_file.side_effect = ValueError("Unsupported file extension")
            with pytest.raises(ValueError):
                ingest_file(Path("file.xyz"))


# --- ingest_directory tests ---

class TestIngestDirectory:
    """Tests for the ingest_directory function."""

    def test_ingest_directory_reads_supported_files(self, tmp_path):
        """Test scanning a directory for supported files."""
        # Create test files
        txt_file = tmp_path / "novel.txt"
        txt_file.write_text("Novel content", encoding="utf-8")
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("PDF content", encoding="utf-8")

        # Use a registry with TxtTestReader
        with patch("novel_writer.processing.ingest.registry") as mock_registry:
            mock_registry.supported_extensions.return_value = [".txt"]
            mock_registry.read_file.return_value = "Novel content"

            results = ingest_directory(tmp_path)

        assert len(results) == 1
        assert results[0][0] == txt_file
        assert results[0][1] == "Novel content"

    def test_ingest_directory_with_specified_extensions(self, tmp_path):
        """Test scanning with specific extensions filter."""
        txt_file = tmp_path / "novel.txt"
        txt_file.write_text("Novel content", encoding="utf-8")
        md_file = tmp_path / "notes.md"
        md_file.write_text("Notes", encoding="utf-8")

        with patch("novel_writer.processing.ingest.registry") as mock_registry:
            mock_registry.read_file.return_value = "Notes"

            results = ingest_directory(tmp_path, extensions=[".md"])

        assert len(results) == 1
        assert results[0][0] == md_file

    def test_ingest_directory_empty_directory(self, tmp_path):
        """Test scanning an empty directory."""
        with patch("novel_writer.processing.ingest.registry") as mock_registry:
            mock_registry.supported_extensions.return_value = [".txt"]
            results = ingest_directory(tmp_path)

        assert results == []

    def test_ingest_directory_skips_failed_files(self, tmp_path):
        """Test that errors for individual files are handled gracefully."""
        txt_file = tmp_path / "good.txt"
        txt_file.write_text("Good content", encoding="utf-8")
        bad_file = tmp_path / "bad.txt"
        bad_file.write_text("Bad content", encoding="utf-8")

        with patch("novel_writer.processing.ingest.registry") as mock_registry:
            mock_registry.supported_extensions.return_value = [".txt"]

            def side_effect(path):
                if "bad" in path.name:
                    raise RuntimeError("Read failed")
                return "Good content"

            mock_registry.read_file.side_effect = side_effect

            results = ingest_directory(tmp_path)

        # Should only return the good file
        assert len(results) == 1
        assert results[0][0] == txt_file
        assert results[0][1] == "Good content"
