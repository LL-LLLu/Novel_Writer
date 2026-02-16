"""End-to-end tests for the file ingestion pipeline."""

import pytest
from pathlib import Path

from novel_writer.processing.ingest import (
    ingest_file,
    ingest_directory,
    registry,
    HTMLReader,
    MarkdownReader,
)


SAMPLE_HTML_CONTENT = """\
<!DOCTYPE html>
<html>
<head><title>The Lost Garden</title></head>
<body>
<h1>Chapter 1: The Gate</h1>
<p>Eleanor pushed through the rusted iron gate, its hinges screaming in protest
after decades of disuse. Beyond lay a garden unlike any she had ever seen.
Overgrown hedges formed labyrinthine corridors that twisted and turned
in impossible patterns. Flowers of every imaginable color bloomed in wild
profusion, their petals glistening with morning dew.</p>

<h2>The First Steps</h2>
<p>"This can't be real," she whispered to herself, stepping onto a crumbling
stone path. The air was thick with the scent of jasmine and something else,
something older, something that spoke of magic and forgotten things. Birds
sang from hidden perches, their melodies weaving together into a symphony
that seemed almost deliberate in its beauty.</p>

<h1>Chapter 2: The Fountain</h1>
<p>At the garden's heart stood a marble fountain, its basin cracked but still
holding water that sparkled with an inner light. Stone figures danced around
its rim, frozen in eternal revelry. Eleanor reached out to touch the cool
marble and felt a tremor run through the stone, as if the fountain recognized
her presence and stirred from a long slumber.</p>
</body>
</html>
"""

SAMPLE_MARKDOWN_CONTENT = """\
# The Clockmaker's Secret

## Part One: Discovery

In the dusty corner of an antique shop on Rue de Rivoli, Thomas found the clock.
It was unremarkable at first glance, a simple mantel clock with a brass face
and Roman numerals. But when he picked it up, he noticed the weight was wrong.
Too heavy for its size, as though it contained something more than gears and springs.

The shopkeeper, a wizened old man with spectacles perched on the tip of his nose,
watched Thomas with keen interest. "You have good taste, monsieur," he said,
his voice carrying a faint tremor. "That clock has not been wound in fifty years."

## Part Two: The Mechanism

Thomas carried the clock to his workshop and set it on the bench beneath the
bright lamp. With practiced hands, he removed the back panel and peered inside.
What he found made his breath catch. The mechanism was unlike anything he had
ever seen in thirty years of clockmaking. Instead of a standard escapement,
there was a complex arrangement of tiny gears within gears, each one engraved
with symbols he did not recognize. At the center sat a small crystal that
pulsed with a faint blue light, as though it contained a captured star.

"Impossible," he murmured, reaching for his magnifying glass. The symbols
were not decorative. They formed a pattern, a sequence, perhaps even a message.
"""


class TestHTMLIngestion:
    """Tests for ingesting HTML files through the pipeline."""

    def test_ingest_single_html_file(self, tmp_path):
        """Ingest a single HTML file and verify tags are stripped."""
        html_file = tmp_path / "novel.html"
        html_file.write_text(SAMPLE_HTML_CONTENT, encoding="utf-8")

        content = ingest_file(html_file)

        # Tags should be stripped
        assert "<h1>" not in content
        assert "<p>" not in content
        assert "</body>" not in content
        assert "</html>" not in content

        # Actual text content should be preserved
        assert "Eleanor" in content
        assert "The Gate" in content
        assert "The Fountain" in content
        assert "rusted iron gate" in content

    def test_ingest_htm_extension(self, tmp_path):
        """Verify .htm extension is also supported."""
        htm_file = tmp_path / "novel.htm"
        htm_file.write_text(SAMPLE_HTML_CONTENT, encoding="utf-8")

        content = ingest_file(htm_file)
        assert "Eleanor" in content
        assert "<h1>" not in content

    def test_html_content_is_clean_text(self, tmp_path):
        """Verify that ingested HTML produces clean readable text."""
        html_file = tmp_path / "story.html"
        html_file.write_text(SAMPLE_HTML_CONTENT, encoding="utf-8")

        content = ingest_file(html_file)

        # Should be non-empty
        assert len(content) > 100

        # Should not contain HTML entities or tags
        assert "&nbsp;" not in content
        assert "&lt;" not in content
        assert "<" not in content or ">" not in content  # no tags

    def test_ingest_html_preserves_paragraph_text(self, tmp_path):
        """Verify paragraph text is extracted intact."""
        html_file = tmp_path / "paragraphs.html"
        html_file.write_text(SAMPLE_HTML_CONTENT, encoding="utf-8")

        content = ingest_file(html_file)

        # Key phrases from paragraphs should appear in output
        assert "rusted iron gate" in content
        assert "scent of jasmine" in content
        assert "marble fountain" in content


class TestMarkdownIngestion:
    """Tests for ingesting Markdown files through the pipeline."""

    def test_ingest_single_markdown_file(self, tmp_path):
        """Ingest a Markdown file and verify formatting is stripped."""
        md_file = tmp_path / "novel.md"
        md_file.write_text(SAMPLE_MARKDOWN_CONTENT, encoding="utf-8")

        content = ingest_file(md_file)

        # Markdown syntax should be converted to plain text
        assert "# " not in content or "## " not in content
        # Actual text should be present
        assert "Thomas" in content
        assert "Clockmaker" in content
        assert "Rue de Rivoli" in content

    def test_markdown_headings_become_text(self, tmp_path):
        """Verify that markdown headings are converted to plain text."""
        md_file = tmp_path / "headings.md"
        md_file.write_text(SAMPLE_MARKDOWN_CONTENT, encoding="utf-8")

        content = ingest_file(md_file)

        # Heading text should be present as plain text
        assert "The Clockmaker" in content
        assert "Part One" in content
        assert "Part Two" in content

    def test_markdown_content_length(self, tmp_path):
        """Verify that substantial content is preserved."""
        md_file = tmp_path / "full.md"
        md_file.write_text(SAMPLE_MARKDOWN_CONTENT, encoding="utf-8")

        content = ingest_file(md_file)

        # Content should be substantial
        assert len(content) > 200

        # Key narrative phrases should be intact
        assert "antique shop" in content
        assert "complex arrangement" in content


class TestDirectoryIngestion:
    """Tests for ingesting entire directories of files."""

    def test_ingest_directory_with_mixed_formats(self, tmp_path):
        """Ingest a directory containing both HTML and Markdown files."""
        # Create files
        html_file = tmp_path / "story1.html"
        html_file.write_text(SAMPLE_HTML_CONTENT, encoding="utf-8")

        md_file = tmp_path / "story2.md"
        md_file.write_text(SAMPLE_MARKDOWN_CONTENT, encoding="utf-8")

        results = ingest_directory(tmp_path)

        assert len(results) == 2

        # Both files should have content
        paths = [r[0] for r in results]
        contents = [r[1] for r in results]

        assert all(len(c) > 100 for c in contents)

    def test_ingest_directory_with_extension_filter(self, tmp_path):
        """Ingest only HTML files when extension filter is applied."""
        html_file = tmp_path / "story.html"
        html_file.write_text(SAMPLE_HTML_CONTENT, encoding="utf-8")

        md_file = tmp_path / "story.md"
        md_file.write_text(SAMPLE_MARKDOWN_CONTENT, encoding="utf-8")

        # Only ingest HTML
        results = ingest_directory(tmp_path, extensions=[".html"])

        assert len(results) == 1
        assert results[0][0].suffix == ".html"
        assert "Eleanor" in results[0][1]

    def test_ingest_directory_ignores_unsupported_files(self, tmp_path):
        """Verify that unsupported file types are skipped."""
        html_file = tmp_path / "story.html"
        html_file.write_text(SAMPLE_HTML_CONTENT, encoding="utf-8")

        # Create an unsupported file type
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\nval1,val2", encoding="utf-8")

        results = ingest_directory(tmp_path)

        # Only the HTML file should be ingested
        assert len(results) == 1
        assert results[0][0].suffix == ".html"

    def test_ingest_empty_directory(self, tmp_path):
        """Verify that ingesting an empty directory returns empty list."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        results = ingest_directory(empty_dir)
        assert results == []

    def test_end_to_end_ingest_and_verify(self, tmp_path):
        """Full end-to-end: create files, ingest, verify all content is valid."""
        # Create multiple files of different types
        for i in range(3):
            html_file = tmp_path / f"chapter_{i}.html"
            html_file.write_text(
                f"<html><body><h1>Chapter {i}</h1>"
                f"<p>This is chapter {i} of the great novel. "
                f"It contains important narrative elements and character development "
                f"that drive the plot forward in unexpected ways.</p></body></html>",
                encoding="utf-8",
            )

        md_file = tmp_path / "epilogue.md"
        md_file.write_text(
            "# Epilogue\n\nThe story came to an end, but the memories lived on. "
            "Every character had found their place in the grand narrative.",
            encoding="utf-8",
        )

        results = ingest_directory(tmp_path)

        assert len(results) == 4

        for path, content in results:
            # Every result should have clean text
            assert len(content) > 10
            assert "<html>" not in content
            assert "<body>" not in content
            assert "<h1>" not in content
