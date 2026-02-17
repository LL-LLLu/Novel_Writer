"""
Scrape novel chapters from 99csw.com using DrissionPage.

Usage:
    python3 scripts/scrape_99csw.py

Fetches all chapters from the configured book URLs and saves them
as text files in data/raw/ for use with the Novel Writer pipeline.

Requirements:
    pip install DrissionPage
"""

import re
import time
from pathlib import Path

from DrissionPage import Chromium


# Book index URLs to scrape
BOOK_URLS = [
    "https://www.99csw.com/book/3546/index.htm",
    "https://www.99csw.com/book/3547/index.htm",
    "https://www.99csw.com/book/3548/index.htm",
    "https://www.99csw.com/book/5015/index.htm",
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DELAY_BETWEEN_CHAPTERS = 1.5  # seconds between chapter fetches


def wait_for_cloudflare(tab, timeout=30):
    """Wait for Cloudflare challenge to resolve."""
    for i in range(timeout):
        title = tab.title or ""
        if "请稍候" not in title and "Just a moment" not in title and "Cloudflare" not in title:
            return True
        time.sleep(1)
    return False


def get_book_info(tab, index_url):
    """Navigate to book index page and extract book title + chapter links."""
    print(f"\nFetching index: {index_url}")
    tab.get(index_url)
    time.sleep(1)

    if not wait_for_cloudflare(tab):
        print("  ERROR: Cloudflare challenge did not resolve")
        return "", []

    title = tab.title or ""
    # Clean title: remove site name suffix
    title = re.sub(r"_在线阅读.*$", "", title)
    title = re.sub(r"_[^_]+$", "", title)  # Remove author suffix
    print(f"  Book: {title}")

    # Extract book ID from URL to filter chapter links
    book_id_match = re.search(r"/book/(\d+)/", index_url)
    book_id = book_id_match.group(1) if book_id_match else ""

    # Get chapter links (filter to this book's chapters only)
    all_links = tab.eles(f"css:a[href*='/book/{book_id}/']")
    chapter_links = []
    seen_hrefs = set()

    for link in all_links:
        href = link.attr("href") or ""
        text = link.text.strip()

        # Only chapter pages (not index), skip duplicates and empty text
        if re.search(r"/book/\d+/\d+\.htm", href) and href not in seen_hrefs and text:
            seen_hrefs.add(href)
            chapter_links.append({"href": href, "text": text})

    print(f"  Found {len(chapter_links)} chapters")
    return title, chapter_links


def get_chapter_content(tab, chapter_url):
    """Navigate to a chapter page and extract the text content."""
    tab.get(chapter_url)
    time.sleep(0.5)

    # Wait briefly for Cloudflare on chapter pages
    wait_for_cloudflare(tab, timeout=10)

    # Try common content selectors
    content = ""
    for selector in ["#content", ".content", ".neirong", "#main-content", ".chapter-content"]:
        try:
            el = tab.ele(f"css:{selector}")
            if el:
                text = el.text.strip()
                if len(text) > 100:
                    content = text
                    break
        except Exception:
            continue

    # Fallback: find largest text block
    if len(content) < 100:
        try:
            paragraphs = tab.eles("css:p")
            if len(paragraphs) > 3:
                content = "\n\n".join(p.text.strip() for p in paragraphs if p.text.strip())
        except Exception:
            pass

    if len(content) < 100:
        try:
            divs = tab.eles("css:div")
            best = ""
            for div in divs:
                text = div.text.strip()
                if len(text) > len(best) and len(text) > 200:
                    best = text
            content = best
        except Exception:
            pass

    return content


def scrape_book(tab, index_url, output_dir):
    """Scrape all chapters from a single book."""
    title, chapter_links = get_book_info(tab, index_url)

    if not chapter_links:
        print(f"  ERROR: No chapter links found for {index_url}")
        return 0

    # Create output filename
    safe_title = re.sub(r'[^\w\u4e00-\u9fff]+', '_', title)[:80].strip('_')
    if not safe_title:
        book_id = re.search(r'/book/(\d+)/', index_url)
        safe_title = f"book_{book_id.group(1)}" if book_id else "unknown_book"

    book_file = output_dir / f"{safe_title}.txt"
    print(f"  Output: {book_file}")

    # Scrape each chapter
    all_text = []
    for i, link in enumerate(chapter_links):
        chapter_url = link["href"]
        chapter_title = link["text"]

        # Ensure full URL
        if not chapter_url.startswith("http"):
            chapter_url = f"https://www.99csw.com{chapter_url}"

        print(f"  [{i+1}/{len(chapter_links)}] {chapter_title[:50]}...", end=" ", flush=True)

        try:
            content = get_chapter_content(tab, chapter_url)
            if content and len(content) > 50:
                all_text.append(f"{chapter_title}\n\n{content}")
                print(f"({len(content):,} chars)")
            else:
                print("(skipped - too short)")
        except Exception as e:
            print(f"(error: {e})")

        time.sleep(DELAY_BETWEEN_CHAPTERS)

    # Write all chapters to file
    if all_text:
        separator = "\n\n" + "=" * 40 + "\n\n"
        full_text = separator.join(all_text)
        book_file.write_text(full_text, encoding="utf-8")
        print(f"\n  Saved {len(all_text)} chapters ({len(full_text):,} chars) -> {book_file.name}")
    else:
        print(f"\n  WARNING: No content extracted for {title}")

    return len(all_text)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Books to scrape: {len(BOOK_URLS)}")

    browser = Chromium()
    tab = browser.latest_tab

    total_chapters = 0
    for url in BOOK_URLS:
        try:
            count = scrape_book(tab, url, OUTPUT_DIR)
            total_chapters += count
        except Exception as e:
            print(f"\n  FAILED to scrape {url}: {e}")
            import traceback
            traceback.print_exc()

    browser.quit()

    print(f"\n{'='*60}")
    print(f"Done! Scraped {total_chapters} total chapters from {len(BOOK_URLS)} books.")
    print(f"Files saved to: {OUTPUT_DIR}")
    if total_chapters > 0:
        print(f"\nNext step: run the pipeline:")
        print(f"  novel-writer pipeline --clean --segment --deduplicate --filter")


if __name__ == "__main__":
    main()
