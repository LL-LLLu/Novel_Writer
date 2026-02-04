import json
from pathlib import Path
from typing import List, Tuple
import hashlib

from datasketch import MinHash, MinHashLSH
from ..utils.logger import setup_logger

logger = setup_logger()

class TextDeduplicator:
    """Remove duplicate and near-duplicate text chunks."""

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.85,
        ngram_size: int = 5
    ):
        """
        Args:
            num_perm: Number of permutations for MinHash
            threshold: Jaccard similarity threshold (0-1)
            ngram_size: Size of character n-grams
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.lsh = MinHashLSH(
            threshold=threshold,
            num_perm=num_perm
        )
        self.seen_hashes = set()

    def create_minhash(self, text: str) -> MinHash:
        """Create MinHash from text."""
        m = MinHash(num_perm=self.num_perm)

        # Character n-grams
        for i in range(len(text) - self.ngram_size + 1):
            ngram = text[i:i + self.ngram_size]
            m.update(ngram.encode('utf-8'))

        return m

    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate of previously seen text."""
        minhash = self.create_minhash(text)

        # Check exact hash first (fast)
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.seen_hashes:
            return True

        # Check LSH buckets
        nearby = self.lsh.query(minhash)
        if nearby:
            # Found similar text - assume duplicate
            return True

        # Not duplicate - add to index
        self.lsh.insert(text_hash, minhash)
        self.seen_hashes.add(text_hash)

        return False

    def deduplicate_texts(
        self,
        texts: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """
        Remove duplicates from list of (id, text) tuples.

        Returns:
            Filtered list of unique texts
        """
        unique_texts = []
        duplicates = 0

        for text_id, text in texts:
            if self.is_duplicate(text):
                duplicates += 1
                logger.debug(f"Duplicate found: {text_id[:50]}...")
            else:
                unique_texts.append((text_id, text))

        logger.info(
            f"Removed {duplicates} duplicates "
            f"({duplicates/len(texts)*100:.1f}%)"
        )

        return unique_texts

    def deduplicate_jsonl(
        self,
        input_file: Path,
        output_file: Path
    ) -> int:
        """
        Deduplicate JSONL dataset.

        Returns:
            Number of unique entries
        """
        entries = []
        total = 0

        # Read entries
        logger.info(f"Reading: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                text = entry.get('output', '')
                entries.append((line, text))
                total += 1

        # Deduplicate
        logger.info("Deduplicating...")
        unique = self.deduplicate_texts(entries)

        # Write unique entries
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for line, _ in unique:
                f.write(line)
                f.write('\n')

        logger.success(
            f"Wrote {len(unique)} unique entries "
            f"(removed {total - len(unique)} duplicates)"
        )

        return len(unique)

def deduplicate_dataset(
    input_file: Path,
    output_file: Path = None,
    threshold: float = 0.85
) -> int:
    """
    Convenience function to deduplicate a dataset.

    Returns:
        Number of unique entries
    """
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_dedup.jsonl"

    deduplicator = TextDeduplicator(threshold=threshold)
    return deduplicator.deduplicate_jsonl(input_file, output_file)
