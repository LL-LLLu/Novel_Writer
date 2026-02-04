#!/usr/bin/env python3
"""Legacy script - use CLI: novel-writer format"""

from novel_writer.processing.format import format_data
from novel_writer.config import Config

if __name__ == "__main__":
    config = Config()
    format_data(config)
