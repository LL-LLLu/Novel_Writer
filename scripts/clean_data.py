#!/usr/bin/env python3
"""Legacy script - use CLI: novel-writer clean"""

from novel_writer.processing.clean import clean_data
from novel_writer.config import Config

if __name__ == "__main__":
    config = Config()
    clean_data(config)
