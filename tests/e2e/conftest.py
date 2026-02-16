"""Shared fixtures for end-to-end tests."""

import json
import pytest
from pathlib import Path

from novel_writer.config import Config, DataConfig


SAMPLE_NOVEL_TEXT = """
第一章 黎明之前

天还没有亮，整个村庄都笼罩在一片寂静之中。远处的山峦在薄雾中若隐若现，仿佛一幅淡墨山水画。
李明站在院子里，深深地吸了一口清晨的空气。今天是个特别的日子，他已经等了整整三年。

"你真的要走吗？"身后传来母亲苍老的声音。

李明没有回头，他知道如果回头，自己可能就再也走不了了。"妈，我会回来的。"

他的声音很轻，却在寂静的清晨显得格外清晰。母亲没有再说什么，只是默默地将一个包袱递到他手中。

第二章 远行

一路向西，李明走了整整七天。他穿过了无数个村庄和城镇，见识了各种各样的人和事。
有热情好客的农家，有精明狡猾的商人，也有孤独的旅人。每个人都有自己的故事，每个故事都让他对这个世界有了新的认识。

"年轻人，你这是要去哪里啊？"路边的一个老者问道。

"去长安。"李明简短地回答。

老者笑了笑，"长安啊，那可是个好地方。不过路还远得很呢。"

李明点了点头，继续前行。他知道前方的路还很长，但他的心中充满了希望和勇气。
"""

SAMPLE_ENGLISH_TEXT = """
Chapter 1: The Beginning

The old library stood at the corner of Elm Street, its weathered brick facade a testament to decades of quiet endurance. Inside, rows upon rows of books lined the shelves, their spines cracked and faded from years of loving use.

Sarah pushed open the heavy oak door and stepped inside. The familiar scent of old paper and wood polish greeted her like an old friend. She had been coming here since she was a child, and every visit felt like coming home.

"Good morning, Sarah," called Mrs. Henderson from behind the circulation desk. "Looking for anything special today?"

"Just browsing," Sarah replied with a smile. But she wasn't just browsing. She had found something last week - a hidden room behind the reference section. A room that held secrets no one had spoken of in years.

Chapter 2: The Discovery

The room was small, no bigger than a closet. But what it contained was extraordinary. Shelves lined every wall, filled not with books but with journals. Handwritten journals dating back over a hundred years.

Sarah carefully lifted one from the shelf. The leather cover was soft with age, and the pages inside were yellowed but still legible. She began to read.
"""


@pytest.fixture
def sample_config(tmp_path):
    """Create a test configuration pointing to tmp directories."""
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    temp_dir = tmp_path / "temp"

    input_dir.mkdir()
    output_dir.mkdir()
    temp_dir.mkdir()

    config = Config(
        data=DataConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            chunk_size=2000,
            overlap=200,
        ),
        log_level="DEBUG",
    )

    return config


@pytest.fixture
def sample_txt_files(sample_config):
    """Create sample text files in the input directory."""
    input_dir = sample_config.data.input_dir

    # Write Chinese novel sample
    chinese_file = input_dir / "sample_chinese.txt"
    chinese_file.write_text(SAMPLE_NOVEL_TEXT, encoding="utf-8")

    # Write English novel sample
    english_file = input_dir / "sample_english.txt"
    english_file.write_text(SAMPLE_ENGLISH_TEXT, encoding="utf-8")

    return [chinese_file, english_file]


@pytest.fixture
def sample_jsonl(sample_config):
    """Create a sample JSONL training file."""
    output_dir = sample_config.data.output_dir
    jsonl_file = output_dir / "train.jsonl"

    entries = [
        {"instruction": "Continue the story:", "output": "The hero walked into the dark forest. Trees towered above, their branches intertwining to block out the moonlight. Every step brought a new sound, a new shadow."},
        {"instruction": "Describe the scene:", "output": "The marketplace buzzed with activity. Vendors called out their wares while customers haggled over prices. The aroma of fresh bread mingled with exotic spices."},
        {"instruction": "Write dialogue:", "output": '"I never thought it would end like this," she whispered. "Neither did I," he replied, his voice heavy with regret. They stood in silence, the weight of their words hanging in the air.'},
        {"instruction": "Continue writing:", "output": "Dawn broke over the mountains, painting the sky in shades of gold and crimson. The village below was just beginning to stir, smoke rising from chimneys as families prepared for another day."},
        {"instruction": "Describe the character:", "output": "Old Master Chen had a face like weathered stone, lined with decades of wisdom and hardship. His eyes, though, sparkled with a youthful mischief that belied his age."},
    ]

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return jsonl_file
