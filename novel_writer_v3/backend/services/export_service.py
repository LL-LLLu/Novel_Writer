"""Export utilities for story content."""

import json


def format_story_txt(title: str, chapters: list[dict]) -> str:
    """Format chapters into a plain text story.

    Args:
        title: Story title
        chapters: List of chapter dicts with 'chapter_number', 'title', 'text'

    Returns:
        Formatted story text
    """
    lines = [title, "=" * len(title), ""]

    for ch in chapters:
        ch_num = ch.get("chapter_number", 0)
        ch_title = ch.get("title", "")
        ch_text = ch.get("text", "")

        lines.append(f"Chapter {ch_num}: {ch_title}")
        lines.append("")
        lines.append(ch_text)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def format_bible_summary(bible_json: str) -> dict:
    """Parse story bible JSON and return a structured summary.

    Args:
        bible_json: JSON string of the story bible

    Returns:
        Dict with formatted sections: characters, plot_threads, timeline, world_notes
    """
    try:
        data = json.loads(bible_json)
    except (json.JSONDecodeError, TypeError):
        return {
            "characters": [],
            "plot_threads": [],
            "timeline": [],
            "world_notes": "",
            "chapter_summaries": [],
            "foreshadowing": [],
        }

    return {
        "characters": data.get("characters", []),
        "plot_threads": data.get("plot_threads", []),
        "timeline": data.get("timeline", []),
        "world_notes": data.get("world_notes", ""),
        "chapter_summaries": data.get("chapter_summaries", []),
        "foreshadowing": data.get("foreshadowing", []),
    }
