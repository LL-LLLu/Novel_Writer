"""Outline Architect agent: parse and generate story outlines."""

from .base import BaseAgent
from ..config import GeminiConfig

SYSTEM_EN = """You are a master story architect. You design compelling, well-structured novel outlines with clear narrative arcs, character development, and thematic depth.

When generating outlines, include for each chapter:
- Chapter number and title
- Key events and plot points
- Characters involved
- Emotional tone and pacing
- How it connects to the overall narrative arc

When breaking down outlines, extract structured data for each chapter."""

SYSTEM_ZH = """你是一位资深的故事架构师。你设计引人入胜、结构严谨的小说大纲，具有清晰的叙事弧线、人物发展和主题深度。

生成大纲时，每章应包含：
- 章节编号和标题
- 关键事件和情节点
- 涉及的角色
- 情感基调和节奏
- 与整体叙事弧线的联系

分解大纲时，提取每章的结构化数据。"""


class OutlineArchitect(BaseAgent):
    def __init__(self, gemini_config: GeminiConfig):
        super().__init__("OutlineArchitect", gemini_config)

    def generate_outline(
        self, premise: str, num_chapters: int, language: str
    ) -> str:
        """Generate a full story outline from a premise."""
        system = SYSTEM_ZH if language == "zh" else SYSTEM_EN
        if language == "zh":
            prompt = (
                f"根据以下故事前提，生成一个{num_chapters}章的详细小说大纲。\n\n"
                f"前提：{premise}\n\n"
                f"要求：\n"
                f"- 每章包含标题、关键事件、涉及角色、情感基调\n"
                f"- 确保整体叙事弧线完整\n"
                f"- 角色发展有层次\n"
                f"- 情节有起伏，节奏合理"
            )
        else:
            prompt = (
                f"Based on the following premise, generate a detailed {num_chapters}-chapter novel outline.\n\n"
                f"Premise: {premise}\n\n"
                f"Requirements:\n"
                f"- Each chapter should have a title, key events, characters involved, and emotional tone\n"
                f"- Ensure a complete narrative arc across all chapters\n"
                f"- Character development should be layered and meaningful\n"
                f"- Plot should have rising action, climax, and resolution with good pacing"
            )
        return self.call_gemini(system, prompt)

    def breakdown_outline(
        self, outline: str, num_chapters: int
    ) -> list[dict]:
        """Break down an outline into structured per-chapter data."""
        system = (
            "You are a story analyst. Extract structured chapter data from outlines. "
            "Return a JSON array where each element has: chapter_number, title, characters (list), "
            "key_events (list), emotional_tone, chapter_goal, connections_to_arc."
        )
        prompt = (
            f"Break down this {num_chapters}-chapter outline into structured data.\n\n"
            f"Outline:\n{outline}\n\n"
            f"Return a JSON array with {num_chapters} elements."
        )
        result = self.call_gemini_json(system, prompt)
        if isinstance(result, dict) and "chapters" in result:
            return result["chapters"]
        if isinstance(result, list):
            return result
        return []
