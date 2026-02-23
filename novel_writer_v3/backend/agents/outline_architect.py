"""Outline Architect agent: parse and generate story outlines."""

from .base import BaseAgent
from ..agent_config import GeminiConfig

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

    def split_into_sections(self, outline: str, num_sections: int, language: str = "en") -> list[dict]:
        """Split a full outline into 3-8 narrative sections/acts.

        Returns list of dicts with keys: section_number, title, summary, chapter_range_hint
        """
        if language == "zh":
            system = "你是一位专业的故事结构规划师。你的任务是将完整的故事大纲拆分成章节组（幕/部分），每个部分代表故事的一个主要阶段。"
            prompt = f"""将以下故事大纲拆分成{num_sections}个叙事段落/幕。

大纲：
{outline}

返回JSON数组，每个元素包含：
- section_number: 段落编号（从1开始）
- title: 段落标题
- summary: 段落摘要（概述本段落包含的主要情节和发展）
- chapter_range_hint: 建议的章节数量

确保所有段落的chapter_range_hint总和合理地覆盖整个故事。
只返回有效的JSON数组，不要包含其他文字。"""
        else:
            system = "You are a professional story structure planner. Your task is to split a full story outline into narrative sections/acts, each representing a major phase of the story."
            prompt = f"""Split the following story outline into {num_sections} narrative sections/acts.

Outline:
{outline}

Return a JSON array where each element contains:
- section_number: section number (starting from 1)
- title: section title
- summary: section summary (overview of major plot points and developments in this section)
- chapter_range_hint: suggested number of chapters for this section

Ensure the total of all chapter_range_hint values reasonably covers the entire story.
Return ONLY a valid JSON array, no other text."""

        result = self.call_gemini_json(system=system, prompt=prompt)
        if isinstance(result, dict) and "sections" in result:
            result = result["sections"]
        return result

    def expand_section_to_chapters(self, section: dict, num_chapters: int, full_outline_context: str, language: str = "en") -> list[dict]:
        """Expand a single section into detailed chapter plans.

        Args:
            section: dict with title, summary, section_number
            num_chapters: how many chapters to create for this section
            full_outline_context: the full outline for context
            language: "en" or "zh"

        Returns list of dicts with keys: chapter_number, title, characters, key_events, emotional_tone, chapter_goal, connections_to_arc
        """
        if language == "zh":
            system = "你是一位专业的章节规划师。根据故事段落的摘要和完整大纲的上下文，你需要将段落扩展为详细的章节计划。"
            prompt = f"""将以下故事段落扩展为{num_chapters}个详细的章节计划。

段落标题：{section.get('title', '')}
段落摘要：{section.get('summary', '')}

完整故事大纲（作为上下文参考）：
{full_outline_context[:3000]}

返回JSON数组，每个元素包含：
- chapter_number: 章节编号（注意：这是在本段落内的相对编号，从1开始）
- title: 章节标题
- characters: 本章出场角色列表
- key_events: 本章关键事件列表
- emotional_tone: 情感基调
- chapter_goal: 章节目标（本章需要推动的情节发展）
- connections_to_arc: 与整体故事弧线的联系

只返回有效的JSON数组，不要包含其他文字。"""
        else:
            system = "You are a professional chapter planner. Given a story section's summary and the full outline context, expand the section into detailed chapter plans."
            prompt = f"""Expand the following story section into {num_chapters} detailed chapter plans.

Section Title: {section.get('title', '')}
Section Summary: {section.get('summary', '')}

Full Story Outline (for context):
{full_outline_context[:3000]}

Return a JSON array where each element contains:
- chapter_number: chapter number (note: this is relative within this section, starting from 1)
- title: chapter title
- characters: list of characters appearing in this chapter
- key_events: list of key events in this chapter
- emotional_tone: emotional tone
- chapter_goal: chapter goal (what plot development this chapter needs to advance)
- connections_to_arc: connections to the overall story arc

Return ONLY a valid JSON array, no other text."""

        result = self.call_gemini_json(system=system, prompt=prompt)
        if isinstance(result, dict) and "chapters" in result:
            result = result["chapters"]
        return result
