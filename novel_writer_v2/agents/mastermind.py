"""Mastermind agent: plan individual chapters with scene-level detail."""

from .base import BaseAgent
from ..config import GeminiConfig
from ..models.story_state import ChapterPlan

SYSTEM_EN = """You are a master chapter planner. Given an outline, story bible context, and style rules, you create detailed scene-by-scene chapter plans that guide the writer.

Each plan should include:
- Scene breakdowns with settings, characters, events, mood
- Emotional arc through the chapter
- Specific writing directions (show don't tell moments, dialogue beats, pacing shifts)
- How the chapter connects to the broader story"""

SYSTEM_ZH = """你是一位章节规划大师。根据大纲、故事圣经上下文和风格规则，你创建详细的场景级章节计划来指导作者。

每个计划应包含：
- 场景分解，包括场景设定、角色、事件、氛围
- 章节内的情感弧线
- 具体的写作指导（展示而非叙述的时刻、对话节拍、节奏变化）
- 章节与更广泛故事的联系"""


class Mastermind(BaseAgent):
    def __init__(self, gemini_config: GeminiConfig):
        super().__init__("Mastermind", gemini_config)

    def plan_chapter(
        self,
        chapter_info: dict,
        bible_context: str,
        style_rules: dict,
        prev_chapter_ending: str,
        language: str,
    ) -> ChapterPlan:
        """Create a detailed chapter plan with scene breakdowns."""
        system = SYSTEM_ZH if language == "zh" else SYSTEM_EN
        style_summary = "\n".join(
            f"- {k}: {v}" for k, v in style_rules.items() if v
        ) if style_rules else "No specific style rules."

        if language == "zh":
            prompt = (
                f"为以下章节创建详细的场景级计划。\n\n"
                f"## 章节信息\n{_format_chapter_info(chapter_info)}\n\n"
                f"## 故事上下文\n{bible_context}\n\n"
                f"## 风格规则\n{style_summary}\n\n"
            )
            if prev_chapter_ending:
                prompt += f"## 上一章结尾\n{prev_chapter_ending}\n\n"
            prompt += (
                "返回JSON，包含：title, scenes (数组，每个场景有 setting, characters, events, mood, "
                "pacing_notes), key_events (数组), emotional_arc, chapter_goal"
            )
        else:
            prompt = (
                f"Create a detailed scene-level plan for this chapter.\n\n"
                f"## Chapter Info\n{_format_chapter_info(chapter_info)}\n\n"
                f"## Story Context\n{bible_context}\n\n"
                f"## Style Rules\n{style_summary}\n\n"
            )
            if prev_chapter_ending:
                prompt += f"## Previous Chapter Ending\n{prev_chapter_ending}\n\n"
            prompt += (
                "Return JSON with: title, scenes (array, each with setting, characters, events, mood, "
                "pacing_notes), key_events (array), emotional_arc, chapter_goal"
            )

        data = self.call_gemini_json(system, prompt)
        return ChapterPlan(
            chapter_number=chapter_info.get("chapter_number", 0),
            title=data.get("title", chapter_info.get("title", "")),
            scenes=data.get("scenes", []),
            key_events=data.get("key_events", []),
            emotional_arc=data.get("emotional_arc", ""),
            chapter_goal=data.get("chapter_goal", ""),
        )


def _format_chapter_info(info: dict) -> str:
    parts = []
    for key, val in info.items():
        if isinstance(val, list):
            parts.append(f"- {key}: {', '.join(str(v) for v in val)}")
        else:
            parts.append(f"- {key}: {val}")
    return "\n".join(parts)
