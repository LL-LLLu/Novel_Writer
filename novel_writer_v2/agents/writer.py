"""Writer agent: generate prose using Qwen 3.5 Plus."""

from .base import BaseAgent
from ..config import GeminiConfig, QwenConfig
from ..models.story_state import ChapterPlan

SYSTEM_EN = """You are a talented fiction writer. You write vivid, engaging prose that brings stories to life. Your writing features:
- Rich sensory details and atmospheric descriptions
- Natural, distinctive character dialogue
- Varied sentence structure and pacing
- Show-don't-tell storytelling
- Emotional depth and psychological realism

Write the chapter as continuous prose. Do not include chapter headers, author notes, or meta-commentary. Just write the story."""

SYSTEM_ZH = """你是一位才华横溢的小说作家。你的写作生动引人入胜，能让故事栩栩如生。你的写作特点：
- 丰富的感官细节和氛围描写
- 自然、有个性的角色对话
- 多变的句式结构和节奏
- 展示而非叙述的讲故事方式
- 情感深度和心理真实感

以连续散文形式写出章节。不要包含章节标题、作者注释或元评论。只写故事本身。"""


class Writer(BaseAgent):
    def __init__(
        self, gemini_config: GeminiConfig, qwen_config: QwenConfig
    ):
        super().__init__("Writer", gemini_config, qwen_config)

    def write_chapter(
        self,
        plan: ChapterPlan,
        bible_context: str,
        style_rules: dict,
        prev_ending: str,
        language: str,
        revision_instructions: str | None = None,
    ) -> str:
        """Write a chapter using Qwen 3.5 Plus.

        When revision_instructions is provided, rewrites incorporating feedback.
        """
        system = SYSTEM_ZH if language == "zh" else SYSTEM_EN

        # Add style rules to system prompt
        if style_rules:
            style_parts = []
            for k, v in style_rules.items():
                if v and k not in ("dos", "donts"):
                    style_parts.append(f"- {k}: {v}")
            if style_rules.get("dos"):
                style_parts.append("DO: " + "; ".join(style_rules["dos"]))
            if style_rules.get("donts"):
                style_parts.append("DON'T: " + "; ".join(style_rules["donts"]))
            if style_parts:
                system += "\n\n## Style Guidelines\n" + "\n".join(style_parts)

        # Build prompt
        scenes_text = ""
        for i, scene in enumerate(plan.scenes, 1):
            scenes_text += f"\nScene {i}:\n"
            for k, v in scene.items():
                scenes_text += f"  - {k}: {v}\n"

        if language == "zh":
            prompt = f"## 章节计划\n标题：{plan.title}\n目标：{plan.chapter_goal}\n情感弧线：{plan.emotional_arc}\n\n## 场景\n{scenes_text}\n"
            prompt += f"\n## 故事上下文\n{bible_context}\n"
            if prev_ending:
                prompt += f"\n## 上一章结尾\n{prev_ending}\n"
            prompt += f"\n请写出5000-8000字的章节内容。"
            if revision_instructions:
                prompt += f"\n\n## 修改要求\n{revision_instructions}\n请根据以上反馈重写章节。"
        else:
            prompt = f"## Chapter Plan\nTitle: {plan.title}\nGoal: {plan.chapter_goal}\nEmotional Arc: {plan.emotional_arc}\n\n## Scenes\n{scenes_text}\n"
            prompt += f"\n## Story Context\n{bible_context}\n"
            if prev_ending:
                prompt += f"\n## Previous Chapter Ending\n{prev_ending}\n"
            prompt += f"\nWrite 5000-8000 characters of chapter prose."
            if revision_instructions:
                prompt += f"\n\n## Revision Instructions\n{revision_instructions}\nPlease rewrite the chapter incorporating this feedback."

        return self.call_qwen(system, prompt)
