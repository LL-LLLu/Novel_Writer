"""Style Critic agent: review chapters for style/guidance compliance."""

from .base import BaseAgent
from ..config import GeminiConfig
from ..models.story_state import CriticFeedback

SYSTEM_EN = """You are an expert literary style critic. You review fiction chapters for:
- Tone consistency with the established style
- Voice quality and narrative perspective adherence
- Pacing (too fast, too slow, well-balanced)
- Dialogue quality (natural, distinctive character voices)
- Prose quality (varied sentences, sensory details, show-don't-tell)
- Guidance adherence (following specific writing rules)

Be specific and constructive. Reference exact passages when possible."""

SYSTEM_ZH = """你是一位专业的文学风格评论家。你审查小说章节的：
- 与既定风格的基调一致性
- 叙述质量和视角的遵守
- 节奏（太快、太慢、平衡）
- 对话质量（自然、有个性的角色声音）
- 散文质量（多变的句式、感官细节、展示而非叙述）
- 指导遵守（遵循特定的写作规则）

要具体且有建设性。尽可能引用确切的段落。"""


class StyleCritic(BaseAgent):
    def __init__(self, gemini_config: GeminiConfig):
        super().__init__("StyleCritic", gemini_config)

    def review(
        self,
        chapter_text: str,
        style_rules: dict,
        language: str,
    ) -> CriticFeedback:
        """Review a chapter for style and guidance compliance."""
        system = SYSTEM_ZH if language == "zh" else SYSTEM_EN
        system += (
            "\n\nReturn JSON with: score (1-10), issues (array of {type, severity, description, suggestion}), "
            "strengths (array of strings), overall_comment (string)."
        )

        style_summary = ""
        if style_rules:
            for k, v in style_rules.items():
                if v:
                    if isinstance(v, list):
                        style_summary += f"- {k}: {', '.join(str(i) for i in v)}\n"
                    else:
                        style_summary += f"- {k}: {v}\n"

        if language == "zh":
            prompt = (
                f"## 风格规则\n{style_summary or '无特定规则'}\n\n"
                f"## 章节文本\n{chapter_text}\n\n"
                f"审查此章节的风格质量和指导遵守情况并评分。"
            )
        else:
            prompt = (
                f"## Style Rules\n{style_summary or 'No specific rules'}\n\n"
                f"## Chapter Text\n{chapter_text}\n\n"
                f"Review this chapter for style quality and guidance compliance, and score it."
            )

        data = self.call_gemini_json(system, prompt)
        return CriticFeedback(
            critic_type="style",
            score=data.get("score", 5),
            issues=data.get("issues", []),
            strengths=data.get("strengths", []),
            overall_comment=data.get("overall_comment", ""),
        )
