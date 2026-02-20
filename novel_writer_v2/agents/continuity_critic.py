"""Continuity Critic agent: review chapters for plot/character/timeline consistency."""

from .base import BaseAgent
from ..config import GeminiConfig
from ..models.story_state import ChapterPlan, CriticFeedback

SYSTEM_EN = """You are a meticulous continuity editor. You review fiction chapters for:
- Character consistency (behavior, knowledge, personality, appearance)
- Plot thread violations (contradictions with established events)
- Timeline issues (chronological errors, impossible sequences)
- Factual contradictions (within the story's own logic)
- Unresolved setups (Chekhov's gun violations)
- Setting consistency (locations, distances, geography)

Be specific about issues found. Reference exact passages when possible."""

SYSTEM_ZH = """你是一位细致的连续性编辑。你审查小说章节的：
- 角色一致性（行为、知识、性格、外貌）
- 情节线索违反（与已确立事件的矛盾）
- 时间线问题（时间顺序错误、不可能的序列）
- 事实矛盾（故事自身逻辑内的矛盾）
- 未解决的铺垫（契诃夫之枪违规）
- 场景一致性（地点、距离、地理）

对发现的问题要具体说明。尽可能引用确切的段落。"""


class ContinuityCritic(BaseAgent):
    def __init__(self, gemini_config: GeminiConfig):
        super().__init__("ContinuityCritic", gemini_config)

    def review(
        self,
        chapter_text: str,
        plan: ChapterPlan,
        bible_context: str,
        language: str,
    ) -> CriticFeedback:
        """Review a chapter for continuity issues."""
        system = SYSTEM_ZH if language == "zh" else SYSTEM_EN
        system += (
            "\n\nReturn JSON with: score (1-10), issues (array of {type, severity, description, suggestion}), "
            "strengths (array of strings), overall_comment (string)."
        )

        plan_summary = f"Title: {plan.title}\nGoal: {plan.chapter_goal}\nKey events: {', '.join(plan.key_events)}"

        if language == "zh":
            prompt = (
                f"## 章节计划\n{plan_summary}\n\n"
                f"## 故事上下文\n{bible_context}\n\n"
                f"## 章节文本\n{chapter_text}\n\n"
                f"审查此章节的连续性问题并评分。"
            )
        else:
            prompt = (
                f"## Chapter Plan\n{plan_summary}\n\n"
                f"## Story Context\n{bible_context}\n\n"
                f"## Chapter Text\n{chapter_text}\n\n"
                f"Review this chapter for continuity issues and score it."
            )

        data = self.call_gemini_json(system, prompt)
        return CriticFeedback(
            critic_type="continuity",
            score=data.get("score", 5),
            issues=data.get("issues", []),
            strengths=data.get("strengths", []),
            overall_comment=data.get("overall_comment", ""),
        )
