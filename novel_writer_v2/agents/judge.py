"""Judge agent: synthesize critic feedback and decide approve/revise."""

from .base import BaseAgent
from ..config import GeminiConfig
from ..models.story_state import CriticFeedback, JudgeVerdict

SYSTEM_EN = """You are a senior editor and final arbiter of chapter quality. You receive feedback from two critics (continuity and style) and must:

1. Weigh both critics' feedback fairly
2. Determine an overall quality score (1-10)
3. Decide whether to APPROVE or REQUEST REVISION
4. If requesting revision, provide clear, actionable instructions

Approval threshold: score >= 7. Be fair but maintain high standards."""

SYSTEM_ZH = """你是一位资深编辑和章节质量的最终仲裁者。你收到两位评论家（连续性和风格）的反馈，必须：

1. 公正地权衡两位评论家的反馈
2. 确定总体质量分数（1-10）
3. 决定是通过还是要求修改
4. 如果要求修改，提供清晰、可操作的修改指示

通过阈值：分数 >= 7。公正但保持高标准。"""


class Judge(BaseAgent):
    def __init__(self, gemini_config: GeminiConfig):
        super().__init__("Judge", gemini_config)

    def evaluate(
        self,
        chapter_text: str,
        continuity_feedback: CriticFeedback,
        style_feedback: CriticFeedback,
        round_number: int,
        max_rounds: int,
        language: str,
    ) -> JudgeVerdict:
        """Evaluate chapter quality and decide approve/revise."""
        system = SYSTEM_ZH if language == "zh" else SYSTEM_EN
        system += (
            "\n\nReturn JSON with: approved (bool), overall_score (1-10), "
            "revision_instructions (string, empty if approved), feedback_summary (string)."
        )

        continuity_issues = "\n".join(
            f"  - [{i.get('severity', 'medium')}] {i.get('description', '')}"
            for i in continuity_feedback.issues
        )
        style_issues = "\n".join(
            f"  - [{i.get('severity', 'medium')}] {i.get('description', '')}"
            for i in style_feedback.issues
        )

        if language == "zh":
            prompt = (
                f"## 审查轮次 {round_number}/{max_rounds}\n\n"
                f"## 连续性评论 (分数: {continuity_feedback.score}/10)\n"
                f"评论：{continuity_feedback.overall_comment}\n"
                f"问题：\n{continuity_issues or '  无'}\n"
                f"优点：{', '.join(continuity_feedback.strengths)}\n\n"
                f"## 风格评论 (分数: {style_feedback.score}/10)\n"
                f"评论：{style_feedback.overall_comment}\n"
                f"问题：\n{style_issues or '  无'}\n"
                f"优点：{', '.join(style_feedback.strengths)}\n\n"
                f"## 章节文本（前2000字）\n{chapter_text[:2000]}\n\n"
                f"做出最终判决。"
            )
        else:
            prompt = (
                f"## Review Round {round_number}/{max_rounds}\n\n"
                f"## Continuity Review (Score: {continuity_feedback.score}/10)\n"
                f"Comment: {continuity_feedback.overall_comment}\n"
                f"Issues:\n{continuity_issues or '  None'}\n"
                f"Strengths: {', '.join(continuity_feedback.strengths)}\n\n"
                f"## Style Review (Score: {style_feedback.score}/10)\n"
                f"Comment: {style_feedback.overall_comment}\n"
                f"Issues:\n{style_issues or '  None'}\n"
                f"Strengths: {', '.join(style_feedback.strengths)}\n\n"
                f"## Chapter Text (first 2000 chars)\n{chapter_text[:2000]}\n\n"
                f"Make your final verdict."
            )

        data = self.call_gemini_json(system, prompt)

        approved = data.get("approved", False)
        score = data.get("overall_score", 5)

        # Force approve on last round
        if round_number >= max_rounds:
            approved = True

        # Auto-approve if score >= 7
        if score >= 7:
            approved = True

        return JudgeVerdict(
            approved=approved,
            overall_score=score,
            revision_instructions=data.get("revision_instructions", ""),
            feedback_summary=data.get("feedback_summary", ""),
            round_number=round_number,
        )
