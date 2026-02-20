"""Style Analyzer agent: extract and generate writing style guidance."""

from .base import BaseAgent
from ..config import GeminiConfig

SYSTEM_EN = """You are an expert literary style analyst. You extract structured writing rules from style guidance documents and can generate guidance for specific writing styles.

When analyzing guidance, extract:
- Tone (e.g., dark, humorous, literary, commercial)
- Voice (first person, third limited, omniscient, etc.)
- POV character(s)
- Pacing preferences
- Dialogue style
- Vocabulary notes
- Do's and Don'ts lists
- Any specific prose techniques mentioned"""

SYSTEM_ZH = """你是一位专业的文学风格分析师。你从写作指导文档中提取结构化的写作规则，也可以为特定写作风格生成指导。

分析指导时，提取：
- 基调（如黑暗、幽默、文学性、通俗等）
- 叙述视角（第一人称、第三人称有限、全知等）
- 视角角色
- 节奏偏好
- 对话风格
- 词汇特点
- 应做和不应做的列表
- 提到的任何特定散文技巧"""


class StyleAnalyzer(BaseAgent):
    def __init__(self, gemini_config: GeminiConfig):
        super().__init__("StyleAnalyzer", gemini_config)

    def analyze_guidance(self, guidance_text: str) -> dict:
        """Extract structured style rules from a guidance document."""
        system = (
            "You are a style analyst. Extract structured writing rules from the guidance text. "
            "Return JSON with keys: tone, voice, pov, pacing, dialogue_style, "
            "vocabulary_notes, dos (list), donts (list), prose_techniques (list)."
        )
        prompt = (
            f"Analyze this writing guidance and extract structured style rules:\n\n"
            f"{guidance_text}"
        )
        return self.call_gemini_json(system, prompt)

    def generate_guidance(
        self, premise: str, style_description: str, language: str
    ) -> str:
        """Generate a writing guidance document from a premise and style description."""
        system = SYSTEM_ZH if language == "zh" else SYSTEM_EN
        if language == "zh":
            prompt = (
                f"根据以下故事前提和风格描述，生成一份详细的写作指导文档。\n\n"
                f"前提：{premise}\n\n"
                f"风格描述：{style_description}\n\n"
                f"请包含：基调、叙述视角、节奏、对话风格、词汇选择、"
                f"应做和不应做的列表、散文技巧等。"
            )
        else:
            prompt = (
                f"Based on the following premise and style description, generate a detailed writing guidance document.\n\n"
                f"Premise: {premise}\n\n"
                f"Style description: {style_description}\n\n"
                f"Include: tone, narrative voice, pacing, dialogue style, vocabulary choices, "
                f"do's and don'ts, prose techniques, and any other relevant guidance."
            )
        return self.call_gemini(system, prompt)
