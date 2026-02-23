"""Data models for story generation state."""

from dataclasses import dataclass, field


@dataclass
class ChapterPlan:
    chapter_number: int = 0
    title: str = ""
    scenes: list[dict] = field(default_factory=list)
    key_events: list[str] = field(default_factory=list)
    emotional_arc: str = ""
    chapter_goal: str = ""


@dataclass
class CriticFeedback:
    critic_type: str = ""
    score: int = 0  # 1-10
    issues: list[dict] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    overall_comment: str = ""


@dataclass
class JudgeVerdict:
    approved: bool = False
    overall_score: int = 0
    revision_instructions: str = ""
    feedback_summary: str = ""
    round_number: int = 0


@dataclass
class DebateRound:
    round_number: int = 0
    chapter_text: str = ""
    continuity_feedback: CriticFeedback | None = None
    style_feedback: CriticFeedback | None = None
    verdict: JudgeVerdict | None = None


@dataclass
class Chapter:
    number: int = 0
    title: str = ""
    text: str = ""
    plan: ChapterPlan | None = None
    debate_rounds: list[DebateRound] = field(default_factory=list)
    final_score: int = 0


@dataclass
class StoryState:
    outline: str = ""
    guidance: str = ""
    style_rules: dict = field(default_factory=dict)
    chapter_plans: list[dict] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    language: str = "auto"
