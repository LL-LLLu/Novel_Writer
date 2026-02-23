from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Project:
    id: int = 0
    title: str = ""
    premise: str = ""
    outline_text: str = ""
    guidance_text: str = ""
    style_rules_json: str = "{}"
    language: str = "auto"
    target_chapters: int = 10
    status: str = "created"  # created, outline_ready, sections_ready, chapters_ready, generating, completed
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Section:
    id: int = 0
    project_id: int = 0
    section_number: int = 0
    title: str = ""
    summary: str = ""
    chapter_count: int = 0
    status: str = "pending"
    created_at: str = ""


@dataclass
class ChapterPlanDB:
    id: int = 0
    project_id: int = 0
    section_id: Optional[int] = None
    chapter_number: int = 0
    title: str = ""
    plan_json: str = "{}"
    status: str = "pending"
    created_at: str = ""


@dataclass
class ChapterDB:
    id: int = 0
    project_id: int = 0
    chapter_plan_id: Optional[int] = None
    chapter_number: int = 0
    title: str = ""
    text: str = ""
    final_score: int = 0
    debate_rounds_json: str = "[]"
    status: str = "pending"
    created_at: str = ""


@dataclass
class StoryBibleDB:
    id: int = 0
    project_id: int = 0
    bible_json: str = "{}"
    updated_at: str = ""


@dataclass
class AgentLog:
    id: int = 0
    project_id: int = 0
    chapter_number: Optional[int] = None
    agent_name: str = ""
    action: str = ""
    prompt_preview: str = ""
    response_preview: str = ""
    elapsed_seconds: float = 0.0
    created_at: str = ""


@dataclass
class Setting:
    id: int = 0
    key: str = ""
    value: str = ""
