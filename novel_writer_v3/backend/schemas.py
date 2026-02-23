from pydantic import BaseModel, Field
from typing import Optional


# Project schemas
class ProjectCreate(BaseModel):
    title: str
    premise: str = ""
    language: str = "auto"
    target_chapters: int = 10


class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    premise: Optional[str] = None
    outline_text: Optional[str] = None
    guidance_text: Optional[str] = None
    style_rules_json: Optional[str] = None
    language: Optional[str] = None
    target_chapters: Optional[int] = None
    status: Optional[str] = None


class ProjectResponse(BaseModel):
    id: int
    title: str
    premise: str
    outline_text: str
    guidance_text: str
    style_rules_json: str
    language: str
    target_chapters: int
    status: str
    created_at: str
    updated_at: str


# Section schemas
class SectionUpdate(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    chapter_count: Optional[int] = None


class SectionResponse(BaseModel):
    id: int
    project_id: int
    section_number: int
    title: str
    summary: str
    chapter_count: int
    status: str
    created_at: str


# Chapter Plan schemas
class ChapterPlanUpdate(BaseModel):
    title: Optional[str] = None
    plan_json: Optional[str] = None


class ChapterPlanResponse(BaseModel):
    id: int
    project_id: int
    section_id: Optional[int]
    chapter_number: int
    title: str
    plan_json: str
    status: str
    created_at: str


# Chapter schemas
class ChapterResponse(BaseModel):
    id: int
    project_id: int
    chapter_plan_id: Optional[int]
    chapter_number: int
    title: str
    text: str
    final_score: int
    debate_rounds_json: str
    status: str
    created_at: str


# Generation schemas
class GenerateRequest(BaseModel):
    start_chapter: int = 1
    end_chapter: Optional[int] = None


class GenerateOutlineRequest(BaseModel):
    num_sections: int = Field(default=5, ge=3, le=8)


class ExpandSectionRequest(BaseModel):
    num_chapters: Optional[int] = None  # If None, auto-calculate from section's chapter_count


# Settings schemas
class SettingsUpdate(BaseModel):
    gemini_api_key: Optional[str] = None
    qwen_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    qwen_model: Optional[str] = None
    qwen_base_url: Optional[str] = None
    gemini_temperature: Optional[float] = None
    qwen_temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    max_debate_rounds: Optional[int] = None
    chapter_min_chars: Optional[int] = None
    chapter_max_chars: Optional[int] = None


class SettingsResponse(BaseModel):
    gemini_api_key: str = ""
    qwen_api_key: str = ""
    gemini_model: str = "gemini-3.1-pro-preview"
    qwen_model: str = "qwen3.5-plus"
    qwen_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    gemini_temperature: float = 0.7
    qwen_temperature: float = 0.8
    max_output_tokens: int = 4096
    max_debate_rounds: int = 3
    chapter_min_chars: int = 5000
    chapter_max_chars: int = 8000


# Story Bible schema
class StoryBibleResponse(BaseModel):
    project_id: int
    bible_json: str
    updated_at: str


# Agent Log schema
class AgentLogResponse(BaseModel):
    id: int
    project_id: int
    chapter_number: Optional[int]
    agent_name: str
    action: str
    prompt_preview: str
    response_preview: str
    elapsed_seconds: float
    created_at: str


# WebSocket progress message
class ProgressMessage(BaseModel):
    type: str = "progress"
    chapter: int = 0
    total: int = 0
    stage: str = ""  # "planning", "writing", "reviewing", "judging", "revising", "updating_bible"
    debate_round: int = 0
    message: str = ""
