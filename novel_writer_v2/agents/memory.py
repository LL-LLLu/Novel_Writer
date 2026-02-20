"""Memory Agent: track story state via a StoryBible across chapters."""

from .base import BaseAgent
from ..config import GeminiConfig
from ..models.story_bible import (
    StoryBible,
    CharacterState,
    PlotThread,
    ChapterSummary,
)


class MemoryAgent(BaseAgent):
    def __init__(self, gemini_config: GeminiConfig):
        super().__init__("MemoryAgent", gemini_config)

    def initialize_from_outline(
        self, outline: str, language: str
    ) -> StoryBible:
        """Seed a StoryBible from the story outline (1 Gemini call)."""
        system = (
            "You are a story analyst. Extract characters, plot threads, and world details from a story outline. "
            "Return JSON with: characters (array of {name, description, relationships}), "
            "plot_threads (array of {name, description, status}), "
            "world_notes (string), foreshadowing (array of strings)."
        )
        if language == "zh":
            prompt = f"分析以下小说大纲，提取角色、情节线索和世界观设定：\n\n{outline}"
        else:
            prompt = f"Analyze this novel outline and extract characters, plot threads, and world details:\n\n{outline}"

        data = self.call_gemini_json(system, prompt)
        bible = StoryBible()

        for c in data.get("characters", []):
            bible.characters.append(
                CharacterState(
                    name=c.get("name", ""),
                    description=c.get("description", ""),
                    relationships=c.get("relationships", {}),
                )
            )

        for pt in data.get("plot_threads", []):
            bible.plot_threads.append(
                PlotThread(
                    name=pt.get("name", ""),
                    description=pt.get("description", ""),
                    status=pt.get("status", "active"),
                )
            )

        bible.world_notes = data.get("world_notes", "")
        bible.foreshadowing = data.get("foreshadowing", [])
        return bible

    def update_after_chapter(
        self,
        chapter_num: int,
        chapter_text: str,
        bible: StoryBible,
        language: str,
    ) -> StoryBible:
        """Update the StoryBible after a chapter is written (1 Gemini call)."""
        system = (
            "You are a story continuity tracker. Given a chapter and current story state, "
            "extract updates. Return JSON with: "
            "chapter_summary (string), key_events (array), characters_present (array of names), "
            "timeline_note (string), character_updates (array of {name, location, emotional_state, "
            "arc_stage, new_knowledge (array)}), plot_updates (array of {name, status, new_events (array)}), "
            "new_foreshadowing (array of strings)."
        )
        current_context = bible.to_context_string(max_chars=2000)
        if language == "zh":
            prompt = (
                f"## 当前故事状态\n{current_context}\n\n"
                f"## 第{chapter_num}章内容\n{chapter_text[:3000]}\n\n"
                f"分析此章节并提取故事状态更新。"
            )
        else:
            prompt = (
                f"## Current Story State\n{current_context}\n\n"
                f"## Chapter {chapter_num} Text\n{chapter_text[:3000]}\n\n"
                f"Analyze this chapter and extract story state updates."
            )

        data = self.call_gemini_json(system, prompt)

        # Add chapter summary
        bible.chapter_summaries.append(
            ChapterSummary(
                number=chapter_num,
                title=f"Chapter {chapter_num}",
                summary=data.get("chapter_summary", ""),
                key_events=data.get("key_events", []),
                characters_present=data.get("characters_present", []),
                timeline_note=data.get("timeline_note", ""),
            )
        )

        if data.get("timeline_note"):
            bible.timeline.append(
                f"Ch{chapter_num}: {data['timeline_note']}"
            )

        # Update characters
        for update in data.get("character_updates", []):
            for char in bible.characters:
                if char.name == update.get("name"):
                    if update.get("location"):
                        char.location = update["location"]
                    if update.get("emotional_state"):
                        char.emotional_state = update["emotional_state"]
                    if update.get("arc_stage"):
                        char.arc_stage = update["arc_stage"]
                    char.knowledge.extend(update.get("new_knowledge", []))
                    char.last_seen_chapter = chapter_num
                    break

        # Update plot threads
        for update in data.get("plot_updates", []):
            for pt in bible.plot_threads:
                if pt.name == update.get("name"):
                    if update.get("status"):
                        pt.status = update["status"]
                    pt.key_events.extend(update.get("new_events", []))
                    if update.get("status") == "resolved":
                        pt.resolved_chapter = chapter_num
                    break

        bible.foreshadowing.extend(data.get("new_foreshadowing", []))
        return bible

    def get_context_for_chapter(
        self, chapter_num: int, bible: StoryBible
    ) -> str:
        """Build context string for a chapter (local, no API call)."""
        return bible.to_context_string(max_chars=3000)
