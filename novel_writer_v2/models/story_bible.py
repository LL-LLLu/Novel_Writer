"""Story Bible data models for tracking narrative state across chapters."""

from dataclasses import dataclass, field


@dataclass
class CharacterState:
    name: str = ""
    description: str = ""
    location: str = ""
    emotional_state: str = ""
    relationships: dict[str, str] = field(default_factory=dict)
    knowledge: list[str] = field(default_factory=list)
    arc_stage: str = ""
    last_seen_chapter: int = 0


@dataclass
class PlotThread:
    name: str = ""
    description: str = ""
    status: str = "active"  # active, resolved, dormant
    introduced_chapter: int = 0
    resolved_chapter: int = 0
    key_events: list[str] = field(default_factory=list)


@dataclass
class ChapterSummary:
    number: int = 0
    title: str = ""
    summary: str = ""
    key_events: list[str] = field(default_factory=list)
    characters_present: list[str] = field(default_factory=list)
    timeline_note: str = ""


@dataclass
class StoryBible:
    characters: list[CharacterState] = field(default_factory=list)
    plot_threads: list[PlotThread] = field(default_factory=list)
    chapter_summaries: list[ChapterSummary] = field(default_factory=list)
    world_notes: str = ""
    timeline: list[str] = field(default_factory=list)
    foreshadowing: list[str] = field(default_factory=list)

    def to_context_string(self, max_chars: int = 3000) -> str:
        """Serialize the story bible into a context string for agent prompts."""
        parts = []

        if self.characters:
            parts.append("## Characters")
            for c in self.characters:
                line = f"- {c.name}: {c.description}"
                if c.location:
                    line += f" (at: {c.location})"
                if c.emotional_state:
                    line += f" [mood: {c.emotional_state}]"
                if c.arc_stage:
                    line += f" [arc: {c.arc_stage}]"
                parts.append(line)
                if c.relationships:
                    for rel_name, rel_desc in c.relationships.items():
                        parts.append(f"  - {rel_name}: {rel_desc}")

        if self.plot_threads:
            parts.append("\n## Plot Threads")
            for pt in self.plot_threads:
                parts.append(f"- [{pt.status}] {pt.name}: {pt.description}")

        if self.chapter_summaries:
            parts.append("\n## Previous Chapters")
            for cs in self.chapter_summaries:
                parts.append(f"- Ch{cs.number} '{cs.title}': {cs.summary}")

        if self.world_notes:
            parts.append(f"\n## World Notes\n{self.world_notes}")

        if self.timeline:
            parts.append("\n## Timeline")
            for t in self.timeline:
                parts.append(f"- {t}")

        if self.foreshadowing:
            parts.append("\n## Foreshadowing")
            for f in self.foreshadowing:
                parts.append(f"- {f}")

        result = "\n".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars - 3] + "..."
        return result
