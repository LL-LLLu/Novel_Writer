"""Multi-round debate pipeline for chapter generation."""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .config import AppConfig
from .models.story_state import (
    Chapter,
    ChapterPlan,
    DebateRound,
    StoryState,
)
from .models.story_bible import StoryBible
from .agents.outline_architect import OutlineArchitect
from .agents.style_analyzer import StyleAnalyzer
from .agents.mastermind import Mastermind
from .agents.memory import MemoryAgent
from .agents.writer import Writer
from .agents.continuity_critic import ContinuityCritic
from .agents.style_critic import StyleCritic
from .agents.judge import Judge
from .utils import detect_language, truncate_text

ProgressCallback = Callable[[str, int, int, int], None]


def _noop_progress(msg: str, ch: int, total: int, rnd: int) -> None:
    pass


class Pipeline:
    """Orchestrates the multi-agent chapter generation pipeline."""

    def __init__(self, config: AppConfig):
        self.config = config
        gc = config.gemini
        qc = config.qwen

        self.outline_architect = OutlineArchitect(gc)
        self.style_analyzer = StyleAnalyzer(gc)
        self.mastermind = Mastermind(gc)
        self.memory = MemoryAgent(gc)
        self.writer = Writer(gc, qc)
        self.continuity_critic = ContinuityCritic(gc)
        self.style_critic = StyleCritic(gc)
        self.judge = Judge(gc)

    @property
    def all_agents(self) -> list:
        return [
            self.outline_architect,
            self.style_analyzer,
            self.mastermind,
            self.memory,
            self.writer,
            self.continuity_critic,
            self.style_critic,
            self.judge,
        ]

    @property
    def all_logs(self) -> list:
        logs = []
        for agent in self.all_agents:
            logs.extend(agent.logs)
        logs.sort(key=lambda l: l.elapsed_seconds, reverse=False)
        return logs

    def generate_outline(
        self, premise: str, num_chapters: int, language: str
    ) -> str:
        if language == "auto":
            language = detect_language(premise)
        return self.outline_architect.generate_outline(
            premise, num_chapters, language
        )

    def breakdown_outline(
        self, outline: str, num_chapters: int
    ) -> list[dict]:
        return self.outline_architect.breakdown_outline(
            outline, num_chapters
        )

    def analyze_guidance(self, guidance_text: str) -> dict:
        return self.style_analyzer.analyze_guidance(guidance_text)

    def generate_guidance(
        self, premise: str, style_description: str, language: str
    ) -> str:
        if language == "auto":
            language = detect_language(premise)
        return self.style_analyzer.generate_guidance(
            premise, style_description, language
        )

    def initialize_bible(self, outline: str, language: str) -> StoryBible:
        if language == "auto":
            language = detect_language(outline)
        return self.memory.initialize_from_outline(outline, language)

    def generate_single_chapter(
        self,
        chapter_num: int,
        chapter_info: dict,
        story_state: StoryState,
        bible: StoryBible,
        progress: ProgressCallback = _noop_progress,
    ) -> tuple[Chapter, StoryBible]:
        """Generate a single chapter with multi-round debate.

        Returns the completed Chapter and updated StoryBible.
        """
        language = story_state.language
        if language == "auto":
            language = detect_language(story_state.outline)

        total = len(story_state.chapter_plans)
        max_rounds = self.config.generation.max_debate_rounds

        # Get previous chapter ending for continuity
        prev_ending = ""
        if story_state.chapters:
            last_ch = story_state.chapters[-1]
            prev_ending = truncate_text(last_ch.text, 1000, from_end=True)

        # Step 1: Plan the chapter
        progress(f"Planning chapter {chapter_num}...", chapter_num, total, 0)
        bible_context = self.memory.get_context_for_chapter(
            chapter_num, bible
        )
        plan = self.mastermind.plan_chapter(
            chapter_info, bible_context, story_state.style_rules,
            prev_ending, language,
        )

        # Step 2: Write initial draft
        progress(f"Writing chapter {chapter_num}...", chapter_num, total, 0)
        chapter_text = self.writer.write_chapter(
            plan, bible_context, story_state.style_rules,
            prev_ending, language,
        )

        debate_rounds: list[DebateRound] = []
        final_score = 0

        # Step 3: Debate loop (up to max_rounds)
        for round_num in range(1, max_rounds + 1):
            progress(
                f"Chapter {chapter_num}: debate round {round_num}/{max_rounds}",
                chapter_num, total, round_num,
            )

            # Run critics in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                cont_future = executor.submit(
                    self.continuity_critic.review,
                    chapter_text, plan, bible_context, language,
                )
                style_future = executor.submit(
                    self.style_critic.review,
                    chapter_text, story_state.style_rules, language,
                )
                continuity_fb = cont_future.result()
                style_fb = style_future.result()

            # Judge evaluates
            progress(
                f"Chapter {chapter_num}: judge evaluating round {round_num}...",
                chapter_num, total, round_num,
            )
            verdict = self.judge.evaluate(
                chapter_text, continuity_fb, style_fb,
                round_num, max_rounds, language,
            )

            debate_rounds.append(
                DebateRound(
                    round_number=round_num,
                    chapter_text=chapter_text,
                    continuity_feedback=continuity_fb,
                    style_feedback=style_fb,
                    verdict=verdict,
                )
            )

            final_score = verdict.overall_score

            if verdict.approved:
                progress(
                    f"Chapter {chapter_num}: approved (score {verdict.overall_score}) at round {round_num}",
                    chapter_num, total, round_num,
                )
                break

            # Revise
            progress(
                f"Chapter {chapter_num}: revising (round {round_num + 1})...",
                chapter_num, total, round_num,
            )
            chapter_text = self.writer.write_chapter(
                plan, bible_context, story_state.style_rules,
                prev_ending, language,
                revision_instructions=verdict.revision_instructions,
            )

        # Step 4: Update story bible
        progress(
            f"Updating story bible after chapter {chapter_num}...",
            chapter_num, total, 0,
        )
        bible = self.memory.update_after_chapter(
            chapter_num, chapter_text, bible, language,
        )

        chapter = Chapter(
            number=chapter_num,
            title=plan.title,
            text=chapter_text,
            plan=plan,
            debate_rounds=debate_rounds,
            final_score=final_score,
        )

        return chapter, bible

    def generate_all_chapters(
        self,
        story_state: StoryState,
        bible: StoryBible,
        progress: ProgressCallback = _noop_progress,
    ) -> tuple[StoryState, StoryBible]:
        """Generate all chapters sequentially.

        Returns updated StoryState and StoryBible.
        """
        for i, ch_info in enumerate(story_state.chapter_plans):
            chapter_num = ch_info.get("chapter_number", i + 1)
            chapter, bible = self.generate_single_chapter(
                chapter_num, ch_info, story_state, bible, progress,
            )
            story_state.chapters.append(chapter)

        return story_state, bible
