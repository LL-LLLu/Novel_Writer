import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Callable

from ..agent_config import GeminiConfig, QwenConfig, AppConfig, GenerationConfig
from ..agents.outline_architect import OutlineArchitect
from ..agents.style_analyzer import StyleAnalyzer
from ..agents.mastermind import Mastermind
from ..agents.memory import MemoryAgent
from ..agents.writer import Writer
from ..agents.continuity_critic import ContinuityCritic
from ..agents.style_critic import StyleCritic
from ..agents.judge import Judge
from ..story_models.story_state import Chapter, ChapterPlan, DebateRound, StoryState
from ..story_models.story_bible import StoryBible, CharacterState, PlotThread, ChapterSummary
from ..database import get_db
from ..utils import detect_language, truncate_text

# Type for progress callback: async function(message, chapter, total, stage, debate_round)
ProgressCallback = Callable  # Will be called with kwargs


async def _get_agent_configs():
    """Build agent configs from DB settings."""
    db = await get_db()
    try:
        settings = {}
        cursor = await db.execute("SELECT key, value FROM settings")
        rows = await cursor.fetchall()
        for row in rows:
            settings[row["key"]] = row["value"]
    finally:
        await db.close()

    if not settings.get("gemini_api_key"):
        raise ValueError("Gemini API key not configured. Go to Settings to add it.")
    if not settings.get("qwen_api_key"):
        raise ValueError("Qwen API key not configured. Go to Settings to add it.")

    gc = GeminiConfig(
        api_key=settings.get("gemini_api_key", ""),
        model=settings.get("gemini_model", "gemini-3.1-pro-preview"),
        temperature=float(settings.get("gemini_temperature", "0.7")),
        max_output_tokens=int(settings.get("max_output_tokens", "4096")),
    )
    qc = QwenConfig(
        api_key=settings.get("qwen_api_key", ""),
        model=settings.get("qwen_model", "qwen3.5-plus"),
        base_url=settings.get("qwen_base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        temperature=float(settings.get("qwen_temperature", "0.8")),
        max_tokens=int(settings.get("max_output_tokens", "4096")),
    )
    gen = GenerationConfig(
        max_debate_rounds=int(settings.get("max_debate_rounds", "3")),
        chapter_min_chars=int(settings.get("chapter_min_chars", "5000")),
        chapter_max_chars=int(settings.get("chapter_max_chars", "8000")),
    )
    return gc, qc, gen


def _bible_to_json(bible: StoryBible) -> str:
    """Serialize a StoryBible to JSON."""
    return json.dumps({
        "characters": [vars(c) for c in bible.characters],
        "plot_threads": [vars(p) for p in bible.plot_threads],
        "chapter_summaries": [vars(s) for s in bible.chapter_summaries],
        "world_notes": bible.world_notes,
        "timeline": bible.timeline,
        "foreshadowing": bible.foreshadowing,
    }, ensure_ascii=False)


def _bible_from_json(bible_json: str) -> StoryBible:
    """Deserialize a StoryBible from JSON."""
    data = json.loads(bible_json)
    bible = StoryBible()
    bible.characters = [CharacterState(**c) for c in data.get("characters", [])]
    bible.plot_threads = [PlotThread(**p) for p in data.get("plot_threads", [])]
    bible.chapter_summaries = [ChapterSummary(**s) for s in data.get("chapter_summaries", [])]
    bible.world_notes = data.get("world_notes", "")
    bible.timeline = data.get("timeline", [])
    bible.foreshadowing = data.get("foreshadowing", [])
    return bible


async def _save_agent_logs(project_id: int, chapter_number: int, agents: list):
    """Save agent logs to DB."""
    db = await get_db()
    try:
        for agent in agents:
            for log in agent.logs:
                await db.execute(
                    "INSERT INTO agent_logs (project_id, chapter_number, agent_name, action, prompt_preview, response_preview, elapsed_seconds) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (project_id, chapter_number, log.agent_name, log.action,
                     log.prompt_preview, log.response_preview, log.elapsed_seconds),
                )
        await db.commit()
    finally:
        await db.close()


def _serialize_debate_rounds(rounds: list[DebateRound]) -> str:
    """Serialize debate rounds to JSON."""
    result = []
    for dr in rounds:
        d = {"round_number": dr.round_number}
        d["chapter_text_preview"] = dr.chapter_text[:500] if dr.chapter_text else ""
        if dr.continuity_feedback:
            d["continuity_feedback"] = {
                "score": dr.continuity_feedback.score,
                "issues": dr.continuity_feedback.issues,
                "strengths": dr.continuity_feedback.strengths,
                "overall_comment": dr.continuity_feedback.overall_comment,
            }
        if dr.style_feedback:
            d["style_feedback"] = {
                "score": dr.style_feedback.score,
                "issues": dr.style_feedback.issues,
                "strengths": dr.style_feedback.strengths,
                "overall_comment": dr.style_feedback.overall_comment,
            }
        if dr.verdict:
            d["verdict"] = {
                "approved": dr.verdict.approved,
                "overall_score": dr.verdict.overall_score,
                "revision_instructions": dr.verdict.revision_instructions,
                "feedback_summary": dr.verdict.feedback_summary,
            }
        result.append(d)
    return json.dumps(result, ensure_ascii=False)


# Global state for tracking active generation
_active_generations: dict[int, bool] = {}  # project_id -> should_continue


def cancel_generation(project_id: int):
    """Signal to stop generation for a project."""
    _active_generations[project_id] = False


async def generate_chapters(
    project_id: int,
    start_chapter: int,
    end_chapter: int | None,
    progress_callback: ProgressCallback | None = None,
):
    """Generate chapters for a project using the debate pipeline.

    This runs the synchronous agent calls in a thread pool.
    """
    _active_generations[project_id] = True

    gc, qc, gen_config = await _get_agent_configs()

    # Build agents
    mastermind = Mastermind(gc)
    memory = MemoryAgent(gc)
    writer = Writer(gc, qc)
    continuity_critic = ContinuityCritic(gc)
    style_critic = StyleCritic(gc)
    judge = Judge(gc)

    # Load project data
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        project = await cursor.fetchone()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        language = project["language"]
        if language == "auto":
            language = detect_language(project["premise"])

        style_rules = json.loads(project["style_rules_json"]) if project["style_rules_json"] else {}

        # Load chapter plans
        cursor = await db.execute(
            "SELECT * FROM chapter_plans WHERE project_id = ? ORDER BY chapter_number",
            (project_id,),
        )
        plans_rows = await cursor.fetchall()
        if not plans_rows:
            raise ValueError("No chapter plans found. Generate outline first.")

        # Load existing chapters for context
        cursor = await db.execute(
            "SELECT * FROM chapters WHERE project_id = ? ORDER BY chapter_number",
            (project_id,),
        )
        existing_chapters = await cursor.fetchall()
        prev_chapter_text = ""
        if existing_chapters:
            last = existing_chapters[-1]
            prev_chapter_text = truncate_text(last["text"], 1000, from_end=True)

        # Load or create story bible
        cursor = await db.execute(
            "SELECT * FROM story_bibles WHERE project_id = ?", (project_id,)
        )
        bible_row = await cursor.fetchone()
        if bible_row:
            bible = _bible_from_json(bible_row["bible_json"])
        else:
            bible = StoryBible()

        # Determine chapter range
        plan_numbers = [r["chapter_number"] for r in plans_rows]
        if end_chapter is None:
            end_chapter = max(plan_numbers)

        plans_by_number = {r["chapter_number"]: r for r in plans_rows}

        # Update project status
        await db.execute(
            "UPDATE projects SET status = 'generating', updated_at = datetime('now') WHERE id = ?",
            (project_id,),
        )
        await db.commit()
    finally:
        await db.close()

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=3)

    async def send_progress(msg, chapter, total, stage, debate_round=0):
        if progress_callback:
            await progress_callback(
                message=msg, chapter=chapter, total=total,
                stage=stage, debate_round=debate_round,
            )

    total = end_chapter - start_chapter + 1

    try:
        for chapter_num in range(start_chapter, end_chapter + 1):
            if not _active_generations.get(project_id, False):
                await send_progress(
                    f"Generation paused at chapter {chapter_num}",
                    chapter_num, total, "paused",
                )
                break

            plan_row = plans_by_number.get(chapter_num)
            if not plan_row:
                continue

            chapter_info = json.loads(plan_row["plan_json"])
            bible_context = memory.get_context_for_chapter(chapter_num, bible)

            # Step 1: Plan
            await send_progress(f"Planning chapter {chapter_num}...", chapter_num, total, "planning")
            plan = await loop.run_in_executor(
                executor,
                mastermind.plan_chapter,
                chapter_info, bible_context, style_rules, prev_chapter_text, language,
            )

            # Step 2: Write
            await send_progress(f"Writing chapter {chapter_num}...", chapter_num, total, "writing")
            chapter_text = await loop.run_in_executor(
                executor,
                writer.write_chapter,
                plan, bible_context, style_rules, prev_chapter_text, language, None,
            )

            debate_rounds = []
            final_score = 0
            max_rounds = gen_config.max_debate_rounds

            # Step 3: Debate loop
            for round_num in range(1, max_rounds + 1):
                if not _active_generations.get(project_id, False):
                    break

                await send_progress(
                    f"Chapter {chapter_num}: reviewing (round {round_num})",
                    chapter_num, total, "reviewing", round_num,
                )

                # Run critics in parallel using threads
                cont_future = loop.run_in_executor(
                    executor,
                    continuity_critic.review,
                    chapter_text, plan, bible_context, language,
                )
                style_future = loop.run_in_executor(
                    executor,
                    style_critic.review,
                    chapter_text, style_rules, language,
                )
                continuity_fb, style_fb = await asyncio.gather(cont_future, style_future)

                # Judge
                await send_progress(
                    f"Chapter {chapter_num}: judging (round {round_num})",
                    chapter_num, total, "judging", round_num,
                )
                verdict = await loop.run_in_executor(
                    executor,
                    judge.evaluate,
                    chapter_text, continuity_fb, style_fb, round_num, max_rounds, language,
                )

                debate_rounds.append(DebateRound(
                    round_number=round_num,
                    chapter_text=chapter_text,
                    continuity_feedback=continuity_fb,
                    style_feedback=style_fb,
                    verdict=verdict,
                ))

                final_score = verdict.overall_score

                if verdict.approved:
                    await send_progress(
                        f"Chapter {chapter_num}: approved (score {verdict.overall_score})",
                        chapter_num, total, "approved", round_num,
                    )
                    break

                # Revise
                await send_progress(
                    f"Chapter {chapter_num}: revising...",
                    chapter_num, total, "revising", round_num,
                )
                chapter_text = await loop.run_in_executor(
                    executor,
                    writer.write_chapter,
                    plan, bible_context, style_rules, prev_chapter_text, language,
                    verdict.revision_instructions,
                )

            # Step 4: Update bible
            await send_progress(
                f"Updating story bible after chapter {chapter_num}...",
                chapter_num, total, "updating_bible",
            )
            bible = await loop.run_in_executor(
                executor,
                memory.update_after_chapter,
                chapter_num, chapter_text, bible, language,
            )

            # Save chapter to DB
            db = await get_db()
            try:
                debate_json = _serialize_debate_rounds(debate_rounds)
                # Delete existing chapter if any
                await db.execute(
                    "DELETE FROM chapters WHERE project_id = ? AND chapter_number = ?",
                    (project_id, chapter_num),
                )
                await db.execute(
                    "INSERT INTO chapters (project_id, chapter_plan_id, chapter_number, title, text, final_score, debate_rounds_json, status) VALUES (?, ?, ?, ?, ?, ?, ?, 'completed')",
                    (project_id, plan_row["id"], chapter_num, plan.title, chapter_text, final_score, debate_json),
                )

                # Save bible
                bible_json = _bible_to_json(bible)
                await db.execute(
                    "INSERT INTO story_bibles (project_id, bible_json) VALUES (?, ?) ON CONFLICT(project_id) DO UPDATE SET bible_json = ?, updated_at = datetime('now')",
                    (project_id, bible_json, bible_json),
                )

                await db.commit()
            finally:
                await db.close()

            # Save agent logs
            await _save_agent_logs(
                project_id, chapter_num,
                [mastermind, memory, writer, continuity_critic, style_critic, judge],
            )
            # Clear logs for next chapter
            for agent in [mastermind, memory, writer, continuity_critic, style_critic, judge]:
                agent.logs.clear()

            prev_chapter_text = truncate_text(chapter_text, 1000, from_end=True)

        # Update project status
        db = await get_db()
        try:
            await db.execute(
                "UPDATE projects SET status = 'completed', updated_at = datetime('now') WHERE id = ?",
                (project_id,),
            )
            await db.commit()
        finally:
            await db.close()

        await send_progress("Generation complete!", end_chapter, total, "complete")

    except Exception as e:
        db = await get_db()
        try:
            await db.execute(
                "UPDATE projects SET status = 'error', updated_at = datetime('now') WHERE id = ?",
                (project_id,),
            )
            await db.commit()
        finally:
            await db.close()
        raise
    finally:
        _active_generations.pop(project_id, None)
        executor.shutdown(wait=False)
