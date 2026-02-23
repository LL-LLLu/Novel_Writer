import json
from ..agent_config import GeminiConfig, QwenConfig, AppConfig, GenerationConfig
from ..agents.outline_architect import OutlineArchitect
from ..agents.style_analyzer import StyleAnalyzer
from ..agents.memory import MemoryAgent
from ..database import get_db
from ..utils import detect_language


async def _get_agent_configs() -> tuple[GeminiConfig, QwenConfig, GenerationConfig]:
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


async def generate_outline(project_id: int) -> str:
    """Generate a full outline for a project using OutlineArchitect."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        project = await cursor.fetchone()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        gc, _, _ = await _get_agent_configs()
        architect = OutlineArchitect(gc)

        language = project["language"]
        if language == "auto":
            language = detect_language(project["premise"])

        outline = architect.generate_outline(
            project["premise"], project["target_chapters"], language
        )

        # Save to project
        await db.execute(
            "UPDATE projects SET outline_text = ?, status = 'outline_ready', updated_at = datetime('now') WHERE id = ?",
            (outline, project_id),
        )
        await db.commit()
        return outline
    finally:
        await db.close()


async def generate_sections(project_id: int, num_sections: int = 5) -> list[dict]:
    """Split the outline into sections (Pass 2)."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        project = await cursor.fetchone()
        if not project:
            raise ValueError(f"Project {project_id} not found")
        if not project["outline_text"]:
            raise ValueError("Project has no outline. Generate outline first.")

        gc, _, _ = await _get_agent_configs()
        architect = OutlineArchitect(gc)

        language = project["language"]
        if language == "auto":
            language = detect_language(project["premise"])

        sections = architect.split_into_sections(
            project["outline_text"], num_sections, language
        )

        # Delete existing sections for this project
        await db.execute("DELETE FROM sections WHERE project_id = ?", (project_id,))

        # Insert new sections
        # Distribute target chapters across sections based on hints
        total_hints = sum(s.get("chapter_range_hint", 1) for s in sections)
        target = project["target_chapters"]

        result = []
        for s in sections:
            hint = s.get("chapter_range_hint", 1)
            chapter_count = max(1, round(target * hint / total_hints))

            cursor = await db.execute(
                "INSERT INTO sections (project_id, section_number, title, summary, chapter_count, status) VALUES (?, ?, ?, ?, ?, 'pending')",
                (project_id, s["section_number"], s.get("title", ""), s.get("summary", ""), chapter_count),
            )
            result.append({
                "id": cursor.lastrowid,
                "project_id": project_id,
                "section_number": s["section_number"],
                "title": s.get("title", ""),
                "summary": s.get("summary", ""),
                "chapter_count": chapter_count,
                "status": "pending",
            })

        await db.execute(
            "UPDATE projects SET status = 'sections_ready', updated_at = datetime('now') WHERE id = ?",
            (project_id,),
        )
        await db.commit()
        return result
    finally:
        await db.close()


async def expand_section(project_id: int, section_id: int, num_chapters: int | None = None) -> list[dict]:
    """Expand a section into chapter plans (Pass 3)."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        project = await cursor.fetchone()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        cursor = await db.execute("SELECT * FROM sections WHERE id = ? AND project_id = ?", (section_id, project_id))
        section = await cursor.fetchone()
        if not section:
            raise ValueError(f"Section {section_id} not found")

        if num_chapters is None:
            num_chapters = section["chapter_count"]

        gc, _, _ = await _get_agent_configs()
        architect = OutlineArchitect(gc)

        language = project["language"]
        if language == "auto":
            language = detect_language(project["premise"])

        section_dict = {
            "title": section["title"],
            "summary": section["summary"],
            "section_number": section["section_number"],
        }

        chapters = architect.expand_section_to_chapters(
            section_dict, num_chapters, project["outline_text"], language
        )

        # Delete existing chapter plans for this section
        await db.execute(
            "DELETE FROM chapter_plans WHERE project_id = ? AND section_id = ?",
            (project_id, section_id),
        )

        # Calculate global chapter offset
        cursor = await db.execute(
            "SELECT COALESCE(SUM(chapter_count), 0) as total FROM sections WHERE project_id = ? AND section_number < ?",
            (project_id, section["section_number"]),
        )
        offset_row = await cursor.fetchone()
        chapter_offset = offset_row["total"]

        result = []
        for ch in chapters:
            global_number = chapter_offset + ch.get("chapter_number", 1)
            plan_json = json.dumps(ch, ensure_ascii=False)
            cursor = await db.execute(
                "INSERT INTO chapter_plans (project_id, section_id, chapter_number, title, plan_json, status) VALUES (?, ?, ?, ?, ?, 'pending')",
                (project_id, section_id, global_number, ch.get("title", ""), plan_json),
            )
            result.append({
                "id": cursor.lastrowid,
                "project_id": project_id,
                "section_id": section_id,
                "chapter_number": global_number,
                "title": ch.get("title", ""),
                "plan_json": plan_json,
                "status": "pending",
            })

        # Update section status
        await db.execute(
            "UPDATE sections SET status = 'expanded', chapter_count = ? WHERE id = ?",
            (num_chapters, section_id),
        )

        # Check if all sections are expanded
        cursor = await db.execute(
            "SELECT COUNT(*) as total FROM sections WHERE project_id = ? AND status != 'expanded'",
            (project_id,),
        )
        remaining = await cursor.fetchone()
        if remaining["total"] == 0:
            await db.execute(
                "UPDATE projects SET status = 'chapters_ready', updated_at = datetime('now') WHERE id = ?",
                (project_id,),
            )

        await db.commit()
        return result
    finally:
        await db.close()


async def generate_guidance(project_id: int, style_description: str = "") -> str:
    """Generate style guidance for a project."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        project = await cursor.fetchone()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        gc, _, _ = await _get_agent_configs()
        analyzer = StyleAnalyzer(gc)

        language = project["language"]
        if language == "auto":
            language = detect_language(project["premise"])

        guidance = analyzer.generate_guidance(project["premise"], style_description, language)

        # Parse into style rules
        style_rules = analyzer.analyze_guidance(guidance)

        await db.execute(
            "UPDATE projects SET guidance_text = ?, style_rules_json = ?, updated_at = datetime('now') WHERE id = ?",
            (guidance, json.dumps(style_rules, ensure_ascii=False), project_id),
        )
        await db.commit()
        return guidance
    finally:
        await db.close()


async def initialize_bible(project_id: int) -> dict:
    """Initialize story bible from the outline."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        project = await cursor.fetchone()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        gc, _, _ = await _get_agent_configs()
        memory = MemoryAgent(gc)

        language = project["language"]
        if language == "auto":
            language = detect_language(project["premise"])

        from ..story_models.story_bible import StoryBible
        bible = memory.initialize_from_outline(project["outline_text"], language)
        bible_json = json.dumps({
            "characters": [vars(c) for c in bible.characters],
            "plot_threads": [vars(p) for p in bible.plot_threads],
            "chapter_summaries": [vars(s) for s in bible.chapter_summaries],
            "world_notes": bible.world_notes,
            "timeline": bible.timeline,
            "foreshadowing": bible.foreshadowing,
        }, ensure_ascii=False)

        await db.execute(
            "INSERT INTO story_bibles (project_id, bible_json) VALUES (?, ?) ON CONFLICT(project_id) DO UPDATE SET bible_json = ?, updated_at = datetime('now')",
            (project_id, bible_json, bible_json),
        )
        await db.commit()
        return {"project_id": project_id, "bible_json": bible_json}
    finally:
        await db.close()
