import json
from fastapi import APIRouter, HTTPException
from ..database import get_db
from ..schemas import (
    SectionResponse, SectionUpdate,
    ChapterPlanResponse, ChapterPlanUpdate,
    GenerateOutlineRequest, ExpandSectionRequest,
    ProjectResponse, StoryBibleResponse,
)
from ..services.outline_service import (
    generate_outline, generate_sections, expand_section,
    generate_guidance, initialize_bible,
)

router = APIRouter(tags=["outline"])


@router.post("/projects/{project_id}/outline/generate")
async def api_generate_outline(project_id: int):
    """Generate outline (Pass 1)."""
    try:
        outline = await generate_outline(project_id)
        return {"outline_text": outline}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/projects/{project_id}/outline")
async def api_save_outline(project_id: int, data: dict):
    """Save edited outline."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT id FROM projects WHERE id = ?", (project_id,))
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Project not found")

        outline_text = data.get("outline_text", "")
        await db.execute(
            "UPDATE projects SET outline_text = ?, status = 'outline_ready', updated_at = datetime('now') WHERE id = ?",
            (outline_text, project_id),
        )
        await db.commit()
        return {"outline_text": outline_text}
    finally:
        await db.close()


@router.post("/projects/{project_id}/sections/generate")
async def api_generate_sections(project_id: int, data: GenerateOutlineRequest = GenerateOutlineRequest()):
    """Split outline into sections (Pass 2)."""
    try:
        sections = await generate_sections(project_id, data.num_sections)
        return sections
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/sections", response_model=list[SectionResponse])
async def api_list_sections(project_id: int):
    """List all sections for a project."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM sections WHERE project_id = ? ORDER BY section_number",
            (project_id,),
        )
        rows = await cursor.fetchall()
        return [SectionResponse(
            id=r["id"], project_id=r["project_id"], section_number=r["section_number"],
            title=r["title"], summary=r["summary"], chapter_count=r["chapter_count"],
            status=r["status"], created_at=r["created_at"],
        ) for r in rows]
    finally:
        await db.close()


@router.put("/projects/{project_id}/sections/{section_id}", response_model=SectionResponse)
async def api_update_section(project_id: int, section_id: int, data: SectionUpdate):
    """Edit a section."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM sections WHERE id = ? AND project_id = ?",
            (section_id, project_id),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Section not found")

        update_dict = data.model_dump(exclude_none=True)
        if update_dict:
            set_clause = ", ".join(f"{k} = ?" for k in update_dict.keys())
            values = list(update_dict.values()) + [section_id]
            await db.execute(f"UPDATE sections SET {set_clause} WHERE id = ?", values)
            await db.commit()

        cursor = await db.execute("SELECT * FROM sections WHERE id = ?", (section_id,))
        row = await cursor.fetchone()
        return SectionResponse(
            id=row["id"], project_id=row["project_id"], section_number=row["section_number"],
            title=row["title"], summary=row["summary"], chapter_count=row["chapter_count"],
            status=row["status"], created_at=row["created_at"],
        )
    finally:
        await db.close()


@router.post("/projects/{project_id}/sections/{section_id}/expand")
async def api_expand_section(project_id: int, section_id: int, data: ExpandSectionRequest = ExpandSectionRequest()):
    """Expand section to chapter plans (Pass 3)."""
    try:
        plans = await expand_section(project_id, section_id, data.num_chapters)
        return plans
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/chapter-plans", response_model=list[ChapterPlanResponse])
async def api_list_chapter_plans(project_id: int):
    """Get all chapter plans for a project."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM chapter_plans WHERE project_id = ? ORDER BY chapter_number",
            (project_id,),
        )
        rows = await cursor.fetchall()
        return [ChapterPlanResponse(
            id=r["id"], project_id=r["project_id"], section_id=r["section_id"],
            chapter_number=r["chapter_number"], title=r["title"],
            plan_json=r["plan_json"], status=r["status"], created_at=r["created_at"],
        ) for r in rows]
    finally:
        await db.close()


@router.put("/projects/{project_id}/chapter-plans/{plan_id}", response_model=ChapterPlanResponse)
async def api_update_chapter_plan(project_id: int, plan_id: int, data: ChapterPlanUpdate):
    """Edit a chapter plan."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM chapter_plans WHERE id = ? AND project_id = ?",
            (plan_id, project_id),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Chapter plan not found")

        update_dict = data.model_dump(exclude_none=True)
        if update_dict:
            set_clause = ", ".join(f"{k} = ?" for k in update_dict.keys())
            values = list(update_dict.values()) + [plan_id]
            await db.execute(f"UPDATE chapter_plans SET {set_clause} WHERE id = ?", values)
            await db.commit()

        cursor = await db.execute("SELECT * FROM chapter_plans WHERE id = ?", (plan_id,))
        row = await cursor.fetchone()
        return ChapterPlanResponse(
            id=row["id"], project_id=row["project_id"], section_id=row["section_id"],
            chapter_number=row["chapter_number"], title=row["title"],
            plan_json=row["plan_json"], status=row["status"], created_at=row["created_at"],
        )
    finally:
        await db.close()


@router.post("/projects/{project_id}/guidance/generate")
async def api_generate_guidance(project_id: int, data: dict = {}):
    """Generate style guidance."""
    try:
        guidance = await generate_guidance(project_id, data.get("style_description", ""))
        return {"guidance_text": guidance}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/bible/initialize")
async def api_initialize_bible(project_id: int):
    """Initialize story bible from outline."""
    try:
        result = await initialize_bible(project_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
