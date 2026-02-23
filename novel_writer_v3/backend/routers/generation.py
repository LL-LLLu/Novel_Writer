import asyncio
import json
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from ..database import get_db
from ..schemas import GenerateRequest, ChapterResponse, AgentLogResponse, StoryBibleResponse
from ..services.generation_service import generate_chapters, cancel_generation

router = APIRouter(tags=["generation"])


@router.post("/projects/{project_id}/generate")
async def api_generate(project_id: int, data: GenerateRequest = GenerateRequest()):
    """Start chapter generation. Returns immediately, generation runs in background."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT id FROM projects WHERE id = ?", (project_id,))
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Project not found")
    finally:
        await db.close()

    # Start generation as background task
    asyncio.create_task(
        generate_chapters(project_id, data.start_chapter, data.end_chapter)
    )
    return {"status": "started", "start_chapter": data.start_chapter, "end_chapter": data.end_chapter}


@router.post("/projects/{project_id}/generate/cancel")
async def api_cancel_generation(project_id: int):
    """Cancel ongoing generation."""
    cancel_generation(project_id)
    return {"status": "cancelling"}


@router.get("/projects/{project_id}/chapters", response_model=list[ChapterResponse])
async def api_list_chapters(project_id: int):
    """List all generated chapters."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM chapters WHERE project_id = ? ORDER BY chapter_number",
            (project_id,),
        )
        rows = await cursor.fetchall()
        return [ChapterResponse(
            id=r["id"], project_id=r["project_id"],
            chapter_plan_id=r["chapter_plan_id"],
            chapter_number=r["chapter_number"], title=r["title"],
            text=r["text"], final_score=r["final_score"],
            debate_rounds_json=r["debate_rounds_json"],
            status=r["status"], created_at=r["created_at"],
        ) for r in rows]
    finally:
        await db.close()


@router.get("/projects/{project_id}/chapters/{chapter_num}", response_model=ChapterResponse)
async def api_get_chapter(project_id: int, chapter_num: int):
    """Get a single chapter."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM chapters WHERE project_id = ? AND chapter_number = ?",
            (project_id, chapter_num),
        )
        r = await cursor.fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Chapter not found")
        return ChapterResponse(
            id=r["id"], project_id=r["project_id"],
            chapter_plan_id=r["chapter_plan_id"],
            chapter_number=r["chapter_number"], title=r["title"],
            text=r["text"], final_score=r["final_score"],
            debate_rounds_json=r["debate_rounds_json"],
            status=r["status"], created_at=r["created_at"],
        )
    finally:
        await db.close()


@router.get("/projects/{project_id}/bible", response_model=StoryBibleResponse)
async def api_get_bible(project_id: int):
    """Get story bible."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM story_bibles WHERE project_id = ?", (project_id,)
        )
        r = await cursor.fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Story bible not found")
        return StoryBibleResponse(
            project_id=r["project_id"],
            bible_json=r["bible_json"],
            updated_at=r["updated_at"],
        )
    finally:
        await db.close()


@router.get("/projects/{project_id}/logs", response_model=list[AgentLogResponse])
async def api_get_logs(project_id: int, chapter: int | None = None, limit: int = 100, offset: int = 0):
    """Get agent logs with optional filtering."""
    db = await get_db()
    try:
        query = "SELECT * FROM agent_logs WHERE project_id = ?"
        params = [project_id]
        if chapter is not None:
            query += " AND chapter_number = ?"
            params.append(chapter)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [AgentLogResponse(
            id=r["id"], project_id=r["project_id"],
            chapter_number=r["chapter_number"],
            agent_name=r["agent_name"], action=r["action"],
            prompt_preview=r["prompt_preview"],
            response_preview=r["response_preview"],
            elapsed_seconds=r["elapsed_seconds"],
            created_at=r["created_at"],
        ) for r in rows]
    finally:
        await db.close()


@router.get("/projects/{project_id}/export")
async def api_export(project_id: int, format: str = "txt"):
    """Export story as text."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        project = await cursor.fetchone()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        cursor = await db.execute(
            "SELECT * FROM chapters WHERE project_id = ? ORDER BY chapter_number",
            (project_id,),
        )
        chapters = await cursor.fetchall()

        lines = [project["title"], "=" * len(project["title"]), ""]
        for ch in chapters:
            lines.append(f"Chapter {ch['chapter_number']}: {ch['title']}")
            lines.append("")
            lines.append(ch["text"])
            lines.append("")
            lines.append("---")
            lines.append("")

        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            content="\n".join(lines),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={project['title']}.txt"},
        )
    finally:
        await db.close()


# WebSocket for real-time progress
@router.websocket("/ws/projects/{project_id}/progress")
async def ws_progress(websocket: WebSocket, project_id: int):
    """WebSocket endpoint for real-time generation progress."""
    await websocket.accept()

    queue: asyncio.Queue = asyncio.Queue()

    async def progress_callback(message: str, chapter: int, total: int, stage: str, debate_round: int = 0):
        await queue.put({
            "type": "progress",
            "chapter": chapter,
            "total": total,
            "stage": stage,
            "debate_round": debate_round,
            "message": message,
        })

    try:
        while True:
            try:
                # Wait for data from client (e.g., start command)
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)

                if data.get("action") == "start":
                    start_ch = data.get("start_chapter", 1)
                    end_ch = data.get("end_chapter")

                    # Start generation with this WebSocket's progress callback
                    asyncio.create_task(
                        generate_chapters(project_id, start_ch, end_ch, progress_callback)
                    )
                elif data.get("action") == "cancel":
                    cancel_generation(project_id)

            except asyncio.TimeoutError:
                pass

            # Drain the queue and send progress updates
            while not queue.empty():
                msg = await queue.get()
                await websocket.send_json(msg)

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        cancel_generation(project_id)
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
