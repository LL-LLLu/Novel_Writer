from fastapi import APIRouter, HTTPException
from ..database import get_db
from ..schemas import ProjectCreate, ProjectUpdate, ProjectResponse

router = APIRouter(tags=["projects"])


def _row_to_project(row) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "premise": row["premise"],
        "outline_text": row["outline_text"],
        "guidance_text": row["guidance_text"],
        "style_rules_json": row["style_rules_json"],
        "language": row["language"],
        "target_chapters": row["target_chapters"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@router.post("/projects", response_model=ProjectResponse)
async def create_project(data: ProjectCreate):
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO projects (title, premise, language, target_chapters) VALUES (?, ?, ?, ?)",
            (data.title, data.premise, data.language, data.target_chapters),
        )
        await db.commit()
        project_id = cursor.lastrowid
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
        return ProjectResponse(**_row_to_project(row))
    finally:
        await db.close()


@router.get("/projects", response_model=list[ProjectResponse])
async def list_projects():
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM projects ORDER BY updated_at DESC")
        rows = await cursor.fetchall()
        return [ProjectResponse(**_row_to_project(row)) for row in rows]
    finally:
        await db.close()


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int):
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")
        return ProjectResponse(**_row_to_project(row))
    finally:
        await db.close()


@router.put("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: int, data: ProjectUpdate):
    db = await get_db()
    try:
        # First check project exists
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")

        update_dict = data.model_dump(exclude_none=True)
        if update_dict:
            set_clause = ", ".join(f"{k} = ?" for k in update_dict.keys())
            set_clause += ", updated_at = datetime('now')"
            values = list(update_dict.values()) + [project_id]
            await db.execute(
                f"UPDATE projects SET {set_clause} WHERE id = ?",
                values,
            )
            await db.commit()

        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
        return ProjectResponse(**_row_to_project(row))
    finally:
        await db.close()


@router.delete("/projects/{project_id}")
async def delete_project(project_id: int):
    db = await get_db()
    try:
        cursor = await db.execute("SELECT id FROM projects WHERE id = ?", (project_id,))
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Project not found")
        await db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        await db.commit()
        return {"status": "deleted"}
    finally:
        await db.close()
