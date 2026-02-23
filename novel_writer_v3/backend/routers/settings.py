from fastapi import APIRouter
from ..database import get_db
from ..schemas import SettingsUpdate, SettingsResponse

router = APIRouter(tags=["settings"])

SETTINGS_KEYS = [
    "gemini_api_key", "qwen_api_key", "gemini_model", "qwen_model",
    "qwen_base_url", "gemini_temperature", "qwen_temperature",
    "max_output_tokens", "max_debate_rounds", "chapter_min_chars", "chapter_max_chars",
]

SETTINGS_DEFAULTS = {
    "gemini_api_key": "",
    "qwen_api_key": "",
    "gemini_model": "gemini-3.1-pro-preview",
    "qwen_model": "qwen3.5-plus",
    "qwen_base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "gemini_temperature": "0.7",
    "qwen_temperature": "0.8",
    "max_output_tokens": "4096",
    "max_debate_rounds": "3",
    "chapter_min_chars": "5000",
    "chapter_max_chars": "8000",
}


@router.get("/settings", response_model=SettingsResponse)
async def get_settings():
    db = await get_db()
    try:
        result = {}
        for key in SETTINGS_KEYS:
            cursor = await db.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = await cursor.fetchone()
            if row:
                result[key] = row["value"]
            else:
                result[key] = SETTINGS_DEFAULTS.get(key, "")
        # Convert numeric fields
        for key in ["gemini_temperature", "qwen_temperature"]:
            result[key] = float(result[key])
        for key in ["max_output_tokens", "max_debate_rounds", "chapter_min_chars", "chapter_max_chars"]:
            result[key] = int(result[key])
        return SettingsResponse(**result)
    finally:
        await db.close()


@router.put("/settings", response_model=SettingsResponse)
async def update_settings(data: SettingsUpdate):
    db = await get_db()
    try:
        update_dict = data.model_dump(exclude_none=True)
        for key, value in update_dict.items():
            await db.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = ?",
                (key, str(value), str(value)),
            )
        await db.commit()
    finally:
        await db.close()
    return await get_settings()
