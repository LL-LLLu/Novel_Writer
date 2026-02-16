from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import torch

from .inference import NovelGenerator
from loguru import logger

# Global generator instance
generator: Optional[NovelGenerator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global generator

    base_model = "unsloth/llama-3-8b-bnb-4bit"
    lora_path = Path("lora_model") if Path("lora_model").exists() else None

    try:
        generator = NovelGenerator(
            base_model_path=base_model,
            lora_path=lora_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        generator = None

    yield

    generator = None

app = FastAPI(
    title="Novel Writer API",
    description="Generate novel text with fine-tuned models",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to continue")
    max_tokens: int = Field(default=500, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)

class GenerationResponse(BaseModel):
    generated_text: str
    prompt_length: int
    generated_length: int

class ChapterRequest(BaseModel):
    context: str = Field(..., description="Previous chapter text")
    max_tokens: int = Field(default=2000, ge=500, le=4096)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate novel text."""
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        generated = generator.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )

        return GenerationResponse(
            generated_text=generated,
            prompt_length=len(request.prompt),
            generated_length=len(generated)
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/chapter")
async def generate_chapter(request: ChapterRequest):
    """Generate a full chapter."""
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        generated = generator.generate_chapter(request.context, request.max_tokens)

        return {
            "chapter": generated,
            "length": len(generated)
        }
    except Exception as e:
        logger.error(f"Chapter generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models."""
    models_dir = Path(".")
    lora_models = [d.name for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("lora")]

    return {
        "base_model": "unsloth/llama-3-8b-bnb-4bit",
        "lora_models": lora_models,
        "current_lora": "lora_model" if Path("lora_model").exists() else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
