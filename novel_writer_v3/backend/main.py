from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and clean up on shutdown."""
    await init_db()
    yield


app = FastAPI(
    title="Novel Writer V3",
    description="Multi-agent novel writing backend",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


# Include routers as they become available
try:
    from .routers import projects, settings, outline, generation

    app.include_router(projects.router, prefix="/api")
    app.include_router(settings.router, prefix="/api")
    app.include_router(outline.router, prefix="/api")
    app.include_router(generation.router, prefix="/api")
except ImportError:
    pass


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
