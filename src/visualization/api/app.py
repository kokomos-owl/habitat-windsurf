"""FastAPI application for visualization service."""

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .router import router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI app."""
    # Startup
    logger.info("Starting visualization service")
    
    # Create visualization directory if it doesn't exist
    vis_dir = Path("visualizations")
    vis_dir.mkdir(exist_ok=True)
    
    yield
    # Shutdown
    logger.info("Shutting down visualization service")

# Create FastAPI app
app = FastAPI(
    title="Habitat Visualization Service",
    description="API for graph visualization and evolution tracking",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount visualization files
vis_dir = Path("visualizations")
app.mount("/visualizations", StaticFiles(directory=str(vis_dir)), name="visualizations")

# Include router
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Serve the visualization interface."""
    return FileResponse(static_dir / "index.html")
