"""FastAPI application for visualization service."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .router import router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Habitat Visualization Service",
    description="API for graph visualization and evolution tracking",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include router
app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Handle application startup."""
    logger.info("Starting visualization service")

@app.on_event("shutdown")
async def shutdown_event():
    """Handle application shutdown."""
    logger.info("Shutting down visualization service")
