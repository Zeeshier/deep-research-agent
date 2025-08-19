"""FastAPI application for the Deep Research Agent API."""
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import using the full module path
from config import settings, get_config
from monitoring.health import health_check, HealthChecker, HealthStatus
from monitoring.logger import get_logger

logger = get_logger("api")

# Create FastAPI app
app = FastAPI(
    title="Deep Research Agent API",
    description="API for the Deep Research Agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=Dict[str, Any], tags=["Health"])
async def get_health() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        health_data = await health_check()
        status_code = 200 if health_data.get("status") == HealthStatus.HEALTHY else 503
        return JSONResponse(content=health_data, status_code=status_code)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"status": HealthStatus.UNHEALTHY, "error": str(e)},
            status_code=500,
        )

# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Welcome to the Deep Research Agent API",
        "version": "1.0.0",
        "docs": "/docs",
    }

# Add startup event
@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    logger.info("Starting Deep Research Agent API...")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Debug mode: {settings.DEBUG}")

# Add shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run shutdown tasks."""
    logger.info("Shutting down Deep Research Agent API...")

# Run with uvicorn if executed directly
if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug",
    )
