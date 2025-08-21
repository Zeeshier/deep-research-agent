"""Health check endpoints for the application."""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get(
    "/healthz",
    summary="Health check endpoint",
    description="Returns the health status of the application.",
    response_description="Application health status",
    response_model=Dict[str, str]
)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint that returns the status of the application.
    
    Returns:
        Dict with status and timestamp
    """
    try:
        # Add any health checks here (e.g., database connection, external services)
        return {"status": "ok"}
    except Exception as e:
        logger.error("Health check failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=503, detail="Service Unavailable")

@router.get(
    "/readyz",
    summary="Readiness check endpoint",
    description="Returns the readiness status of the application.",
    response_description="Application readiness status",
    response_model=Dict[str, str]
)
async def readiness_check() -> Dict[str, str]:
    """
    Readiness check endpoint that verifies all required services are available.
    
    Returns:
        Dict with status and timestamp
    """
    try:
        # Add readiness checks here (e.g., database connection, external services)
        return {"status": "ready"}
    except Exception as e:
        logger.error("Readiness check failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=503, detail="Service Not Ready")

# Example of how to add the router to your FastAPI app:
# from fastapi import FastAPI
# app = FastAPI()
# app.include_router(health_router, prefix="/api")
