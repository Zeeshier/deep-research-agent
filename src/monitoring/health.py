"""Health check module for the Deep Research Agent."""
import asyncio
import time
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Awaitable
import httpx
from pydantic import BaseModel, Field

from config import settings
from monitoring.logger import get_logger
from tools.llm import llm as llm_client
from tools.composio_tools import web_search

logger = get_logger("health")

class HealthStatus(str, Enum):
    """Health status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class HealthCheckResult(BaseModel):
    """Result of a health check."""
    status: HealthStatus = Field(..., description="Health status of the component")
    component: str = Field(..., description="Name of the component being checked")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details about the health check"
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Timestamp when the check was performed"
    )

    class Config:
        json_encoders = {
            "timestamp": lambda v: v.isoformat() if hasattr(v, "isoformat") else v
        }

class HealthCheckManager:
    """Manages health checks for the application."""
    
    def __init__(self):
        self.checks: List[Callable[[], Awaitable[HealthCheckResult]]] = []
    
    def register_check(self, check: Callable[[], Awaitable[HealthCheckResult]]) -> None:
        """Register a health check function.
        
        Args:
            check: Async function that returns a HealthCheckResult
        """
        self.checks.append(check)
        
    def register_checks(self, *checks: Callable[[], Awaitable[HealthCheckResult]]) -> None:
        """Register multiple health check functions.
        
        Args:
            *checks: Async functions that return HealthCheckResult objects
        """
        self.checks.extend(checks)

    async def get_health_status(self) -> Dict:
        """Get the overall health status of the application."""
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        # Run all checks in parallel
        async def run_check(check):
            try:
                result = await check()
                results[result.component] = result
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status != HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.DEGRADED
            except Exception as e:
                logger.error(f"Health check '{check.__name__}' failed: {str(e)}", exc_info=True)
                results[check.__name__] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    component=check.__name__,
                    details={"error": str(e)},
                )
                overall_status = HealthStatus.UNHEALTHY
        
        await asyncio.gather(*[run_check(check) for check in self.checks])
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "services": {k: v.dict() for k, v in results.items()},
        }

async def check_app_status() -> HealthCheckResult:
    """Check basic application status."""
    return HealthCheckResult(
        status=HealthStatus.HEALTHY,
        component="application",
        details={
            "version": "1.0.0",  # TODO: Get from package version
            "environment": "development" if settings.DEBUG else "production",
            "debug": settings.DEBUG,
        },
    )

async def check_llm_health() -> HealthCheckResult:
    """Check if the LLM service is healthy."""
    try:
        # Simple test to check if LLM is responding
        # This is a placeholder - implement actual LLM health check
        await asyncio.sleep(0.1)  # Simulate network delay
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            component="llm",
            details={
                "model": settings.LLM_MODEL,
                "temperature": settings.LLM_TEMPERATURE,
                "max_tokens": settings.LLM_MAX_TOKENS
            }
        )
    except Exception as e:
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            component="llm",
            details={"error": str(e)}
        )

async def check_web_search() -> HealthCheckResult:
    """Check if the web search functionality is working."""
    if not settings.WEB_SEARCH_ENABLED:
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            component="web_search",
            details={"enabled": False, "reason": "Web search is disabled in settings"},
        )
            
    try:
        results = web_search(query="test", max_results=1)
        return HealthCheckResult(
            status=HealthStatus.HEALTHY if results else HealthStatus.DEGRADED,
            component="web_search",
            details={
                "results_count": len(results) if results else 0,
            },
        )
    except Exception as e:
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            component="web_search",
            details={
                "error": str(e),
            },
        )

async def check_google_docs() -> HealthCheckResult:
    """Check if Google Docs integration is working."""
    if not settings.GOOGLE_DOCS_ENABLED:
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            component="google_docs",
            details={"enabled": False, "reason": "Google Docs is disabled in settings"},
        )
            
    return HealthCheckResult(
        status=HealthStatus.HEALTHY,
        component="google_docs",
        details={"enabled": True},
    )

# Create a singleton instance
health_checker = HealthCheckManager()
health_checker.register_checks(
    check_app_status,
    check_llm_health,
    check_web_search,
    check_google_docs,
)

# FastAPI/Starlette compatible health check endpoint
async def health_check() -> Dict:
    """Health check endpoint for monitoring."""
    return await health_checker.get_health_status()
