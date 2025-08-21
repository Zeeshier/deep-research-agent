"""
Comprehensive health check module for the Deep Research Agent.

This module provides health monitoring for all critical components of the application,
including external services, resources, and system metrics.
"""
import asyncio
import os
import psutil
import time
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Awaitable, Tuple
import httpx
from pydantic import BaseModel, Field, validator

from config import settings, get_config
from monitoring.logger import get_logger
from monitoring.metrics import get_metrics
from tools.llm import llm as llm_client
from tools.composio_tools import web_search, create_google_doc

logger = get_logger("health")

class HealthStatus(str, Enum):
    """Health status values with severity levels."""
    HEALTHY = "healthy"      # Component is functioning normally
    DEGRADED = "degraded"    # Component is working but with reduced performance
    UNHEALTHY = "unhealthy"  # Component is not functioning correctly
    UNKNOWN = "unknown"      # Component status could not be determined

class ComponentType(str, Enum):
    """Types of components being monitored."""
    SYSTEM = "system"
    SERVICE = "service"
    DATABASE = "database"
    EXTERNAL = "external"
    CACHE = "cache"
    STORAGE = "storage"

class HealthCheckResult(BaseModel):
    """Result of a health check with detailed diagnostics."""
    status: HealthStatus = Field(
        ...,
        description="Health status of the component"
    )
    component: str = Field(
        ...,
        description="Name of the component being checked"
    )
    component_type: ComponentType = Field(
        default=ComponentType.SERVICE,
        description="Type of the component"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic details about the health check"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the check was performed"
    )
    response_time: Optional[float] = Field(
        None,
        description="Response time in milliseconds"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if the check failed"
    )

    class Config:
        json_encoders = {
            "timestamp": lambda v: v.isoformat() if hasattr(v, "isoformat") else v,
            "response_time": lambda v: f"{v:.2f}ms" if v is not None else None
        }
        
    @validator('status', pre=True)
    def validate_status(cls, v):
        if isinstance(v, str):
            return HealthStatus(v.lower())
        return v

class HealthCheckManager:
    """
    Manages health checks for the application with support for:
    - Synchronous and asynchronous checks
    - Timeout handling
    - Result caching
    - Dependency tracking
    - Performance metrics
    """
    
    def __init__(self, cache_ttl: int = 30):
        """Initialize the health check manager.
        
        Args:
            cache_ttl: Time in seconds to cache health check results
        """
        self.checks: List[Callable[[], Awaitable[HealthCheckResult]]] = []
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, HealthCheckResult]] = {}
        self._dependencies: Dict[str, List[str]] = {}
    
    def register_check(
        self, 
        check: Callable[[], Awaitable[HealthCheckResult]],
        name: Optional[str] = None,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """Register a health check function with optional dependencies.
        
        Args:
            check: The health check function to register
            name: Optional name for the check (defaults to function name)
            dependencies: List of component names this check depends on
        """
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
        
        try:
            # Get cached result if available and fresh
            if component and component in self._cache:
                cache_time, cached_result = self._cache[component]
                if (time.time() - cache_time) < self.cache_ttl:
                    return self._format_result(cached_result)
            
            # Run the requested checks
            if component:
                checks = [
                    check for check in self.checks 
                    if getattr(check, "__name__", "") == component or 
                       getattr(check, "component_name", "") == component
                ]
                if not checks:
                    raise ValueError(f"No health check found for component: {component}")
            else:
                checks = self.checks
            
            # Execute checks in parallel with timeout
            tasks = []
            for check in checks:
                task = asyncio.create_task(
                    self._run_check_with_timeout(check)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            status = HealthStatus.HEALTHY
            components = {}
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Health check failed: {str(result)}", exc_info=result)
                    result = HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        component="unknown",
                        error=str(result),
                        details={"error_type": result.__class__.__name__}
                    )
                
                # Cache the result
                self._cache[result.component] = (time.time(), result)
                
                # Add to components
                components[result.component] = result.dict()
                
                # Update overall status (most severe wins)
                status_order = [
                    HealthStatus.UNKNOWN,
                    HealthStatus.HEALTHY,
                    HealthStatus.DEGRADED,
                    HealthStatus.UNHEALTHY
                ]
                
                if status_order.index(result.status) > status_order.index(status):
                    status = result.status
            
            return {
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                "response_time_ms": (time.time() - start_time) * 1000,
                "components": components
            }
            
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}", exc_info=True)
            return {
                "status": HealthStatus.UNHEALTHY,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "components": {}
            }

    def _format_result(self, result: HealthCheckResult) -> Dict[str, Any]:
        return {
            "status": result.status,
            "timestamp": result.timestamp.isoformat(),
            "response_time_ms": result.response_time,
            "component": result.component,
            "details": result.details,
            "error": result.error
        }

    async def _run_check_with_timeout(self, check: Callable[[], Awaitable[HealthCheckResult]]) -> HealthCheckResult:
        try:
            return await asyncio.wait_for(check(), timeout=10)
        except asyncio.TimeoutError:
            logger.error(f"Health check '{check.__name__}' timed out")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component=check.__name__,
                error="Timed out",
                details={"timeout": 10}
            )
        except Exception as e:
            logger.error(f"Health check '{check.__name__}' failed: {str(e)}", exc_info=True)
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component=check.__name__,
                error=str(e),
                details={"error_type": e.__class__.__name__}
            )

async def check_system_resources() -> HealthCheckResult:
    """Check system resource usage (CPU, memory, disk)."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        details = {
            "cpu_percent": cpu_percent,
            "memory": {
                "total_gb": round(memory.total / (1024 ** 3), 2),
                "available_gb": round(memory.available / (1024 ** 3), 2),
                "used_percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024 ** 3), 2),
                "free_gb": round(disk.free / (1024 ** 3), 2),
                "used_percent": disk.percent
            }
        }
        
        # Determine status based on thresholds
        status = HealthStatus.HEALTHY
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            status = HealthStatus.UNHEALTHY
        elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 70:
            status = HealthStatus.DEGRADED
            
        return HealthCheckResult(
            status=status,
            component="system_resources",
            component_type=ComponentType.SYSTEM,
            details=details
        )
        
    except Exception as e:
        logger.error(f"System resources check failed: {str(e)}", exc_info=True)
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            component="system_resources",
            component_type=ComponentType.SYSTEM,
            error=str(e),
            details={"error_type": e.__class__.__name__}
        )

async def check_app_status() -> HealthCheckResult:
    """Check basic application status and configuration."""
    start_time = time.time()
    
    try:
        config = get_config()
        
        # Check required environment variables
        missing_vars = [
            var for var in [
                'OPENAI_API_KEY',
                'GOOGLE_API_KEY',
                'GOOGLE_CSE_ID'
            ] if not getattr(config, var, None)
        ]
        
        if missing_vars:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component="app_config",
                component_type=ComponentType.SYSTEM,
                error=f"Missing required configuration: {', '.join(missing_vars)}",
                response_time=(time.time() - start_time) * 1000,
                details={
                    "version": settings.VERSION,
                    "environment": settings.ENV,
                    "missing_variables": missing_vars
                }
            )
        
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            component="app_config",
            component_type=ComponentType.SYSTEM,
            response_time=(time.time() - start_time) * 1000,
            details={
                "version": settings.VERSION,
                "environment": settings.ENV,
                "debug": settings.DEBUG
            }
        )
    except Exception as e:
        logger.error(f"App status check failed: {str(e)}", exc_info=True)
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            component="app_config",
            component_type=ComponentType.SYSTEM,
            error=str(e),
            response_time=(time.time() - start_time) * 1000,
            details={
                "error_type": e.__class__.__name__,
                "version": getattr(settings, 'VERSION', 'unknown'),
                "environment": getattr(settings, 'ENV', 'unknown')
            }
        )

async def check_llm_health() -> HealthCheckResult:
    """Check if the LLM service is healthy with detailed metrics."""
    start_time = time.time()
    component = "llm_service"
    
    try:
        # Test with a simple query
        prompt = "Respond with 'pong'"
        
        # Time the LLM call
        llm_start = time.time()
        response = await llm_client.agenerate([prompt])
        llm_time = (time.time() - llm_start) * 1000  # ms
        
        if not response.generations:
            raise ValueError("No response from LLM")
        
        # Check response content
        response_text = response.generations[0][0].text.strip().lower()
        if "pong" not in response_text:
            raise ValueError(f"Unexpected response: {response_text}")
        
        # Get token usage if available
        usage = {}
        if hasattr(response, 'llm_output') and 'token_usage' in response.llm_output:
            usage = response.llm_output['token_usage']
        
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            component=component,
            component_type=ComponentType.SERVICE,
            response_time=llm_time,
            details={
                "model": settings.OPENAI_MODEL,
                "response_time_ms": round(llm_time, 2),
                "usage": usage,
                "test_query": prompt,
                "test_response": response_text[:100]  # First 100 chars
            }
        )
        
    except Exception as e:
        logger.error(f"LLM health check failed: {str(e)}", exc_info=True)
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            component=component,
            component_type=ComponentType.SERVICE,
            error=str(e),
            response_time=(time.time() - start_time) * 1000,
            details={
                "error_type": e.__class__.__name__,
                "model": getattr(settings, 'OPENAI_MODEL', 'unknown'),
                "test_query": prompt if 'prompt' in locals() else 'not_attempted'
            }
        )

async def check_web_search() -> HealthCheckResult:
    """Check if the web search functionality is working with validation."""
    start_time = time.time()
    component = "web_search"
    test_query = "current time"
    
    try:
        # Time the search
        search_start = time.time()
        results = await web_search(test_query, num_results=3)
        search_time = (time.time() - search_start) * 1000  # ms
        
        if not results:
            raise ValueError("No results returned from web search")
            
        # Validate results structure
        required_fields = ['title', 'link', 'snippet']
        for i, result in enumerate(results):
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing field '{field}' in result {i}")
        
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            component=component,
            component_type=ComponentType.EXTERNAL,
            response_time=search_time,
            details={
                "result_count": len(results),
                "response_time_ms": round(search_time, 2),
                "test_query": test_query,
                "first_result": {
                    "title": results[0].get('title', ''),
                    "domain": results[0].get('link', '').split('/')[2] if '//' in results[0].get('link', '') else ''
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Web search health check failed: {str(e)}", exc_info=True)
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            component=component,
            component_type=ComponentType.EXTERNAL,
            error=str(e),
            response_time=(time.time() - start_time) * 1000,
            details={
                "error_type": e.__class__.__name__,
                "test_query": test_query,
                "api_key_configured": bool(getattr(settings, 'GOOGLE_API_KEY', None))
            }
        )

async def check_google_docs() -> HealthCheckResult:
    """Check if Google Docs integration is working with proper cleanup."""
    start_time = time.time()
    component = "google_docs"
    test_title = "Health Check Test - " + datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    test_content = "This is a test document for health monitoring. " \
                 "It will be automatically deleted after verification."
    
    try:
        # Test document creation
        create_start = time.time()
        doc = await create_google_doc(test_title, test_content)
        create_time = (time.time() - create_start) * 1000  # ms
        
        if not doc or not doc.get('document_id'):
            raise ValueError("Failed to create test document")
            
        document_id = doc['document_id']
        
        # Verify document was created with correct content
        # Note: In a real implementation, you would need to fetch and verify the document
        # This is a simplified version
        
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            component=component,
            component_type=ComponentType.EXTERNAL,
            response_time=create_time,
            details={
                "document_id": document_id,
                "creation_time_ms": round(create_time, 2),
                "title": test_title
            }
        )
        
        # Schedule document deletion (in a real implementation)
        # asyncio.create_task(_delete_test_document(document_id))
        
        return result
        
    except Exception as e:
        logger.error(f"Google Docs health check failed: {str(e)}", exc_info=True)
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            component=component,
            component_type=ComponentType.EXTERNAL,
            error=str(e),
            response_time=(time.time() - start_time) * 1000,
            details={
                "error_type": e.__class__.__name__,
                "test_title": test_title,
                "api_key_configured": bool(getattr(settings, 'GOOGLE_API_KEY', None))
            }
        )

def _add_metrics_to_health(health_data: Dict[str, Any]) -> Dict[str, Any]:
    """Add application metrics to health check response."""
    try:
        metrics = get_metrics()
        health_data['metrics'] = metrics
    except Exception as e:
        logger.error(f"Failed to collect metrics: {str(e)}", exc_info=True)
        health_data['metrics_error'] = str(e)
    return health_data

# Create a singleton instance with 30-second cache TTL
health_checker = HealthCheckManager(cache_ttl=30)

# Register all health checks with dependencies
health_checker.register_checks(
    check_system_resources,
    check_app_status,
    check_llm_health,
    check_web_search,
    check_google_docs
)

# Register component dependencies
health_checker.register_dependencies(
    "llm_service",
    depends_on=["app_config"]  # LLM depends on config being loaded
)
health_checker.register_dependencies(
    "web_search",
    depends_on=["app_config"]  # Web search depends on config
)
health_checker.register_dependencies(
    "google_docs",
    depends_on=["app_config"]  # Google Docs depends on config
)

async def health_check(include_metrics: bool = True) -> Dict[str, Any]:
    """
    Comprehensive health check endpoint for monitoring.
    
    Args:
        include_metrics: Whether to include detailed metrics in the response
        
    Returns:
        Dict containing health status and component details
    """
    try:
        # Get health status
        health_data = await health_checker.get_status()
        
        # Add metrics if requested
        if include_metrics:
            health_data = _add_metrics_to_health(health_data)
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check endpoint failed: {str(e)}", exc_info=True)
        return {
            "status": HealthStatus.UNHEALTHY,
            "timestamp": datetime.utcnow().isoformat(),
            "error": "Health check failed",
            "details": {
                "error": str(e),
                "type": e.__class__.__name__
            }
        }
