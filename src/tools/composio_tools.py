"""Composio Tools with enhanced resilience patterns."""
import os
import json
import logging
from typing import List, Dict, Any, Optional, TypeVar, Callable, Type, cast
from functools import wraps
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState,
    before_sleep_log,
)
from composio_langchain import ComposioToolSet, Action
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
MAX_RETRIES = 3
TIMEOUT = 30  # seconds
MIN_RETRY_DELAY = 1  # second
MAX_RETRY_DELAY = 4  # seconds

# Get API key
COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
if not COMPOSIO_API_KEY:
    raise RuntimeError("Missing COMPOSIO_API_KEY")

# Initialize toolset with timeout
toolset = ComposioToolSet(
    api_key=COMPOSIO_API_KEY,
    timeout=TIMEOUT
)

# Custom exceptions
class ComposioToolError(Exception):
    """Base exception for Composio tool errors."""
    pass

class WebSearchError(ComposioToolError):
    """Exception raised for web search related errors."""
    pass

class GoogleDocsError(ComposioToolError):
    """Exception raised for Google Docs related errors."""
    pass

def log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log the retry attempt."""
    if retry_state.attempt_number > 1:
        logger.warning(
            "Retrying %s: attempt %s",
            retry_state.fn.__name__,
            retry_state.attempt_number,
            exc_info=retry_state.outcome.exception() if retry_state.outcome else None,
        )

# Configure retry decorator
retry_decorator = retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=MIN_RETRY_DELAY, max=MAX_RETRY_DELAY),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError, TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

def handle_composio_response(resp: Any) -> Any:
    """Handle and validate Composio API response."""
    if isinstance(resp, str):
        try:
            resp = json.loads(resp)
        except json.JSONDecodeError as e:
            raise ComposioToolError(f"Invalid JSON response: {resp}") from e
    
    if not isinstance(resp, dict):
        raise ComposioToolError(f"Unexpected response type: {type(resp).__name__}")
    
    return resp

@retry_decorator
def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Perform a web search using Tavily.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        List of search results
        
    Raises:
        WebSearchError: If the search fails after all retries
    """
    try:
        resp = toolset.execute_action(
            action=Action.TAVILY_TAVILY_SEARCH,
            params={
                "query": query,
                "max_results": max_results,
                "timeout": TIMEOUT * 1000  # Convert to milliseconds
            }
        )
        
        resp = handle_composio_response(resp)
        return resp.get("data", [])
        
    except Exception as e:
        error_msg = f"Web search failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if isinstance(e, (httpx.RequestError, httpx.HTTPStatusError, TimeoutError)):
            raise  # Let the retry decorator handle these
        raise WebSearchError(error_msg) from e

@retry_decorator
def create_google_doc(title: str, body: str) -> str:
    """
    Create a Google Doc with the given title and content.
    
    Args:
        title: The title of the document
        body: The markdown content of the document
        
    Returns:
        The web view link to the created document
        
    Raises:
        GoogleDocsError: If document creation fails after all retries
    """
    try:
        resp = toolset.execute_action(
            action=Action.GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN,
            params={
                "title": title,
                "body": body,
                "timeout": TIMEOUT * 1000  # Convert to milliseconds
            }
        )
        
        resp = handle_composio_response(resp)
        return resp.get("data", {}).get("webViewLink", "")
        
    except Exception as e:
        error_msg = f"Google Docs creation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if isinstance(e, (httpx.RequestError, httpx.HTTPStatusError, TimeoutError)):
            raise  # Let the retry decorator handle these
        raise GoogleDocsError(error_msg) from e