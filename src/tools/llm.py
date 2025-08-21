import os
import logging
from functools import wraps
from typing import Any, Callable, TypeVar, cast
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState,
    before_sleep_log,
)
from langchain_groq import ChatGroq
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY")

# Custom exception for LLM errors
class LLMError(Exception):
    """Custom exception for LLM-related errors."""
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

# Initialize the LLM client with timeout
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    api_key=GROQ_API_KEY,
    timeout=TIMEOUT,
    request_timeout=TIMEOUT,
)

# Wrap the LLM's _generate method with retry logic
original_generate = llm._generate

@wraps(original_generate)
@retry_decorator
async def wrapped_generate(*args: Any, **kwargs: Any) -> Any:
    try:
        # Set timeout if not provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = TIMEOUT
        return await original_generate(*args, **kwargs)
    except Exception as e:
        logger.error("LLM generation failed: %s", str(e), exc_info=True)
        raise LLMError(f"Failed to generate response: {str(e)}") from e

# Apply the wrapped method
llm._generate = wrapped_generate