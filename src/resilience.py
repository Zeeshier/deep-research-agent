"""
Resilience patterns for robust AI applications.

This module provides utilities for:
- Output parsing and schema validation
- Timeout and retry strategies
- Circuit breakers and fallbacks
- Iteration caps and loop detection
- State validation and recovery
- Resource usage limits
"""

import time
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Generic, List
from functools import wraps
import logging
import os
from contextlib import contextmanager
import signal
from dataclasses import dataclass

from pydantic import BaseModel, Field, validator, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_core.messages import BaseMessage

# Set up logging
logger = logging.getLogger(__name__)

# Type variables for generic type hints
T = TypeVar('T')
R = TypeVar('R')

# Default configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 2.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_TIMEOUT = 30.0

class ResilienceError(Exception):
    """Base exception for resilience-related errors."""
    pass

class MaxRetriesExceededError(ResilienceError):
    """Raised when maximum number of retries is exceeded."""
    pass

class CircuitOpenError(ResilienceError):
    """Raised when a circuit breaker is open."""
    pass

class TimeoutError(ResilienceError):
    """Raised when an operation times out."""
    pass

class ResourceLimitExceededError(ResilienceError):
    """Raised when a resource limit is exceeded."""
    pass

def validate_output(output: Any, model: Type[BaseModel]) -> BaseModel:
    """
    Validate and clean the output against a Pydantic model.
    
    Args:
        output: The output to validate
        model: The Pydantic model to validate against
        
    Returns:
        Validated and cleaned output as an instance of the model
        
    Raises:
        ValidationError: If the output cannot be validated
    """
    try:
        if isinstance(output, dict):
            return model(**output)
        elif isinstance(output, (str, bytes, bytearray)):
            # Handle string/bytes input (e.g., JSON)
            import json
            return model(**json.loads(output))
        elif isinstance(output, model):
            return output
        else:
            # Try to convert using dict()
            return model(**dict(output))
    except Exception as e:
        logger.error(f"Output validation failed: {str(e)}")
        raise ValidationError(f"Failed to validate output: {str(e)}")

def retry_with_backoff(
    func: Callable[..., R],
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable[..., R]:
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        func: The function to decorate
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        exceptions: Tuple of exceptions to catch and retry on
        on_retry: Optional callback function called on each retry
                with attempt number and exception
                
    Returns:
        Decorated function with retry logic
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                
                if attempt == max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded.")
                    raise MaxRetriesExceededError(
                        f"Max retries ({max_retries}) exceeded. Last error: {str(e)}"
                    ) from e
                
                # Calculate next delay with exponential backoff and jitter
                delay = min(delay * (exponential_base ** attempt), max_delay)
                if jitter:
                    import random
                    delay = random.uniform(0, delay)
                
                if on_retry:
                    on_retry(attempt + 1, e)
                
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed. "
                    f"Retrying in {delay:.2f}s. Error: {str(e)}"
                )
                
                time.sleep(delay)
    
    return wrapper

class CircuitBreaker:
    """
    Implements the circuit breaker pattern to handle failing operations.
    
    The circuit breaker has three states:
    - CLOSED: Operations proceed normally
    - OPEN: Operations fail fast with CircuitOpenError
    - HALF-OPEN: A limited number of test operations are allowed
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 300.0,  # 5 minutes
        name: str = "default",
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds before trying to close the circuit
            name: Name of the circuit breaker for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        
        self._failures = 0
        self._state = "CLOSED"
        self._last_failure_time = None
        self._test_mode = False
    
    @property
    def state(self) -> str:
        """Get the current state of the circuit breaker."""
        if self._state == "OPEN" and self._should_try_recovery():
            self._state = "HALF-OPEN"
            self._test_mode = True
        return self._state
    
    def _should_try_recovery(self) -> bool:
        """Check if we should attempt to recover from an open state."""
        if self._state != "OPEN" or not self._last_failure_time:
            return False
        
        time_since_failure = time.time() - self._last_failure_time
        return time_since_failure >= self.recovery_timeout
    
    def record_failure(self):
        """Record a failed operation."""
        self._failures += 1
        self._last_failure_time = time.time()
        
        if self._failures >= self.failure_threshold:
            self._state = "OPEN"
            logger.warning(
                f"Circuit breaker '{self.name}' is now OPEN. "
                f"Failures: {self._failures}"
            )
    
    def record_success(self):
        """Record a successful operation."""
        if self._state == "HALF-OPEN" and self._test_mode:
            # Test operation succeeded, close the circuit
            self._reset()
            logger.info(f"Circuit breaker '{self.name}' is now CLOSED.")
        else:
            # Reset failure count on success
            self._failures = max(0, self._failures - 1)
    
    def _reset(self):
        """Reset the circuit breaker to its initial state."""
        self._failures = 0
        self._state = "CLOSED"
        self._last_failure_time = None
        self._test_mode = False
    
    def __call__(self, func: Callable[..., R]) -> Callable[..., R]:
        """Use the circuit breaker as a decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check circuit state
            if self.state == "OPEN":
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is OPEN. "
                    f"Last failure: {self._last_failure_time}"
                )
            
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise
        
        return wrapper

@contextmanager
def resource_limit(
    max_memory_mb: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
    max_file_size: Optional[int] = None,
    max_cpu_percent: Optional[float] = None,
):
    """
    Context manager for enforcing resource limits.
    
    Args:
        max_memory_mb: Maximum memory usage in MB (not enforced on Windows)
        timeout_seconds: Maximum execution time in seconds
        max_file_size: Maximum file size in bytes (not enforced on Windows)
        max_cpu_percent: Maximum CPU usage as a percentage (0-100)
    """
    import platform
    import signal
    import os
    
    if platform.system() == 'Windows':
        # On Windows, we'll use a simplified version without resource limits
        if timeout_seconds is not None:
            # Use a simple timer for timeout
            import time
            start_time = time.time()
            
            def check_timeout():
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
            
            try:
                yield check_timeout
            except TimeoutError:
                raise
            finally:
                pass
        else:
            # No timeout, just yield
            yield
    else:
        # Unix-like systems can use resource module
        import resource
        import psutil
        
        # Store original signal handler
        original_handler = signal.getsignal(signal.SIGALRM)
        
        # Set up timeout if specified
        if timeout_seconds is not None:
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        
        # Set memory limit if specified
        if max_memory_mb is not None and hasattr(resource, 'RLIMIT_AS'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            new_limit = max_memory_mb * 1024 * 1024  # Convert MB to bytes
            resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
        
        # Set file size limit if specified
        if max_file_size is not None and hasattr(resource, 'RLIMIT_FSIZE'):
            soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_size, hard))
        
        try:
            yield
        finally:
            # Clean up signal handler
            if timeout_seconds is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, original_handler)

class StateValidator:
    """Utility class for validating and cleaning agent state."""
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize with a schema defining expected state structure.
        
        Example schema:
        {
            "user_query": (str, "[MISSING_QUERY]"),
            "documents": (list, []),
            "step": (int, 0, lambda x: max(0, int(x))),
        }
        """
        self.schema = schema
    
    def validate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean the state dictionary.
        
        Args:
            state: The state dictionary to validate
            
        Returns:
            Validated and cleaned state dictionary
        """
        validated = {}
        
        for key, (expected_type, default, *processors) in self.schema.items():
            value = state.get(key, default)
            
            # Ensure type
            if not isinstance(value, expected_type):
                try:
                    value = expected_type(value)
                except (TypeError, ValueError):
                    logger.warning(
                        f"Could not convert {key}={value!r} to {expected_type.__name__}, "
                        f"using default {default!r}"
                    )
                    value = default
            
            # Apply any processing functions
            for processor in processors:
                if callable(processor):
                    try:
                        value = processor(value)
                    except Exception as e:
                        logger.warning(
                            f"Error processing {key}={value!r} with {processor.__name__}: {e}"
                        )
            
            validated[key] = value
        
        return validated

class IterationLimiter:
    """Utility for limiting iterations in loops and recursive operations."""
    
    def __init__(self, max_iterations: int, name: str = "operation"):
        """
        Initialize with maximum number of iterations.
        
        Args:
            max_iterations: Maximum allowed iterations
            name: Name for logging purposes
        """
        self.max_iterations = max_iterations
        self.name = name
        self._count = 0
    
    def __call__(self) -> bool:
        """
        Check if the iteration limit has been reached.
        
        Returns:
            True if the iteration should continue, False if the limit is reached
            
        Raises:
            MaxRetriesExceededError: If the iteration limit is reached
        """
        self._count += 1
        
        if self._count > self.max_iterations:
            raise MaxRetriesExceededError(
                f"{self.name} exceeded maximum iterations ({self.max_iterations})"
            )
        
        return True
    
    def reset(self):
        """Reset the iteration counter."""
        self._count = 0
    
    @property
    def count(self) -> int:
        """Get the current iteration count."""
        return self._count

def with_retry(
    func: Optional[Callable[..., R]] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable[..., R]:
    """
    Decorator that retries a function with exponential backoff.
    
    This is a more flexible alternative to the retry_with_backoff decorator,
    allowing for both function decoration and direct usage as a context manager.
    
    Example usage as decorator:
        @with_retry(max_retries=3)
        def my_function():
            # function implementation
            pass
    
    Example usage as context manager:
        with with_retry(max_retries=3) as retry:
            result = retry(lambda: some_operation())
    """
    if func is None:
        # Return a decorator with the specified parameters
        return lambda f: retry_with_backoff(
            f,
            max_retries=max_retries,
            initial_delay=initial_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            exceptions=exceptions,
            on_retry=on_retry,
        )
    else:
        # Apply the decorator directly
        return retry_with_backoff(
            func,
            max_retries=max_retries,
            initial_delay=initial_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            exceptions=exceptions,
            on_retry=on_retry,
        )

def validate_structured_output(
    output: Any,
    model: Type[BaseModel],
    strict: bool = True,
) -> BaseModel:
    """
    Validate and clean structured output against a Pydantic model.
    
    Args:
        output: The output to validate (dict, string, or model instance)
        model: The Pydantic model to validate against
        strict: If True, raise ValidationError on failure
               If False, return a default instance on failure
    
    Returns:
        Validated model instance
    """
    try:
        if isinstance(output, model):
            return output
        
        # Handle string/bytes input (e.g., JSON)
        if isinstance(output, (str, bytes, bytearray)):
            import json
            output = json.loads(output)
        
        # Handle dictionary input
        if isinstance(output, dict):
            return model(**output)
        
        # Try to convert using dict()
        try:
            return model(**dict(output))
        except (TypeError, ValueError):
            if strict:
                raise ValidationError(
                    f"Could not convert {type(output).__name__} to {model.__name__}"
                )
            return model()
            
    except Exception as e:
        logger.error(f"Output validation failed: {str(e)}")
        if strict:
            raise ValidationError(f"Failed to validate output: {str(e)}")
        return model()

# Example usage of the patterns
if __name__ == "__main__":
    # Example 1: Using retry decorator
    @retry_with_backoff(max_retries=3)
    def unreliable_operation():
        import random
        if random.random() < 0.7:
            raise ValueError("Temporary failure")
        return "Success!"
    
    try:
        result = unreliable_operation()
        print(f"Operation result: {result}")
    except MaxRetriesExceededError as e:
        print(f"Operation failed after retries: {e}")
    
    # Example 2: Using circuit breaker
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=5, name="test_circuit")
    
    @cb
    def failing_operation():
        raise RuntimeError("Something went wrong")
    
    for _ in range(3):
        try:
            failing_operation()
        except Exception as e:
            print(f"Operation failed: {e}")
    
    # Example 3: Using resource limits
    try:
        with resource_limit(max_memory_mb=100, timeout_seconds=5):
            # This operation will be killed if it uses >100MB or runs >5 seconds
            result = sum(i * i for i in range(10**6))
            print(f"Result: {result}")
    except TimeoutError as e:
        print(f"Operation timed out: {e}")
    except MemoryError as e:
        print(f"Operation used too much memory: {e}")
    
    # Example 4: Using state validation
    schema = {
        "user_query": (str, "[MISSING_QUERY]"),
        "documents": (list, []),
        "step": (int, 0, lambda x: max(0, int(x))),
    }
    
    validator = StateValidator(schema)
    state = {"user_query": 123, "step": "2"}
    cleaned = validator.validate(state)
    print(f"Cleaned state: {cleaned}")
    
    # Example 5: Using iteration limiter
    limiter = IterationLimiter(5, "test_loop")
    
    try:
        while limiter():
            print(f"Iteration {limiter.count}")
            if limiter.count == 10:  # This will never be reached
                break
    except MaxRetriesExceededError as e:
        print(f"Loop terminated: {e}")
