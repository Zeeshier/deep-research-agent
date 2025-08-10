import functools
import time
from monitoring.logger import get_logger

logger = get_logger("metrics")

def timed(func):
    """
    Decorator that logs execution time of any callable.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.info(f"{func.__name__} took {elapsed:.3f}s")
    return wrapper