"""
Enhanced logging configuration with support for both text and JSON formatting.

Environment Variables:
    LOG_LEVEL: Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    LOG_FORMAT: Set to 'json' for JSON formatted logs, or anything else for text
"""
import os
import sys
import json
import logging
import logging.handlers
from typing import Any, Dict, Optional, Union
from datetime import datetime

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Configure default log level based on environment
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "").lower()

class JsonFormatter(logging.Formatter):
    """Custom formatter for JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_record: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        
        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in [
                'args', 'asctime', 'created', 'exc_info', 'exc_text',
                'filename', 'funcName', 'id', 'levelname', 'levelno',
                'lineno', 'module', 'msecs', 'msecs', 'message', 'msg',
                'name', 'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'thread', 'threadName', 'extra'
            ] and not key.startswith('_'):
                log_record[key] = value
        
        # Handle extra attributes from the 'extra' parameter
        if hasattr(record, 'extra') and isinstance(record.extra, dict):
            log_record.update(record.extra)
        
        return json.dumps(log_record, ensure_ascii=False)

def get_logger(
    name: str,
    level: Optional[Union[str, int]] = None,
    json_format: Optional[bool] = None
) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Name of the logger (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, use JSON formatting. If None, use LOG_FORMAT env var
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Don't add handlers if they're already configured
    if logger.handlers:
        return logger
    
    # Set log level
    if level is None:
        level = LOG_LEVEL
    
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logger.setLevel(level)
    
    # Determine format
    use_json = json_format if json_format is not None else (LOG_FORMAT == 'json')
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Set formatter
    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    # Prevent duplicate logs in some environments
    logger.propagate = False
    
    return logger

# Example usage:
if __name__ == "__main__":
    # Basic usage
    logger = get_logger(__name__)
    logger.info("This is a test log message")
    
    # With extra fields (only in JSON mode)
    logger.info("User logged in", extra={"user_id": 123, "ip": "192.168.1.1"})
    
    # Force JSON format
    json_logger = get_logger(f"{__name__}.json", json_format=True)
    json_logger.warning("This is a JSON formatted log")
    json_logger.error("Something went wrong", extra={"error_code": 500})