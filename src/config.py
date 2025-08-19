"""Configuration management for the Deep Research Agent."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings with environment variable fallbacks."""
    
    # Application settings
    APP_NAME: str = Field(default="Deep Research Agent", env="APP_NAME")
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API settings
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    CORS_ALLOWED_ORIGINS: List[str] = Field(
        default_factory=lambda: ["*"],
        env="CORS_ALLOWED_ORIGINS"
    )
    
    # LLM settings
    LLM_MODEL: str = Field(default="llama-4-scout-17b-16e-instruct", env="LLM_MODEL")
    LLM_TEMPERATURE: float = Field(default=0.7, env="LLM_TEMPERATURE")
    LLM_MAX_TOKENS: int = Field(default=2000, env="LLM_MAX_TOKENS")
    
    # Web search settings
    WEB_SEARCH_ENABLED: bool = Field(default=True, env="WEB_SEARCH_ENABLED")
    WEB_SEARCH_MAX_RESULTS: int = Field(default=3, env="WEB_SEARCH_MAX_RESULTS")
    WEB_SEARCH_TIMEOUT: int = Field(default=10, env="WEB_SEARCH_TIMEOUT")
    
    # Retry settings
    MAX_RETRIES: int = Field(default=3, env="MAX_RETRIES")
    RETRY_DELAY: float = Field(default=1.0, env="RETRY_DELAY")
    
    # Performance settings
    CACHE_ENABLED: bool = Field(default=True, env="CACHE_ENABLED")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # Health check settings
    HEALTH_CHECK_TIMEOUT: int = Field(default=30, env="HEALTH_CHECK_TIMEOUT")
    
    # Google Docs settings
    GOOGLE_DOCS_ENABLED: bool = Field(default=True, env="GOOGLE_DOCS_ENABLED")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }

# Create settings instance
settings = Settings()

def get_config() -> Settings:
    """Get the current configuration."""
    return settings

def update_config(**kwargs) -> None:
    """Update configuration settings."""
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
