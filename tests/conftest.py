"""
Pytest configuration and shared fixtures.
"""
import os
import pytest
from dotenv import load_dotenv

# Load ONLY the .env.example so tests never use real secrets
load_dotenv(".env.example")


@pytest.fixture(scope="session", autouse=True)
def mock_env():
    """
    Ensure every test runs with deterministic / fake credentials.
    """
    os.environ["GROQ_API_KEY"] = "fake-groq-key"
    os.environ["COMPOSIO_API_KEY"] = "fake-composio-key"