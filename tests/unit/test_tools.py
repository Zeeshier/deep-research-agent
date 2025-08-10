import pytest
from unittest.mock import patch
from src.tools.composio_tools import web_search_tool

@patch("src.tools.composio_tools.web_search_tool.run")
def test_web_search_tool(mock_run):
    mock_run.return_value = [{"title": "Test", "url": "http://example.com"}]
    result = web_search_tool.run("query")
    assert len(result) == 1
    assert result[0]["title"] == "Test"