import pytest
from src.nodes.nodes import research_agent_node

def test_research_agent_node_returns_dict():
    state = {"topic": "AI in health", "domain": "Health"}
    result = research_agent_node(state)
    assert isinstance(result, dict)
    assert "questions" in result or "report" in result