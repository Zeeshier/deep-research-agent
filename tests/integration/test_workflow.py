from src.graph import build_graph

def test_workflow_end_to_end():
    graph = build_graph()
    state = graph.invoke({"topic": "AI in health", "domain": "Health"})
    assert "report" in state
    assert "<html>" in state["report"].lower()