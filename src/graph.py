from langgraph.graph import StateGraph, END
from state import GraphState
from nodes.nodes import (
    generate_questions_node,
    research_agent_node,
    save_report_node,
)

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("generate_questions", generate_questions_node)
    workflow.add_node("research", research_agent_node)
    workflow.add_node("save_report", save_report_node)

    workflow.set_entry_point("generate_questions")
    workflow.add_edge("generate_questions", "research")
    workflow.add_edge("research", "save_report")
    workflow.add_edge("save_report", END)

    return workflow.compile()