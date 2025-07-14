from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from state import GraphState
from nodes.nodes import agent_node, tools

def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools=tools))
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    builder.add_edge(START, "agent")
    return builder
