from tools.composio_tools import get_composio_tools
from tools.llm import llm

tools = get_composio_tools()
llm_with_tools = llm.bind_tools(tools)

def agent_node(state):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = tools
