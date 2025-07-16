from composio_langgraph import Action, ComposioToolSet

def get_composio_tools():
    toolset = ComposioToolSet()
    tools = toolset.get_tools(actions=[
        Action.COMPOSIO_SEARCH_TAVILY_SEARCH,
        Action.GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN
    ])
    return tools
