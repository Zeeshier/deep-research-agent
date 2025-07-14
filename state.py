from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
