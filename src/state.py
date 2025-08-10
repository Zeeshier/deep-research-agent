from typing import TypedDict, List, Optional

class GraphState(TypedDict):
    topic: str
    domain: str
    questions: Optional[List[str]]
    findings: Optional[List[str]]
    report: Optional[str]
    error: Optional[str]