from typing import List
from langchain.schema import Document
from tools.llm import llm
from tools.composio_tools import web_search, create_google_doc
from state import GraphState
from monitoring.logger import get_logger
from monitoring.metrics import timed
from guardrails.input_validator import validate_input
from guardrails.prompt_injection import detect_prompt_injection

logger = get_logger("nodes")

SYSTEM_PROMPT_QUESTIONS = """\
You are a research strategist. Given a topic and domain, output exactly 3 yes/no research questions.
Return them as a simple numbered list (1. 2. 3.) without extra text.
"""

SYSTEM_PROMPT_RESEARCH = """\
You are a McKinsey analyst. Using the provided context, write a concise, professional report in raw HTML.
Report structure:
<h1>Executive Summary</h1>
<h2>Key Findings</h2>
<ul>...</ul>
<h2>Recommendations</h2>
<ol>...</ol>
<h2>Sources</h2>
<ul>...</ul>
"""

@timed
def generate_questions_node(state: GraphState) -> GraphState:
    topic = state["topic"]
    domain = state["domain"]
    prompt = f"Topic: {topic}\nDomain: {domain}"
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT_QUESTIONS},
            {"role": "user", "content": prompt},
        ]
    )
    raw = response.content.strip()
    questions = [q[2:].strip() for q in raw.splitlines() if q.strip()]
    if len(questions) != 3:
        questions = questions[:3] + ["Is this trend sustainable?"] * (3 - len(questions))
    logger.info("Generated questions: %s", questions)
    return {**state, "questions": questions}

@timed
def research_agent_node(state: GraphState) -> GraphState:
    topic = state["topic"]
    if not validate_input(topic) or detect_prompt_injection(topic):
        raise ValueError("Unsafe input detected")

    questions = state.get("questions", [])
    findings: List[str] = []
    for q in questions:
        query = f"{topic} {q}"
        try:
            results = web_search(query=f"{topic} {q}", max_results=3)
            snippet = " ".join([r.get("content", "")[:300] for r in results])
            findings.append(f"{q}: {snippet}")

        except Exception as exc:
            logger.warning("Search failed for query %s: %s", query, exc)
            findings.append(f"{q}: No reliable data found.")

    full_context = "\n\n".join(findings)
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT_RESEARCH},
            {"role": "user", "content": full_context},
        ]
    )
    report = response.content.strip()
    logger.info("Report generated, length=%d chars", len(report))
    return {**state, "report": report}

@timed
def save_report_node(state: GraphState) -> GraphState:
    report = state.get("report", "")
    if not report:
        state["error"] = "Empty report"
        return state

    try:
        doc_url = create_google_doc(title=f"Report: {state['topic']}", body=report)
        logger.info("Report saved to Google Docs: %s", doc_url)
    except Exception as exc:
        logger.warning("Google Docs save failed: %s", exc)
        state["error"] = str(exc)
    return state