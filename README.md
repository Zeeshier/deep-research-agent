# Deep Research Agent 🧠


---

## Introduction 🌟

The Deep Research Agent is a sophisticated multi-agent system designed to conduct comprehensive research on user-specified topics within a chosen domain. It leverages AI-driven tools, including the Meta LLaMA model, to generate targeted research questions, perform in-depth analysis, and compile findings into a professional report. This project demonstrates the system's production readiness, safety, and usability, making it an ideal tool for researchers, analysts, and professionals seeking data-driven insights.

## Project Overview 📝

The Deep Research Agent is a powerful, AI-powered tool for conducting deep research. It integrates the Meta LLaMA large language model with external tools like Tavily for web searches, ensuring accurate and up-to-date information. The system is built to be user-friendly, supporting iterative research refinement through a Streamlit-based interface. It automates the entire research pipeline, from question generation to report creation and storage in Google Docs.

<img src="images\screenshot.png" alt="screenshot" width="700" height ="370"  />


### Project Structure 🗂️

The project is organized in a clean, modular structure for easy maintenance and scalability. Here's the directory tree:

```
deep-research-agent/
├── .env.example          # Example environment variables file for configuration
├── .gitignore            # Git ignore file to exclude unnecessary files
├── .gitattributes        # Git attributes for handling file types
├── LICENSE               # Project license file (e.g., MIT or Apache)
├── README.md             # Main README with setup instructions and usage
├── Dockerfile            # Docker configuration for containerization
├── docker-compose.yml    # Docker Compose for multi-container setups
├── pytest.ini            # Configuration for pytest testing framework
├── requirements.txt      # List of Python dependencies
├── notebook/             # Jupyter notebooks for experimentation
│   └── deep_research.ipynb  # Notebook for interactive deep research demos
├── tests/                # Test suite directory
│   ├── __init__.py       # Init file for tests package
│   ├── conftest.py       # Pytest fixtures and configurations
│   ├── unit/             # Unit tests
│   │   ├── __init__.py
│   │   ├── test_nodes.py  # Tests for individual nodes
│   │   └── test_tools.py  # Tests for tools like LLM and Composio
│   ├── integration/      # Integration tests
│   │   ├── __init__.py
│   │   └── test_workflow.py  # Tests for workflow integration
│   └── system/           # System/end-to-end tests
│       ├── __init__.py
│       └── test_e2e.py   # Full system tests including UI
└── src/                  # Source code directory
    ├── __init__.py       # Init file for src package
    ├── app.py            # Streamlit application entry point
    ├── graph.py          # LangGraph workflow definition
    ├── state.py          # State management for the agent
    ├── monitoring/       # Monitoring tools
    │   ├── __init__.py
    │   ├── logger.py     # Custom logging module
    │   └── metrics.py    # Metrics tracking (e.g., performance)
    ├── guardrails/       # Safety and validation modules
    │   ├── __init__.py
    │   ├── input_validator.py  # Input validation logic
    │   └── prompt_injection.py # Prompt injection detection
    ├── nodes/            # Agent nodes (research steps)
    │   ├── __init__.py
    │   └── nodes.py      # Definitions of nodes like research_agent_node
    ├── tools/            # AI tools and integrations
    │   ├── __init__.py
    │   ├── llm.py        # LLM wrapper (e.g., for Meta LLaMA)
    │   └── composio_tools.py  # Tools for external integrations like Google Docs
    └── prompts.py        # Prompt templates for the LLM
```

This structure follows best practices: separating concerns into source code, tests, and documentation.

### Key Features ✨

- **Input Flexibility** 📝: Users can specify a research topic and domain via an intuitive Streamlit web interface, allowing for customized research.
- **Automated Question Generation** ❓: The system generates specific yes/no research questions tailored to the topic and domain, ensuring focused analysis.
- **AI-Powered Research** 🤖: Utilizes the Meta LLaMA model for reasoning and Tavily for real-time web searches to gather accurate, verifiable data.
- **Professional Reporting** 📄: Compiles findings into a polished, HTML-formatted report styled like McKinsey consulting reports, with sections for executive summary, findings, and recommendations.
- **Google Docs Integration** 📑: Automatically exports and saves reports to Google Docs for easy sharing and collaboration.
- **State Management** 🧮: Leverages LangGraph for orchestrating the multi-agent workflow, maintaining state across steps for seamless execution and memory retention.

## Flowchart 📈

Below is a flowchart illustrating the workflow of the Deep Research Agent. It starts with user input, proceeds through question generation, research, and ends with report creation and storage.

<img src="images\flowchart.png" alt="flowchart" height ="600" width="400" text-align:center  />

The workflow is cyclic if refinements are needed, ensuring iterative improvement.

## Enhancements for Production-Readiness 🛠️

### Testing 🧪

Comprehensive testing ensures reliability. Tests are divided into unit, integration, and system levels using pytest.

#### Unit Tests

Unit tests isolate and verify individual components.

```python
# tests/unit/test_nodes.py
from src.nodes.nodes import research_agent_node

def test_research_agent_node_returns_dict():
    state = {"topic": "AI in health", "domain": "Health"}
    result = research_agent_node(state)
    assert isinstance(result, dict)
    assert "questions" in result or "report" in result
```

**Explanation**: This test checks if the `research_agent_node` function processes a state dictionary (containing topic and domain) and returns a dictionary with expected keys like "questions" or "report". It ensures the node outputs the correct data structure without side effects.

#### Integration Tests

Integration tests verify component interactions.

```python
# tests/integration/test_workflow.py
from src.graph import build_graph

def test_workflow_end_to_end():
    graph = build_graph()
    state = graph.invoke({"topic": "AI in health", "domain": "Health"})
    assert "report" in state
    assert "<html>" in state["report"].lower()
```

**Explanation**: This test builds the full LangGraph workflow, invokes it with sample input, and asserts that the output state includes a "report" key with HTML content. It simulates the entire pipeline to catch integration issues.

#### System Tests

System tests validate the end-to-end application, including the UI.

```python
# tests/system/test_e2e.py
import subprocess
import time
import requests

def test_streamlit_ui_loads():
    proc = subprocess.Popen(["streamlit", "run", "src/app.py", "--server.headless=true"])
    time.sleep(10)  # give it time to boot
    try:
        resp = requests.get("http://localhost:8501", timeout=5)
        assert resp.status_code == 200
        assert "Deep Research Agent" in resp.text
    finally:
        proc.terminate()
        proc.wait()
```

**Explanation**: This test launches the Streamlit app in headless mode, waits for it to start, and sends an HTTP request to verify the UI loads correctly (status 200) and contains the expected title. It ensures the full system, including the web interface, functions as intended.

### Safety Features 🔒

#### Input Validation

Prevents invalid or malicious inputs.

```python
# guardrails/input_validator.py
def validate_input(user_text: str) -> bool:
    MAX_LEN = 200
    REJECTED_KEYWORDS = ["drop table", "delete from", "<script"]
    if not isinstance(user_text, str) or len(user_text) > MAX_LEN:
        return False
    lowered = user_text.lower()
    return not any(bad in lowered for bad in REJECTED_KEYWORDS)
```

**Explanation**: This function checks if the input is a string under 200 characters and doesn't contain SQL injection or XSS keywords. It returns True for safe inputs, blocking potential attacks early.

#### Prompt Injection Protection

Detects and blocks prompt injection attempts.

```python
# guardrails/prompt_injection.py
import re

PROMPT_INJ_PATTERN = re.compile(r"\b(ignore|disregard|forget|override).*\b(previous|instruction|prompt)\b", re.IGNORECASE)

def detect_prompt_injection(text: str) -> bool:
    return bool(PROMPT_INJ_PATTERN.search(text))
```

**Explanation**: Uses regex to scan for patterns like "ignore previous instructions," common in prompt injections. Returns True if detected, allowing the system to block harmful prompts.

### User Interface Design 🖥️

The Streamlit UI provides a simple, guided experience.

```python
# src/app.py
import streamlit as st
from graph import build_graph
from monitoring.logger import get_logger
from guardrails.input_validator import validate_input
from guardrails.prompt_injection import detect_prompt_injection

logger = get_logger("streamlit_ui")

st.set_page_config(page_title="Deep Research Agent – M3", page_icon="🧠", layout="centered")

st.title("🧠 Deep Research Agent – Production Ready")
st.markdown("Module-3 Capstone – Agentic AI Developer Certification")

topic = st.text_input("Research topic", placeholder="e.g., AI in healthcare")
domain = st.text_input("Domain / Industry", placeholder="e.g., Health")

if st.button("Start Research"):
    if not topic.strip():
        st.error("Topic cannot be empty.")
        st.stop()
    if not validate_input(topic) or not validate_input(domain):
        st.error("Invalid or too-long input detected.")
        st.stop()
    if detect_prompt_injection(topic) or detect_prompt_injection(domain):
        st.error("Potential prompt injection detected – request blocked.")
        logger.warning("Blocked suspicious input: %s | %s", topic, domain)
        st.stop()

    logger.info("User started research: topic=%s domain=%s", topic, domain)

    with st.spinner("🔄 Running multi-agent workflow…"):
        try:
            graph = build_graph()
            state = graph.invoke({"topic": topic.strip(), "domain": domain.strip()})
            report = state.get("report", "")
            if not report:
                st.error("No report generated.")
                st.stop()

            st.success("Report ready!")
            st.download_button(
                label="📄 Download HTML Report",
                data=report,
                file_name=f"{topic.replace(' ', '_')}_report.html",
                mime="text/html",
            )

            st.components.v1.html(report, height=800, scrolling=True)

        except Exception as exc:
            logger.exception("Workflow failed")
            st.error(f"Something went wrong: {exc}")
```

**Explanation**: This script sets up the Streamlit app with input fields for topic and domain. On button click, it validates inputs, runs the workflow via LangGraph, displays a spinner during processing, and renders the HTML report with a download option. Safety checks are integrated to prevent errors or attacks.

### Failure Handling and Monitoring 🛡️

#### Error Handling

Catches and logs exceptions gracefully.

```python
# src/nodes/nodes.py
from monitoring.logger import get_logger

logger = get_logger("nodes")

def research_agent_node(state: dict) -> dict:
    try:
        # Research logic here (e.g., LLM calls, web searches)
        # Example: questions = llm.generate_questions(state["topic"], state["domain"])
        return {"questions": questions}  # Placeholder for actual logic
    except Exception as exc:
        logger.error("Error during research: %s", exc)
        return {"error": str(exc)}
```

**Explanation**: In each node (like research_agent_node), a try-except block wraps the core logic. If an error occurs (e.g., API failure), it's logged, and an error state is returned, preventing full system crashes.

#### Logging

Provides detailed, formatted logs for debugging.

```python
# monitoring/logger.py
import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
```

**Explanation**: This factory function creates a logger with INFO level, streaming output to console in a timestamped format. It's used across modules for consistent monitoring of events, warnings, and errors.

### Deployment 🚀

#### Docker

Enables containerized, scalable deployment.

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
```

**Explanation**: This Dockerfile uses a slim Python base image, installs dependencies, copies the project, exposes the Streamlit port (8501), and runs the app. It allows easy deployment on platforms like Docker Hub, Kubernetes, or cloud services.

## Conclusion 🎉

The Deep Research Agent is a production-ready, AI-driven system that streamlines research workflows while prioritizing safety and usability. It showcases advanced agentic AI capabilities, from multi-agent orchestration to secure integrations. Future enhancements could include support for more LLMs, advanced analytics visualizations, or multi-domain expansions. This capstone project highlights readiness for real-world applications in research and analysis. 🚀