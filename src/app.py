import streamlit as st
from graph import build_graph
from monitoring.logger import get_logger
from guardrails.input_validator import validate_input
from guardrails.prompt_injection import detect_prompt_injection

logger = get_logger("streamlit_ui")

st.set_page_config(
    page_title="Deep Research Agent", page_icon="🧠", layout="centered"
)

st.title("🧠 Deep Research Agent")

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