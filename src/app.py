import streamlit as st
import time
from typing import Dict, Any, Optional

from graph import build_graph
from monitoring.logger import get_logger
from guardrails.input_validator import validate_input
from guardrails.prompt_injection import detect_prompt_injection
from nodes.nodes import NodeError, MaxRetriesExceededError

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

logger = get_logger("streamlit_ui")

def initialize_session_state() -> None:
    """Initialize the session state variables if they don't exist."""
    if 'research_started' not in st.session_state:
        st.session_state.research_started = False
    if 'research_complete' not in st.session_state:
        st.session_state.research_complete = False
    if 'error_occurred' not in st.session_state:
        st.session_state.error_occurred = False
    if 'error_message' not in st.session_state:
        st.session_state.error_message = ""

def reset_research_state() -> None:
    """Reset the research state variables."""
    st.session_state.research_started = False
    st.session_state.research_complete = False
    st.session_state.error_occurred = False
    st.session_state.error_message = ""

def validate_inputs(topic: str, domain: str) -> tuple[bool, str]:
    """Validate user inputs.
    
    Args:
        topic: The research topic
        domain: The research domain/industry
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not topic.strip():
        return False, "Topic cannot be empty."
    if not domain.strip():
        return False, "Domain cannot be empty."
    if not validate_input(topic) or not validate_input(domain):
        return False, "Invalid input detected. Please avoid special characters."
    if detect_prompt_injection(topic) or detect_prompt_injection(domain):
        logger.warning("Blocked suspicious input: %s | %s", topic, domain)
        return False, "Invalid input detected. Please try different keywords."
    return True, ""

def run_research_workflow(topic: str, domain: str) -> Dict[str, Any]:
    """Run the research workflow with retry logic.
    
    Args:
        topic: The research topic
        domain: The research domain/industry
        
    Returns:
        dict: Result containing report, doc_url, or error information
    """
    retry_count = 0
    last_error = None
    
    while retry_count < MAX_RETRIES:
        try:
            graph = build_graph()
            state = graph.invoke({"topic": topic.strip(), "domain": domain.strip()})
            
            # Check if we have a report or an error
            if "report" in state and state["report"]:
                return {
                    "success": True,
                    "report": state["report"],
                    "doc_url": state.get("doc_url", ""),
                    "error": state.get("error", "")
                }
            else:
                error_msg = state.get("error", "No report was generated.")
                logger.error("Research failed: %s", error_msg)
                return {"success": False, "error": error_msg}
                
        except MaxRetriesExceededError as e:
            last_error = f"Operation failed after multiple attempts: {str(e)}"
            logger.error(last_error, exc_info=True)
            break
            
        except NodeError as e:
            last_error = f"Research error: {str(e)}"
            logger.error(last_error, exc_info=True)
            break
            
        except Exception as e:
            last_error = f"An unexpected error occurred: {str(e)}"
            logger.error(last_error, exc_info=True)
            retry_count += 1
            if retry_count < MAX_RETRIES:
                time.sleep(RETRY_DELAY * retry_count)  # Exponential backoff
            
    return {"success": False, "error": last_error or "An unknown error occurred"}

def main() -> None:
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Configure page
    st.set_page_config(
        page_title="Deep Research Agent", 
        page_icon="🧠", 
        layout="centered"
    )
    
    # Header
    st.title("🧠 Deep Research Agent")
    st.markdown("---")
    
    # Display error message if any
    if st.session_state.error_occurred and st.session_state.error_message:
        st.error(st.session_state.error_message)
    
    # Research form
    with st.form("research_form"):
        st.subheader("Start New Research")
        
        topic = st.text_input(
            "Research topic", 
            placeholder="e.g., AI in healthcare",
            key="topic_input"
        )
        domain = st.text_input(
            "Domain / Industry", 
            placeholder="e.g., Health",
            key="domain_input"
        )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            start_research = st.form_submit_button("Start Research")
        with col2:
            if st.form_submit_button("Reset"):
                reset_research_state()
                st.rerun()
    
    # Handle form submission
    if start_research and not st.session_state.research_started:
        # Validate inputs
        is_valid, error_msg = validate_inputs(topic, domain)
        if not is_valid:
            st.session_state.error_occurred = True
            st.session_state.error_message = error_msg
            st.rerun()
        
        # Start research
        st.session_state.research_started = True
        st.session_state.error_occurred = False
        st.session_state.error_message = ""
        
        # Show progress and run research
        with st.spinner("🚀 Starting research... This may take a few minutes."):
            result = run_research_workflow(topic, domain)
            
            if result["success"]:
                st.session_state.research_complete = True
                st.session_state.report = result["report"]
                st.session_state.doc_url = result["doc_url"]
                
                if result.get("error"):
                    st.session_state.warning_message = result["error"]
            else:
                st.session_state.error_occurred = True
                st.session_state.error_message = result["error"]
        
        st.rerun()
    
    # Display results if research is complete
    if st.session_state.research_complete and hasattr(st.session_state, 'report'):
        st.success("✅ Research complete!")
        
        if hasattr(st.session_state, 'warning_message') and st.session_state.warning_message:
            st.warning(st.session_state.warning_message)
        
        # Show download button
        st.download_button(
            label="📄 Download HTML Report",
            data=st.session_state.report,
            file_name=f"{topic.replace(' ', '_')}_report.html",
            mime="text/html",
        )
        
        # Show Google Docs link if available
        if hasattr(st.session_state, 'doc_url') and st.session_state.doc_url:
            st.markdown(f"📝 [View in Google Docs]({st.session_state.doc_url})")
        
        # Display the report
        st.markdown("### Research Report")
        st.components.v1.html(st.session_state.report, height=800, scrolling=True)

if __name__ == "__main__":
    main()