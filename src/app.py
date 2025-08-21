import streamlit as st
import time
import json
import traceback
from typing import Dict, Any, Optional, List, Tuple

from graph import WorkflowManager, WorkflowError, WorkflowState
from monitoring.logger import get_logger
from guardrails.input_validator import validate_input
from guardrails.prompt_injection import detect_prompt_injection
from nodes.nodes import NodeError, MaxRetriesExceededError

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Initialize workflow manager
workflow_manager = WorkflowManager()
logger = get_logger("streamlit_ui")

def initialize_session_state() -> None:
    """Initialize the session state variables if they don't exist."""
    defaults = {
        'research_started': False,
        'research_complete': False,
        'error_occurred': False,
        'error_message': "",
        'error_count': 0,
        'last_error': None,
        'error_traceback': None,
        'show_technical_details': False,
        'workflow_state': None,
        'execution_metrics': {
            'start_time': None,
            'end_time': None,
            'duration': None,
            'nodes_executed': 0,
            'retries': 0
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_research_state() -> None:
    """Reset the research state variables."""
    st.session_state.research_started = False
    st.session_state.research_complete = False
    st.session_state.error_occurred = False
    st.session_state.error_message = ""
    st.session_state.error_count = 0
    st.session_state.last_error = None
    st.session_state.error_traceback = None
    st.session_state.workflow_state = None
    st.session_state.execution_metrics = {
        'start_time': None,
        'end_time': None,
        'duration': None,
        'nodes_executed': 0,
        'retries': 0
    }

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
    """Run the research workflow with enhanced resilience patterns.
    
    Args:
        topic: The research topic
        domain: The research domain/industry
        
    Returns:
        dict: Result containing report, doc_url, or error information
    """
    # Update execution metrics
    st.session_state.execution_metrics['start_time'] = time.time()
    
    try:
        # Execute the workflow using the workflow manager
        result = workflow_manager.execute_workflow(topic.strip(), domain.strip())
        
        # Update execution metrics
        st.session_state.execution_metrics['end_time'] = time.time()
        st.session_state.execution_metrics['duration'] = (
            st.session_state.execution_metrics['end_time'] - 
            st.session_state.execution_metrics['start_time']
        )
        
        if result.get('node_statuses'):
            st.session_state.execution_metrics['nodes_executed'] = len([
                node for node in result['node_statuses'].values() 
                if node['status'] in ['completed', 'failed']
            ])
            st.session_state.execution_metrics['retries'] = sum(
                node.get('attempts', 1) - 1 
                for node in result['node_statuses'].values()
            )
        
        # Store workflow state for debugging
        st.session_state.workflow_state = result
        
        # Log the workflow execution status
        if result["success"]:
            logger.info("Workflow completed successfully")
        else:
            logger.error("Workflow failed: %s", result.get('error', 'Unknown error'))
            
        return result

        
        
    except WorkflowError as e:
        error_msg = f"Workflow execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False, 
            "error": error_msg,
            "type": "workflow_error"
        }
        
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        error_traceback = traceback.format_exc()
        logger.error("%s\n%s", error_msg, error_traceback)
        
        # Store error details for technical view
        st.session_state.last_error = str(e)
        st.session_state.error_traceback = error_traceback
        
        return {
            "success": False, 
            "error": error_msg,
            "type": "unexpected_error"
        }

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
        with st.container():
            st.error(st.session_state.error_message)
            
            # Show technical details toggle if there was an error
            if st.session_state.last_error or st.session_state.error_traceback:
                st.session_state.show_technical_details = st.checkbox(
                    "Show technical details",
                    value=st.session_state.show_technical_details
                )
                
                if st.session_state.show_technical_details:
                    with st.expander("Technical Details", expanded=False):
                        st.markdown("### Error Details")
                        st.code(st.session_state.last_error or "No error details available")
                        
                        if st.session_state.error_traceback:
                            st.markdown("### Stack Trace")
                            st.code(st.session_state.error_traceback)
                        
                        if st.session_state.workflow_state:
                            st.markdown("### Workflow State")
                            st.json({
                                k: v for k, v in st.session_state.workflow_state.items()
                                if k not in ['report', 'doc_url']
                            })
                        
                        if st.session_state.execution_metrics:
                            st.markdown("### Execution Metrics")
                            metrics = st.session_state.execution_metrics.copy()
                            if 'start_time' in metrics:
                                metrics['start_time'] = time.strftime(
                                    '%Y-%m-%d %H:%M:%S', 
                                    time.localtime(metrics['start_time'])
                                )
                            if 'end_time' in metrics:
                                metrics['end_time'] = time.strftime(
                                    '%Y-%m-%d %H:%M:%S', 
                                    time.localtime(metrics['end_time'])
                                ) if metrics['end_time'] else None
                            if 'duration' in metrics and metrics['duration'] is not None:
                                metrics['duration'] = f"{metrics['duration']:.2f} seconds"
                            st.json(metrics)
    
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
        # Reset error state
        st.session_state.error_occurred = False
        st.session_state.error_message = ""
        
        # Validate inputs
        is_valid, error_msg = validate_inputs(topic, domain)
        if not is_valid:
            st.session_state.error_occurred = True
            st.session_state.error_message = error_msg
            st.session_state.error_count += 1
            st.rerun()
        
        # Start research
        st.session_state.research_started = True
        
        # Show progress and run research
        with st.spinner("🚀 Starting research... This may take a few minutes."):
            try:
                result = run_research_workflow(topic, domain)
                
                if result["success"]:
                    st.session_state.research_complete = True
                    st.session_state.report = result.get("report", "")
                    st.session_state.doc_url = result.get("doc_url")
                    
                    if result.get("error"):
                        st.session_state.warning_message = result["error"]
                else:
                    st.session_state.error_occurred = True
                    st.session_state.error_message = result["error"]
                    st.session_state.error_count += 1
                    
                    # If we've had multiple failures, show a more prominent warning
                    if st.session_state.error_count >= 3:
                        st.session_state.error_message = (
                            "We're having trouble completing your request. "
                            "Please try again later or contact support if the issue persists.\n\n"
                            f"Error: {result['error']}"
                        )
                        
            except Exception as e:
                st.session_state.error_occurred = True
                st.session_state.error_message = f"An unexpected error occurred: {str(e)}"
                st.session_state.error_traceback = traceback.format_exc()
                st.session_state.error_count += 1
                logger.error(
                    "Unexpected error in workflow execution: %s",
                    str(e),
                    exc_info=True
                )
        
        st.rerun()
    
    # Display results if research is complete
    if st.session_state.research_complete and hasattr(st.session_state, 'report'):
        st.success("✅ Research complete!")
        
        # Show execution metrics if available
        if st.session_state.execution_metrics and st.session_state.execution_metrics.get('duration') is not None:
            with st.expander("Execution Metrics", expanded=False):
                metrics = st.session_state.execution_metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Duration", f"{metrics['duration']:.2f} seconds")
                with col2:
                    st.metric("Nodes Executed", metrics.get('nodes_executed', 0))
                with col3:
                    st.metric("Retries", metrics.get('retries', 0))
        
        if hasattr(st.session_state, 'warning_message') and st.session_state.warning_message:
            st.warning(st.session_state.warning_message)
        
        # Show download button if we have a report
        if st.session_state.report:
            st.download_button(
                label="📄 Download HTML Report",
                data=st.session_state.report,
                file_name=f"{topic.replace(' ', '_')}_report.html",
                mime="text/html",
            )
        
        # Show Google Docs link if available
        if hasattr(st.session_state, 'doc_url') and st.session_state.doc_url:
            st.markdown(f"📝 [View in Google Docs]({st.session_state.doc_url})")
        
        # Display the report if available
        if hasattr(st.session_state, 'report') and st.session_state.report:
            st.markdown("### Research Report")
            st.components.v1.html(st.session_state.report, height=800, scrolling=True)
        else:
            st.warning("No report was generated. Please check the error details for more information.")

if __name__ == "__main__":
    main()