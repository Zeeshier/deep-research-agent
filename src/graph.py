"""
Workflow implementation with enhanced resilience patterns.

This module provides a resilient workflow implementation that integrates with
the resilience patterns defined in the resilience module.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Type, TypedDict, Union
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, validator

from resilience import (
    retry_with_backoff,
    CircuitBreaker,
    resource_limit,
    StateValidator,
    IterationLimiter,
    validate_structured_output,
    ResilienceError,
    MaxRetriesExceededError,
    CircuitOpenError,
    TimeoutError,
    ResourceLimitExceededError
)
from state import GraphState
from monitoring.logger import get_logger
from monitoring.metrics import timed
from config import get_config

logger = get_logger("workflow")

# Default configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0  # seconds
DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_MAX_MEMORY_MB = 512  # 512MB

class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class NodeExecution:
    """Tracks the execution state of a node."""
    name: str
    status: NodeStatus = NodeStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    attempts: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowState:
    """Extended workflow state with resilience metadata."""
    # Original fields from GraphState
    topic: str
    domain: str
    questions: Optional[List[str]] = None
    findings: Optional[List[str]] = None
    report: Optional[str] = None
    error: Optional[str] = None
    
    # Resilience metadata
    node_statuses: Dict[str, NodeExecution] = field(default_factory=dict)
    circuit_breakers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_graph_state(self) -> GraphState:
        """Convert to a GraphState dictionary."""
        return {
            'topic': self.topic,
            'domain': self.domain,
            'questions': self.questions,
            'findings': self.findings,
            'report': self.report,
            'error': self.error
        }
    
    def update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()
    
    def get_node_status(self, node_name: str) -> NodeExecution:
        """Get or create a node status."""
        if node_name not in self.node_statuses:
            self.node_statuses[node_name] = NodeExecution(name=node_name)
        return self.node_statuses[node_name]
    
    def start_node(self, node_name: str) -> None:
        """Mark a node as started."""
        node = self.get_node_status(node_name)
        node.status = NodeStatus.RUNNING
        node.start_time = datetime.utcnow()
        node.attempts += 1
        self.update_timestamp()
    
    def complete_node(self, node_name: str, metadata: Optional[Dict] = None) -> None:
        """Mark a node as completed."""
        node = self.get_node_status(node_name)
        node.status = NodeStatus.COMPLETED
        node.end_time = datetime.utcnow()
        if metadata:
            node.metadata.update(metadata)
        self.update_timestamp()
    
    def fail_node(self, node_name: str, error: Exception) -> None:
        """Mark a node as failed."""
        node = self.get_node_status(node_name)
        node.status = NodeStatus.FAILED
        node.end_time = datetime.utcnow()
        node.last_error = str(error)
        self.error = str(error)
        self.update_timestamp()

class WorkflowError(Exception):
    """Base exception for workflow-related errors."""
    pass

class WorkflowTimeoutError(WorkflowError):
    """Raised when a workflow operation times out."""
    pass

class WorkflowValidationError(WorkflowError):
    """Raised when workflow validation fails."""
    pass

class WorkflowManager:
    """Manages workflow execution with resilience patterns."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the workflow manager.
        
        Args:
            config: Optional configuration overrides
        """
        self.config = get_config()
        if config:
            self.config.update(config)
        
        # Get max_iterations from config or use default (8 as per requirements)
        self.max_iterations = self.config.get('max_iterations', 8)
        
        # Initialize circuit breakers for external services
        self.circuit_breakers = {
            "llm": CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=300,  # 5 minutes
                name="llm_service"
            ),
            "web_search": CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=300,  # 5 minutes
                name="web_search"
            ),
            "google_docs": CircuitBreaker(
                failure_threshold=2,
                recovery_timeout=600,  # 10 minutes
                name="google_docs"
            )
        }
        
        # State validation schema - now validates the underlying dictionary
        self.state_validator = StateValidator({
            "topic": (str, "[MISSING_TOPIC]"),
            "domain": (str, "[MISSING_DOMAIN]"),
            "questions": (list, []),
            "findings": (list, []),
            "report": (str, ""),
            "error": (str, ""),
        })
    
    def validate_state(self, state: WorkflowState) -> bool:
        """Validate the workflow state.
        
        Args:
            state: The workflow state to validate
            
        Returns:
            bool: True if the state is valid
            
        Raises:
            WorkflowValidationError: If the state is invalid
        """
        try:
            self.state_validator.validate(state)
            return True
        except Exception as e:
            raise WorkflowValidationError(f"Invalid workflow state: {str(e)}") from e
    
    def retry_decorator(func):
        return retry_with_backoff(
            func,
            max_retries=3,
            initial_delay=1.0,
            max_delay=10.0
        )
    
    @retry_decorator
    def execute_node(
        self,
        node_func: Callable[[WorkflowState], WorkflowState],
        state: WorkflowState,
        node_name: str,
        timeout: Optional[float] = None
    ) -> WorkflowState:
        """Execute a workflow node with resilience patterns.
        
        Args:
            node_func: The node function to execute
            state: The current workflow state
            node_name: Name of the node for tracking
            timeout: Optional timeout in seconds
            
        Returns:
            Updated workflow state
            
        Raises:
            WorkflowError: If the node execution fails
        """
        node_status = state.get_node_status(node_name)
        
        # Skip if already completed
        if node_status.status == NodeStatus.COMPLETED:
            return state
            
        # Mark as running
        state.start_node(node_name)
        
        try:
            # Execute with resource limits
            with resource_limit(
                max_memory_mb=DEFAULT_MAX_MEMORY_MB,
                timeout_seconds=timeout or DEFAULT_TIMEOUT
            ):
                # Execute the node function
                result = node_func(state)
                
                # Validate the result
                self.validate_state(result)
                
                # Mark as completed
                state.complete_node(node_name, {"execution_time": (datetime.utcnow() - node_status.start_time).total_seconds()
                })
                
                return result
                
        except Exception as e:
            # Handle specific errors
            if isinstance(e, (TimeoutError, ResourceLimitExceededError)):
                error_msg = f"{node_name} timed out or exceeded resource limits: {str(e)}"
            elif isinstance(e, CircuitOpenError):
                error_msg = f"Service unavailable for {node_name}: {str(e)}"
            else:
                error_msg = f"Error in {node_name}: {str(e)}"
            
            # Update node status
            state.fail_node(node_name, WorkflowError(error_msg))
            
            # Reraise with additional context
            raise WorkflowError(error_msg) from e
    
    def build_workflow(self) -> Callable[[WorkflowState], WorkflowState]:
        """Build the workflow with resilience patterns."""
        from langgraph.graph import StateGraph, END
        
        # Import node functions
        from nodes.nodes import (
            generate_questions_node,
            research_agent_node,
            save_report_node,
        )
        
        # Create the workflow
        workflow = StateGraph(WorkflowState)
        
        # Add nodes with resilience wrappers
        workflow.add_node(
            "generate_questions",
            lambda state: self.execute_node(generate_questions_node, state, "generate_questions")
        )
        
        workflow.add_node(
            "research",
            lambda state: self.execute_node(research_agent_node, state, "research")
        )
        
        workflow.add_node(
            "save_report",
            lambda state: self.execute_node(save_report_node, state, "save_report")
        )
        
        # Add iteration counter to state
        def increment_iteration_counter(state: WorkflowState) -> WorkflowState:
            """Increment the iteration counter and check against max_iterations."""
            if not hasattr(state, 'iteration_count'):
                state.iteration_count = 0
            state.iteration_count += 1
            
            if state.iteration_count > self.max_iterations:
                raise WorkflowError(
                    f"Maximum iterations ({self.max_iterations}) exceeded. "
                    "The workflow is taking too long to complete and has been terminated."
                )
            return state
        
        # Add a check before each node execution
        def check_iteration_limit(state: WorkflowState) -> WorkflowState:
            """Check if we've exceeded the maximum number of iterations."""
            if hasattr(state, 'iteration_count') and state.iteration_count >= self.max_iterations:
                raise WorkflowError(
                    f"Maximum iterations ({self.max_iterations}) exceeded. "
                    "The workflow is taking too long to complete and has been terminated."
                )
            return state
        
        # Add the iteration counter to the workflow
        workflow.add_node("increment_iteration", increment_iteration_counter)
        workflow.add_node("check_iteration_limit", check_iteration_limit)
        
        # Define the workflow edges with iteration checks
        workflow.set_entry_point("increment_iteration")
        workflow.add_edge("increment_iteration", "check_iteration_limit")
        workflow.add_edge("check_iteration_limit", "generate_questions")
        workflow.add_edge("generate_questions", "research")
        workflow.add_edge("research", "save_report")
        workflow.add_edge("save_report", END)
        
        return workflow.compile()
    
    def execute_workflow(self, topic: str, domain: str) -> Dict[str, Any]:
        """Execute the workflow with the given inputs.
        
        Args:
            topic: The research topic
            domain: The research domain
            
        Returns:
            Dict containing the workflow results
        """
        # Initialize the workflow state
        state = WorkflowState(
            topic=topic,
            domain=domain,
            questions=[],
            findings=[],
            report="",
            error=""
        )
        
        # Build and execute the workflow
        try:
            workflow = self.build_workflow()
            result = workflow.invoke(state)
            
            # Return the results
            return {
                "success": True,
                "report": result.get("report", ""),
                "doc_url": result.get("doc_url", ""),
                "error": result.get("error", ""),
                "execution_time": (datetime.utcnow() - state.created_at).total_seconds(),
                "node_statuses": {
                    name: {
                        "status": ns.status.value,
                        "attempts": ns.attempts,
                        "duration": (ns.end_time - ns.start_time).total_seconds() if ns.end_time and ns.start_time else None,
                        "error": ns.last_error
                    }
                    for name, ns in state.node_statuses.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "node_statuses": {
                    name: {
                        "status": ns.status.value,
                        "attempts": ns.attempts,
                        "duration": (ns.end_time - ns.start_time).total_seconds() if ns.end_time and ns.start_time else None,
                        "error": ns.last_error
                    }
                    for name, ns in state.node_statuses.items()
                }
            }
