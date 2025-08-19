from typing import List, Dict, Any, Optional
import time
from functools import wraps, lru_cache
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast, Tuple
from datetime import datetime
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage

from tools.llm import llm
from tools.composio_tools import web_search, create_google_doc
from state import GraphState
from monitoring.logger import get_logger
from monitoring.metrics import timed
from guardrails.input_validator import validate_input
from guardrails.prompt_injection import detect_prompt_injection
from config import settings, get_config

logger = get_logger("nodes")

class NodeError(Exception):
    """Base exception for node-related errors."""
    pass

class MaxRetriesExceededError(NodeError):
    """Raised when max retries are exceeded."""
    pass

class ValidationError(NodeError):
    """Raised when input validation fails."""
    pass

def retry_on_exception(
    max_retries: Optional[int] = None,
    retry_delay: Optional[float] = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator to retry a function on specified exceptions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts (uses config if None)
        retry_delay: Base delay between retries in seconds (uses config if None)
        exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = get_config()
            _max_retries = max_retries if max_retries is not None else config.MAX_RETRIES
            _retry_delay = retry_delay if retry_delay is not None else config.RETRY_DELAY
            
            last_exception = None
            for attempt in range(_max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < _max_retries:
                        wait_time = _retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Attempt {attempt + 1}/{_max_retries} failed for {func.__name__}: "
                            f"{str(e)[:200]}... Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    continue
            
            error_msg = f"Max retries ({_max_retries}) exceeded for {func.__name__}"
            if last_exception:
                error_msg += f": {str(last_exception)}"
            raise MaxRetriesExceededError(error_msg) from last_exception
        return wrapper
    return decorator

# Cached prompts to avoid repeated LLM calls for the same input
@lru_cache(maxsize=128)
def get_system_prompt_questions() -> str:
    """Get the system prompt for generating research questions."""
    return """\
You are a research strategist. Given a topic and domain, output exactly 3 yes/no research questions.
Return them as a simple numbered list (1. 2. 3.) without extra text.
"""

@lru_cache(maxsize=128)
def get_system_prompt_research() -> str:
    """Get the system prompt for generating research reports."""
    return """\
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

# Cache for storing expensive computations
class ResearchCache:
    _instance = None
    _cache: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResearchCache, cls).__new__(cls)
        return cls._instance
    
    @staticmethod
    def get_key(prefix: str, *args) -> str:
        """Generate a cache key from prefix and arguments."""
        return f"{prefix}:{':'.join(str(arg) for arg in args)}"
    
    def get(self, key: str) -> Any:
        """Get a value from the cache."""
        if not settings.CACHE_ENABLED:
            return None
            
        cached = self._cache.get(key)
        if cached and (time.time() - cached['timestamp']) < settings.CACHE_TTL:
            return cached['value']
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache."""
        if not settings.CACHE_ENABLED:
            return
            
        self._cache[key] = {
            'value': value,
            'timestamp': time.time()
        }

# Initialize cache
cache = ResearchCache()

def validate_research_input(topic: str, domain: str) -> None:
    """Validate research input parameters.
    
    Args:
        topic: Research topic to validate
        domain: Research domain to validate
        
    Raises:
        ValidationError: If validation fails
    """
    if not topic or not isinstance(topic, str) or not topic.strip():
        raise ValidationError("Topic must be a non-empty string")
    if not domain or not isinstance(domain, str):
        raise ValidationError("Domain must be a string")
        
    # Check length limits
    max_length = 200
    if len(topic) > max_length:
        raise ValidationError(f"Topic must be less than {max_length} characters")
    if len(domain) > max_length:
        raise ValidationError(f"Domain must be less than {max_length} characters")
        
    # Security validations
    if not validate_input(topic) or not validate_input(domain):
        raise ValidationError("Invalid input detected")
    if detect_prompt_injection(topic) or detect_prompt_injection(domain):
        raise ValidationError("Potential prompt injection detected")

@timed
@retry_on_exception()
def generate_questions_node(state: GraphState) -> GraphState:
    """Generate research questions based on topic and domain.
    
    Args:
        state: Current graph state containing 'topic' and 'domain'
        
    Returns:
        Updated state with 'questions' key containing list of questions
        
    Raises:
        NodeError: If question generation fails
        ValidationError: If input validation fails
    """
    try:
        topic = state["topic"].strip()
        domain = state["domain"].strip()
        validate_research_input(topic, domain)
        
        # Check cache first
        cache_key = f"questions:{topic}:{domain}"
        cached_questions = cache.get(cache_key)
        if cached_questions:
            logger.debug("Using cached questions for topic: %s", topic)
            return {**state, "questions": cached_questions}
        
        # Generate new questions
        prompt = f"Topic: {topic}\nDomain: {domain}"
        messages = [
            {"role": "system", "content": get_system_prompt_questions()},
            {"role": "user", "content": prompt},
        ]
        
        # Use streaming for better UX with long-running generations
        response = llm.invoke(messages)
        
        # Process response
        raw = response.content.strip()
        questions = []
        for line in raw.splitlines():
            line = line.strip()
            if line and (line.startswith(('1.', '2.', '3.')) or line[0].isdigit() and line[1] in ('.', ')')):
                # Extract question text after the number
                question = line[2:].lstrip(' .)')
                if question:
                    questions.append(question)
        
        # Ensure we have exactly 3 questions
        if not questions:
            questions = [
                f"What are the key aspects of {topic} in {domain}?",
                f"How does {topic} impact {domain}?",
                f"What are the future trends of {topic} in {domain}?"
            ]
        elif len(questions) < 3:
            questions.extend([
                f"What are the implications of {topic} in {domain}?"
                for _ in range(3 - len(questions))
            ])
        else:
            questions = questions[:3]
        
        # Cache the results
        cache.set(cache_key, questions)
        
        logger.info("Generated %d research questions for topic: %s", len(questions), topic)
        return {**state, "questions": questions}
        
    except ValidationError as ve:
        logger.warning("Validation error in generate_questions_node: %s", str(ve))
        raise
    except Exception as e:
        logger.error("Error in generate_questions_node: %s", str(e), exc_info=True)
        raise NodeError(f"Failed to generate questions: {str(e)}") from e

@timed
@retry_on_exception(max_retries=2)
def research_agent_node(state: GraphState) -> GraphState:
    """Conduct research based on generated questions.
    
    Args:
        state: Current graph state containing 'topic', 'domain', and 'questions'
        
    Returns:
        Updated state with 'report' key containing research findings
        
    Raises:
        NodeError: If research fails or no questions are provided
    """
    try:
        topic = state["topic"].strip()
        questions = state.get("questions", [])
        
        if not questions:
            raise NodeError("No research questions provided")
        
        # Check cache for existing report
        questions_hash = hash(tuple(questions))
        cache_key = f"report:{topic}:{questions_hash}"
        cached_report = cache.get(cache_key)
        
        if cached_report:
            logger.info("Using cached report for topic: %s", topic)
            return {**state, "report": cached_report}
            
        # Process questions in parallel for better performance
        findings = process_questions_parallel(topic, questions)
        
        if not findings:
            raise NodeError("No research findings were generated")
            
        # Generate the report
        report = generate_research_report(findings)
        
        # Cache the report
        cache.set(cache_key, report)
        
        logger.info("Successfully generated report, length=%d chars", len(report))
        return {**state, "report": report}
        
    except Exception as e:
        logger.error("Error in research_agent_node: %s", str(e), exc_info=True)
        raise NodeError(f"Research failed: {str(e)}") from e

def process_questions_parallel(topic: str, questions: List[str]) -> List[str]:
    """Process research questions in parallel.
    
    Args:
        topic: The research topic
        questions: List of research questions
        
    Returns:
        List of research findings
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    findings = []
    
    def process_question(i: int, question: str) -> Tuple[int, str]:
        """Process a single research question."""
        try:
            query = f"{topic} {question}"
            cache_key = f"search:{query}"
            cached_result = cache.get(cache_key)
            
            if cached_result:
                return i, cached_result
                
            results = web_search(
                query=query, 
                max_results=settings.WEB_SEARCH_MAX_RESULTS
            )
            
            if not results:
                return i, f"{i}. {question}: No results found."
                
            snippets = []
            for j, result in enumerate(results, 1):
                content = result.get("content", "").strip()
                if content:
                    # Truncate content to a reasonable length
                    max_length = 500
                    if len(content) > max_length:
                        content = content[:max_length] + "..."
                    snippets.append(f"{j}. {content}")
            
            if snippets:
                result = f"{i}. {question}:\n" + "\n\n".join(snippets)
            else:
                result = f"{i}. {question}: No content available in search results."
            
            # Cache the result
            cache.set(cache_key, result)
            return i, result
            
        except Exception as e:
            logger.warning("Search failed for question '%s': %s", question, str(e))
            return i, f"{i}. {question}: Error retrieving data - {str(e)}"
    
    # Process questions in parallel
    with ThreadPoolExecutor(max_workers=min(5, len(questions))) as executor:
        futures = [
            executor.submit(process_question, i+1, q) 
            for i, q in enumerate(questions)
        ]
        
        # Collect results in order
        results = [None] * len(questions)
        for future in as_completed(futures):
            i, result = future.result()
            results[i-1] = result
    
    return [r for r in results if r is not None]

def generate_research_report(findings: List[str]) -> str:
    """Generate a research report from findings.
    
    Args:
        findings: List of research findings
        
    Returns:
        Generated report as HTML
    """
    full_context = "\n\n".join(findings)
    
    # Truncate context if too long to avoid API limits
    max_context_length = 10000
    if len(full_context) > max_context_length:
        logger.warning("Truncating research context from %d to %d characters", 
                      len(full_context), max_context_length)
        full_context = full_context[:max_context_length]
    
    messages = [
        {"role": "system", "content": get_system_prompt_research()},
        {"role": "user", "content": full_context},
    ]
    
    try:
        response = llm.invoke(messages)
        report = response.content.strip()
        
        if not report:
            raise NodeError("Generated report is empty")
            
        return report
        
    except Exception as e:
        logger.error("Error generating report: %s", str(e), exc_info=True)
        # Fallback to a simple report if LLM generation fails
        return "<h1>Research Report</h1>\n" + \
               "<p>We encountered an error generating the full report. Here are the raw findings:</p>\n" + \
               "<pre>" + "\n\n".join(findings) + "</pre>"

@timed
@retry_on_exception(max_retries=2)
def save_report_node(state: GraphState) -> GraphState:
    """Save the generated report to Google Docs.
    
    Args:
        state: Current graph state containing 'report', 'topic', and 'domain'
        
    Returns:
        Updated state with 'doc_url' if successful, or 'error' if failed
    """
    try:
        report = state.get("report", "")
        topic = state.get("topic", "Untitled").strip()
        
        if not report:
            error_msg = "Cannot save empty report"
            logger.error(error_msg)
            return {**state, "error": error_msg}
        
        # Check if Google Docs is enabled in config
        if not settings.GOOGLE_DOCS_ENABLED:
            logger.info("Google Docs integration is disabled in settings")
            return {**state, "doc_url": "", "error": None}
        
        # Generate a unique document name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        doc_title = f"Research: {topic} - {timestamp}"
        
        # Truncate title if too long (Google Docs has a 100 char limit)
        if len(doc_title) > 80:
            doc_title = doc_title[:77] + "..."
        
        # Save to Google Docs
        doc_url = create_google_doc(title=doc_title, body=report)
        
        if not doc_url:
            raise NodeError("Failed to get document URL from Google Docs API")
            
        logger.info("Successfully saved report to Google Docs: %s", doc_url)
        return {**state, "doc_url": doc_url, "error": None}
        
    except Exception as e:
        error_msg = f"Failed to save to Google Docs: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Don't fail the entire process if Google Docs save fails
        # Just log the error and continue
        return {**state, "doc_url": "", "error": error_msg}