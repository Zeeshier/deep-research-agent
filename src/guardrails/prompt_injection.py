import re

PROMPT_INJ_PATTERN = re.compile(
    r"\b(ignore|disregard|forget|override).*\b(previous|instruction|prompt)\b", re.IGNORECASE
)

def detect_prompt_injection(text: str) -> bool:
    """
    Simple regex heuristic for prompt-injection attempts.
    Returns True if injection suspected.
    """
    return bool(PROMPT_INJ_PATTERN.search(text))