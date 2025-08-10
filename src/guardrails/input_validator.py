import re

MAX_LEN = 200
REJECTED_KEYWORDS = ["drop table", "delete from", "<script", "os.system", "exec("]

def validate_input(user_text: str) -> bool:
    """
    Basic sanity checks for topic/domain strings.
    """
    if not isinstance(user_text, str):
        return False
    if len(user_text) > MAX_LEN:
        return False
    lowered = user_text.lower()
    return not any(bad in lowered for bad in REJECTED_KEYWORDS)