# src/tools/composio_tools.py
import os
from typing import List, Dict
from composio_langchain import ComposioToolSet, Action
from dotenv import load_dotenv
import json

load_dotenv()

COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
if not COMPOSIO_API_KEY:
    raise RuntimeError("Missing COMPOSIO_API_KEY")

toolset = ComposioToolSet(api_key=COMPOSIO_API_KEY)

# Tavily search

def web_search(query: str, max_results: int = 5) -> List[Dict]:
    try:
        resp = toolset.execute_action(
            action=Action.TAVILY_TAVILY_SEARCH,
            params={"query": query, "max_results": max_results}
        )
        # Ensure the response is a JSON object
        if isinstance(resp, str):
            resp = json.loads(resp)
        return resp.get("data", [])
    except Exception as e:
        print(f"Error during search: {e}")
        return []


# Google Docs create

def create_google_doc(title: str, body: str) -> str:
    try:
        resp = toolset.execute_action(
            action=Action.GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN,
            params={"title": title, "body": body}
        )
        # Ensure the response is a JSON object
        if isinstance(resp, str):
            resp = json.loads(resp)
        return resp.get("data", {}).get("webViewLink", "")
    except Exception as e:
        print(f"Error during Google Docs creation: {e}")
        return ""