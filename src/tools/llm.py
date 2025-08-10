import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY")

llm = ChatGroq( model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0, api_key=GROQ_API_KEY)