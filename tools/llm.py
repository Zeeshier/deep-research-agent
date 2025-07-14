import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq( model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0, api_key=os.getenv("GROQ_API_KEY"))