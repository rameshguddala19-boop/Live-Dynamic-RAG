import os
from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")

DB_PATH = os.getenv("RAG_DB_PATH", "rag.db")
CHROMA_PATH = os.getenv("RAG_CHROMA_PATH", "chroma_store")

COLLECTION = "employees"

TOP_K = int(os.getenv("RAG_TOP_K", "12"))