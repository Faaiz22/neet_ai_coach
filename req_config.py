
# File: requirements.txt
CONFIG = {
    "model_name": "mistral",
    "db_backend": "chroma",
    "max_context_tokens": 2048,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "index_path": "memory/index",
    "mcq_path": "data/mcqs",
    "ncert_path": "data/books",
    "session_memory": True
}

"""
REQUIREMENTS:
pip install transformers sentence-transformers langchain faiss-cpu chromadb PyMuPDF pandas streamlit python-docx sympy ollama
"""

