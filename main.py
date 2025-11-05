# Root entrypoint used by Railway. It imports the RAG FastAPI app that defines /rag.
try:
    # if your app lives at api/trust_rag_api.py
    from api.trust_rag_api import app
except Exception:
    # fallback if it lives at trust_rag_api.py in the root
    from trust_rag_api import app
