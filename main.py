# Railway entrypoint: import the FastAPI app that defines /rag
try:
    from api.trust_rag_api import app  # if your app file is at api/trust_rag_api.py
except Exception:
    from trust_rag_api import app      # fallback if it’s at root/trust_rag_api.py
