# main.py — Railway entrypoint. Imports the API app no matter where it lives.
cat > main.py <<'PY'
try:
    from api.trust_rag_api import app  # prefer api/ folder
except Exception:
    from trust_rag_api import app      # fallback to root file
PY

# trust_rag_api.py — put at root if you don't have api/trust_rag_api.py
# (If you already have it under api/, skip this step.)
[ -f trust_rag_api.py ] || cat > trust_rag_api.py <<'PY'
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from openai import OpenAI
import os

try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

app = FastAPI(title="Private Trust Fiduciary Advisor API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = os.getenv("PINECONE_INDEX", "").strip()
host = os.getenv("PINECONE_HOST", "").strip()
if host:
    idx = pc.Index(host=host)
elif index_name:
    idx = pc.Index(index_name)
else:
    raise RuntimeError("Set PINECONE_HOST or PINECONE_INDEX")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/rag")
def rag_endpoint(question: str = Query(..., min_length=3)):
    emb = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
    results = idx.query(vector=emb, top_k=5, include_metadata=True)
    matches = results["matches"] if isinstance(results, dict) else getattr(results, "matches", [])
    lines = []
    for m in matches:
        meta = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        score = m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0)
        title = meta.get("title") or meta.get("doc_parent") or "Unknown"
        level = meta.get("doc_level", "N/A")
        page  = meta.get("page", "?")
        lines.append(f"{title} (Level {level} p.{page} – score {float(score):.3f})")
    sources = "\n".join(lines) if lines else "No matches."
    return {"response": f"🧾 SOURCES\n{sourses}\n\n💬 ANSWER\n{question}"}
PY

# Procfile — keep uvicorn bound to Railway's $PORT
cat > Procfile <<'PROC'
web: uvicorn main:app --host 0.0.0.0 --port $PORT
PROC

# requirements.txt — use the exact versions you told me
cat > requirements.txt <<'REQ'
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9
PyJWT==2.9.0
openai==1.43.0
pinecone-client==5.0.1
python-dotenv==1.0.1
REQ
