from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from openai import OpenAI
import os

# --- Load environment (.env harmless on Railway) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- SCRUB PROXY VARS to prevent OpenAI client 'proxies' kw errors ---
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)

app = FastAPI(title="Private Trust Fiduciary Advisor API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pinecone client ---
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = os.getenv("PINECONE_INDEX", "").strip()
host = os.getenv("PINECONE_HOST", "").strip()
if host:
    idx = pc.Index(host=host)
elif index_name:
    idx = pc.Index(index_name)
else:
    raise RuntimeError("Set PINECONE_HOST or PINECONE_INDEX")

# --- OpenAI client (NO proxy kw) ---
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/rag")
def rag_endpoint(question: str = Query(..., min_length=3)):
    # 1) Embed
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    # 2) Query Pinecone
    results = idx.query(vector=emb, top_k=5, include_metadata=True)
    matches = results["matches"] if isinstance(results, dict) else getattr(results, "matches", [])

    # 3) Summarize sources
    lines = []
    for m in matches:
        meta = m.get("metadata", {}) if isinstance(m, dict) else (getattr(m, "metadata", {}) or {})
        score = m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0)
        title = meta.get("title") or meta.get("doc_parent") or "Unknown"
        level = meta.get("doc_level", "N/A")
        page  = meta.get("page", "?")
        lines.append(f"{title} (Level {level} p.{page} – score {float(score):.3f})")

    sources = "\n".join(lines) if lines else "No matches."
    return {"response": f"🧾 SOURCES\n{sources}\n\n💬 ANSWER\n{question}"}
