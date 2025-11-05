from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from openai import OpenAI
import httpx
import os, traceback

# --- Load .env if present (harmless on Railway) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- SCRUB ALL PROXY VARS (prevents 'proxies' kw leaks into OpenAI client) ---
for k in (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
    "OPENAI_PROXY", "OPENAI_HTTP_PROXY", "OPENAI_HTTPS_PROXY"
):
    os.environ.pop(k, None)
os.environ.setdefault("NO_PROXY", "*")

app = FastAPI(title="Private Trust Fiduciary Advisor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pinecone setup ---
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.getenv("PINECONE_INDEX", "").strip()
host = os.getenv("PINECONE_HOST", "").strip()

if host:
    idx = pc.Index(host=host)
elif index_name:
    idx = pc.Index(index_name)
else:
    raise RuntimeError("Set PINECONE_HOST or PINECONE_INDEX")

# --- OpenAI client (explicit httpx client with NO proxy) ---
http_client = httpx.Client(timeout=60.0)  # no proxies arg
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], http_client=http_client)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/diag")
def diag():
    """Quick diagnostics so 500s become visible."""
    info = {
        "has_PINECONE_API_KEY": bool(os.getenv("PINECONE_API_KEY")),
        "has_OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "PINECONE_INDEX": index_name or None,
        "PINECONE_HOST": host or None,
    }
    try:
        # light control-plane ping
        lst = pc.list_indexes()
        info["pinecone_list_indexes_ok"] = True
        info["pinecone_indexes_count"] = len(lst or [])
    except Exception as e:
        info["pinecone_list_indexes_ok"] = False
        info["pinecone_error"] = str(e)

    try:
        # tiny embed test (cheap) just to verify openai client
        _ = client.embeddings.create(model="text-embedding-3-small", input="ping").data[0].embedding
        info["openai_embeddings_ok"] = True
    except Exception as e:
        info["openai_embeddings_ok"] = False
        info["openai_error"] = str(e)

    return info

@app.get("/rag")
def rag_endpoint(question: str = Query(..., min_length=3)):
    try:
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

    except Exception as e:
        # print full traceback to Railway logs and surface the message in JSON
        traceback.print_exc()
        return {"response": f"Error: {e.__class__.__name__}: {e}"}
