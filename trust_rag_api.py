from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from openai import OpenAI
import httpx
import os, traceback

# --------- ENV HYGIENE (runs before any clients init) ----------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Kill all proxy vars so neither httpx nor OpenAI can see them
for k in (
    "HTTP_PROXY","HTTPS_PROXY","ALL_PROXY",
    "http_proxy","https_proxy","all_proxy",
    "OPENAI_PROXY","OPENAI_HTTP_PROXY","OPENAI_HTTPS_PROXY"
):
    os.environ.pop(k, None)

# Ensure no proxy is applied by libcurl/httpx for any host
os.environ["NO_PROXY"] = "*"

# Optional: if someone set a custom base URL by mistake, clear it
if os.getenv("OPENAI_BASE_URL", "").strip().lower() in ("", "none", "null"):
    os.environ.pop("OPENAI_BASE_URL", None)

app = FastAPI(title="Private Trust Fiduciary Advisor API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --------- Clients ----------
# Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = os.getenv("PINECONE_INDEX", "").strip()
host       = os.getenv("PINECONE_HOST", "").strip()
if host:
    idx = pc.Index(host=host)
elif index_name:
    idx = pc.Index(index_name)
else:
    raise RuntimeError("Set PINECONE_HOST or PINECONE_INDEX")

# OpenAI — provide our OWN httpx client with trust_env=False (ignores env proxies)
openai_http = httpx.Client(timeout=60.0, trust_env=False)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], http_client=openai_http)

# --------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/diag")
def diag():
    info = {
        "has_PINECONE_API_KEY": bool(os.getenv("PINECONE_API_KEY")),
        "has_OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "PINECONE_INDEX": index_name or None,
        "PINECONE_HOST": host or None,
        "NO_PROXY": os.getenv("NO_PROXY"),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
    }
    # Pinecone control-plane ping
    try:
        lst = pc.list_indexes()
        info["pinecone_list_indexes_ok"] = True
        info["pinecone_indexes_count"] = len(lst or [])
    except Exception as e:
        info["pinecone_list_indexes_ok"] = False
        info["pinecone_error"] = str(e)

    # OpenAI plain HTTP sanity (also trust_env=False)
    try:
        r = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            timeout=20.0,
            trust_env=False,
        )
        info["openai_http_ok"] = (200 <= r.status_code < 500)
        info["openai_http_status"] = r.status_code
        if r.status_code != 200:
            info["openai_http_body"] = r.text[:400]
    except Exception as e:
        info["openai_http_ok"] = False
        info["openai_http_error"] = str(e)

    # SDK embeddings check (uses our custom no-proxy client)
    try:
        _ = client.embeddings.create(model="text-embedding-3-small", input="ping").data[0].embedding
        info["openai_embeddings_ok"] = True
    except Exception as e:
        info["openai_embeddings_ok"] = False
        info["openai_error"] = str(e)

    return info

@app.get("/rag")
def rag_endpoint(question: str = Query(..., min_length=3)):
    try:
        emb = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
        results = idx.query(vector=emb, top_k=5, include_metadata=True)
        matches = results["matches"] if isinstance(results, dict) else getattr(results, "matches", [])
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
        traceback.print_exc()
        return {"response": f"Error: {e.__class__.__name__}: {e}"}
