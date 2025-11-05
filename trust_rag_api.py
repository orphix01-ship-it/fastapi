# trust_rag_api.py
from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from openai import OpenAI
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import HTMLResponse
import httpx
import os, time, traceback
from collections import deque

# -------------------- ENV / SETUP --------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Scrub proxy env so neither httpx nor the OpenAI SDK injects proxies
for k in (
    "HTTP_PROXY","HTTPS_PROXY","ALL_PROXY",
    "http_proxy","https_proxy","all_proxy",
    "OPENAI_PROXY","OPENAI_HTTP_PROXY","OPENAI_HTTPS_PROXY"
):
    os.environ.pop(k, None)
os.environ.setdefault("NO_PROXY", "*")

# Optional: if OPENAI_BASE_URL is accidentally set to junk, remove it
if os.getenv("OPENAI_BASE_URL", "").strip().lower() in ("", "none", "null"):
    os.environ.pop("OPENAI_BASE_URL", None)

API_TOKEN = os.getenv("API_TOKEN", "")  # set in Railway to protect /rag

# -------------------- APP --------------------
app = FastAPI(title="Private Trust Fiduciary Advisor API")

# CORS (relax now; you can lock to your domain later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# -------------------- AUTH / RATE LIMIT --------------------
def require_auth(auth_header: str | None):
    if not API_TOKEN:  # if you haven't set a token, skip auth
        return
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth_header.split(" ", 1)[1].strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

REQUESTS = deque(maxlen=50)  # simple in-memory sliding window
RATE_WINDOW = 10             # seconds
RATE_LIMIT = 20              # max requests per window

def check_rate_limit():
    now = time.time()
    while REQUESTS and now - REQUESTS[0] > RATE_WINDOW:
        REQUESTS.popleft()
    if len(REQUESTS) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too Many Requests")
    REQUESTS.append(now)

# -------------------- CLIENTS --------------------
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

# OpenAI — custom httpx with trust_env=False so no env proxies are used
def _clean_openai_key(raw: str) -> str:
    s = (raw or "").strip()
    if not s.startswith("sk-"):
        parts = [t.strip() for t in s.replace("=", " ").split() if t.strip().startswith("sk-")]
        if parts:
            s = parts[-1]
    if not s.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY is malformed — set only the raw 'sk-...' value.")
    return s

_openai_key = _clean_openai_key(os.getenv("OPENAI_API_KEY", ""))
openai_http = httpx.Client(timeout=60.0, trust_env=False)
client = OpenAI(api_key=_openai_key, http_client=openai_http)

# -------------------- SIMPLE WIDGET PAGE --------------------
WIDGET_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Private Trust Fiduciary Advisor</title>
  <style>
    body{font-family:system-ui,Arial,sans-serif;margin:0;padding:20px;background:#f7f7f8}
    .wrap{max-width:900px;margin:0 auto}
    h1{font-size:22px;margin:0 0 12px}
    form{display:flex;gap:8px;margin:12px 0}
    input[type=text]{flex:1;padding:12px;border:1px solid #d0d0d6;border-radius:8px}
    button{padding:12px 16px;border:none;border-radius:8px;background:#0B3B5C;color:#fff;cursor:pointer}
    .card{background:#fff;border:1px solid #e5e5ea;border-radius:12px;padding:14px;margin-top:12px}
    .src{font-size:13px;color:#555;margin-top:8px}
    .muted{color:#777}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Private Trust Fiduciary Advisor</h1>
    <form id="f">
      <input id="q" type="text" placeholder="Ask a fiduciary/trust question…" required />
      <button type="submit">Ask</button>
    </form>
    <div id="out" class="muted">Ask something to see results.</div>
  </div>
  <script>
    const OUT = document.getElementById('out');
    const F = document.getElementById('f');
    const Q = document.getElementById('q');

    async function ask(q) {
      OUT.innerHTML = '<div class="card">Working…</div>';
      const u = new URL('/rag', location.origin);
      u.searchParams.set('question', q);
      u.searchParams.set('top_k', '5');

      const res = await fetch(u, {
        headers: {
          // If you enabled API_TOKEN on the server, uncomment next line and paste the token:
          // 'Authorization': 'Bearer YOUR_API_TOKEN'
        }
      });
      const data = await res.json();

      const answer = data.answer || data.response || '(no answer)';
      const sources = data.sources || [];

      let html = '<div class="card"><strong>Answer</strong><br>' + escapeHtml(answer).replaceAll('\\n','<br>') + '</div>';
      if (sources.length) {
        html += '<div class="card"><strong>Sources</strong><div class="src"><ul>';
        for (const s of sources) {
          html += `<li>${escapeHtml(s.title || '')} (Level ${escapeHtml(s.level||'')}, p.${escapeHtml(String(s.page||'?'))}, score ${Number(s.score||0).toFixed(3)})</li>`;
        }
        html += '</ul></div></div>';
      }
      OUT.innerHTML = html;
    }

    F.addEventListener('submit', (e) => {
      e.preventDefault();
      ask(Q.value.trim());
    });

    function escapeHtml(s){return s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
  </script>
</body>
</html>"""

@app.get("/widget", response_class=HTMLResponse)
def widget():
    return HTMLResponse(WIDGET_HTML)

# -------------------- ENDPOINTS --------------------
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
    try:
        lst = pc.list_indexes()
        info["pinecone_list_indexes_ok"] = True
        info["pinecone_indexes_count"] = len(lst or [])
    except Exception as e:
        info["pinecone_list_indexes_ok"] = False
        info["pinecone_error"] = str(e)

    try:
        r = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {_openai_key}"},
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

    try:
        _ = client.embeddings.create(model="text-embedding-3-small", input="ping").data[0].embedding
        info["openai_embeddings_ok"] = True
    except Exception as e:
        info["openai_embeddings_ok"] = False
        info["openai_error"] = str(e)

    return info

@app.get("/rag")
def rag_endpoint(
    question: str = Query(..., min_length=3),
    top_k: int = Query(5, ge=1, le=20),
    level: str | None = Query(None),
    authorization: str | None = Header(default=None),
):
    # Guard & rate-limit
    require_auth(authorization)
    check_rate_limit()

    t0 = time.time()
    try:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        ).data[0].embedding

        flt = {"doc_level": {"$eq": level}} if level else None
        results = idx.query(vector=emb, top_k=top_k, include_metadata=True, filter=flt)
        matches = results["matches"] if isinstance(results, dict) else getattr(results, "matches", [])

        sources = []
        for m in matches:
            meta  = m.get("metadata", {}) if isinstance(m, dict) else (getattr(m, "metadata", {}) or {})
            title = meta.get("title") or meta.get("doc_parent") or "Unknown"
            lvl   = meta.get("doc_level", "N/A")
            page  = meta.get("page", "?")
            score = float(m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0))
            sources.append({"title": title, "level": lvl, "page": str(page), "score": score})

        answer = question
        return {"answer": answer, "sources": sources, "t_ms": int((time.time()-t0)*1000)}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{e.__class__.__name__}: {e}")
