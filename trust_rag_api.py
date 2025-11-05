# trust_rag_api.py
from fastapi import FastAPI, Query, Header, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pinecone import Pinecone
from openai import OpenAI
import httpx, zipfile, io, re, os, time, traceback
from collections import deque

# -------------------- ENV / SETUP --------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Scrub proxy env so neither httpx nor the OpenAI SDK picks up proxies
for k in (
    "HTTP_PROXY","HTTPS_PROXY","ALL_PROXY",
    "http_proxy","https_proxy","all_proxy",
    "OPENAI_PROXY","OPENAI_HTTP_PROXY","OPENAI_HTTPS_PROXY"
):
    os.environ.pop(k, None)
os.environ.setdefault("NO_PROXY", "*")

# Remove bogus overrides
if os.getenv("OPENAI_BASE_URL", "").strip().lower() in ("", "none", "null"):
    os.environ.pop("OPENAI_BASE_URL", None)

API_TOKEN = os.getenv("API_TOKEN", "")  # optional bearer for /rag and /review
SYNTH_MODEL = os.getenv("SYNTH_MODEL", "gpt-4o-mini")
MAX_SNIPPETS = int(os.getenv("MAX_SNIPPETS", "20"))               # generous for comprehensive answers
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))  # large context budget
UPLOAD_MAX_BYTES = 12 * 1024 * 1024  # 12MB cap per file

# -------------------- APP --------------------
app = FastAPI(title="Private Trust Fiduciary Advisor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Optional metrics (won’t crash if missing)
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
except Exception:
    pass

# -------------------- AUTH / RATE LIMIT --------------------
def require_auth(auth_header: str | None):
    if not API_TOKEN:
        return
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth_header.split(" ", 1)[1].strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

REQUESTS = deque(maxlen=120)
RATE_WINDOW = 10
RATE_LIMIT = 100  # more permissive for longer answers

def check_rate_limit():
    now = time.time()
    while REQUESTS and now - REQUESTS[0] > RATE_WINDOW:
        REQUESTS.popleft()
    if len(REQUESTS) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too Many Requests")
    REQUESTS.append(now)

# -------------------- CLIENTS --------------------
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.getenv("PINECONE_INDEX", "").strip()
host       = os.getenv("PINECONE_HOST", "").strip()
if host:
    idx = pc.Index(host=host)
elif index_name:
    idx = pc.Index(index_name)
else:
    raise RuntimeError("Set PINECONE_HOST or PINECONE_INDEX")

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
openai_http = httpx.Client(timeout=120.0, trust_env=False)
client = OpenAI(api_key=_openai_key, http_client=openai_http)

# -------------------- RAG utilities --------------------
def _extract_snippet(meta: dict) -> str:
    for k in ("text", "chunk", "content", "body", "passage"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _clean_title(title: str) -> str:
    # strip level prefixes, hashes, OCR tags, long filenames; keep human title only
    t = title or "Unknown"
    t = re.sub(r'^[Ll]\d[_\-:\s]+', '', t)
    t = re.sub(r'(?i)\bocr\b', '', t)
    t = re.sub(r'[0-9a-f]{8,}', '', t, flags=re.I)
    if ' -- ' in t:
        first, *rest = t.split(' -- ')
        if len(first) >= 6:
            t = first
    t = t.replace('_',' ')
    t = re.sub(r'\s+', ' ', t).strip(' -–—\t\r\n')
    return t or "Unknown"

def _dedup_and_rank_sources(matches, top_k: int):
    """Unique sources with L1→L5 precedence, highest score per (title, level, page, version)."""
    rank = {"L1":1,"L2":2,"L3":3,"L4":4,"L5":5}
    best = {}
    for m in matches:
        meta = m.get("metadata", {}) if isinstance(m, dict) else (getattr(m, "metadata", {}) or {})
        raw_title = (meta.get("title") or meta.get("doc_parent") or "Unknown")
        title = _clean_title(raw_title)
        lvl   = (meta.get("doc_level") or meta.get("level") or "N/A").strip()
        page  = str(meta.get("page", "?"))
        ver   = str(meta.get("version", meta.get("v", ""))) if meta.get("version", meta.get("v", "")) else ""
        score = float(m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0))
        key = (title, lvl, page, ver)
        if key not in best or score > best[key]["score"]:
            best[key] = {"title": title, "level": lvl, "page": page, "version": ver, "score": score, "meta": meta}
    unique = list(best.values())
    unique.sort(key=lambda s: (rank.get(s["level"], 99), -s["score"]))
    return unique[:top_k]

def _citations_titles_only(unique):
    """Return deduplicated title list in precedence order (titles only)."""
    seen = set()
    lines = []
    for s in unique:
        t = s["title"]
        if t in seen:
            continue
        seen.add(t)
        lines.append(t)
    return lines

# -------------------- Synthesis: HTML only, no forced memo style --------------------
def synthesize_answer_html(question: str, unique_sources: list[dict], snippets: list[str]) -> str:
    """
    Produce comprehensive HTML following the user's requested format (e.g., draft a document
    if the prompt asks for one). Use ONLY the provided context. If context is empty/insufficient,
    return exactly: 'No relevant material found in the Trust-Law knowledge base.'
    Append one 'Citations' block (titles only, deduped) and the standard disclaimer.
    """
    if not snippets and not unique_sources:
        return "<p>No relevant material found in the Trust-Law knowledge base.</p>"

    # Build large context
    context, used = "", 0
    kept = 0
    for s in snippets:
        if not s: 
            continue
        s = s.strip()
        if not s: 
            continue
        if used + len(s) > MAX_CONTEXT_CHARS:
            break
        context += f"\n---\n{s}"
        used += len(s)
        kept += 1
        if kept >= MAX_SNIPPETS:
            break

    titles = _citations_titles_only(unique_sources)
    titles_html = "<ul>" + "".join(f"<li>{re.escape(t) if '<' in t else t}</li>" for t in titles) + "</ul>" if titles else "<p>No relevant material found in the Trust-Law knowledge base.</p>"

    # System: no style override; HTML only; follow user’s requested genre/format
    system_msg = (
        "You are the 'Private Trust Fiduciary Advisor'. Produce a comprehensive, formal legal output that matches the user's requested format "
        "(e.g., if asked to draft a resolution/contract/letter, draft it). Output strictly as clean HTML (no Markdown, no asterisks). "
        "Use semantic tags: <h1-4>, <p>, <ul><li>, <strong>, <em>, <blockquote>, <hr>, <sup>…</sup>. "
        "Use ONLY the provided context; do not rely on any internal knowledge."
    )

    # User: pass the question verbatim and the raw context; ask for one citations block (titles only) + disclaimer
    user_msg = (
        f"<h2>Request</h2><p>{question}</p>"
        f"<h3>Context</h3><pre>{(context or '').strip()}</pre>"
        f"<p>Instructions:</p>"
        f"<ul>"
        f"<li>Use only the context above; if insufficient, respond exactly: <code>No relevant material found in the Trust-Law knowledge base.</code></li>"
        f"<li>Do not force a memorandum style; follow the user's requested format and genre.</li>"
        f"<li>Do not include any separate sources table; include exactly one <strong>Citations</strong> block at the end with titles only (deduplicated).</li>"
        f"<li>After citations, append this one-line disclaimer verbatim: "
        f"<em>This response is provided solely for educational and informational purposes. It does not constitute legal, tax, or financial advice, nor does it establish an attorney-client or fiduciary relationship. Users must consult qualified counsel or a CPA for application of law to specific facts.</em></li>"
        f"</ul>"
        f"<hr><h3>Citations</h3>{titles_html}"
    )

    try:
        chat = client.chat.completions.create(
            model=SYNTH_MODEL,
            temperature=0.15,
            max_tokens=2200,   # allow full, formatted output
            messages=[{"role":"system","content":system_msg},
                      {"role":"user","content":user_msg}],
        )
        html = (chat.choices[0].message.content or "").strip()
        if not html:
            return "<p>No relevant material found in the Trust-Law knowledge base.</p>"
        # If model returned plain text, wrap into HTML
        if "<" not in html:
            html = "<div><p>" + html.replace("\n","<br>") + "</p></div>"
        return html
    except Exception as e:
        return f"<p><em>(Synthesis unavailable: {e})</em></p>"

# -------------------- Widget (multiline textarea + auto-resize, black text) --------------------
WIDGET_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Private Trust Fiduciary Advisor</title>
<style>
  :root { --ink:#000; --bg:#f7f7f8; --card:#fff; --muted:#444; --border:#e5e5ea; --brand:#0B3B5C; }
  html,body{height:100%}
  body{font-family:system-ui,Arial,sans-serif;margin:0;padding:20px;background:var(--bg);color:var(--ink)}
  .wrap{max-width:1000px;margin:0 auto;min-height:calc(100vh - 40px);display:flex;flex-direction:column}
  h1{font-size:22px;margin:0 0 12px;color:var(--ink)}
  form{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start;margin:12px 0}
  textarea{flex:1;min-height:140px;max-height:50vh;resize:vertical;padding:12px;border:1px solid #d0d0d6;border-radius:10px;color:var(--ink);background:#fff;line-height:1.5}
  textarea::placeholder{color:#888}
  input[type=file]{padding:10px;border:1px dashed #c8ccd3;border-radius:10px;background:#fff;color:var(--ink)}
  button{padding:12px 18px;border:none;border-radius:10px;background:var(--brand);color:#fff;cursor:pointer;white-space:nowrap}
  .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px;margin-top:14px;color:var(--ink)}
  .out{flex:1;overflow:auto}
  .muted{color:var(--muted)}
</style>
</head>
<body>
  <div class="wrap">
    <h1>Private Trust Fiduciary Advisor</h1>
    <form id="f">
      <textarea id="q" placeholder="Type your fiduciary/trust question here… (multi-line supported). If you need a specific document (e.g., Trustee Resolution, Certificate of Trust, §642(c)(2) set-aside memo), say so explicitly." required></textarea>
      <input id="file" type="file" multiple accept=".pdf,.txt,.docx"/>
      <button type="submit">Ask</button>
    </form>
    <div id="out" class="out">
      <div class="card muted">Type your question (box expands as you type), optionally attach PDF/TXT/DOCX, then click <strong>Ask</strong>.</div>
    </div>
  </div>
<script>
const OUT=document.getElementById('out'),F=document.getElementById('f'),Q=document.getElementById('q'),FILES=document.getElementById('file');
function autoresize(){ Q.style.height='auto'; const max=window.innerHeight*0.5; Q.style.height=Math.min(Q.scrollHeight,max)+'px'; }
Q.addEventListener('input',autoresize); window.addEventListener('resize',autoresize); setTimeout(autoresize,60);

async function askRag(q){
  const u=new URL('/rag',location.origin);u.searchParams.set('question',q);u.searchParams.set('top_k','12');
  const r=await fetch(u,{headers:{}}); return r.json();
}
async function askReview(q,files){
  const fd=new FormData(); fd.append('question',q); for(const f of files) fd.append('files',f);
  const r=await fetch('/review',{method:'POST',body:fd,headers:{}}); return r.json();
}

F.addEventListener('submit',async(e)=>{
  e.preventDefault();
  OUT.innerHTML='<div class="card">Working…</div>';
  try{
    let data;
    if(FILES.files && FILES.files.length>0){ data=await askReview(Q.value.trim(),FILES.files);}
    else { data=await askRag(Q.value.trim());}
    const ans=data.answer||data.response||'(no answer)';
    // render as HTML to preserve headings, bold, lists, dividers
    OUT.innerHTML='<div class="card" style="color:#000;">'+ans+'</div>';
  }catch(err){
    OUT.innerHTML='<div class="card">Error: '+String(err)+'</div>';
  }
});
</script>
</body>
</html>"""

@app.get("/widget", response_class=HTMLResponse)
def widget():
    return HTMLResponse(WIDGET_HTML)

# -------------------- Health/Diag --------------------
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
        r = httpx.get("https://api.openai.com/v1/models",
                      headers={"Authorization": f"Bearer {_openai_key}"},
                      timeout=20.0, trust_env=False)
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

# -------------------- RAG Answer (HTML, no forced memo style) --------------------
@app.get("/rag")
def rag_endpoint(
    question: str = Query(..., min_length=3),
    top_k: int = Query(12, ge=1, le=30),
    level: str | None = Query(None),
    authorization: str | None = Header(default=None),
):
    require_auth(authorization)
    check_rate_limit()
    t0 = time.time()
    try:
        emb = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
        flt = {"doc_level": {"$eq": level}} if level else None
        results = idx.query(vector=emb, top_k=max(top_k, 12), include_metadata=True, filter=flt)
        matches = results["matches"] if isinstance(results, dict) else getattr(results, "matches", [])
        unique = _dedup_and_rank_sources(matches, top_k=top_k)

        # harvest snippets
        snippets = []
        for s in unique:
            sn = _extract_snippet(s["meta"])
            if sn: snippets.append(sn)

        html = synthesize_answer_html(question, unique, snippets)
        if not html.strip():
            html = "<p>No relevant material found in the Trust-Law knowledge base.</p>"
        return {"answer": html, "t_ms": int((time.time()-t0)*1000)}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{e.__class__.__name__}: {e}")

# -------------------- Document Review (PDF/TXT/DOCX) --------------------
@app.post("/review")
def review_endpoint(
    authorization: str | None = Header(default=None),
    question: str = Form("Please produce a comprehensive, formal legal output based strictly on the attached materials."),
    files: list[UploadFile] = File(default=[]),
):
    require_auth(authorization)
    check_rate_limit()
    t0 = time.time()
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")
        texts = []
        for uf in files:
            name = (uf.filename or "").lower()
            raw = uf.file.read(UPLOAD_MAX_BYTES + 1)
            if len(raw) > UPLOAD_MAX_BYTES:
                raise HTTPException(status_code=413, detail=f"{uf.filename} exceeds {UPLOAD_MAX_BYTES//1024//1024}MB limit.")
            if name.endswith(".pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(raw))
                    pages = []
                    for p in reader.pages:
                        try: pages.append(p.extract_text() or "")
                        except Exception: pages.append("")
                    texts.append("\n".join(pages))
                except Exception as e:
                    raise HTTPException(status_code=415, detail=f"Failed to parse PDF: {uf.filename} ({e})")
            elif name.endswith(".txt"):
                try: texts.append(raw.decode("utf-8", errors="ignore"))
                except Exception: texts.append(raw.decode("latin-1", errors="ignore"))
            elif name.endswith(".docx"):
                try:
                    try:
                        import docx  # optional
                        doc = docx.Document(io.BytesIO(raw))
                        paras = [p.text for p in doc.paragraphs if p.text]
                        texts.append("\n".join(paras))
                    except Exception:
                        with zipfile.ZipFile(io.BytesIO(raw)) as z:
                            xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
                            stripped = re.sub(r"<[^>]+>", " ", xml)
                            stripped = re.sub(r"\s+", " ", stripped).strip()
                            texts.append(stripped)
                except Exception as e:
                    raise HTTPException(status_code=415, detail=f"Failed to parse DOCX: {uf.filename} ({e})")
            else:
                raise HTTPException(status_code=415, detail=f"Unsupported type: {uf.filename} (only PDF/TXT/DOCX)")
        merged = "\n---\n".join([t for t in texts if t.strip()])
        chunks, CHUNK = [], 2000
        for i in range(0, len(merged), CHUNK):
            chunks.append(merged[i:i+CHUNK])
            if len(chunks) >= MAX_SNIPPETS:
                break
        pseudo_sources = [{"title":"Uploaded Document","level":"L5","page":"?","version":"","score":1.0}]
        html = synthesize_answer_html(question, pseudo_sources, chunks)
        if not html.strip():
            html = "<p>No relevant material found in the Trust-Law knowledge base.</p>"
        return {"answer": html, "t_ms": int((time.time()-t0)*1000)}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{e.__class__.__name__}: {e}")
