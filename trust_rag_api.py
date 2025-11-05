# trust_rag_api.py
from fastapi import FastAPI, Query, Header, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pinecone import Pinecone
from openai import OpenAI
import httpx, zipfile, io, re, os, time, traceback
from collections import deque

# ========== ENV / SETUP ==========
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# remove proxies so SDK never injects `proxies=` automatically
for _k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy",
           "OPENAI_PROXY","OPENAI_HTTP_PROXY","OPENAI_HTTPS_PROXY"):
    os.environ.pop(_k, None)
os.environ.setdefault("NO_PROXY", "*")

# ignore broken custom base url
if os.getenv("OPENAI_BASE_URL", "").strip().lower() in ("", "none", "null"):
    os.environ.pop("OPENAI_BASE_URL", None)

API_TOKEN         = os.getenv("API_TOKEN", "")      # optional bearer for /rag & /review
SYNTH_MODEL       = os.getenv("SYNTH_MODEL", "gpt-4o-mini")
MAX_SNIPPETS      = int(os.getenv("MAX_SNIPPETS", "20"))          # generous context count
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))  # large context
UPLOAD_MAX_BYTES  = 12 * 1024 * 1024                               # 12 MB per file

app = FastAPI(title="Private Trust Fiduciary Advisor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# optional metrics; harmless if package not installed
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
except Exception:
    pass

# ========== AUTH / RATE LIMIT ==========
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
RATE_LIMIT  = 100

def check_rate_limit():
    now = time.time()
    while REQUESTS and now - REQUESTS[0] > RATE_WINDOW:
        REQUESTS.popleft()
    if len(REQUESTS) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too Many Requests")
    REQUESTS.append(now)

# ========== CLIENTS ==========
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.getenv("PINECONE_INDEX", "").strip()
host       = os.getenv("PINECONE_HOST", "").strip()
idx = pc.Index(host=host) if host else pc.Index(index_name)

def _clean_openai_key(raw: str) -> str:
    s = (raw or "").trim() if hasattr(raw, "trim") else (raw or "").strip()
    if not s.startswith("sk-"):
        parts = [t.strip() for t in s.replace("=", " ").split() if t.strip().startswith("sk-")]
        if parts:
            s = parts[-1]
    if not s.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY appears malformed.")
    return s

_openai_key = _clean_openai_key(os.getenv("OPENAI_API_KEY", ""))
openai_http = httpx.Client(timeout=120.0, trust_env=False)
client      = OpenAI(api_key=_openai_key, http_client=openai_http)

# ========== RAG HELPERS ==========
def _extract_snippet(meta: dict) -> str:
    for k in ("text","chunk","content","body","passage"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _clean_title(title: str) -> str:
    # strip level prefixes, hashes, OCR suffixes, long filenames; keep clean human title
    t = (title or "Unknown")
    t = re.sub(r'^[Ll]\d[_\-:\s]+', '', t)
    t = re.sub(r'(?i)\bocr\b', '', t)
    t = re.sub(r'[0-9a-f]{8,}', '', t)
    if " -- " in t:
        first, *_ = t.split(" -- ")
        if len(first) >= 6:
            t = first
    return re.sub(r"\s+", " ", t.replace("_", " ")).strip(" -–—")

def _dedup_and_rank_sources(matches, top_k: int):
    """Unique by (title, level, page, version), precedence L1→L5, highest score within each key."""
    rank = {"L1":1,"L2":2,"L3":3,"L4":4,"L5":5}
    best = {}
    for m in (matches or []):
        meta  = m.get("metadata", {}) if isinstance(m, dict) else (getattr(m, "metadata", {}) or {})
        title = _clean_title(meta.get("title") or meta.get("doc_parent") or "Unknown")
        lvl   = (meta.get("doc_level") or meta.get("level") or "N/A").strip()
        page  = str(meta.get("page", "?"))
        ver   = str(meta.get("version", meta.get("v", ""))) if meta.get("version", meta.get("v", "")) else ""
        score = float(m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0))
        key   = (title, lvl, page, ver)
        if key not in best or score > best[key]["score"]:
            best[key] = {"title": title, "level": lvl, "page": page, "version": ver, "score": score, "meta": meta}
    uniq = list(best.values())
    uniq.sort(key=lambda s: (rank.get(s["level"], 99), -s["score"]))
    return uniq[:top_k]

def _titles_only(uniq_sources: list[dict]) -> list[str]:
    seen, out = set(), []
    for s in uniq_sources:
        t = s["title"]
        if t in seen: 
            continue
        seen.add(t)
        out.append(t)
    return out

# ========== SYNTHESIS (NO SYSTEM MESSAGE) ==========
def synthesize_html(question: str, uniq_sources: list[dict], snippets: list[str]) -> str:
    """
    Send only a single user message (no system message), so your Custom GPT’s own policy governs style.
    We just provide the question + raw context and a *separate* citation title list.
    """
    if not snippets and not uniq_sources:
        return "<p>No relevant material found in the Trust-Law knowledge base.</p>"

    # Build large context block
    buf, used, kept = [], 0, 0
    for s in snippets:
        s = s.strip()
        if not s: 
            continue
        if used + len(s) > MAX_CONTEXT_CHARS:
            break
        buf.append(s)
        used += len(s); kept += 1
        if kept >= MAX_SNIPPETS:
            break
    context = "\n---\n".join(buf)

    # hand over titles only (deduped, precedence applied)
    titles = _titles_only(uniq_sources)
    titles_html = "<ul>" + "".join(f"<li>{t}</li>" for t in titles) + "</ul>" if titles else "<p></p>"

    # One and only user message: your question + raw context + a plain citations block.
    # No style instructions, no memo framing, no extra rules.
    user_msg = (
        f"<h2>Question</h2>\n<p>{question}</p>\n"
        f"<h3>Context</h3>\n<pre>{context}</pre>\n"
        f"<h3>Citations</h3>\n{titles_html}"
    )

    try:
        res = client.chat.completions.create(
            model=SYNTH_MODEL,
            temperature=0.15,
            max_tokens=2200,
            messages=[{"role": "user", "content": user_msg}],
        )
        html = (res.data[0].message.content if hasattr(res, "data") else res.choices[0].message.content).strip()
        if not html:
            return "<p>No relevant material found in the Trust-Law knowledge base.</p>"
        # If model replies in plain text, wrap for display
        if "<" not in html:
            html = "<div><p>" + html.replace("\n", "<br>") + "</p></div>"
        return html
    except Exception as e:
        return f"<p><em>(Synthesis unavailable: {e})</em></p>"

# ========== SUPER-MINIMAL WHITE WIDGET ==========
WIDGET_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Private Trust Fiduciary Advisor</title>
<style>
  html,body{margin:0;padding:0;background:#fff;color:#000;font:16px/1.6 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:960px;margin:0 auto;padding:16px}
  form{display:flex;gap:8px;align-items:flex-start;margin:12px 0}
  textarea{flex:1;min-height:140px;max-height:60vh;resize:vertical;border:1px solid #ddd;border-radius:6px;padding:10px;font:16px/1.5 inherit;color:#000;background:#fff}
  input[type=file]{border:1px solid #ddd;border-radius:6px;padding:8px;background:#fff;color:#000}
  button{border:1px solid #ddd;background:#fff;color:#000;padding:10px 16px;border-radius:6px;cursor:pointer}
  .out{margin-top:12px}
  .card{border:1px solid #eee;border-radius:6px;padding:16px;background:#fff;color:#000;}
</style>
</head>
<body>
  <div class="wrap">
    <form id="f" onsubmit="return false;">
      <textarea id="q" placeholder="Type your question or paste text. The box expands as you type." required></textarea>
      <input id="file" type="file" multiple accept=".pdf,.txt,.docx"/>
      <button id="ask">Ask</button>
    </form>
    <div id="out" class="out"><div class="card">Enter your question and click <strong>Ask</strong>.</div></div>
  </div>

<script>
// super minimal: only expand textarea, fetch, and render HTML (or rendered markdown)
const q   = document.getElementById('q');
const out = document.getElementById('out');
const btn = document.getElementById('ask');
const files = document.getElementById('file');

function autoresize() {
  q.style.height = 'auto';
  q.style.height = Math.min(q.scrollHeight, window.innerHeight*0.6) + 'px';
}
q.addEventListener('input', autoresize);
window.addEventListener('resize', autoresize);

async function askRag(text) {
  const url = new URL('/rag', location.origin);
  url.searchParams.set('question', text);
  url.searchParams.set('top_k', '12');
  const res = await fetch(url, { method:'GET' });
  return res.json();
}
async function askReview(text, fileList) {
  const fd = new FormData();
  fd.append('question', text);
  for (const f of fileList) fd.append('files', f);
  const res = await fetch('/review', { method:'POST', body: fd });
  return res.json();
}

// tiny markdown-to-HTML renderer (headings, bold/italic, code blocks, lists, hr)
function mdToHtml(md){
  if(!md) return '';
  let h = md.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  // code fences
  h = h.replace(/```([\\s\\S]*?)```/g, (_,code)=>`<pre><code>${code.replace(/</g,'&lt;')}</code></pre>`);
  // headings
  h = h.replace(/^######\\s+(.*)$/gm,'<h6>$1</h6>')
       .replace(/^#####\\s+(.*)$/gm,'<h5>$1</h5>')
       .replace(/^####\\s+(.*)$/gm,'<h4>$1</h4>')
       .replace(/^###\\s+(.*)$/gm,'<h3>$1</h3>')
       .replace(/^##\\s+(.*)$/gm,'<h2>$1</h2>')
       .replace(/^#\\s+(.*)$/gm,'<h1>$1</h1>');
  // horizontal rule
  h = h.replace(/^---$/gm,'<hr>');
  // bold/italic
  h = h.replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>')
       .replace(/\\*(.+?)\\*/g,'<em>$1</em>'); 
  // simple lists
  h = h.replace(/(?:^|\\n)[*-]\\s+(.*)/g, (m, item)=>`<li>${item}</li>`)
  h = h.replace(/(<li>.*<\\/li>)(\\n?)+/gs, m => `<ul>${m}</ul>`);
  // paragraphs
  h = h.replace(/\\n{2,}/g,'</p><p>').replace(/^(?!<h\\d|<ul|<pre|<hr)(.+)$/gm,'<p>$1</p>');
  return h;
}

btn.addEventListener('click', async () => {
  const text = q.value.trim();
  if (!text) return;
  out.innerHTML = '<div class="card">Working…</div>';
  try {
    let data;
    if (files.files && files.files.length > 0) {
      data = await askReview(text, files.files);
    } else {
      data = await askRag(text);
    }
    let html = data && data.answer ? data.answer : '';
    // If API returned plain HTML from GPT, use it; if it seems markdown, render to HTML.
    const looksHtml = typeof html === 'string' && /<\\w+[^>]*>/.test(html);
    const rendered = looksHtml ? html : mdToHtml(String(html || ''));
    out.innerHTML = '<div class="card">' + rendered + '</div>';
  } catch (e) {
    out.innerHTML = '<div class="card">Error: ' + (e && e.message ? e.message : String(e)) + '</div>';
  }
});
</script>
</body>
</html>
"""

@app.get("/widget", response_class=HTMLResponse)
def widget():
    return HTMLResponse(WIDGET_HTML)

# ========== Health / Diag ==========
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
    }
    try:
        lst = pc.list_indexes()
        info["pinecone_ok"] = True
        info["index_count"] = len(lst or [])
    except Exception as e:
        info["pinecone_ok"] = False
        info["error"] = str(e)
    return info

# ========== RAG ==========
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
        res = idx.query(vector=emb, top_k=max(top_k, 12), include_metadata=True, filter=flt)
        matches = res["matches"] if isinstance(res, dict) else getattr(res, "matches", [])
        uniq = _dedup_and_rank_sources(matches, top_k=top_k)
        snippets = [s for s in (_extract_snippet(u["meta"]) for u in uniq) if s]
        html = synthesize_html(question, uniq, snippets)
        return {"answer": html, "t_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ========== Review (PDF/TXT/DOCX) ==========
@app.post("/review")
def review_endpoint(
    authorization: str | None = Header(default=None),
    question: str = Form(""),
    files: list[UploadFile] = File(default=[]),
):
    require_auth(authorization)
    check_rate_limit()
    t0 = time.time()
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")
        texts = []
        for f in files:
            name = (f.filename or "").lower()
            raw  = f.file.read(UPLOAD_MAX_BYTES + 1)
            if len(raw) > UPLOAD_MAX_BYTES:
                raise HTTPException(status_code=413, detail=f"{f.filename} exceeds {UPLOAD_MAX_BYTES//1024//1024}MB limit.")
            if name.endswith(".pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(raw))
                    pages = []
                    for p in reader.pages:
                        try:
                            pages.append(p.extract_text() or "")
                        except Exception:
                            pages.append("")
                    texts.append("\n".join(pages))
                except Exception as e:
                    raise HTTPException(status_code=415, detail=f"Failed to parse PDF: {f.filename} ({e})")
            elif name.endswith(".txt"):
                try:
                    texts.append(raw.decode("utf-8", errors="ignore"))
                except Exception:
                    texts.append(raw.decode("latin-1", errors="ignore"))
            elif name.endswith(".docx"):
                try:
                    try:
                        import docx
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
                    raise HTTPException(status_code=415, detail=f"Failed to parse DOCX: {f.filename} ({e})")
            else:
                raise HTTPException(status_code=415, detail=f"Unsupported file type: {f.filename} (only PDF/TXT/DOCX)")
        merged  = "\n---\n".join([t for t in texts if t.strip()])
        chunks  = [merged[i:i+2000] for i in range(0, len(merged), 2000)][:MAX_SNIPPETS]
        pseudo  = [{"title": "Uploaded Document", "level": "L5", "page": "?", "version": "", "score": 1.0, "meta": {}}]
        html    = synthesize_html(question or "Please analyze the attached materials.", pseudo, chunks)
        return {"answer": html, "t_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
