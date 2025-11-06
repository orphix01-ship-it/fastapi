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

API_TOKEN         = os.getenv("API_TOKEN", "")      # optional bearer for /search, /rag & /review
SYNTH_MODEL       = os.getenv("SYNTH_MODEL", "gpt-4o-mini")
MAX_SNIPPETS      = int(os.getenv("MAX_SNIPPETS", "20"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))
UPLOAD_MAX_BYTES  = 12 * 1024 * 1024  # 12 MB

app = FastAPI(title="Private Trust Fiduciary Advisor API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# optional metrics
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
    s = (raw or "").strip()
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
    rank = {"L1":1,"L2":2,"L3":3,"L4":4,"L5":5}
    best = {}
    for m in (matches or []):
        meta  = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
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
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

# ========== SYNTHESIS ==========
def synthesize_html(question: str, uniq_sources: list[dict], snippets: list[str]) -> str:
    if not snippets and not uniq_sources:
        return "<p>No relevant material found in the Trust-Law knowledge base.</p>"

    buf, used, kept = [], 0, 0
    for s in snippets:
        s = s.strip()
        if not s:
            continue
        if used + len(s) > MAX_CONTEXT_CHARS:
            break
        buf.append(s); used += len(s); kept += 1
        if kept >= MAX_SNIPPETS:
            break
    context = "\n---\n".join(buf)
    titles = _titles_only(uniq_sources)
    titles_html = "<ul>" + "".join(f"<li>{t}</li>" for t in titles) + "</ul>" if titles else "<p></p>"

    user_msg = (
        f"<h2>Question</h2>\n<p>{question}</p>\n"
        f"<h3>Context</h3>\n<pre>{context}</pre>\n"
        f"<h3>Citations</h3>\n{titles_html}"
    )
    try:
        res = client.chat.completions.create(
            model=SYNTH_MODEL, temperature=0.15, max_tokens=2200,
            messages=[{"role": "user", "content": user_msg}],
        )
        html = (getattr(res, "choices", None) or getattr(res, "data"))[0].message.content.strip()
        if not html:
            return "<p>No relevant material found in the Trust-Law knowledge base.</p>"
        if "<" not in html:
            html = "<div><p>" + html.replace("\n", "<br>") + "</p></div>"
        return html
    except Exception as e:
        return f"<p><em>(Synthesis unavailable: {e})</em></p>"

# ========== WIDGET (no avatars, no initial advisor bubble, user light-blue, advisor unboxed) ==========
WIDGET_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Private Trust Fiduciary Advisor</title>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#ffffff; --text:#000000; --border:#e5e5e5; --ring:#d9d9d9;
    --user:#e8f1ff;
    --shadow:0 1px 2px rgba(0,0,0,.03), 0 8px 24px rgba(0,0,0,.04);
    --font: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial;
    --title:"Cinzel",serif;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 var(--font)}
  .app{display:flex;flex-direction:column;height:100vh;width:100%}
  .header{background:#fff;border-bottom:1px solid var(--border)}
  .header .inner{max-width:900px;margin:0 auto;padding:14px 16px;text-align:center}
  .title{font-family:var(--title);font-weight:300;letter-spacing:.2px;color:#000;font-size:20px}

  .main{flex:1;overflow:auto;padding:24px 12px 140px}
  .container{max-width:900px;margin:0 auto}
  .thread{display:flex;flex-direction:column;gap:16px}

  .msg{padding:0;border:0;background:transparent}
  .msg .bubble{display:inline-block;max-width:80%}
  .msg.user .bubble{
    background:var(--user); border:1px solid var(--border);
    border-radius:14px; padding:12px 14px; box-shadow:var(--shadow);
  }
  .msg.advisor .bubble{background:transparent; padding:0; max-width:100%}
  .meta{font-size:12px;margin-bottom:6px;color:#000}

  /* Rich content defaults */
  .bubble h1,.bubble h2,.bubble h3{margin:.6em 0 .4em}
  .bubble p{margin:.6em 0}
  .bubble ul, .bubble ol{margin:.4em 0 .6em 1.4em}
  .bubble blockquote{margin:.6em 0; padding:.4em .8em; border-left:3px solid #ddd; background:#f9f9f9}
  .bubble code{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,"Cascadia Mono","Segoe UI Mono","Roboto Mono","Oxygen Mono","Ubuntu Mono","Courier New",monospace;background:#fff;border:1px solid var(--border);padding:.1em .3em;border-radius:6px;color:#000}
  .bubble pre{background:#fff;color:#000;border:1px solid var(--border);padding:12px;border-radius:12px;overflow:auto}
  .bubble a{color:#065fd4;text-decoration:underline}
  .bubble strong{font-weight:700}
  .bubble em{font-style:italic}
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <div class="inner"><div class="title">Private Trust Fiduciary Advisor</div></div>
  </div>

  <main class="main">
    <div class="container">
      <div id="thread" class="thread"></div>
    </div>
  </main>

  <div class="composer">
    <div class="inner">
      <div class="bar" style="display:flex;align-items:flex-end;gap:8px;background:#fff;border:1px solid var(--ring);border-radius:22px;padding:8px;box-shadow:var(--shadow)">
        <input id="file" type="file" multiple accept=".pdf,.txt,.docx" />
        <div id="input" class="input" role="textbox" aria-multiline="true" contenteditable="true" data-placeholder="Message the Advisor… (Shift+Enter for newline)" style="flex:1;min-height:24px;max-height:160px;overflow:auto;outline:none;padding:8px 10px;font:16px/1.5 var(--font);color:#000"></div>
        <button id="send" class="send" title="Send" style="padding:8px 14px;border-radius:12px;background:#000;color:#fff;border:1px solid #000;cursor:pointer">Send</button>
      </div>
      <div style="margin-top:8px;color:#000;font-size:12px;">
        State your inquiry to receive formal trust, fiduciary, and contractual analysis with strategic guidance.
      </div>
    </div>
  </div>
</div>

<script>
  const elThread = document.getElementById('thread');
  const elInput  = document.getElementById('input');
  const elSend   = document.getElementById('send');
  const elFile   = document.getElementById('file');

  let lastQuestion = "";

  function now(){ return new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}) }

  function addMessage(role, html){
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + (role === 'user' ? 'user' : 'advisor');
    const meta = `<div class="meta">${role==='user'?'You':'Advisor'} · ${now()}</div>`;
    wrap.innerHTML = meta + `<div class="bubble">${html}</div>`;
    elThread.appendChild(wrap);
    elThread.scrollTop = elThread.scrollHeight;
  }

  // Sanitize URLs to avoid javascript: schemes
  function safeUrl(url){
    try{
      const u = new URL(url, window.location.origin);
      const ok = ['http:','https:','mailto:','tel:'].includes(u.protocol);
      return ok ? u.href : '#';
    }catch(_){ return '#'; }
  }

  // Robust MD -> HTML so output never shows raw asterisks and renders bold/italic/links, etc.
  function mdToHtml(md){
    if(!md) return '';
    // If it already looks like HTML, trust it (so <strong>, <em>, <a>, etc. render).
    if (/<\w+[^>]*>/.test(md)) return md;

    // Escape
    let h = md.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

    // Code blocks (triple backticks)
    h = h.replace(/```(\w+)?\n([\s\S]*?)```/g, (_,lang,code)=> {
      const c = (code||'').replace(/</g,'&lt;');
      return `<pre><code>${c}</code></pre>`;
    });

    // Blockquotes
    h = h.replace(/^(>\s?.+)(\n(>\s?.+))*/gm, (m)=> {
      const inner = m.replace(/^>\s?/gm,'');
      return `<blockquote>${inner}</blockquote>`;
    });

    // Ordered lists
    h = h.replace(/^(?:\s*\d+\.\s.+)(?:\n\s*\d+\.\s.+)*/gm, (block)=>{
      const items = block.split(/\n/).map(l=>l.replace(/^\s*\d+\.\s/, '')).map(t=>`<li>${t}</li>`).join('');
      return `<ol>${items}</ol>`;
    });

    // Unordered lists
    h = h.replace(/^(?:\s*[*-]\s.+)(?:\n\s*[*-]\s.+)*/gm, (block)=>{
      const items = block.split(/\n/).map(l=>l.replace(/^\s*[*-]\s/, '')).map(t=>`<li>${t}</li>`).join('');
      return `<ul>${items}</ul>`;
    });

    // Inline code
    h = h.replace(/`([^`\n]+)`/g, '<code>$1</code>');

    // Bold-Italic (***text*** or ___text___)
    h = h.replace(/(\*\*\*|___)(.+?)\1/g, '<strong><em>$2</em></strong>');

    // Bold (**text** or __text__)
    h = h.replace(/(\*\*|__)(.+?)\1/g, '<strong>$2</strong>');

    // Italic (*text* or _text_)
    h = h.replace(/(^|[^\*])\*(?!\s)(.+?)\*(?!\w)/g, '$1<em>$2</em>');
    h = h.replace(/(^|[^_])_(?!\s)(.+?)_(?!\w)/g, '$1<em>$2</em>');

    // Links: [text](url)
    h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_,text,url)=>{
      return `<a href="${safeUrl(url)}" target="_blank" rel="noopener noreferrer">${text}</a>`;
    });

    // Paragraphs (avoid wrapping existing block tags)
    h = h.replace(/(^|\n)(?!<h\d|<ul>|<ol>|<pre>|<blockquote>|<hr>|<p>|<\/)([^\n]+)(?=\n|$)/g, (m, lead, line)=>{
      const trimmed = line.trim();
      if (!trimmed) return '';
      return `${lead}<p>${trimmed}</p>`;
    });

    // Horizontal rule
    h = h.replace(/^\s*---\s*$/gm, '<hr>');

    return h;
  }

  async function callRag(q){
    const url=new URL('/rag', location.origin);
    url.searchParams.set('question', q);
    url.searchParams.set('top_k','12');
    const r = await fetch(url,{method:'GET'});
    if(!r.ok) throw new Error('RAG failed: '+r.status);
    return r.json();
  }

  async function callReview(q, files){
    const fd = new FormData();
    fd.append('question', q);
    for(const f of files) fd.append('files', f);
    const r = await fetch('/review',{method:'POST', body:fd});
    if(!r.ok) throw new Error('Review failed: '+r.status);
    return r.json();
  }

  function readInput(){
    const tmp = elInput.cloneNode(true);
    tmp.querySelectorAll('div').forEach(d=>{
      if (d.innerHTML === "<br>") d.innerHTML = "\\n";
    });
    const text = tmp.innerText.replace(/\\u00A0/g,' ').trim();
    return text;
  }

  async function handleSend(q){
    if(!q) return;
    // USER bubble (light blue)
    addMessage('user', q.replace(/\\n/g,'<br>'));
    lastQuestion = q;

    const work = document.createElement('div');
    work.className = 'msg advisor';
    work.innerHTML = `<div class="meta">Advisor · thinking…</div><div class="bubble"><p>Working…</p></div>`;
    elThread.appendChild(work); elThread.scrollTop = elThread.scrollHeight;

    try{
      const files = Array.from(elFile.files || []);
      const data = files.length ? await callReview(q, files) : await callRag(q);
      let html = (data && data.answer) ? data.answer : '';
      const looksHtml = typeof html==='string' && /<\\w+[^>]*>/.test(html);
      const rendered = looksHtml ? html : mdToHtml(String(html||''));
      work.querySelector('.meta').textContent = 'Advisor · ' + now();
      work.querySelector('.bubble').outerHTML = `<div class="bubble">${rendered}</div>`;
    }catch(e){
      work.querySelector('.meta').textContent = 'Advisor · error';
      work.querySelector('.bubble').innerHTML = '<p style="color:#b91c1c">Error: '+(e && e.message ? e.message : String(e))+'</p>';
    }
  }

  // Send on Enter; Shift+Enter inserts newline
  elInput.addEventListener('keydown', (ev)=>{
    if (ev.key === 'Enter' && !ev.shiftKey){
      ev.preventDefault();
      const q = readInput();
      if (!q) return;
      elInput.innerHTML = '';
      handleSend(q);
    }
  });
  elSend.addEventListener('click', ()=>{
    const q = readInput();
    if (!q) return;
    elInput.innerHTML = '';
    handleSend(q);
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

# ========== /search ==========
@app.get("/search")
def search_endpoint(
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
        titles = _titles_only(uniq)
        rows = []
        for s in uniq:
            meta = s["meta"] or {}
            rows.append({
                "title":   s["title"],
                "level":   s["level"],
                "page":    s["page"],
                "version": s.get("version",""),
                "score":   s["score"],
                "snippet": _extract_snippet(meta) or ""
            })
        return {"question": question, "titles": titles, "matches": rows, "t_ms": int((time.time()-t0)*1000)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ========== /rag ==========
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

# ========== /review ==========
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
                        try: pages.append(p.extract_text() or "")
                        except Exception: pages.append("")
                    texts.append("\n".join(pages))
                except Exception as e:
                    raise HTTPException(status_code=415, detail=f"Failed to parse PDF: {f.filename} ({e})")
            elif name.endswith(".txt"):
                try: texts.append(raw.decode("utf-8", errors="ignore"))
                except Exception: texts.append(raw.decode("latin-1", errors="ignore"))
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
        merged = "\n---\n".join([t for t in texts if t.strip()])
        chunks = [merged[i:i+2000] for i in range(0, len(merged), 2000)][:MAX_SNIPPETS]
        pseudo = [{"title": "Uploaded Document", "level": "L5", "page": "?", "version": "", "score": 1.0, "meta": {}}]
        html = synthesize_html(question or "Please analyze the attached materials.", pseudo, chunks)
        return {"answer": html, "t_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
