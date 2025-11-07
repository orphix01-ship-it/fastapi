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
SYNTH_MODEL       = os.getenv("SYNTH_MODEL", "gpt-4o")
MAX_SNIPPETS      = int(os.getenv("MAX_SNIPPETS", "20"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))
MAX_OUT_TOKENS = int(os.getenv("MAX_OUT_TOKENS", "16384"))  # pick a high, sane default
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
        if parts: s = parts[-1]
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
            seen.add(t); out.append(t)
    return out

# ========== SYNTHESIS ==========
def synthesize_html(question: str, uniq_sources: list[dict], snippets: list[str]) -> str:
    """
    Synthesizes a clean HTML answer using a system message that enforces:
    - HTML-only output (no markdown asterisks)
    - Proper tags for bold/italic/headings/lists/links
    """
    if not snippets and not uniq_sources:
        return "<p>No relevant material found in the Trust-Law knowledge base.</p>"

    # Build large context block (snippets joined by ---)
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

    # titles only (deduped, precedence applied)
    titles = _titles_only(uniq_sources)
    titles_html = "<ul>" + "".join(f"<li>{t}</li>" for t in titles) + "</ul>" if titles else "<p></p>"

    # Single user message: question + raw context + citation titles
    user_msg = (
        f"<h2>Question</h2>\n<p>{question}</p>\n"
        f"<h3>Context</h3>\n<pre>{context}</pre>\n"
        f"<h3>Citations</h3>\n{titles_html}"
    )

    # 🔹 NEW: System message with your additional instructions
    system_msg = (
    )

    try:
        res = client.chat.completions.create(
            model=SYNTH_MODEL,
            temperature=0.15,
            max_tokens=MAX_OUT_TOKENS,
            messages=[
                {"role": "system", "content": system_msg},   # ← additional instructions live here
                {"role": "user", "content": user_msg},
            ],
        )
        html = (getattr(res, "choices", None) or getattr(res, "data"))[0].message.content.strip()
        if not html:
            return "<p>No relevant material found in the Trust-Law knowledge base.</p>"
        # Guard: if model returns plain text, minimally wrap to HTML
        if "<" not in html:
            html = "<div><p>" + html.replace("\n", "<br>") + "</p></div>"
        return html
    except Exception as e:
        return f"<p><em>(Synthesis unavailable: {e})</em></p>"

# ========== WIDGET ==========
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
  .header{background:#fff}
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

  .bubble h1,.bubble h2,.bubble h3{margin:.6em 0 .4em}
  .bubble p{margin:.6em 0}
  .bubble ul, .bubble ol{margin:.4em 0 .6em 1.4em}
  .bubble a{color:#000;text-decoration:underline}
  .bubble strong{font-weight:700}
  .bubble em{font-style:italic}
  .bubble code{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,"Cascadia Mono","Segoe UI Mono","Roboto Mono","Oxygen Mono","Ubuntu Mono","Courier New",monospace;background:#fff;border:1px solid var(--border);padding:.1em .3em;border-radius:6px;color:#000}
  .bubble pre{background:#fff;color:#000;border:1px solid var(--border);padding:12px;border-radius:12px;overflow:auto}
  .bubble blockquote{border-left:3px solid #000;padding:6px 12px;margin:8px 0;background:#fafafa}

  /* Composer with NO top divider line */
  .composer{position:fixed;bottom:0;left:0;right:0;background:#fff;padding:18px 12px;border-top:none}
  .composer .inner{max-width:900px;margin:0 auto}
  .bar{display:flex;align-items:flex-end;gap:8px;background:#fff;border:1px solid var(--ring);border-radius:22px;padding:8px;box-shadow:var(--shadow)}
  .input{flex:1;min-height:24px;max-height:160px;overflow:auto;outline:none;padding:8px 10px;font:16px/1.5 var(--font);color:#000}
  .input:empty:before{content:attr(data-placeholder);color:#000}
  .btn{cursor:pointer}
  .send{padding:8px 12px;border-radius:12px;background:#000;color:#fff;border:1px solid #000}
  .attach{padding:8px 10px;border-radius:12px;background:#fff;color:#000;border:1px solid var(--border)}

  a{color:#000;text-decoration:underline}
  a:hover{text-decoration:none}

  #file{display:none}
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
      <div class="bar">
        <input id="file" type="file" multiple accept=".pdf,.txt,.docx" />
        <div id="input" class="input" role="textbox" aria-multiline="true" contenteditable="true" data-placeholder="Message the Advisor… (Shift+Enter for newline)"></div>
        <button id="attach" class="attach btn" type="button" title="Add files">+</button>
        <button id="send" class="send btn" type="button" title="Send" aria-label="Send">
          <!-- paper-plane icon -->
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="m5 12 14-7-4 14-3-5-7-2z" stroke="#fff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>
      <div id="filehint" style="margin-top:8px;color:#000;font-size:12px;">
        State your inquiry to receive formal trust, fiduciary, and contractual analysis with strategic guidance.
      </div>
    </div>
  </div>
</div>

<script>
  const elThread = document.getElementById('thread');
  const elInput  = document.getElementById('input');
  const elSend   = document.getElementById('send');
  const elAttach = document.getElementById('attach');
  const elFile   = document.getElementById('file');
  const elHint   = document.getElementById('filehint');

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

  // Normalize common trust/contract bold templates into colon-style lines (no **asterisks** visuals)
  function normalizeTrustDoc(html){
    let out = html;

    // Convert all-caps bold title line to heading + blank line
    out = out.replace(/<p>\s*<strong>\s*([A-Z0-9][A-Z0-9\s\-&,.'()]+?)\s*<\/strong>\s*<\/p>/g,
                      '<h2>$1</h2><p></p>');

    // Map bold labels to "Label: " (no bold)
    const labelMap = [
      {re:/<strong>\s*TRUST\s*NAME\s*:\s*<\/strong>/gi, rep:'Trust: '},
      {re:/<strong>\s*DATE\s*:\s*<\/strong>/gi, rep:'Date: '},
      {re:/<strong>\s*TAX\s*YEAR\s*:\s*<\/strong>/gi, rep:'Tax Year: '},
      {re:/<strong>\s*TRUSTEE\(S\)\s*:\s*<\/strong>/gi, rep:'Trustee(s): '},
      {re:/<strong>\s*LOCATION\s*:\s*<\/strong>/gi, rep:'Location: '},
    ];
    labelMap.forEach(({re,rep})=>{ out = out.replace(re, rep); });

    // Remove any remaining bold-wrapped labels like <strong>Label:</strong>
    out = out.replace(/<strong>\s*([A-Za-z][A-Za-z()\s]+:)\s*<\/strong>\s*/g, '$1 ');

    // Also convert markdown-style **LABEL:** if it slipped through
    out = out.replace(/\*\*\s*TRUST\s*NAME\s*:\s*\*\*/gi, 'Trust: ')
             .replace(/\*\*\s*DATE\s*:\s*\*\*/gi, 'Date: ')
             .replace(/\*\*\s*TAX\s*YEAR\s*:\s*\*\*/gi, 'Tax Year: ')
             .replace(/\*\*\s*TRUSTEE\(S\)\s*:\s*\*\*/gi, 'Trustee(s): ')
             .replace(/\*\*\s*LOCATION\s*:\s*\*\*/gi, 'Location: ');

    return out;
  }

  // Markdown -> HTML with full formatting (bold, italic, links, blockquotes, code, lists).
  function mdToHtml(md){
    if(!md) return '';
    // If already HTML-like, trust it so <strong>/<em>/<a> are preserved
    if (/<\\w+[^>]*>/.test(md)) return md;

    let h = md;

    // Escape first
    h = h.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

    // Code blocks
    h = h.replace(/```([\\s\\S]*?)```/g, (_,c)=>`<pre><code>${c.replace(/</g,'&lt;')}</code></pre>`);
    // Inline code
    h = h.replace(/`([^`]+?)`/g, '<code>$1</code>');

    // Headings
    h = h.replace(/^######\\s+(.*)$/gm,'<h6>$1</h6>').replace(/^#####\\s+(.*)$/gm,'<h5>$1</h5>')
         .replace(/^####\\s+(.*)$/gm,'<h4>$1</h4>').replace(/^###\\s+(.*)$/gm,'<h3>$1</h3>')
         .replace(/^##\\s+(.*)$/gm,'<h2>$1</h2>').replace(/^#\\s+(.*)$/gm,'<h1>$1</h1>');

    // Blockquotes
    h = h.replace(/^>\\s?(.*)$/gm, '<blockquote>$1</blockquote>');

    // Bold / italic (turn **...** and *...* into <strong>/<em>)
    h = h.replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>')
         .replace(/__(.+?)__/g,'<strong>$1</strong>')
         .replace(/\\*(?!\\s)(.+?)\\*/g,'<em>$1</em>')
         .replace(/_(?!\\s)(.+?)_/g,'<em>$1</em>');

    // Links
    h = h.replace(/\\[([^\\]]+)\\]\\((https?:\\/\\/[^\\s)]+)\\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    // Autolink
    h = h.replace(/(^|\\s)(https?:\\/\\/[^\\s<]+)(?=\\s|$)/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');

    // Ordered lists
    h = h.replace(/(?:^|\\n)(\\d+)\\.\\s+(.+)(?:(?=\\n\\d+\\.\\s)|$)/gms, (m)=>{
      const items = m.trim().split(/\\n(?=\\d+\\.\\s)/).map(it=>it.replace(/^\\d+\\.\\s+/, '')).map(t=>`<li>${t}</li>`).join('');
      return `<ol>${items}</ol>`;
    });
    // Unordered lists
    h = h.replace(/(?:^|\\n)[*-]\\s+(.+)(?:(?=\\n[*-]\\s)|$)/gms, (m)=>{
      const items = m.trim().split(/\\n(?=[*-]\\s)/).map(it=>it.replace(/^[*-]\\s+/, '')).map(t=>`<li>${t}</li>`).join('');
      return `<ul>${items}</ul>`;
    });

    // Paragraphs
    h = h.replace(/\\n{2,}/g,'</p><p>').replace(/^(?!<h\\d|<ul|<ol|<pre|<hr|<p|<blockquote|<table)(.+)$/gm,'<p>$1</p>');

    return h;
  }

  // Convert leftover **bold**/*italic* that still appear inside HTML (runs after md/html detection)
  function applyInlineFormatting(html) {
    if (!html) return '';
    let out = String(html);

    // Protect <code>, <pre>, <a> blocks
    const slots = [];
    function protect(tag) {
      const re = new RegExp(`<${tag}\\b[^>]*>[\\s\\S]*?<\\/${tag}>`, 'gi');
      out = out.replace(re, m => {
        const key = `__SLOT_${tag.toUpperCase()}_${slots.length}__`;
        slots.push({ key, val: m });
        return key;
      });
    }
    protect('code'); protect('pre'); protect('a');

    // Replace markdown bold/italic in remaining text
    out = out
      .replace(/\\*\\*([^*]+?)\\*\\*/g, '<strong>$1</strong>')
      .replace(/__([^_]+?)__/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\\*([^*\\n]+?)\\*/g, '$1<em>$2</em>')
      .replace(/(^|[^_])_([^_\\n]+?)_/g, '$1<em>$2</em>');

    // Restore protected blocks
    for (const { key, val } of slots) out = out.replace(key, val);
    return out;
  }

  // === API helpers ===
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

      // If model already returned HTML, use it; otherwise convert markdown to HTML
      const looksHtml = typeof html==='string' && /<\\w+[^>]*>/.test(html);
      let rendered = looksHtml ? html : mdToHtml(String(html||''));

      // Convert any leftover **/*** to <strong>/<em>, then normalize trust doc labels
      rendered = applyInlineFormatting(rendered);
      rendered = normalizeTrustDoc(rendered);

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

  // Click handlers
  elSend.addEventListener('click', ()=>{
    const q = readInput();
    if (!q) return;
    elInput.innerHTML = '';
    handleSend(q);
  });

  // "+" attach button opens file dialog; show selected file count
  elAttach.addEventListener('click', ()=>{ elFile.click(); });
  elFile.addEventListener('change', ()=>{
    if (elFile.files && elFile.files.length){
      elHint.textContent = `${elFile.files.length} file${elFile.files.length>1?'s':''} selected.`;
    }else{
      elHint.textContent = 'State your inquiry to receive formal trust, fiduciary, and contractual analysis with strategic guidance.';
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
