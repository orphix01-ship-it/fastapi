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
        buf.append(s); used += len(s); kept += 1
        if kept >= MAX_SNIPPETS:
            break
    context = "\n---\n".join(buf)

    # titles only (deduped, precedence applied)
    titles = _titles_only(uniq_sources)
    titles_html = "<ul>" + "".join(f"<li>{t}</li>" for t in titles) + "</ul>" if titles else "<p></p>"

    # Only a user message (no system); no style instructions.
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
        # openai>=1.40: choices[0] path preserved for back-compat
        html = (getattr(res, "choices", None) or getattr(res, "data"))[0].message.content.strip()
        if not html:
            return "<p>No relevant material found in the Trust-Law knowledge base.</p>"
        if "<" not in html:
            html = "<div><p>" + html.replace("\n", "<br>") + "</p></div>"
        return html
    except Exception as e:
        return f"<p><em>(Synthesis unavailable: {e})</em></p>"

WIDGET_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Private Trust Fiduciary Advisor</title>
<style>
  /* ====== ChatGPT-like font stack ======
     If you have a Söhne webfont license, uncomment and point src: to your files.
  @font-face{
    font-family: "Söhne";
    src: url("/fonts/soehne-var.woff2") format("woff2");
    font-weight: 100 900; font-style: normal; font-display: swap;
  }
  @font-face{
    font-family: "Söhne";
    src: url("/fonts/soehne-italic-var.woff2") format("woff2");
    font-weight: 100 900; font-style: italic; font-display: swap;
  }
  */
  :root{
    --bg:#f7f7f8;
    --panel:#ffffff;
    --text:#0c0c0d;
    --muted:#6b7280;
    --border:#e5e7eb;
    --ring:#d1d5db;
    --user:#e5f0ff;
    --assistant:#ffffff;
    --radius:16px;
    --radius-sm:12px;
    --shadow: 0 1px 2px rgba(0,0,0,.04), 0 8px 24px rgba(0,0,0,.06);
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", "Oxygen Mono", "Ubuntu Mono", "Courier New", monospace;
    --font: "Söhne", ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{
    margin:0; background:var(--bg); color:var(--text);
    font: 16px/1.6 var(--font);
  }

  .app{
    display:flex; flex-direction:column; height:100vh; width:100%;
  }
  .header{
    position:sticky; top:0; backdrop-filter:saturate(180%) blur(8px);
    background:rgba(247,247,248,.7); border-bottom:1px solid var(--border);
    z-index:10;
  }
  .header .inner{
    max-width: 900px; margin:0 auto; padding:12px 16px;
    display:flex; align-items:center; gap:8px;
  }
  .logo{width:28px;height:28px;border-radius:8px;background:#111;}
  .title{font-weight:600}

  .main{
    flex:1; overflow:auto; padding: 24px 12px 140px; /* space for composer */
  }
  .container{max-width:900px;margin:0 auto}

  /* Conversation */
  .thread{display:flex; flex-direction:column; gap:16px}
  .msg{
    display:grid; grid-template-columns: 40px 1fr; gap:12px;
    padding:16px; border:1px solid var(--border); border-radius: var(--radius-sm);
    background:var(--assistant); box-shadow: var(--shadow);
  }
  .msg.user{ background:var(--user) }
  .avatar{
    width:40px; height:40px; border-radius:8px; background:#111; color:#fff;
    display:flex; align-items:center; justify-content:center; font-weight:700;
  }
  .avatar.user{ background:#1d4ed8 }
  .content{min-width:0}
  .meta{
    display:flex; align-items:center; gap:8px; font-size:12px; color:var(--muted); margin-bottom:8px;
  }
  .actions{ display:flex; gap:8px; margin-top:10px; }
  .btn{
    border:1px solid var(--border); background:#fff; color:#111;
    padding:6px 10px; border-radius:10px; font-size:12px; cursor:pointer;
  }
  .btn:active{ transform: translateY(1px) }

  /* Markdown-ish rendering */
  .content h1,.content h2,.content h3{margin:.6em 0 .4em}
  .content p{margin:.6em 0}
  .content ul{margin:.4em 0 .6em 1.2em}
  .content code{font-family:var(--mono); background:#f3f4f6; padding:.1em .3em; border-radius:6px}
  .content pre{background:#0b1020; color:#e6edf3; padding:12px; border-radius:12px; overflow:auto}
  .codebar{display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:6px; font-size:12px; color:#c7cbd1}
  .copy{border:1px solid #2b3245; background:#111827; color:#e6edf3; padding:4px 8px; border-radius:8px; cursor:pointer}

  /* Sticky composer (ChatGPT-like) */
  .composer{
    position:fixed; bottom:0; left:0; right:0; background:linear-gradient(to top, rgba(247,247,248,1), rgba(247,247,248,.8) 60%, rgba(247,247,248,0));
    padding:18px 12px; border-top:1px solid var(--border);
  }
  .composer .inner{max-width:900px; margin:0 auto}
  .bar{
    display:flex; align-items:flex-end; gap:8px; background:var(--panel);
    border:1px solid var(--ring); border-radius: 22px; padding:8px; box-shadow: var(--shadow);
  }
  .input{
    flex:1; min-height:24px; max-height:160px; overflow:auto; outline:none;
    padding:8px 10px; border-radius:16px; font: 16px/1.5 var(--font);
  }
  .input:empty:before{content:attr(data-placeholder); color:#9ca3af}
  .iconbtn{
    width:36px; height:36px; border-radius:12px; border:1px solid var(--border);
    background:#fff; display:flex; align-items:center; justify-content:center; cursor:pointer;
  }
  .send{
    padding:8px 14px; border-radius:12px; background:#111; color:#fff; border:1px solid #111; cursor:pointer;
  }

  /* Subtle link style (like ChatGPT) */
  a{color:#065fd4; text-decoration:none}
  a:hover{text-decoration:underline}

  /* Hidden native file input */
  #file{position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(1px,1px,1px,1px)}
  .filelabel{cursor:pointer}

  /* Utility */
  .hidden{display:none}
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <div class="inner">
      <div class="logo" aria-hidden="true"></div>
      <div class="title">Private Trust Fiduciary Advisor</div>
    </div>
  </div>

  <main class="main">
    <div class="container">
      <div id="thread" class="thread">
        <div class="msg assistant">
          <div class="avatar">A</div>
          <div class="content">
            <div class="meta">Assistant · Ready</div>
            <p>Ask anything about trusts. Attach a PDF/DOCX/TXT if you want me to review it.</p>
          </div>
        </div>
      </div>
    </div>
  </main>

  <!-- Composer -->
  <div class="composer">
    <div class="inner">
      <div class="bar">
        <label class="iconbtn filelabel" title="Attach file">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M8 12v5a4 4 0 1 0 8 0V7a3 3 0 0 0-6 0v8a2 2 0 1 0 4 0V9" stroke="#111" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          <input id="file" type="file" multiple accept=".pdf,.txt,.docx"/>
        </label>
        <div id="input" class="input" role="textbox" aria-multiline="true" contenteditable="true" data-placeholder="Message ChatGPT… (Shift+Enter for newline)"></div>
        <button id="send" class="send" title="Send">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="m5 12 14-7-4 14-3-5-7-2z" stroke="#fff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>
      <div style="margin-top:8px; color:#6b7280; font-size:12px;">We don’t store files. Generation may use your private vector index only.</div>
    </div>
  </div>
</div>

<script>
  const elThread = document.getElementById('thread');
  const elInput  = document.getElementById('input');
  const elSend   = document.getElementById('send');
  const elFile   = document.getElementById('file');

  let lastQuestion = "";

  function now() {
    return new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  }

  function addMessage(role, html, rawMd=null) {
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + (role === 'user' ? 'user' : 'assistant');
    wrap.innerHTML = `
      <div class="avatar ${role==='user'?'user':''}">${role==='user'?'U':'A'}</div>
      <div class="content">
        <div class="meta">${role==='user'?'You':'Assistant'} · ${now()}</div>
        <div class="body">${html}</div>
        ${role==='assistant' ? `
          <div class="actions">
            <button class="btn" data-action="copy-answer">Copy</button>
            <button class="btn" data-action="regenerate">Regenerate</button>
            ${rawMd ? `<button class="btn" data-action="show-sources">Citations</button>` : ``}
          </div>
        ` : ``}
      </div>`;
    elThread.appendChild(wrap);
    elThread.scrollTop = elThread.scrollHeight;

    // Wire up action buttons
    wrap.querySelectorAll('.btn').forEach(btn=>{
      btn.addEventListener('click', ()=>{
        const act = btn.dataset.action;
        if (act === 'copy-answer') {
          const text = wrap.querySelector('.body').innerText;
          navigator.clipboard.writeText(text);
          btn.textContent = 'Copied';
          setTimeout(()=>btn.textContent='Copy', 1200);
        } else if (act === 'regenerate') {
          if (lastQuestion) send(lastQuestion, true);
        } else if (act === 'show-sources') {
          // Toggle citations list if present
          const c = wrap.querySelector('.citations');
          if (c) c.classList.toggle('hidden');
        }
      });
    });

    // Enhance code blocks with copy bars
    wrap.querySelectorAll('pre').forEach(pre=>{
      if (pre.dataset.wired) return;
      pre.dataset.wired = "1";
      const bar = document.createElement('div');
      bar.className = 'codebar';
      bar.innerHTML = `<span>Code</span><button class="copy">Copy code</button>`;
      pre.parentNode.insertBefore(bar, pre);
      bar.querySelector('.copy').addEventListener('click', ()=>{
        const text = pre.innerText;
        navigator.clipboard.writeText(text);
        bar.querySelector('.copy').textContent = 'Copied';
        setTimeout(()=>bar.querySelector('.copy').textContent='Copy code', 1200);
      });
    });
  }

  // Lightweight MD→HTML (keeps your passthrough HTML)
  function mdToHtml(md){
    if(!md) return '';
    // If it already looks like HTML, just trust it.
    if (/<\\w+[^>]*>/.test(md)) return md;
    let h = md.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    h = h.replace(/```([\\s\\S]*?)```/g,(_,c)=>`<pre><code>${c.replace(/</g,'&lt;')}</code></pre>`);
    h = h.replace(/^######\\s+(.*)$/gm,'<h6>$1</h6>').replace(/^#####\\s+(.*)$/gm,'<h5>$1</h5>')
         .replace(/^####\\s+(.*)$/gm,'<h4>$1</h4>').replace(/^###\\s+(.*)$/gm,'<h3>$1</h3>')
         .replace(/^##\\s+(.*)$/gm,'<h2>$1</h2>').replace(/^#\\s+(.*)$/gm,'<h1>$1</h1>');
    h = h.replace(/^---$/gm,'<hr>');
    h = h.replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>').replace(/\\*(.+?)\\*/g,'<em>$1</em>');
    h = h.replace(/(?:^|\\n)[*-]\\s+(.*)/g,(m,i)=>`<li>${i}</li>`).replace(/(<li>.*<\\/li>)(\\n?)+/gs,m=>`<ul>${m}</ul>`);
    h = h.replace(/\\n{2,}/g,'</p><p>').replace(/^(?!<h\\d|<ul|<pre|<hr)(.+)$/gm,'<p>$1</p>');
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
    // Normalize <div><br></div> to newlines
    const tmp = elInput.cloneNode(true);
    tmp.querySelectorAll('div').forEach(d=>{
      if (d.innerHTML === "<br>") d.innerHTML = "\\n";
    });
    const text = tmp.innerText.replace(/\\u00A0/g,' ').trim();
    return text;
  }

  async function send(q, isRegen=false){
    if(!q) return;
    if(!isRegen){
      lastQuestion = q;
      // Show user bubble
      addMessage('user', q.replace(/\\n/g,'<br>'));
    }
    // Show working assistant bubble
    const work = document.createElement('div');
    work.className = 'msg assistant';
    work.innerHTML = '<div class="avatar">A</div><div class="content"><div class="meta">Assistant · thinking…</div><div class="body"><p>Working…</p></div></div>';
    elThread.appendChild(work); elThread.scrollTop = elThread.scrollHeight;

    try{
      const files = Array.from(elFile.files || []);
      const data = files.length ? await callReview(q, files) : await callRag(q);
      let html = (data && data.answer) ? data.answer : '';
      const looksHtml = typeof html==='string' && /<\\w+[^>]*>/.test(html);
      const rendered = looksHtml ? html : mdToHtml(String(html||''));

      // Replace the working bubble with final answer (plus hidden citations list if present in html)
      work.querySelector('.meta').textContent = 'Assistant · ' + now();
      work.querySelector('.body').innerHTML = rendered;

      // If your synthesis includes a <ul> of citations, optionally mark it for toggle
      const cites = work.querySelector('h3, h4, h5, h6');
      if (cites && /citation/i.test(cites.textContent)) {
        const list = cites.nextElementSibling;
        if (list && (list.tagName === 'UL' || list.tagName === 'OL')) {
          list.classList.add('citations','hidden');
        }
        // Add action row if missing
        let actions = work.querySelector('.actions');
        if (!actions) {
          actions = document.createElement('div');
          actions.className = 'actions';
          work.querySelector('.content').appendChild(actions);
        }
        const btn = document.createElement('button');
        btn.className = 'btn'; btn.dataset.action='show-sources'; btn.textContent='Citations';
        actions.appendChild(btn);
        btn.addEventListener('click', ()=>{
          const c = work.querySelector('.citations');
          if (c) c.classList.toggle('hidden');
        });
      }

      // Wire copy/regenerate for this bubble
      let actions = work.querySelector('.actions');
      if (!actions) {
        actions = document.createElement('div');
        actions.className = 'actions';
        work.querySelector('.content').appendChild(actions);
      }
      const copyBtn = document.createElement('button'); copyBtn.className='btn'; copyBtn.dataset.action='copy-answer'; copyBtn.textContent='Copy';
      const regenBtn = document.createElement('button'); regenBtn.className='btn'; regenBtn.dataset.action='regenerate'; regenBtn.textContent='Regenerate';
      actions.appendChild(copyBtn); actions.appendChild(regenBtn);
      [copyBtn, regenBtn].forEach(b=>b.addEventListener('click', ()=>{
        const act = b.dataset.action;
        if (act==='copy-answer'){
          const txt = work.querySelector('.body').innerText;
          navigator.clipboard.writeText(txt); b.textContent='Copied'; setTimeout(()=>b.textContent='Copy',1200);
        } else if (act==='regenerate'){
          if (lastQuestion) send(lastQuestion, true);
        }
      }));

      // Add code copy bars
      work.querySelectorAll('pre').forEach(pre=>{
        if (pre.dataset.wired) return;
        pre.dataset.wired = "1";
        const bar = document.createElement('div');
        bar.className = 'codebar';
        bar.innerHTML = `<span>Code</span><button class="copy">Copy code</button>`;
        pre.parentNode.insertBefore(bar, pre);
        bar.querySelector('.copy').addEventListener('click', ()=>{
          navigator.clipboard.writeText(pre.innerText);
          bar.querySelector('.copy').textContent='Copied';
          setTimeout(()=>bar.querySelector('.copy').textContent='Copy code',1200);
        });
      });

    }catch(e){
      work.querySelector('.meta').textContent = 'Assistant · error';
      work.querySelector('.body').innerHTML = '<p style="color:#b91c1c">Error: '+(e && e.message ? e.message : String(e))+'</p>';
    }
  }

  // Send on Enter, newline on Shift+Enter — like ChatGPT
  elInput.addEventListener('keydown', (ev)=>{
    if (ev.key === 'Enter' && !ev.shiftKey){
      ev.preventDefault();
      const q = readInput();
      if (!q) return;
      elInput.innerHTML = '';
      send(q);
    }
  });
  elSend.addEventListener('click', ()=>{
    const q = readInput();
    if (!q) return;
    elInput.innerHTML = '';
    send(q);
  });
</script>
</body>
</html>
"""
# ========== /search (RAW CONTEXT MODE) ==========
@app.get("/search")
def search_endpoint(
    question: str = Query(..., min_length=3),
    top_k: int = Query(12, ge=1, le=30),
    level: str | None = Query(None),
    authorization: str | None = Header(default=None),
):
    """
    Raw/context mode:
    - Returns only retrieved context for your GPT to synthesize.
    - Response: { question, titles[], matches[], t_ms }
      matches[] items: { title, level, page, version, score, snippet }
    """
    require_auth(authorization)
    check_rate_limit()
    t0 = time.time()
    try:
        emb = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
        flt = {"doc_level": {"$eq": level}} if level else None
        res = idx.query(vector=emb, top_k=max(top_k, 12), include_metadata=True, filter=flt)
        matches = res["matches"] if isinstance(res, dict) else getattr(res, "matches", [])
        uniq = _dedup_and_rank_sources(matches, top_k=top_k)

        # build raw payload
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

# ========== /rag (SYNTHESIZED HTML ANSWER) ==========
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

# ========== /review (PDF/TXT/DOCX) ==========
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
