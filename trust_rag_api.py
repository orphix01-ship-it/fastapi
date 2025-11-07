# trust_rag_api.py
from fastapi import FastAPI, Query, Header, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pinecone import Pinecone
from openai import OpenAI
import httpx, zipfile, io, re, os, time, traceback, sqlite3, json, uuid
from datetime import datetime
from collections import deque
from typing import Optional

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
MAX_OUT_TOKENS    = int(os.getenv("MAX_OUT_TOKENS", "16384"))  # high, sane default
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

# Simple user extractor: prefer X-User-Id (from your app auth), else "demo"
def get_current_user(authorization: Optional[str] = Header(None),
                     x_user_id: Optional[str] = Header(None)) -> str:
    # Wire your real auth provider here if needed
    if x_user_id and x_user_id.strip():
        return x_user_id.strip()
    return "demo"  # safe fallback for development

# ========== DB (SQLite) ==========
DB_PATH = os.getenv("TRUST_RAG_DB", "trust_rag.db")

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      email TEXT,
      name TEXT,
      role TEXT,
      settings_json TEXT,
      created_at TEXT,
      last_login_at TEXT
    );

    CREATE TABLE IF NOT EXISTS chats (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      title TEXT,
      archived INTEGER DEFAULT 0,
      created_at TEXT,
      updated_at TEXT
    );

    CREATE TABLE IF NOT EXISTS messages (
      id TEXT PRIMARY KEY,
      chat_id TEXT NOT NULL,
      user_id TEXT,
      role TEXT NOT NULL,               -- 'user' | 'advisor' | 'system'
      content_html TEXT NOT NULL,
      content_raw  TEXT,
      meta_json    TEXT,
      created_at   TEXT
    );

    CREATE TABLE IF NOT EXISTS files (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      chat_id TEXT,
      name TEXT,
      mime TEXT,
      size INTEGER,
      storage_url TEXT,
      created_at TEXT
    );
    """)
    conn.commit()
    conn.close()
init_db()

def iso_now():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

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
    - Professional legal formatting
    """
    if not snippets and not uniq_sources:
        return "<p>No relevant material found in the Trust-Law knowledge base.</p>"

    buf, used, kept = [], 0, 0
    for s in snippets:
        s = s.strip()
        if not s: continue
        if used + len(s) > MAX_CONTEXT_CHARS: break
        buf.append(s); used += len(s); kept += 1
        if kept >= MAX_SNIPPETS: break
    context = "\n---\n".join(buf)

    titles = _titles_only(uniq_sources)
    titles_html = "<ul>" + "".join(f"<li>{t}</li>" for t in titles) + "</ul>" if titles else "<p></p>"

    user_msg = (
        f"<h2>Question</h2>\n<p>{question}</p>\n"
        f"<h3>Context</h3>\n<pre>{context}</pre>\n"
        f"<h3>Citations</h3>\n{titles_html}"
    )

    system_msg = (
        "You are the Private Trust Fiduciary Advisor. "
        "Always respond using clean, valid HTML (no markdown asterisks). "
        "Use <strong> for bold, <em> for italics, <h1>-<h6> for headings, "
        "<ul>/<ol> for lists, <pre><code> for code, and <a> for links. "
        "Prefer professional, legal-style formatting suitable for trust "
        "and fiduciary documents. If the content resembles a formal instrument "
        "(resolutions, certificates), format labels as plain lines like "
        "“Date: …”, “Trust: …”, “Tax Year: …”, “Location: …”."
    )

    try:
        res = client.chat.completions.create(
            model=SYNTH_MODEL,
            temperature=0.15,
            max_tokens=MAX_OUT_TOKENS,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        html = (getattr(res, "choices", None) or getattr(res, "data"))[0].message.content.strip()
        if not html:
            return "<p>No relevant material found in the Trust-Law knowledge base.</p>"
        if "<" not in html:
            html = "<div><p>" + html.replace("\n", "<br>") + "</p></div>"
        return html
    except Exception as e:
        return f"<p><em>(Synthesis unavailable: {e})</em></p>"

# ========== CHAT & HISTORY (API) ==========
def ensure_chat(conn, user_id: str, chat_id: Optional[str]) -> str:
    cur = conn.cursor()
    if chat_id:
        row = cur.execute("SELECT id FROM chats WHERE id=? AND user_id=?", (chat_id, user_id)).fetchone()
        if row:
            return chat_id
    new_id = str(uuid.uuid4())
    now = iso_now()
    cur.execute("INSERT INTO chats (id, user_id, title, archived, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                (new_id, user_id, "New chat", 0, now, now))
    conn.commit()
    return new_id

def insert_message(conn, chat_id: str, user_id: Optional[str], role: str,
                   content_html: str, content_raw: Optional[str], meta: dict):
    cur = conn.cursor()
    mid = str(uuid.uuid4())
    now = iso_now()
    cur.execute("""INSERT INTO messages (id, chat_id, user_id, role, content_html, content_raw, meta_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (mid, chat_id, user_id, role, content_html, content_raw, json.dumps(meta or {}), now))
    cur.execute("UPDATE chats SET updated_at=? WHERE id=?", (now, chat_id))
    conn.commit()
    return mid

@app.post("/chats")
def create_chat(user_id: str = Depends(get_current_user)):
    conn = db()
    cid = ensure_chat(conn, user_id, None)
    conn.close()
    return {"chat_id": cid, "title": "New chat"}

@app.get("/chats")
def list_chats(page: int = 1, size: int = 30, user_id: str = Depends(get_current_user)):
    off = max((page-1),0)*size
    conn = db()
    cur = conn.cursor()
    rows = cur.execute("""SELECT id, title, archived, created_at, updated_at
                          FROM chats WHERE user_id=? ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
                       (user_id, size, off)).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows], "page": page, "size": size}

@app.get("/chats/{chat_id}")
def get_chat(chat_id: str, user_id: str = Depends(get_current_user)):
    conn = db()
    cur = conn.cursor()
    chat = cur.execute("SELECT id, title, archived, created_at, updated_at FROM chats WHERE id=? AND user_id=?",
                       (chat_id, user_id)).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(404, "Chat not found")
    msgs = cur.execute("""SELECT id, role, content_html, created_at
                          FROM messages WHERE chat_id=? ORDER BY created_at ASC LIMIT 500""", (chat_id,)).fetchall()
    conn.close()
    return {"chat": dict(chat), "messages": [dict(m) for m in msgs]}

@app.get("/chats/{chat_id}/messages")
def list_messages(chat_id: str, page: int = 1, size: int = 100, user_id: str = Depends(get_current_user)):
    off = max((page-1),0)*size
    conn = db()
    cur = conn.cursor()
    ok = cur.execute("SELECT 1 FROM chats WHERE id=? AND user_id=?", (chat_id, user_id)).fetchone()
    if not ok:
        conn.close()
        raise HTTPException(404, "Chat not found")
    msgs = cur.execute("""SELECT id, role, content_html, created_at
                          FROM messages WHERE chat_id=? ORDER BY created_at ASC LIMIT ? OFFSET ?""",
                       (chat_id, size, off)).fetchall()
    conn.close()
    return {"messages": [dict(m) for m in msgs], "page": page, "size": size}

@app.post("/chats/{chat_id}/title")
def rename_chat(chat_id: str, title: str = Form(...), user_id: str = Depends(get_current_user)):
    conn = db()
    cur = conn.cursor()
    cur.execute("UPDATE chats SET title=?, updated_at=? WHERE id=? AND user_id=?", (title[:200], iso_now(), chat_id, user_id))
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(404, "Chat not found")
    conn.commit()
    conn.close()
    return {"ok": True}

@app.post("/chats/{chat_id}/archive")
def archive_chat(chat_id: str, archived: int = Form(1), user_id: str = Depends(get_current_user)):
    conn = db()
    cur = conn.cursor()
    cur.execute("UPDATE chats SET archived=?, updated_at=? WHERE id=? AND user_id=?", (1 if archived else 0, iso_now(), chat_id, user_id))
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(404, "Chat not found")
    conn.commit()
    conn.close()
    return {"ok": True}

@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str, user_id: str = Depends(get_current_user)):
    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    cur.execute("DELETE FROM chats WHERE id=? AND user_id=?", (chat_id, user_id))
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(404, "Chat not found or not yours")
    conn.commit()
    conn.close()
    return {"ok": True}

# ========== WIDGET (Sidebar with Profile + Chat Tabs) ==========
WIDGET_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Private Trust Fiduciary Advisor</title>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#fff; --text:#000; --border:#e5e5e5; --ring:#d9d9d9; --user:#e8f1ff;
    --shadow:0 1px 2px rgba(0,0,0,.03), 0 8px 24px rgba(0,0,0,.04);
    --font: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial;
    --title:"Cinzel",serif;
    --rail: #fafafa;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 var(--font)}

  .app{display:grid; grid-template-columns: 280px 1fr; height:100vh; width:100%}
  .rail{
    border-right:1px solid var(--border); background:var(--rail); display:flex; flex-direction:column; min-width:0;
  }
  .brand{padding:14px 16px; text-align:center; border-bottom:1px solid var(--border)}
  .brand .title{font-family:var(--title); font-weight:300; font-size:18px}
  .section{padding:12px 12px; border-bottom:1px solid var(--border)}
  .label{font-size:12px; color:#6b7280; margin-bottom:6px}
  .row{display:flex; gap:6px}
  .input{flex:1; border:1px solid var(--border); border-radius:10px; padding:6px 8px; background:#fff; color:#000}
  .btn{cursor:pointer; border:1px solid var(--border); background:#fff; color:#000; padding:6px 10px; border-radius:10px}
  .btn.primary{background:#000; color:#fff; border-color:#000}
  .chats{flex:1; overflow:auto; padding:8px}
  .chatitem{border:1px solid var(--border); background:#fff; padding:8px 10px; border-radius:10px; margin-bottom:8px; cursor:pointer}
  .chatitem.active{outline:2px solid #000}
  .chatmeta{font-size:12px; color:#6b7280; margin-top:2px}

  .main{display:flex; flex-direction:column; min-width:0}
  .header{background:#fff; padding:12px 16px; border-bottom:1px solid var(--border)}
  .header .title{font-family:var(--title); font-weight:300; font-size:20px; text-align:center}

  .content{flex:1; overflow:auto; padding:24px 12px 140px}
  .container{max-width:900px; margin:0 auto}
  .thread{display:flex; flex-direction:column; gap:16px}
  .msg{padding:0; border:0; background:transparent}
  .msg .bubble{display:inline-block; max-width:80%}
  .msg.user .bubble{background:var(--user); border:1px solid var(--border); border-radius:14px; padding:12px 14px; box-shadow:var(--shadow)}
  .msg.advisor .bubble{background:transparent; padding:0; max-width:100%}
  .meta{font-size:12px; margin-bottom:6px; color:#000}

  .bubble h1,.bubble h2,.bubble h3{margin:.6em 0 .4em}
  .bubble p{margin:.6em 0}
  .bubble ul, .bubble ol{margin:.4em 0 .6em 1.4em}
  .bubble a{color:#000; text-decoration:underline}
  .bubble strong{font-weight:700}
  .bubble em{font-style:italic}
  .bubble code{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,"Cascadia Mono","Segoe UI Mono","Roboto Mono","Oxygen Mono","Ubuntu Mono","Courier New",monospace;background:#fff;border:1px solid var(--border);padding:.1em .3em;border-radius:6px;color:#000}
  .bubble pre{background:#fff;color:#000;border:1px solid var(--border);padding:12px;border-radius:12px;overflow:auto}
  .bubble blockquote{border-left:3px solid #000;padding:6px 12px;margin:8px 0;background:#fafafa}

  /* Composer: NO page-wide divider */
  .composer{position:fixed; left:280px; right:0; bottom:0; background:#fff; padding:18px 12px; border-top:none}
  .bar{display:flex; align-items:flex-end; gap:8px; background:#fff; border:1px solid var(--ring); border-radius:22px; padding:8px; box-shadow:var(--shadow); max-width:900px; margin:0 auto}
  .cin{flex:1; min-height:24px; max-height:160px; overflow:auto; outline:none; padding:8px 10px; font:16px/1.5 var(--font); color:#000}
  .cin:empty:before{content:attr(data-placeholder); color:#000}
  .iconbtn{cursor:pointer; border:1px solid var(--border); background:#fff; color:#000; padding:8px 10px; border-radius:12px}
  .send{border:1px solid #000; background:#000; color:#fff; border-radius:12px; padding:8px 12px}
  #file{display:none}
</style>
</head>
<body>
  <div class="app">
    <!-- Left rail -->
    <aside class="rail">
      <div class="brand"><div class="title">Private Trust Fiduciary Advisor</div></div>

      <div class="section">
        <div class="label">Profile</div>
        <div class="row" style="margin-bottom:6px">
          <input id="pf-id" class="input" placeholder="Client ID (required)"/>
        </div>
        <div class="row" style="margin-bottom:6px">
          <input id="pf-email" class="input" placeholder="Email (optional)"/>
        </div>
        <div class="row">
          <input id="pf-token" class="input" placeholder="API Token (optional)"/>
          <button id="pf-save" class="btn primary">Save</button>
        </div>
        <div id="pf-msg" class="label" style="margin-top:6px;"></div>
      </div>

      <div class="section">
        <div class="row">
          <button id="btn-newchat" class="btn primary">New Chat</button>
        </div>
      </div>

      <div class="section" style="border-bottom:none; padding-bottom:0"><div class="label">Chats</div></div>
      <div id="chatlist" class="chats"></div>
    </aside>

    <!-- Main pane -->
    <main class="main">
      <div class="header"><div class="title">Advisor</div></div>

      <div class="content">
        <div class="container">
          <div id="thread" class="thread"></div>
        </div>
      </div>
    </main>
  </div>

  <!-- Composer -->
  <div class="composer">
    <div class="bar">
      <input id="file" type="file" multiple accept=".pdf,.txt,.docx"/>
      <div id="input" class="cin" contenteditable="true" data-placeholder="Message the Advisor… (Shift+Enter for newline)"></div>
      <button id="attach" class="iconbtn" title="Add files">+</button>
      <button id="send" class="send" title="Send" aria-label="Send">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="m5 12 14-7-4 14-3-5-7-2z" stroke="#fff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
    </div>
    <div id="filehint" style="max-width:900px; margin:8px auto 0; color:#000; font-size:12px;">
      State your inquiry to receive formal trust, fiduciary, and contractual analysis with strategic guidance.
    </div>
  </div>

<script>
  // ====== State ======
  let currentChatId = null;
  let state = {
    userId: localStorage.getItem('userId') || 'demo',
    email:  localStorage.getItem('email')  || '',
    token:  localStorage.getItem('apiToken') || ''
  };

  // ====== Elements ======
  const elThread  = document.getElementById('thread');
  const elInput   = document.getElementById('input');
  const elSend    = document.getElementById('send');
  const elAttach  = document.getElementById('attach');
  const elFile    = document.getElementById('file');
  const elHint    = document.getElementById('filehint');
  const elPfId    = document.getElementById('pf-id');
  const elPfEmail = document.getElementById('pf-email');
  const elPfToken = document.getElementById('pf-token');
  const elPfSave  = document.getElementById('pf-save');
  const elPfMsg   = document.getElementById('pf-msg');
  const elChatList= document.getElementById('chatlist');

  // Prefill profile inputs
  elPfId.value    = state.userId || '';
  elPfEmail.value = state.email || '';
  elPfToken.value = state.token || '';

  function now(){ return new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}) }

  function addMessage(role, html){
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + (role === 'user' ? 'user' : 'advisor');
    const meta = `<div class="meta">${role==='user'?'You':'Advisor'} · ${now()}</div>`;
    wrap.innerHTML = meta + `<div class="bubble">${html}</div>`;
    elThread.appendChild(wrap);
    elThread.scrollTop = elThread.scrollHeight;
  }

  // ====== Formatting helpers ======
  function mdToHtml(md){
    if(!md) return '';
    if (/<\\w+[^>]*>/.test(md)) return md;
    let h = md.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    h = h.replace(/```([\\s\\S]*?)```/g,(_,c)=>`<pre><code>${c.replace(/</g,'&lt;')}</code></pre>`);
    h = h.replace(/`([^`]+?)`/g,'<code>$1</code>');
    h = h.replace(/^######\\s+(.*)$/gm,'<h6>$1</h6>').replace(/^#####\\s+(.*)$/gm,'<h5>$1</h5>').replace(/^####\\s+(.*)$/gm,'<h4>$1</h4>').replace(/^###\\s+(.*)$/gm,'<h3>$1</h3>').replace(/^##\\s+(.*)$/gm,'<h2>$1</h2>').replace(/^#\\s+(.*)$/gm,'<h1>$1</h1>');
    h = h.replace(/^>\\s?(.*)$/gm,'<blockquote>$1</blockquote>');
    h = h.replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>').replace(/__(.+?)__/g,'<strong>$1</strong>').replace(/\\*(?!\\s)(.+?)\\*/g,'<em>$1</em>').replace(/_(?!\\s)(.+?)_/g,'<em>$1</em>');
    h = h.replace(/\\[([^\\]]+)\\]\\((https?:\\/\\/[^\\s)]+)\\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
    h = h.replace(/(^|\\s)(https?:\\/\\/[^\\s<]+)(?=\\s|$)/g,'$1<a href="$2" target="_blank" rel="noopener">$2</a>');
    h = h.replace(/(?:^|\\n)(\\d+)\\.\\s+(.+)(?:(?=\\n\\d+\\.\\s)|$)/gms,(m)=>{const items=m.trim().split(/\\n(?=\\d+\\.\\s)/).map(it=>it.replace(/^\\d+\\.\\s+/,'')).map(t=>`<li>${t}</li>`).join('');return `<ol>${items}</ol>`;});
    h = h.replace(/(?:^|\\n)[*-]\\s+(.+)(?:(?=\\n[*-]\\s)|$)/gms,(m)=>{const items=m.trim().split(/\\n(?=[*-]\\s)/).map(it=>it.replace(/^[*-]\\s+/,'')).map(t=>`<li>${t}</li>`).join('');return `<ul>${items}</ul>`;});
    h = h.replace(/\\n{2,}/g,'</p><p>').replace(/^(?!<h\\d|<ul|<ol|<pre|<hr|<p|<blockquote|<table)(.+)$/gm,'<p>$1</p>');
    return h;
  }
  function applyInlineFormatting(html){
    if(!html) return ''; let out=String(html); const slots=[];
    function protect(tag){ const re=new RegExp(`<${tag}\\\\b[^>]*>[\\\\s\\\\S]*?<\\\\/${tag}>`,'gi'); out=out.replace(re,m=>{const key=`__SLOT_${tag.toUpperCase()}_${slots.length}__`; slots.push({key,val:m}); return key;});}
    protect('code'); protect('pre'); protect('a');
    out = out.replace(/\\*\\*([^*]+?)\\*\\*/g,'<strong>$1</strong>')
             .replace(/__([^_]+?)__/g,'<strong>$1</strong>')
             .replace(/(^|[^*])\\*([^*\\n]+?)\\*/g,'$1<em>$2</em>')
             .replace(/(^|[^_])_([^_\\n]+?)_/g,'$1<em>$2</em>');
    for(const {key,val} of slots) out=out.replace(key,val);
    return out;
  }
  function normalizeTrustDoc(html){
    let out=html;
    out=out.replace(/<p>\\s*<strong>\\s*([A-Z0-9][A-Z0-9\\s\\-&,.'()]+?)\\s*<\\/strong>\\s*<\\/p>/g,'<h2>$1</h2><p></p>');
    const map=[{re:/<strong>\\s*TRUST\\s*NAME\\s*:\\s*<\\/strong>/gi,rep:'Trust: '},{re:/<strong>\\s*DATE\\s*:\\s*<\\/strong>/gi,rep:'Date: '},{re:/<strong>\\s*TAX\\s*YEAR\\s*:\\s*<\\/strong>/gi,rep:'Tax Year: '},{re:/<strong>\\s*TRUSTEE\\(S\\)\\s*:\\s*<\\/strong>/gi,rep:'Trustee(s): '},{re:/<strong>\\s*LOCATION\\s*:\\s*<\\/strong>/gi,rep:'Location: '}];
    map.forEach(({re,rep})=>{ out=out.replace(re,rep) });
    out=out.replace(/<strong>\\s*([A-Za-z][A-Za-z()\\s]+:)\\s*<\\/strong>\\s*/g,'$1 ');
    out=out.replace(/\\*\\*\\s*TRUST\\s*NAME\\s*:\\s*\\*\\*/gi,'Trust: ').replace(/\\*\\*\\s*DATE\\s*:\\s*\\*\\*/gi,'Date: ').replace(/\\*\\*\\s*TAX\\s*YEAR\\s*:\\s*\\*\\*/gi,'Tax Year: ').replace(/\\*\\*\\s*TRUSTEE\\(S\\)\\s*:\\s*\\*\\*/gi,'Trustee(s): ').replace(/\\*\\*\\s*LOCATION\\s*:\\s*\\*\\*/gi,'Location: ');
    return out;
  }

  // ====== Fetch helpers with headers ======
  function hdrs(){
    const h = { 'X-User-Id': state.userId };
    if (state.token) h['Authorization'] = 'Bearer ' + state.token;
    return h;
  }
  async function getJSON(url){
    const r = await fetch(url, {headers: hdrs()});
    if(!r.ok) throw new Error(url+': '+r.status);
    return r.json();
  }
  async function postForm(url, form){
    const r = await fetch(url, {method:'POST', headers: hdrs(), body: form});
    if(!r.ok) throw new Error(url+': '+r.status);
    return r.json();
  }

  // ====== Profile save ======
  document.getElementById('pf-save').addEventListener('click', ()=>{
    const uid = elPfId.value.trim();
    const em  = elPfEmail.value.trim();
    const tk  = elPfToken.value.trim();
    if(!uid){ elPfMsg.textContent='Client ID is required.'; return; }
    state.userId = uid; state.email = em; state.token = tk;
    localStorage.setItem('userId', uid);
    localStorage.setItem('email', em);
    localStorage.setItem('apiToken', tk);
    elPfMsg.textContent = 'Saved.';
    // reload chat list for this user
    loadChats();
  });

  // ====== Chat list / tabs ======
  async function loadChats(){
    elChatList.innerHTML = '';
    try{
      const data = await getJSON('/chats');
      (data.items || []).forEach(ch=>{
        const div = document.createElement('div');
        div.className = 'chatitem' + (currentChatId===ch.id ? ' active' : '');
        div.innerHTML = `<div>${(ch.title||'Untitled')}</div><div class="chatmeta">${new Date(ch.updated_at).toLocaleString()}</div>`;
        div.addEventListener('click', ()=> openChat(ch.id));
        elChatList.appendChild(div);
      });
    }catch(e){
      elChatList.innerHTML = '<div class="label">Failed to load chats.</div>';
    }
  }

  async function openChat(chatId){
    currentChatId = chatId;
    // highlight active
    document.querySelectorAll('.chatitem').forEach(x=>x.classList.remove('active'));
    const items = Array.from(document.querySelectorAll('.chatitem'));
    const idx = (items.findIndex(x => (x.textContent||'').includes(chatId)) ); // not reliable; refresh list
    await loadChats(); // refresh then highlight
    // load messages
    elThread.innerHTML = '';
    try{
      const data = await getJSON(`/chats/${chatId}`);
      (data.messages || []).forEach(m=>{
        addMessage(m.role === 'user' ? 'user' : 'advisor', m.content_html);
      });
    }catch(e){
      addMessage('advisor', `<p style="color:#b91c1c">Failed to load chat: ${e}</p>`);
    }
  }

  // New chat
  document.getElementById('btn-newchat').addEventListener('click', async ()=>{
    try{
      const form = new FormData();
      const res  = await postForm('/chats', form); // POST /chats returns {chat_id}
      currentChatId = res.chat_id;
      await loadChats();
      elThread.innerHTML = '';
      addMessage('advisor', '<p>New chat created. How may I assist?</p>');
    }catch(e){
      addMessage('advisor', `<p style="color:#b91c1c">Failed to create chat: ${e}</p>`);
    }
  });

  // ====== RAG calls ======
  function readInput(){
    const tmp = elInput.cloneNode(true);
    tmp.querySelectorAll('div').forEach(d=>{ if (d.innerHTML === "<br>") d.innerHTML = "\\n"; });
    return tmp.innerText.replace(/\\u00A0/g,' ').trim();
  }

  async function callRag(q, chatId){
    const url = new URL('/rag', location.origin);
    url.searchParams.set('question', q);
    if (chatId) url.searchParams.set('chat_id', chatId);
    url.searchParams.set('top_k','12');
    const r = await fetch(url, {method:'GET', headers: hdrs()});
    if(!r.ok) throw new Error('RAG failed: '+r.status);
    return r.json();
  }

  async function callReview(q, files, chatId){
    const fd = new FormData();
    fd.append('question', q);
    for(const f of files) fd.append('files', f);
    if (chatId) fd.append('chat_id', chatId);
    const r = await fetch('/review', {method:'POST', headers: hdrs(), body: fd});
    if(!r.ok) throw new Error('Review failed: '+r.status);
    return r.json();
  }

  async function handleSend(q){
    if(!q) return;
    addMessage('user', q.replace(/\\n/g,'<br>'));
    const work = document.createElement('div');
    work.className = 'msg advisor';
    work.innerHTML = `<div class="meta">Advisor · thinking…</div><div class="bubble"><p>Working…</p></div>`;
    elThread.appendChild(work); elThread.scrollTop = elThread.scrollHeight;

    try{
      const files = Array.from(elFile.files || []);
      const data  = files.length ? await callReview(q, files, currentChatId) : await callRag(q, currentChatId);
      if (data && data.chat_id) currentChatId = data.chat_id;
      // refresh chat list to move this chat to top
      loadChats();

      let html = (data && data.answer) ? data.answer : '';
      const looksHtml = typeof html==='string' && /<\\w+[^>]*>/.test(html);
      let rendered = looksHtml ? html : mdToHtml(String(html||''));
      rendered = applyInlineFormatting(rendered);
      rendered = normalizeTrustDoc(rendered);

      work.querySelector('.meta').textContent = 'Advisor · ' + now();
      work.querySelector('.bubble').outerHTML = `<div class="bubble">${rendered}</div>`;
    }catch(e){
      work.querySelector('.meta').textContent = 'Advisor · error';
      work.querySelector('.bubble').innerHTML = '<p style="color:#b91c1c">Error: '+(e && e.message ? e.message : String(e))+'</p>';
    }
  }

  // Send & attach events
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
  elAttach.addEventListener('click', ()=> elFile.click());
  elFile.addEventListener('change', ()=>{
    if (elFile.files && elFile.files.length){
      elHint.textContent = `${elFile.files.length} file${elFile.files.length>1?'s':''} selected.`;
    }else{
      elHint.textContent = 'State your inquiry to receive formal trust, fiduciary, and contractual analysis with strategic guidance.';
    }
  });

  // Initial load
  loadChats();
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
        "db_path": DB_PATH,
    }
    try:
        lst = pc.list_indexes()
        info["pinecone_ok"] = True
        info["index_count"] = len(lst or [])
    except Exception as e:
        info["pinecone_ok"] = False
        info["error"] = str(e)
    return info

# ========== /search (RAW CONTEXT) ==========
@app.get("/search")
def search_endpoint(
    question: str = Query(..., min_length=3),
    top_k: int = Query(12, ge=1, le=30),
    level: str | None = Query(None),
    authorization: str | None = Header(default=None),
    user_id: str = Depends(get_current_user),
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
            meta = s.get("meta") if isinstance(s, dict) else {}
            if not meta and isinstance(s, dict):
                meta = s.get("meta", {})
            rows.append({
                "title":   s["title"],
                "level":   s["level"],
                "page":    s["page"],
                "version": s.get("version",""),
                "score":   s["score"],
                "snippet": _extract_snippet(meta or {}) or ""
            })
        return {"question": question, "titles": titles, "matches": rows, "t_ms": int((time.time()-t0)*1000)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ========== /rag (SYNTHESIS + PERSISTENCE) ==========
@app.get("/rag")
def rag_endpoint(
    question: str = Query(..., min_length=3),
    chat_id: Optional[str] = Query(None),
    top_k: int = Query(12, ge=1, le=30),
    level: str | None = Query(None),
    authorization: str | None = Header(default=None),
    user_id: str = Depends(get_current_user),
):
    require_auth(authorization)
    check_rate_limit()
    conn = db()
    try:
        chat_id = ensure_chat(conn, user_id, chat_id)
        insert_message(conn, chat_id, user_id, "user", content_html=f"<p>{question}</p>", content_raw=question, meta={"t_ms":0})
        t0 = time.time()
        emb = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
        flt = {"doc_level": {"$eq": level}} if level else None
        res = idx.query(vector=emb, top_k=max(top_k, 12), include_metadata=True, filter=flt)
        matches = res["matches"] if isinstance(res, dict) else getattr(res, "matches", [])
        uniq = _dedup_and_rank_sources(matches, top_k=top_k)
        snippets = [s for s in (_extract_snippet(u.get("meta", {})) for u in uniq) if s]
        html = synthesize_html(question, uniq, snippets)
        elapsed = int((time.time()-t0)*1000)
        insert_message(conn, chat_id, None, "advisor", content_html=html, content_raw=None, meta={"t_ms": elapsed})
        return {"answer": html, "t_ms": elapsed, "chat_id": chat_id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# ========== /review (PDF/TXT/DOCX) + PERSISTENCE ==========
@app.post("/review")
def review_endpoint(
    authorization: str | None = Header(default=None),
    chat_id: Optional[str] = Form(None),
    question: str = Form(""),
    files: list[UploadFile] = File(default=[]),
    user_id: str = Depends(get_current_user),
):
    require_auth(authorization)
    check_rate_limit()
    conn = db()
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")
        chat_id = ensure_chat(conn, user_id, chat_id)
        insert_message(conn, chat_id, user_id, "user", content_html=f"<p>{question}</p>", content_raw=question, meta={"upload": True})

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
        chunks  = [merged[i:i+2000] for i in range(0, len(merged), 2000)][:MAX_SNIPPETS]
        pseudo  = [{"title": "Uploaded Document", "level": "L5", "page": "?", "version": "", "score": 1.0, "meta": {}}]
        html    = synthesize_html(question or "Please analyze the attached materials.", pseudo, chunks)

        insert_message(conn, chat_id, None, "advisor", content_html=html, content_raw=None, meta={"upload": True})
        return {"answer": html, "t_ms": 0, "chat_id": chat_id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
