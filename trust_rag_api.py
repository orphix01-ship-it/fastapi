# trust_rag_api.py
from fastapi import FastAPI, Query, Header, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pinecone import Pinecone
from openai import OpenAI
import httpx, zipfile, io, re, os, time, traceback, sqlite3, json, uuid
from datetime import datetime
from collections import deque
from typing import Optional, List, Dict, Any

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

API_TOKEN         = os.getenv("API_TOKEN", "")      # optional bearer for /search, /rag & /review (leave empty to disable auth)
SYNTH_MODEL       = os.getenv("SYNTH_MODEL", "gpt-4o")
MAX_SNIPPETS      = int(os.getenv("MAX_SNIPPETS", "20"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))
MAX_OUT_TOKENS    = int(os.getenv("MAX_OUT_TOKENS", "16384"))
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
def require_auth(auth_header: Optional[str]):
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

# Resolve current user from header; default to "demo" if none
def get_current_user(authorization: Optional[str] = Header(None),
                     x_user_id: Optional[str] = Header(None)) -> str:
    if x_user_id and x_user_id.strip():
        return x_user_id.strip()
    return "demo"

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

def iso_now() -> str:
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
def _extract_snippet(meta: Dict[str, Any]) -> str:
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

def _listing_title(meta: Dict[str, Any]) -> str:
    return _clean_title(meta.get("title") or meta.get("doc_parent") or "Unknown")

def _dedup_and_rank_sources(matches: List[Dict[str, Any]], top_k: int):
    rank = {"L1":1,"L2":2,"L3":3,"L4":4,"L5":5}
    best: Dict[Any, Dict[str, Any]] = {}
    for m in (matches or []):
        meta  = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        title = _listing_title(meta)
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

def _titles_only(uniq_sources: List[Dict[str, Any]]) -> List[str]:
    seen, out = set(), []
    for s in uniq_sources:
        t = s["title"]
        if t not in seen:
            seen.add(t); out.append(t)
    return out

# ========== SYNTHESIS ==========
def synthesize_html(question: str, uniq_sources: List[Dict[str, Any]], snippets: List[str]) -> str:
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
        "Always respond using clean, valid HTML."
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
            return "<p>No relevant material found.</p>"
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
                   content_html: str, content_raw: Optional[str], meta: Dict[str, Any]):
    cur = conn.cursor()
    mid = str(uuid.uuid4())
    now = iso_now()
    cur.execute("""INSERT INTO messages (id, chat_id, user_id, role, content_html, content_raw, meta_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (mid, chat_id, user_id, role, content_html, content_raw, json.dumps(meta or {}), now))
    cur.execute("UPDATE chats SET updated_at=?, title=COALESCE(title,'New chat') WHERE id=?", (now, chat_id))
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

# ========== WIDGET ==========
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
    --rail:#fafafa;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 var(--font)}

  /* Grid that truly collapses columns without overflow */
  .app{display:grid; grid-template-columns: 280px 1fr 360px; height:100vh}
  body.sidebar-closed .app{grid-template-columns: 0 1fr 360px}
  body.tree-closed .app{grid-template-columns: 280px 1fr 0}
  body.sidebar-closed.tree-closed .app{grid-template-columns: 0 1fr 0}

  .rail{border-right:1px solid var(--border); background:var(--rail); display:flex; flex-direction:column; min-width:0}
  .brand{padding:12px 12px 8px; border-bottom:1px solid var(--border)}
  .brand .title{
    font-family:var(--title); font-weight:300; font-size:20px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; display:block
  }
  .title-actions{display:flex; gap:8px; margin-top:8px; flex-wrap:wrap}

  .section{padding:12px; border-bottom:1px solid var(--border)}
  .label{font-size:12px; color:#6b7280; margin-bottom:6px}
  .row{display:flex; gap:6px}
  .input{flex:1; border:1px solid var(--border); border-radius:10px; padding:6px 8px; background:#fff}
  .btn{cursor:pointer; border:1px solid var(--border); background:#fff; color:#000; padding:6px 10px; border-radius:10px}
  .btn.primary{background:#000; color:#fff; border-color:#000}
  .chats{flex:1; overflow:auto; padding:8px}
  .chatitem{display:flex; justify-content:space-between; align-items:center; border:1px solid var(--border); background:#fff; padding:8px 10px; border-radius:10px; margin-bottom:8px}
  .chatitem .meta{font-size:12px; color:#6b7280}
  .chatitem .title{max-width:170px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; cursor:pointer}
  .chatitem .actions{display:flex; gap:6px}
  .chatitem.active{outline:2px solid #000}

  .main{display:flex; flex-direction:column; min-width:0}
  .content{flex:1; overflow:auto; padding:24px 12px 140px}
  .container{max-width:900px; margin:0 auto}
  .thread{display:flex; flex-direction:column; gap:16px}
  .msg .bubble{display:inline-block; max-width:80%}
  .msg.user .bubble{background:var(--user); border:1px solid var(--border); border-radius:14px; padding:12px 14px; box-shadow:var(--shadow)}
  .msg.advisor .bubble{background:transparent; padding:0; max-width:100%}
  .meta{font-size:12px; margin-bottom:6px; color:#000}
  .bubble p{margin:.6em 0}

  /* Composer follows column widths */
  .composer{position:fixed; left:280px; right:360px; bottom:0; background:#fff; padding:18px 12px; transition:left .18s,right .18s; border-top:1px solid var(--border)}
  body.sidebar-closed .composer{left:0}
  body.tree-closed .composer{right:0}
  .bar{display:flex; align-items:flex-end; gap:8px; background:#fff; border:1px solid var(--ring); border-radius:22px; padding:8px; box-shadow:var(--shadow); max-width:900px; margin:0 auto}
  .cin{flex:1; min-height:24px; max-height:160px; overflow:auto; outline:none; padding:8px 10px}

  /* Right panel: add air so it isn't cramped */
  .tree{border-left:1px solid var(--border); background:#fff; display:flex; flex-direction:column; min-width:0}
  .tree-head{display:flex; align-items:center; justify-content:space-between; padding:14px 16px; border-bottom:1px solid var(--border)}
  .tree-title{font-weight:700}
  .tree-actions{display:flex; gap:8px}
  .tree-body{flex:1; overflow:auto; padding:14px 16px 18px}
  .tree-item{padding:6px 8px; border-radius:8px; cursor:pointer}
  .tree-item:hover{background:#fafafa}
  .tree-item.active{outline:2px solid #000}
  .node-row{display:flex; align-items:center; gap:6px}
  .node-actions{display:flex; gap:6px; margin-left:auto}

  /* Top-right global toggles */
  .top-controls{position:fixed; top:10px; right:10px; display:flex; gap:8px; z-index:40}
</style>
</head>
<body>

  <!-- Always-visible controls -->
  <div class="top-controls">
    <button id="btn-left-toggle" class="btn" title="Toggle left sidebar">☰</button>
    <button id="btn-right-toggle" class="btn" title="Toggle conversation tree">⟷</button>
  </div>

  <div class="app">
    <!-- Left rail -->
    <aside class="rail" id="rail">
      <div class="brand">
        <div class="title">Private Trust Fiduciary Advisor</div>
        <div class="title-actions">
          <button id="btn-rail-collapse" class="btn">Collapse</button>
          <button id="btn-rail-open" class="btn">Open</button>
        </div>
      </div>

      <div class="section">
        <div class="label">Profile</div>
        <div class="row" style="margin-bottom:6px">
          <input id="pf-id" class="input" placeholder="Client ID (required)"/>
        </div>
        <div class="row">
          <input id="pf-email" class="input" placeholder="Email (optional)"/>
        </div>
        <div class="row" style="gap:8px; margin-top:8px">
          <button id="pf-save" class="btn primary">Save</button>
          <button id="pf-logout" class="btn">Log out</button>
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

    <!-- Main -->
    <main class="main">
      <div class="content">
        <div class="container">
          <div id="thread" class="thread"></div>
        </div>
      </div>
    </main>

    <!-- Right panel -->
    <aside class="tree" id="treePane" aria-label="Conversation Tree">
      <div class="tree-head">
        <div class="tree-title">Conversation Tree</div>
        <div class="tree-actions">
          <button id="btn-tree-toggle" class="btn">Toggle</button>
        </div>
      </div>
      <div class="tree-body">
        <div class="label" style="margin-bottom:6px">Roots</div>
        <div id="roots" style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px"></div>
        <div class="label" style="margin-top:8px">Nodes</div>
        <div id="treeList"></div>
      </div>
    </aside>
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
  // Grab ALL DOM references FIRST so handlers don't reference undefined vars
  const elRail           = document.getElementById('rail');
  const elTreePane       = document.getElementById('treePane');
  const elLeftToggle     = document.getElementById('btn-left-toggle');
  const elRightToggle    = document.getElementById('btn-right-toggle');
  const elRailCollapse   = document.getElementById('btn-rail-collapse');
  const elRailOpen       = document.getElementById('btn-rail-open');
  const elTreeToggle     = document.getElementById('btn-tree-toggle');

  const elThread         = document.getElementById('thread');
  const elInput          = document.getElementById('input');
  const elSend           = document.getElementById('send');
  const elAttach         = document.getElementById('attach');
  const elFile           = document.getElementById('file');
  const elHint           = document.getElementById('filehint');

  const elPfId           = document.getElementById('pf-id');
  const elPfEmail        = document.getElementById('pf-email');
  const elPfSave         = document.getElementById('pf-save');
  const elPfLogout       = document.getElementById('pf-logout');
  const elPfMsg          = document.getElementById('pf-msg');
  const elChatList       = document.getElementById('chatlist');

  const elRoots          = document.getElementById('roots');
  const elTreeList       = document.getElementById('treeList');

  // === UI persistent state (sidebar + tree) ===
  const UIKEY = 'ui.layout.v1';
  const ui = (() => {
    const def = { sidebarOpen: true, treeOpen: true };
    try { return Object.assign(def, JSON.parse(localStorage.getItem(UIKEY) || '{}')); }
    catch(_){ return def; }
  })();
  function saveUI(){ localStorage.setItem(UIKEY, JSON.stringify(ui)); }
  function applyLayout(){
    document.body.classList.toggle('sidebar-closed', !ui.sidebarOpen);
    document.body.classList.toggle('tree-closed',    !ui.treeOpen);
  }

  // Wire buttons (explicit listeners)
  if (elRailCollapse) elRailCollapse.addEventListener('click', ()=>{ ui.sidebarOpen=false; saveUI(); applyLayout(); });
  if (elRailOpen)     elRailOpen.addEventListener('click',     ()=>{ ui.sidebarOpen=true;  saveUI(); applyLayout(); });
  if (elTreeToggle)   elTreeToggle.addEventListener('click',   ()=>{ ui.treeOpen=!ui.treeOpen; saveUI(); applyLayout(); });
  if (elLeftToggle)   elLeftToggle.addEventListener('click',   ()=>{ ui.sidebarOpen=!ui.sidebarOpen; saveUI(); applyLayout(); });
  if (elRightToggle)  elRightToggle.addEventListener('click',  ()=>{ ui.treeOpen=!ui.treeOpen; saveUI(); applyLayout(); });

  // ====== State ======
  let currentChatId = null;
  let state = {
    userId: localStorage.getItem('userId') || '',
    email:  localStorage.getItem('email')  || ''
  };

  // ====== Conversation Tree Model ======
  const TREEKEY = 'conversation.tree.v1';
  const uuidv4 = () => 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random()*16|0, v = c === 'x' ? r : (r&0x3|0x8); return v.toString(16);
  });

  /** @typedef {{ id:string, rootId:string, parentId:(string|null), role:'user'|'assistant'|'system', content:string, createdAt:string, children:string[], meta?:{ title?:string, tags?:string[], chatId?:string } }} MessageNode */
  /** @typedef {{ id:string, title:string, createdAt:string, firstNodeId:(string|null) }} ConversationRoot */
  /** @typedef {{ roots:string[], nodesById:Record<string,MessageNode>, rootsById:Record<string,ConversationRoot>, currentRootId:(string|null), currentNodeId:(string|null) }} TreeIndex */

  /** @type {TreeIndex} */
  let tree = (() => {
    try{
      const t = JSON.parse(localStorage.getItem(TREEKEY) || '{}');
      if (t && t.roots && t.nodesById && t.rootsById) return t;
    }catch(_){}
    return { roots: [], nodesById: {}, rootsById: {}, currentRootId: null, currentNodeId: null };
  })();

  function saveTree(){ localStorage.setItem(TREEKEY, JSON.stringify(tree)); }
  function newRoot(title){
    const id = uuidv4();
    tree.rootsById[id] = { id, title: title || 'Untitled Root', createdAt: new Date().toISOString(), firstNodeId: null };
    tree.roots.unshift(id);
    tree.currentRootId = id; tree.currentNodeId = null;
    saveTree(); renderTree();
    return id;
  }
  function newNode({rootId, parentId=null, role='user', content='', title}){
    const id = uuidv4();
    const node = { id, rootId, parentId, role, content, createdAt: new Date().toISOString(), children: [], meta:{ title: title || undefined } };
    tree.nodesById[id] = node;
    if (parentId){ tree.nodesById[parentId].children.push(id); }
    const r = tree.rootsById[rootId]; if (!r.firstNodeId) r.firstNodeId = id;
    tree.currentNodeId = id;
    saveTree(); renderTree();
    return id;
  }
  function selectRoot(rootId){ tree.currentRootId = rootId; saveTree(); renderTree(); }
  function selectNode(nodeId){
    tree.currentNodeId = nodeId; saveTree(); renderTree();
    const node = tree.nodesById[nodeId];
    if (node && node.meta && node.meta.chatId){ openChat(node.meta.chatId); }
  }
  function branchFrom(nodeId, title){
    const parent = tree.nodesById[nodeId]; if (!parent) return null;
    return newNode({ rootId: parent.rootId, parentId: parent.id, role:'user', content:'', title: title || 'New branch' });
  }
  function setNodeChat(nodeId, chatId){
    const n = tree.nodesById[nodeId]; if (!n) return;
    n.meta = n.meta || {}; n.meta.chatId = chatId; saveTree();
  }
  function labelFor(node){ return (node.meta && node.meta.title) || (node.content ? node.content.slice(0,80) : (node.role === 'assistant' ? 'Assistant' : 'User')); }

  // ====== Elements already grabbed above ======

  // Prefill profile inputs
  elPfId.value    = state.userId || '';
  elPfEmail.value = state.email || '';

  function now(){ return new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}) }

  function addMessage(role, html){
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + (role === 'user' ? 'user' : 'advisor');
    const meta = `<div class="meta">${role==='user'?'You':'Advisor'} · ${now()}</div>`;
    wrap.innerHTML = meta + `<div class="bubble">${html}</div>`;
    elThread.appendChild(wrap);
    elThread.scrollTop = elThread.scrollHeight;
  }

  // ====== Tree Rendering ======
  function renderTree(){
    if (elRoots){
      elRoots.innerHTML = '';
      (tree.roots || []).forEach(rid=>{
        const r = tree.rootsById[rid];
        const b = document.createElement('button');
        b.className = 'btn' + (tree.currentRootId === rid ? ' primary' : '');
        b.textContent = r.title || 'Untitled Root';
        b.addEventListener('click', ()=> selectRoot(rid));
        elRoots.appendChild(b);
      });
    }
    if (!tree.currentRootId){ elTreeList.innerHTML = '<div class="label">No root selected.</div>'; return; }
    const rootId = tree.currentRootId;
    const build = (id, depth) => {
      const node = tree.nodesById[id]; if (!node) return [];
      const row = document.createElement('div');
      row.className = 'tree-item ' + (tree.currentNodeId===id ? 'active' : '');
      row.innerHTML = `<div class="node-row" style="margin-left:${Math.min(depth,4)*14}px">
        <span>${labelFor(node)}</span>
        <div class="node-actions">
          <button class="btn" data-act="open" data-id="${id}">Open</button>
          <button class="btn" data-act="branch" data-id="${id}">Branch</button>
        </div>
      </div>`;
      row.querySelector('[data-act="open"]').addEventListener('click', (e)=>{ e.stopPropagation(); selectNode(id); });
      row.querySelector('[data-act="branch"]').addEventListener('click', (e)=>{
        e.stopPropagation();
        const title = prompt('Branch title? (optional)') || 'New branch';
        const cid = branchFrom(id, title);
        if (cid) selectNode(cid);
      });
      row.addEventListener('click', ()=> selectNode(id));
      const out = [row];
      (node.children || []).forEach(childId => { out.push(...build(childId, depth+1)); });
      return out;
    };
    const top = [];
    Object.values(tree.nodesById).forEach(n=>{ if (n.rootId === rootId && !n.parentId) top.push(n.id); });
    top.sort((a,b)=> new Date(tree.nodesById[a].createdAt) - new Date(tree.nodesById[b].createdAt));
    elTreeList.innerHTML = '';
    top.forEach(tid => build(tid, 0).forEach(el => elTreeList.appendChild(el)));
  }

  // ====== Formatting helpers ======
  function mdToHtml(md){
    if(!md) return '';
    if (/<\w+[^>]*>/.test(md)) return md;
    let h = md.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    h = h.replace(/```([\s\S]*?)```/g,(_,c)=>`<pre><code>${c.replace(/</g,'&lt;')}</code></pre>`);
    h = h.replace(/`([^`]+?)`/g,'<code>$1</code>');
    h = h.replace(/\n{2,}/g,'</p><p>');
    return h;
  }
  function applyInlineFormatting(html){ return html || ''; }
  function normalizeTrustDoc(html){ return html || ''; }

  // ====== Fetch helpers with headers ======
  function hdrs(){ const h = {}; if (state.userId) h['X-User-Id'] = state.userId; return h; }
  async function getJSON(url){ const r = await fetch(url, {headers: hdrs()}); if(!r.ok) throw new Error(url+': '+r.status); return r.json(); }
  async function postForm(url, form){ const r = await fetch(url, {method:'POST', headers: hdrs(), body: form}); if(!r.ok) throw new Error(url+': '+r.status); return r.json(); }
  async function del(url){ const r = await fetch(url, {method:'DELETE', headers: hdrs()}); if(!r.ok) throw new Error(url+': '+r.status); return r.json(); }

  // ====== Profile save / logout ======
  elPfSave.addEventListener('click', ()=>{
    const uid = elPfId.value.trim();
    const em  = elPfEmail.value.trim();
    if(!uid){ elPfMsg.textContent='Client ID is required.'; return; }
    state.userId = uid; state.email = em;
    localStorage.setItem('userId', uid);
    localStorage.setItem('email', em);
    elPfMsg.textContent = 'Saved. Loading your chats…';
    currentChatId = null;
    elThread.innerHTML = '';
    loadChats();
  });

  elPfLogout.addEventListener('click', ()=>{
    localStorage.removeItem('userId');
    localStorage.removeItem('email');
    state.userId = '';
    state.email  = '';
    elPfId.value = '';
    elPfEmail.value = '';
    elPfMsg.textContent = 'Logged out. Enter Client ID to log in.';
    currentChatId = null;
    elChatList.innerHTML = '';
    elThread.innerHTML = '<div class="msg advisor"><div class="meta">Advisor ·</div><div class="bubble"><p>Please sign in with your Client ID to view your chats.</p></div></div>';
  });

  // ====== Chat list / tabs, rename, delete ======
  async function loadChats(){
    elChatList.innerHTML = '';
    if(!state.userId){
      elChatList.innerHTML = '<div class="label">Not signed in.</div>';
      return;
    }
    try{
      const data = await getJSON('/chats');
      (data.items || []).forEach(ch=>{
        const row = document.createElement('div');
        row.className = 'chatitem' + (currentChatId===ch.id ? ' active' : '');
        row.dataset.id = ch.id;

        const left = document.createElement('div');
        const title = document.createElement('div');
        title.className = 'title';
        title.textContent = ch.title || 'Untitled';
        title.title = ch.title || 'Untitled';
        title.addEventListener('click', ()=> openChat(ch.id));
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.textContent = new Date(ch.updated_at).toLocaleString();
        left.appendChild(title);
        left.appendChild(meta);

        const actions = document.createElement('div');
        actions.className = 'actions';

        const btnRename = document.createElement('button');
        btnRename.className = 'btn';
        btnRename.textContent = '✎';
        btnRename.title = 'Rename';
        btnRename.addEventListener('click', async (e)=>{
          e.stopPropagation();
          const newTitle = prompt('New chat title:', ch.title || '');
          if(newTitle !== null){
            const fd = new FormData();
            fd.append('title', newTitle);
            try { await postForm(`/chats/${ch.id}/title`, fd); loadChats(); }
            catch(err){ alert('Rename failed: '+err); }
          }
        });

        const btnDelete = document.createElement('button');
        btnDelete.className = 'btn';
        btnDelete.textContent = '🗑';
        btnDelete.title = 'Delete';
        btnDelete.addEventListener('click', async (e)=>{
          e.stopPropagation();
          if(confirm('Delete this chat?')){
            try { await del(`/chats/${ch.id}`); if (currentChatId === ch.id) { currentChatId = null; elThread.innerHTML=''; } loadChats(); }
            catch(err){ alert('Delete failed: '+err); }
          }
        });

        actions.appendChild(btnRename);
        actions.appendChild(btnDelete);

        row.appendChild(left);
        row.appendChild(actions);
        elChatList.appendChild(row);
      });
      if (!currentChatId && data.items && data.items.length) {
        currentChatId = data.items[0].id;
        openChat(currentChatId);
      }
      elPfMsg.textContent = '';
    }catch(e){
      elChatList.innerHTML = '<div class="label">Failed to load chats.</div>';
    }
  }

  async function openChat(chatId){
    currentChatId = chatId;
    document.querySelectorAll('.chatitem').forEach(x=>{
      x.classList.toggle('active', x.dataset.id === chatId);
    });
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
    if(!state.userId){ elPfMsg.textContent='Please enter Client ID and Save first.'; return; }
    if(!q) return;
    addMessage('user', q.replace(/\\n/g,'<br>'));
    if (!tree.currentRootId){ newRoot('General'); }
    if (!tree.currentNodeId){
      newNode({ rootId: tree.currentRootId, parentId: null, role: 'user', content: q, title: q.slice(0,80) });
    } else {
      const branchedId = branchFrom(tree.currentNodeId, q.slice(0,80));
      if (branchedId){ const n = tree.nodesById[branchedId]; n.content = q; saveTree(); renderTree(); }
    }
    const pendingNodeId = tree.currentNodeId;

    if (currentChatId) {
      const fd = new FormData(); fd.append('title', q.slice(0, 60));
      postForm(`/chats/${currentChatId}/title`, fd).catch(()=>{});
    }

    const work = document.createElement('div');
    work.className = 'msg advisor';
    work.innerHTML = `<div class="meta">Advisor · thinking…</div><div class="bubble"><p>Working…</p></div>`;
    elThread.appendChild(work); elThread.scrollTop = elThread.scrollHeight;

    try{
      const files = Array.from(elFile.files || []);
      const data  = files.length ? await callReview(q, files, currentChatId) : await callRag(q, currentChatId);
      if (data && data.chat_id) currentChatId = data.chat_id;
      loadChats();
      if (pendingNodeId && currentChatId){ setNodeChat(pendingNodeId, currentChatId); }

      let html = (data && data.answer) ? data.answer : '';
      const looksHtml = typeof html==='string' && /<\\w+[^>]*>/.test(html);
      let rendered = looksHtml ? html : mdToHtml(String(html||'')); rendered = applyInlineFormatting(rendered); rendered = normalizeTrustDoc(rendered);
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
      const q = readInput(); if (!q) return;
      elInput.innerHTML = '';
      handleSend(q);
    }
  });
  elSend.addEventListener('click', ()=>{
    const q = readInput(); if (!q) return;
    elInput.innerHTML = '';
    handleSend(q);
  });
  elAttach.addEventListener('click', ()=> elFile.click());
  elFile.addEventListener('change', ()=>{
    elHint.textContent = (elFile.files && elFile.files.length)
      ? `${elFile.files.length} file${elFile.files.length>1?'s':''} selected.`
      : 'State your inquiry to receive formal trust, fiduciary, and contractual analysis with strategic guidance.';
  });

  // Init layout + tree + chats
  applyLayout();
  renderTree();
  if (state.userId) { loadChats(); }
  else { elThread.innerHTML = '<div class="msg advisor"><div class="meta">Advisor ·</div><div class="bubble"><p>Welcome. Enter your Client ID on the left and click “Save” to begin.</p></div></div>'; }
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

# ========== /search ==========
@app.get("/search")
def search_endpoint(
    question: str = Query(..., min_length=3),
    top_k: int = Query(12, ge=1, le=30),
    level: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
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

# ========== /rag ==========
@app.get("/rag")
def rag_endpoint(
    question: str = Query(..., min_length=3),
    chat_id: Optional[str] = Query(None),
    top_k: int = Query(12, ge=1, le=30),
    level: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
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

# ========== /review ==========
@app.post("/review")
def review_endpoint(
    authorization: Optional[str] = Header(None),
    chat_id: Optional[str] = Form(None),
    question: str = Form(""),
    files: List[UploadFile] = File(default=[]),
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

        texts: List[str] = []
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
