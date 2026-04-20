# trust_rag_api.py
# Private Trust Fiduciary Advisor API
#
# Notable changes vs. prior revision:
#   - Message-level branching: `messages.parent_id` + `/chats/{id}/tree` endpoint,
#     plus `parent_id` on /rag & /review so a send truly forks from any point.
#   - /rag now returns `sources` so /draft and /chat actually have citations.
#   - /draft and /chat no longer HTTP-call the public hostname to reach themselves;
#     they call the internal `_run_rag()` helper directly.
#   - Modern widget: design tokens, dark mode, SVG icons, toasts, modal dialogs,
#     tree with visual connectors, proper path-based thread rendering.

from fastapi import FastAPI, Query, Header, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pinecone import Pinecone
from openai import OpenAI
import httpx, zipfile, io, re, os, time, traceback, sqlite3, json, uuid
from datetime import datetime
from collections import deque
from typing import Optional, List, Dict, Any, Tuple

# ========== ENV / SETUP ==========
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Strip proxy env so the OpenAI SDK doesn't inject `proxies=` automatically
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
           "OPENAI_PROXY", "OPENAI_HTTP_PROXY", "OPENAI_HTTPS_PROXY"):
    os.environ.pop(_k, None)
os.environ.setdefault("NO_PROXY", "*")

# Ignore a broken custom base url
if os.getenv("OPENAI_BASE_URL", "").strip().lower() in ("", "none", "null"):
    os.environ.pop("OPENAI_BASE_URL", None)

API_TOKEN         = os.getenv("API_TOKEN", "")
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

# Optional metrics
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

    CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id, updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, created_at);
    """)
    # Migration: add parent_id for message-level branching
    cur.execute("PRAGMA table_info(messages)")
    cols = [r[1] for r in cur.fetchall()]
    if "parent_id" not in cols:
        cur.execute("ALTER TABLE messages ADD COLUMN parent_id TEXT")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_id)")
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
        if parts:
            s = parts[-1]
    if not s.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY appears malformed.")
    return s

_openai_key = _clean_openai_key(os.getenv("OPENAI_API_KEY", ""))
openai_http = httpx.Client(timeout=120.0, trust_env=False)
client      = OpenAI(api_key=_openai_key, http_client=openai_http)

# ========== RAG HELPERS ==========
def _extract_snippet(meta: Dict[str, Any]) -> str:
    for k in ("text", "chunk", "content", "body", "passage"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _clean_title(title: str) -> str:
    t = (title or "Unknown")
    t = re.sub(r"^[Ll]\d[_\-:\s]+", "", t)
    t = re.sub(r"(?i)\bocr\b", "", t)
    t = re.sub(r"[0-9a-f]{8,}", "", t)
    if " -- " in t:
        first, *_ = t.split(" -- ")
        if len(first) >= 6:
            t = first
    return re.sub(r"\s+", " ", t.replace("_", " ")).strip(" -–—")

def _listing_title(meta: Dict[str, Any]) -> str:
    return _clean_title(meta.get("title") or meta.get("doc_parent") or "Unknown")

def _dedup_and_rank_sources(matches: List[Dict[str, Any]], top_k: int):
    rank = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}
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
            best[key] = {"title": title, "level": lvl, "page": page, "version": ver,
                         "score": score, "meta": meta}
    uniq = list(best.values())
    uniq.sort(key=lambda s: (rank.get(s["level"], 99), -s["score"]))
    return uniq[:top_k]

def _titles_only(uniq_sources: List[Dict[str, Any]]) -> List[str]:
    seen, out = set(), []
    for s in uniq_sources:
        t = s["title"]
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

# ========== SYNTHESIS ==========
def synthesize_html(question: str, uniq_sources: List[Dict[str, Any]], snippets: List[str]) -> str:
    if not snippets and not uniq_sources:
        return "<p>No relevant material found in the Trust-Law knowledge base.</p>"

    buf, used, kept = [], 0, 0
    for s in snippets:
        s = s.strip()
        if not s:
            continue
        if used + len(s) > MAX_CONTEXT_CHARS:
            break
        buf.append(s)
        used += len(s)
        kept += 1
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

    system_msg = (
        "You are the Private Trust Fiduciary Advisor. "
        "Always respond using clean, valid HTML (no markdown asterisks). "
        "Use <strong> for bold, <em> for italics, <h1>-<h6> for headings, "
        "<ul>/<ol> for lists, <pre><code> for code, and <a> for links. "
        "Prefer professional, legal-style formatting suitable for trust "
        "and fiduciary documents. If the content resembles a formal instrument "
        "(resolutions, certificates), format labels as plain lines like "
        "\"Date: …\", \"Trust: …\", \"Tax Year: …\", \"Location: …\"."
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

# ========== INTERNAL RAG CORE (shared by /rag, /draft, /chat) ==========
def _run_rag(question: str, top_k: int = 12, level: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute the retrieval + synthesis pipeline and return both HTML answer
    and citation metadata. Shared by /rag, /draft and /chat so we never
    HTTP-call ourselves across the network.
    """
    t0 = time.time()
    emb = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
    flt = {"doc_level": {"$eq": level}} if level else None
    res = idx.query(vector=emb, top_k=max(top_k, 12), include_metadata=True, filter=flt)
    matches = res["matches"] if isinstance(res, dict) else getattr(res, "matches", [])
    uniq = _dedup_and_rank_sources(matches, top_k=top_k)
    snippets = [s for s in (_extract_snippet(u.get("meta", {})) for u in uniq) if s]
    html = synthesize_html(question, uniq, snippets)
    sources = [
        {"title": s["title"], "level": s["level"], "page": s["page"],
         "version": s.get("version", ""), "score": s["score"]}
        for s in uniq
    ]
    return {
        "answer": html,
        "sources": sources,
        "titles": _titles_only(uniq),
        "t_ms": int((time.time() - t0) * 1000),
    }

# ========== CHAT & HISTORY (helpers) ==========
def ensure_chat(conn, user_id: str, chat_id: Optional[str]) -> str:
    cur = conn.cursor()
    if chat_id:
        row = cur.execute("SELECT id FROM chats WHERE id=? AND user_id=?",
                          (chat_id, user_id)).fetchone()
        if row:
            return chat_id
    new_id = str(uuid.uuid4())
    now = iso_now()
    cur.execute(
        "INSERT INTO chats (id, user_id, title, archived, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (new_id, user_id, "New chat", 0, now, now),
    )
    conn.commit()
    return new_id

def insert_message(conn, chat_id: str, user_id: Optional[str], role: str,
                   content_html: str, content_raw: Optional[str],
                   meta: Dict[str, Any], parent_id: Optional[str] = None) -> str:
    cur = conn.cursor()
    mid = str(uuid.uuid4())
    now = iso_now()
    cur.execute(
        """INSERT INTO messages (id, chat_id, user_id, role, content_html, content_raw,
                                 meta_json, created_at, parent_id)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (mid, chat_id, user_id, role, content_html, content_raw,
         json.dumps(meta or {}), now, parent_id),
    )
    cur.execute(
        "UPDATE chats SET updated_at=?, title=COALESCE(NULLIF(title,''),'New chat') WHERE id=?",
        (now, chat_id),
    )
    conn.commit()
    return mid

def _latest_leaf(conn, chat_id: str) -> Optional[str]:
    """Return the id of the chronologically latest message in a chat, or None."""
    row = conn.cursor().execute(
        "SELECT id FROM messages WHERE chat_id=? ORDER BY created_at DESC LIMIT 1",
        (chat_id,),
    ).fetchone()
    return row["id"] if row else None

# ========== CHAT CRUD API ==========
@app.post("/chats")
def create_chat(user_id: str = Depends(get_current_user)):
    conn = db()
    cid = ensure_chat(conn, user_id, None)
    conn.close()
    return {"chat_id": cid, "title": "New chat"}

@app.get("/chats")
def list_chats(page: int = 1, size: int = 30, user_id: str = Depends(get_current_user)):
    off = max((page - 1), 0) * size
    conn = db()
    rows = conn.cursor().execute(
        """SELECT id, title, archived, created_at, updated_at
           FROM chats WHERE user_id=? ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
        (user_id, size, off),
    ).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows], "page": page, "size": size}

@app.get("/chats/{chat_id}")
def get_chat(chat_id: str, user_id: str = Depends(get_current_user)):
    conn = db()
    cur = conn.cursor()
    chat = cur.execute(
        "SELECT id, title, archived, created_at, updated_at FROM chats WHERE id=? AND user_id=?",
        (chat_id, user_id),
    ).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(404, "Chat not found")
    msgs = cur.execute(
        """SELECT id, role, content_html, created_at, parent_id
           FROM messages WHERE chat_id=? ORDER BY created_at ASC LIMIT 1000""",
        (chat_id,),
    ).fetchall()
    conn.close()
    return {"chat": dict(chat), "messages": [dict(m) for m in msgs]}

@app.get("/chats/{chat_id}/messages")
def list_messages(chat_id: str, page: int = 1, size: int = 200,
                  user_id: str = Depends(get_current_user)):
    off = max((page - 1), 0) * size
    conn = db()
    cur = conn.cursor()
    ok = cur.execute("SELECT 1 FROM chats WHERE id=? AND user_id=?", (chat_id, user_id)).fetchone()
    if not ok:
        conn.close()
        raise HTTPException(404, "Chat not found")
    msgs = cur.execute(
        """SELECT id, role, content_html, content_raw, created_at, parent_id
           FROM messages WHERE chat_id=? ORDER BY created_at ASC LIMIT ? OFFSET ?""",
        (chat_id, size, off),
    ).fetchall()
    conn.close()
    return {"messages": [dict(m) for m in msgs], "page": page, "size": size}

@app.get("/chats/{chat_id}/tree")
def get_chat_tree(chat_id: str, user_id: str = Depends(get_current_user)):
    """
    Return the full branching tree for a chat: every message with its parent_id,
    so the widget can render the conversation graph.
    """
    conn = db()
    cur = conn.cursor()
    ok = cur.execute("SELECT 1 FROM chats WHERE id=? AND user_id=?", (chat_id, user_id)).fetchone()
    if not ok:
        conn.close()
        raise HTTPException(404, "Chat not found")
    rows = cur.execute(
        """SELECT id, role, content_html, content_raw, created_at, parent_id
           FROM messages WHERE chat_id=? ORDER BY created_at ASC""",
        (chat_id,),
    ).fetchall()
    conn.close()
    return {"chat_id": chat_id, "nodes": [dict(r) for r in rows]}

@app.post("/chats/{chat_id}/title")
def rename_chat(chat_id: str, title: str = Form(...), user_id: str = Depends(get_current_user)):
    conn = db()
    cur = conn.cursor()
    cur.execute("UPDATE chats SET title=?, updated_at=? WHERE id=? AND user_id=?",
                (title[:200], iso_now(), chat_id, user_id))
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(404, "Chat not found")
    conn.commit()
    conn.close()
    return {"ok": True}

@app.post("/chats/{chat_id}/archive")
def archive_chat(chat_id: str, archived: int = Form(1),
                 user_id: str = Depends(get_current_user)):
    conn = db()
    cur = conn.cursor()
    cur.execute("UPDATE chats SET archived=?, updated_at=? WHERE id=? AND user_id=?",
                (1 if archived else 0, iso_now(), chat_id, user_id))
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

# ========== WIDGET (redesigned) ==========
WIDGET_HTML = r"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Private Trust Fiduciary Advisor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  /* ============ Design tokens ============ */
  :root{
    --font-brand: "Cinzel", Georgia, serif;
    --font-sans:  "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono:  "JetBrains Mono", ui-monospace, Menlo, Monaco, monospace;

    --bg:            #fafaf7;
    --surface:       #ffffff;
    --surface-2:     #f5f4ef;
    --surface-3:     #eeece4;
    --text:          #1a1a1a;
    --text-muted:    #6b6862;
    --text-subtle:   #9e9a93;
    --border:        #e8e6e1;
    --border-strong: #d4d1c9;
    --accent:        #8b6f3e;
    --accent-hover:  #755a30;
    --accent-bg:     #f5f0e5;
    --user-bubble:   #f1ede2;
    --error:         #8b2635;
    --success:       #2d5a3d;
    --ring:          rgba(139,111,62,.22);
    --shadow-sm:     0 1px 2px rgba(15,15,15,.04);
    --shadow:        0 2px 8px rgba(15,15,15,.06), 0 1px 2px rgba(15,15,15,.04);
    --shadow-lg:     0 16px 48px rgba(15,15,15,.10), 0 2px 8px rgba(15,15,15,.04);

    --radius-sm: 6px;
    --radius:    10px;
    --radius-lg: 14px;

    --t-fast: 140ms cubic-bezier(.3,.1,.3,1);
    --t-base: 220ms cubic-bezier(.3,.1,.3,1);

    --rail-w: 288px;
    --tree-w: 340px;
  }
  [data-theme="dark"]{
    --bg:            #0b0c0e;
    --surface:       #15171a;
    --surface-2:     #1a1c20;
    --surface-3:     #22252a;
    --text:          #ececec;
    --text-muted:    #a8a5a0;
    --text-subtle:   #6b6862;
    --border:        #2a2c31;
    --border-strong: #3a3c42;
    --accent:        #c9a668;
    --accent-hover:  #d8b677;
    --accent-bg:     #2a2313;
    --user-bubble:   #1f2126;
    --error:         #c44554;
    --success:       #4ea36a;
    --ring:          rgba(201,166,104,.28);
    --shadow-sm:     0 1px 2px rgba(0,0,0,.35);
    --shadow:        0 2px 8px rgba(0,0,0,.45), 0 1px 2px rgba(0,0,0,.3);
    --shadow-lg:     0 16px 48px rgba(0,0,0,.6);
  }

  *,*::before,*::after{ box-sizing:border-box }
  html,body{ margin:0; padding:0; height:100% }
  body{
    background: var(--bg);
    color: var(--text);
    font: 15px/1.6 var(--font-sans);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    overflow: hidden;
  }
  ::selection{ background: var(--accent-bg); color: var(--text) }
  button{ font: inherit; color: inherit; cursor: pointer; background: transparent; border: 0; padding: 0 }
  input,textarea{ font: inherit; color: inherit }
  a{ color: var(--accent); text-decoration: underline; text-underline-offset: 2px }

  /* ============ Layout ============ */
  .app{
    display: grid;
    grid-template-columns: var(--rail-w) 1fr var(--tree-w);
    height: 100vh;
    transition: grid-template-columns var(--t-base);
  }
  body.sidebar-closed .app{ grid-template-columns: 0 1fr var(--tree-w) }
  body.tree-closed    .app{ grid-template-columns: var(--rail-w) 1fr 0 }
  body.sidebar-closed.tree-closed .app{ grid-template-columns: 0 1fr 0 }

  .rail, .tree-pane{
    background: var(--surface);
    display: flex; flex-direction: column;
    overflow: hidden;
    transition: border-color var(--t-base), opacity var(--t-base);
  }
  .rail{ border-right: 1px solid var(--border) }
  .tree-pane{ border-left: 1px solid var(--border) }
  body.sidebar-closed .rail, body.tree-closed .tree-pane{ border-color: transparent; opacity: 0; pointer-events: none }

  .main{ display: flex; flex-direction: column; min-width: 0; background: var(--bg) }

  /* ============ Edge reopen tabs ============ */
  .edge-tab{
    position: fixed; top: 50%; transform: translateY(-50%);
    width: 18px; height: 56px;
    background: var(--surface); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    box-shadow: var(--shadow-sm);
    color: var(--text-muted);
    z-index: 40;
    opacity: 0; pointer-events: none;
    transition: opacity var(--t-base), color var(--t-fast), background var(--t-fast);
  }
  .edge-tab:hover{ color: var(--accent); background: var(--surface-2) }
  .edge-tab.left{ left: 0; border-left: 0; border-radius: 0 10px 10px 0 }
  .edge-tab.right{ right: 0; border-right: 0; border-radius: 10px 0 0 10px }
  body.sidebar-closed .edge-tab.left{ opacity: 1; pointer-events: auto }
  body.tree-closed    .edge-tab.right{ opacity: 1; pointer-events: auto }

  /* ============ Rail ============ */
  .brand{
    padding: 18px 20px 16px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
  }
  .brand-title{
    font-family: var(--font-brand);
    font-weight: 500; font-size: 15px; letter-spacing: .08em;
    line-height: 1.2;
  }
  .brand-sub{
    font-size: 10.5px; letter-spacing: .18em; text-transform: uppercase;
    color: var(--text-subtle); margin-top: 2px;
  }

  .icon-btn{
    width: 32px; height: 32px; border-radius: var(--radius-sm);
    display: inline-flex; align-items: center; justify-content: center;
    color: var(--text-muted);
    transition: background var(--t-fast), color var(--t-fast);
  }
  .icon-btn:hover{ background: var(--surface-2); color: var(--text) }
  .icon-btn svg{ width: 16px; height: 16px }

  .rail-section{ padding: 14px 16px; border-bottom: 1px solid var(--border) }
  .rail-section:last-child{ border-bottom: 0 }

  .field-label{
    font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
    color: var(--text-subtle); margin-bottom: 8px;
  }
  .field{
    display: flex; align-items: center; gap: 8px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    padding: 8px 12px;
    transition: border-color var(--t-fast), box-shadow var(--t-fast);
  }
  .field:focus-within{ border-color: var(--accent); box-shadow: 0 0 0 3px var(--ring) }
  .field input{
    flex: 1; border: 0; outline: 0; background: transparent;
    font-size: 14px; min-width: 0;
  }
  .field input::placeholder{ color: var(--text-subtle) }
  .row{ display: flex; gap: 8px }
  .row + .row{ margin-top: 8px }

  .btn{
    display: inline-flex; align-items: center; justify-content: center; gap: 8px;
    height: 36px; padding: 0 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    color: var(--text);
    font-size: 13px; font-weight: 500;
    transition: background var(--t-fast), border-color var(--t-fast), color var(--t-fast), transform var(--t-fast);
  }
  .btn:hover{ background: var(--surface-2); border-color: var(--border-strong) }
  .btn:active{ transform: translateY(1px) }
  .btn.primary{ background: var(--text); color: var(--bg); border-color: var(--text) }
  .btn.primary:hover{ background: var(--accent); border-color: var(--accent); color: #fff }
  .btn.accent{ background: var(--accent); color: #fff; border-color: var(--accent) }
  .btn.accent:hover{ background: var(--accent-hover); border-color: var(--accent-hover) }
  .btn.ghost{ border-color: transparent; background: transparent }
  .btn.ghost:hover{ background: var(--surface-2) }
  .btn.full{ width: 100% }
  .btn.sm{ height: 30px; padding: 0 10px; font-size: 12px }
  .btn svg{ width: 14px; height: 14px }

  .msg-line{ margin-top: 8px; font-size: 12px; color: var(--text-muted); min-height: 16px }
  .msg-line.error{ color: var(--error) }
  .msg-line.success{ color: var(--success) }

  /* Chat list */
  .chat-list{ flex: 1; overflow: auto; padding: 8px 10px 16px }
  .chat-list-header{
    padding: 6px 8px 8px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .chat-item{
    display: block;
    padding: 10px 12px;
    border: 1px solid transparent;
    border-radius: var(--radius);
    margin-bottom: 4px;
    cursor: pointer;
    transition: background var(--t-fast), border-color var(--t-fast);
    position: relative;
  }
  .chat-item:hover{ background: var(--surface-2) }
  .chat-item.active{ background: var(--accent-bg); border-color: var(--border-strong) }
  .chat-item .title{
    font-size: 13.5px; font-weight: 500;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    padding-right: 52px;
  }
  .chat-item .meta{ font-size: 11px; color: var(--text-subtle); margin-top: 2px }
  .chat-item .actions{
    position: absolute; top: 8px; right: 8px;
    display: flex; gap: 2px;
    opacity: 0; transition: opacity var(--t-fast);
  }
  .chat-item:hover .actions, .chat-item.active .actions{ opacity: 1 }
  .chat-item .actions .icon-btn{ width: 26px; height: 26px }
  .chat-item .actions .icon-btn svg{ width: 13px; height: 13px }

  .empty-state{
    padding: 20px 16px; text-align: center; color: var(--text-subtle); font-size: 13px;
  }

  /* ============ Main ============ */
  .main-header{
    height: 56px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center;
    padding: 0 18px;
    gap: 12px;
    background: var(--surface);
    flex-shrink: 0;
  }
  .main-title{
    font-family: var(--font-brand);
    font-weight: 500; font-size: 15px; letter-spacing: .08em;
    flex: 1; text-align: center;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .main-title .muted{ color: var(--text-subtle); font-weight: 400 }

  .thread-scroll{ flex: 1; overflow: auto; scroll-behavior: smooth }
  .thread{
    max-width: 860px; margin: 0 auto;
    padding: 32px 20px 180px;
    display: flex; flex-direction: column; gap: 22px;
  }
  .msg{ display: flex; flex-direction: column; animation: fadeUp .28s ease }
  @keyframes fadeUp{ from{ opacity: 0; transform: translateY(4px) } to{ opacity: 1; transform: none } }

  .msg .bubble-wrap{ display: flex; align-items: flex-start; gap: 10px }
  .msg.user .bubble-wrap{ justify-content: flex-end }
  .avatar{
    width: 28px; height: 28px; border-radius: 50%;
    background: var(--surface-3);
    color: var(--text-muted);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600; letter-spacing: .05em;
    flex-shrink: 0; margin-top: 2px;
    border: 1px solid var(--border);
  }
  .msg.advisor .avatar{ background: var(--accent-bg); color: var(--accent); border-color: var(--border-strong) }

  .bubble-content{ min-width: 0; max-width: 100% }
  .msg-meta{
    font-size: 11px; color: var(--text-subtle);
    letter-spacing: .05em; margin-bottom: 6px;
    display: flex; align-items: center; gap: 8px;
  }
  .msg-meta .branch-mark{
    display: inline-flex; align-items: center; gap: 4px;
    padding: 1px 8px; border-radius: 999px;
    background: var(--surface-3); color: var(--text-muted);
    font-size: 10px;
  }
  .msg-meta .branch-mark svg{ width: 10px; height: 10px }

  .bubble{
    padding: 14px 16px;
    border-radius: var(--radius-lg);
    line-height: 1.65;
    overflow-wrap: break-word;
  }
  .msg.user .bubble{
    background: var(--user-bubble);
    border: 1px solid var(--border);
    max-width: 720px;
    white-space: pre-wrap;
  }
  .msg.advisor .bubble{
    background: var(--surface);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
  }
  .msg.advisor .bubble.thinking{
    padding: 12px 16px;
    display: inline-flex; align-items: center; gap: 10px;
    color: var(--text-muted); font-style: italic;
  }
  .dots{ display: inline-flex; gap: 3px }
  .dots span{
    width: 5px; height: 5px; border-radius: 50%;
    background: currentColor; opacity: .4;
    animation: dot 1.2s infinite ease-in-out;
  }
  .dots span:nth-child(2){ animation-delay: .15s }
  .dots span:nth-child(3){ animation-delay: .3s }
  @keyframes dot{ 0%,80%,100%{ opacity: .2 } 40%{ opacity: .95 } }

  /* Message actions */
  .bubble-actions{
    display: flex; gap: 2px; margin-top: 6px;
    opacity: 0; transition: opacity var(--t-fast);
  }
  .msg:hover .bubble-actions{ opacity: 1 }
  .bubble-actions .icon-btn{ width: 28px; height: 28px; color: var(--text-subtle) }
  .bubble-actions .icon-btn:hover{ color: var(--accent) }
  .bubble-actions .icon-btn svg{ width: 13px; height: 13px }

  /* Typography inside bubble */
  .bubble h1{ font-family: var(--font-brand); font-weight: 500; font-size: 22px; letter-spacing: .05em; margin: 4px 0 12px }
  .bubble h2{ font-family: var(--font-brand); font-weight: 500; font-size: 18px; letter-spacing: .04em; margin: 18px 0 8px }
  .bubble h3{ font-weight: 600; font-size: 15.5px; margin: 14px 0 6px }
  .bubble h4,.bubble h5,.bubble h6{ font-weight: 600; font-size: 14px; margin: 12px 0 4px }
  .bubble p{ margin: 8px 0 }
  .bubble p:first-child{ margin-top: 0 }
  .bubble p:last-child{ margin-bottom: 0 }
  .bubble ul,.bubble ol{ margin: 8px 0 8px 22px }
  .bubble li{ margin: 3px 0 }
  .bubble strong{ font-weight: 600 }
  .bubble code{
    font-family: var(--font-mono); font-size: .9em;
    background: var(--surface-2); border: 1px solid var(--border);
    padding: 1px 6px; border-radius: 4px;
  }
  .bubble pre{
    background: var(--surface-2); border: 1px solid var(--border);
    padding: 12px 14px; border-radius: var(--radius);
    font-family: var(--font-mono); font-size: 13px;
    overflow: auto; margin: 10px 0;
  }
  .bubble blockquote{
    border-left: 3px solid var(--accent);
    padding: 4px 14px; margin: 10px 0;
    background: var(--surface-2);
    color: var(--text-muted);
  }
  .bubble table{ width: 100%; border-collapse: collapse; margin: 10px 0 }
  .bubble th,.bubble td{ border: 1px solid var(--border); padding: 8px 10px; text-align: left; font-size: 13.5px }
  .bubble th{ background: var(--surface-2); font-weight: 600 }

  /* Welcome card */
  .welcome{
    text-align: center;
    padding: 72px 20px 20px;
    color: var(--text-muted);
  }
  .welcome-title{
    font-family: var(--font-brand);
    font-weight: 500; font-size: 28px; letter-spacing: .08em;
    color: var(--text);
    margin-bottom: 8px;
  }
  .welcome-sub{ font-size: 14px; max-width: 540px; margin: 0 auto; line-height: 1.7 }
  .suggestions{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 10px; max-width: 720px; margin: 28px auto 0;
  }
  .suggestion{
    padding: 14px 16px; border: 1px solid var(--border);
    border-radius: var(--radius); background: var(--surface);
    text-align: left; cursor: pointer; font-size: 13.5px;
    transition: border-color var(--t-fast), background var(--t-fast), transform var(--t-fast);
    color: var(--text);
  }
  .suggestion:hover{ border-color: var(--accent); background: var(--accent-bg); transform: translateY(-1px) }
  .suggestion .title{ font-weight: 600; margin-bottom: 4px }
  .suggestion .desc{ color: var(--text-muted); font-size: 12.5px }

  /* ============ Composer ============ */
  .composer-wrap{
    position: fixed; bottom: 0;
    left: var(--rail-w); right: var(--tree-w);
    padding: 18px 20px 22px;
    background: linear-gradient(to top, var(--bg) 60%, transparent);
    transition: left var(--t-base), right var(--t-base);
    pointer-events: none;
  }
  body.sidebar-closed .composer-wrap{ left: 0 }
  body.tree-closed    .composer-wrap{ right: 0 }

  .composer{
    max-width: 860px; margin: 0 auto;
    pointer-events: auto;
  }
  .branch-indicator{
    display: none;
    align-items: center; gap: 8px;
    padding: 6px 12px; margin-bottom: 8px;
    background: var(--accent-bg); border: 1px solid var(--border-strong);
    border-radius: var(--radius);
    font-size: 12px; color: var(--text-muted);
  }
  .branch-indicator.show{ display: inline-flex }
  .branch-indicator svg{ width: 12px; height: 12px; color: var(--accent) }
  .branch-indicator button{
    margin-left: auto; color: var(--text-muted); font-size: 11px;
    text-decoration: underline;
  }

  .composer-bar{
    display: flex; align-items: flex-end; gap: 8px;
    padding: 10px 10px 10px 16px;
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 20px;
    box-shadow: var(--shadow);
    transition: border-color var(--t-fast), box-shadow var(--t-fast);
  }
  .composer-bar:focus-within{ border-color: var(--accent); box-shadow: 0 0 0 3px var(--ring), var(--shadow) }

  .composer-input{
    flex: 1;
    min-height: 28px; max-height: 200px;
    outline: 0;
    font: 15px/1.55 var(--font-sans); color: var(--text);
    padding: 6px 0;
    overflow-y: auto;
    overflow-wrap: break-word;
    word-break: break-word;
  }
  .composer-input:empty::before{
    content: attr(data-placeholder);
    color: var(--text-subtle);
  }

  .composer-actions{ display: flex; align-items: center; gap: 4px }
  .composer-actions .icon-btn{ width: 36px; height: 36px; border-radius: 10px }
  .send-btn{
    width: 38px; height: 38px; border-radius: 10px;
    background: var(--text); color: var(--bg);
    display: inline-flex; align-items: center; justify-content: center;
    transition: background var(--t-fast), transform var(--t-fast);
  }
  .send-btn:hover{ background: var(--accent); color: #fff }
  .send-btn:disabled{ background: var(--surface-3); color: var(--text-subtle); cursor: not-allowed }
  .send-btn svg{ width: 16px; height: 16px }

  .composer-foot{
    max-width: 860px; margin: 8px auto 0;
    font-size: 11.5px; color: var(--text-subtle);
    display: flex; align-items: center; justify-content: space-between; gap: 10px;
  }
  .file-chips{ display: flex; flex-wrap: wrap; gap: 6px }
  .chip{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 8px; border: 1px solid var(--border);
    border-radius: 999px; background: var(--surface);
    font-size: 11px;
  }
  .chip svg{ width: 11px; height: 11px }
  .chip button{ color: var(--text-subtle); line-height: 1 }
  .chip button:hover{ color: var(--error) }

  #file{ display: none }

  /* ============ Tree pane ============ */
  .tree-head{
    height: 56px; padding: 0 16px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .tree-title{
    font-size: 12px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--text-muted); font-weight: 600;
  }
  .tree-body{ flex: 1; overflow: auto; padding: 14px 12px 24px }

  .tree-empty{ padding: 28px 16px; text-align: center; color: var(--text-subtle); font-size: 13px }

  /* Tree nodes — indent + connector */
  .tree-node{
    position: relative;
    padding: 6px 8px 6px 22px;
    margin: 2px 0;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background var(--t-fast);
  }
  .tree-node:hover{ background: var(--surface-2) }
  .tree-node.active{ background: var(--accent-bg) }
  .tree-node.in-path{ background: linear-gradient(to right, var(--surface-2), transparent 60%) }

  /* vertical connector line */
  .tree-node::before{
    content: ""; position: absolute;
    left: 10px; top: 0; bottom: 0;
    width: 1px; background: var(--border);
  }
  /* horizontal tick */
  .tree-node::after{
    content: ""; position: absolute;
    left: 10px; top: 18px; width: 10px; height: 1px;
    background: var(--border);
  }
  .tree-node[data-depth="0"]::before,
  .tree-node[data-depth="0"]::after{ display: none }
  .tree-node[data-depth="0"]{ padding-left: 8px }

  .tree-node .dot{
    position: absolute; left: 6px; top: 13px;
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--surface); border: 1.5px solid var(--border-strong);
    z-index: 1;
  }
  .tree-node[data-depth="0"] .dot{ left: auto; position: static; margin-right: 8px; display: inline-block; vertical-align: middle }
  .tree-node.active .dot,
  .tree-node.in-path .dot{ background: var(--accent); border-color: var(--accent) }
  .tree-node.assistant .dot{ background: var(--surface); border-color: var(--accent); border-style: dashed }
  .tree-node.assistant.active .dot{ background: var(--accent) }

  .tree-row{
    display: flex; align-items: center; gap: 8px;
    min-width: 0;
  }
  .tree-label{
    flex: 1; font-size: 13px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    color: var(--text);
  }
  .tree-node.assistant .tree-label{ color: var(--text-muted); font-size: 12.5px }
  .tree-node .role-tag{
    font-size: 9.5px; letter-spacing: .1em; text-transform: uppercase;
    color: var(--text-subtle);
    margin-right: 6px;
  }
  .tree-actions{
    display: flex; gap: 2px;
    opacity: 0; transition: opacity var(--t-fast);
  }
  .tree-node:hover .tree-actions,
  .tree-node.active .tree-actions{ opacity: 1 }
  .tree-actions .icon-btn{ width: 24px; height: 24px }
  .tree-actions .icon-btn svg{ width: 12px; height: 12px }

  /* ============ Modal ============ */
  .modal-root{
    position: fixed; inset: 0; z-index: 60;
    display: none; align-items: center; justify-content: center;
    background: rgba(15,15,15,.34);
    backdrop-filter: blur(4px);
    animation: fadeIn .18s ease;
  }
  .modal-root.show{ display: flex }
  .modal{
    width: min(420px, 92vw);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-lg);
    padding: 22px 22px 18px;
  }
  .modal h3{ font-family: var(--font-brand); font-weight: 500; font-size: 17px; letter-spacing: .05em; margin-bottom: 6px }
  .modal p{ color: var(--text-muted); font-size: 13.5px; margin-bottom: 14px }
  .modal .field{ margin-bottom: 14px }
  .modal-actions{ display: flex; justify-content: flex-end; gap: 8px }
  @keyframes fadeIn{ from{ opacity: 0 } to{ opacity: 1 } }

  /* ============ Toasts ============ */
  .toast-root{
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    display: flex; flex-direction: column; gap: 8px; z-index: 70;
    pointer-events: none;
  }
  .toast{
    background: var(--surface); border: 1px solid var(--border);
    box-shadow: var(--shadow-lg);
    border-radius: var(--radius);
    padding: 10px 14px; font-size: 13px;
    display: flex; align-items: center; gap: 10px;
    animation: toastIn .22s ease;
    pointer-events: auto;
  }
  .toast.error{ border-color: var(--error); color: var(--error) }
  .toast.success{ border-color: var(--success); color: var(--success) }
  .toast svg{ width: 14px; height: 14px }
  @keyframes toastIn{ from{ opacity: 0; transform: translateY(6px) } to{ opacity: 1; transform: none } }

  /* ============ Scrollbars ============ */
  ::-webkit-scrollbar{ width: 10px; height: 10px }
  ::-webkit-scrollbar-track{ background: transparent }
  ::-webkit-scrollbar-thumb{ background: var(--border-strong); border: 2px solid transparent; background-clip: padding-box; border-radius: 10px }
  ::-webkit-scrollbar-thumb:hover{ background: var(--text-subtle); background-clip: padding-box; border: 2px solid transparent }

  /* ============ Mobile ============ */
  @media (max-width: 980px){
    :root{ --rail-w: 0px; --tree-w: 0px }
    .edge-tab{ display: none }
    .composer-wrap{ left: 0; right: 0 }
  }
</style>
</head>
<body>
  <div class="app">
    <!-- ============ LEFT RAIL ============ -->
    <aside class="rail" id="rail">
      <div class="brand">
        <div>
          <div class="brand-title">Fiduciary Advisor</div>
          <div class="brand-sub">Private Trust Counsel</div>
        </div>
        <button class="icon-btn" id="btn-rail-collapse" title="Collapse sidebar" aria-label="Collapse sidebar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg>
        </button>
      </div>

      <div class="rail-section">
        <div class="field-label">Client Profile</div>
        <div class="row">
          <div class="field">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:14px;height:14px;color:var(--text-subtle)"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
            <input id="pf-id" placeholder="Client ID (required)"/>
          </div>
        </div>
        <div class="row">
          <div class="field">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:14px;height:14px;color:var(--text-subtle)"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><path d="M22 6l-10 7L2 6"/></svg>
            <input id="pf-email" placeholder="Email (optional)"/>
          </div>
        </div>
        <div class="row" style="margin-top:10px">
          <button class="btn primary full" id="pf-save">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
            Save
          </button>
          <button class="btn ghost" id="pf-logout" title="Log out">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
          </button>
        </div>
        <div class="msg-line" id="pf-msg"></div>
      </div>

      <div class="rail-section" style="padding-top:10px; padding-bottom:10px">
        <button class="btn accent full" id="btn-newchat">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          New Chat
        </button>
      </div>

      <div class="chat-list-header" style="padding: 10px 16px 0">
        <div class="field-label" style="margin:0">Conversations</div>
      </div>
      <div class="chat-list" id="chatlist"></div>

      <div class="rail-section" style="display:flex; justify-content:space-between; align-items:center">
        <div style="font-size: 11px; color: var(--text-subtle); letter-spacing: .06em">api.fctadvisor.com</div>
        <button class="icon-btn" id="btn-theme" title="Toggle theme">
          <svg id="ic-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>
        </button>
      </div>
    </aside>

    <!-- ============ MAIN ============ -->
    <main class="main">
      <div class="main-header">
        <button class="icon-btn" id="btn-toggle-rail-m" title="Toggle sidebar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
        </button>
        <div class="main-title" id="chat-title">
          <span class="muted">Advisor</span>
        </div>
        <button class="icon-btn" id="btn-toggle-tree-m" title="Toggle tree">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="2"/><circle cx="18" cy="12" r="2"/><circle cx="6" cy="18" r="2"/><path d="M8 6h4a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H8"/><path d="M16 12H8"/></svg>
        </button>
      </div>

      <div class="thread-scroll" id="thread-scroll">
        <div class="thread" id="thread"></div>
      </div>
    </main>

    <!-- ============ TREE PANE ============ -->
    <aside class="tree-pane" id="treePane">
      <div class="tree-head">
        <div class="tree-title">Conversation Tree</div>
        <button class="icon-btn" id="btn-tree-collapse" title="Collapse tree">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"/></svg>
        </button>
      </div>
      <div class="tree-body" id="tree-body">
        <div class="tree-empty">Send a message to begin a thread.</div>
      </div>
    </aside>

    <!-- Edge tabs -->
    <button class="edge-tab left"  id="left-tab"  title="Open sidebar" aria-label="Open sidebar">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"/></svg>
    </button>
    <button class="edge-tab right" id="right-tab" title="Open tree" aria-label="Open tree">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg>
    </button>
  </div>

  <!-- Composer -->
  <div class="composer-wrap">
    <div class="composer">
      <div class="branch-indicator" id="branch-ind">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>
        <span id="branch-ind-text">Branching from a previous message…</span>
        <button id="branch-clear">Cancel</button>
      </div>
      <div class="composer-bar">
        <div class="composer-input" id="input" contenteditable="true" data-placeholder="Pose an inquiry to the Advisor… (Shift+Enter for new line)"></div>
        <div class="composer-actions">
          <button class="icon-btn" id="attach" title="Attach files">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
          </button>
          <input id="file" type="file" multiple accept=".pdf,.txt,.docx"/>
          <button class="send-btn" id="send" title="Send" aria-label="Send">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          </button>
        </div>
      </div>
      <div class="composer-foot">
        <div id="foot-hint">Formal trust, fiduciary, and contractual analysis with strategic guidance.</div>
        <div class="file-chips" id="file-chips"></div>
      </div>
    </div>
  </div>

  <!-- Modal root -->
  <div class="modal-root" id="modal-root">
    <div class="modal" id="modal">
      <h3 id="modal-title">Rename</h3>
      <p id="modal-desc"></p>
      <div class="field">
        <input id="modal-input" placeholder=""/>
      </div>
      <div class="modal-actions">
        <button class="btn ghost" id="modal-cancel">Cancel</button>
        <button class="btn primary" id="modal-ok">Confirm</button>
      </div>
    </div>
  </div>

  <!-- Toast root -->
  <div class="toast-root" id="toast-root"></div>

<script>
/* ===================== State ===================== */
const LS = {
  UI:   'fct.ui.v2',
  USER: 'fct.userId',
  EMAIL:'fct.email',
  THEME:'fct.theme'
};
const ui = (() => {
  const def = { sidebarOpen: true, treeOpen: true };
  try { return Object.assign(def, JSON.parse(localStorage.getItem(LS.UI) || '{}')); }
  catch(_){ return def; }
})();
const saveUI = () => localStorage.setItem(LS.UI, JSON.stringify(ui));

const state = {
  userId:  localStorage.getItem(LS.USER) || '',
  email:   localStorage.getItem(LS.EMAIL) || '',
  theme:   localStorage.getItem(LS.THEME) || 'light',
  currentChatId: null,
  currentChatTitle: '',
  treeNodes: [],         // raw message list for current chat
  activeLeafId: null,    // the leaf representing the currently-viewed path
  branchFromId: null,    // if set, next send will fork under this msg
};

/* ===================== Theme ===================== */
function applyTheme(){
  document.documentElement.setAttribute('data-theme', state.theme);
  const ic = document.getElementById('ic-sun');
  if (state.theme === 'dark'){
    ic.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
  } else {
    ic.innerHTML = '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>';
  }
}

/* ===================== Layout ===================== */
function applyLayout(){
  document.body.classList.toggle('sidebar-closed', !ui.sidebarOpen);
  document.body.classList.toggle('tree-closed', !ui.treeOpen);
}

/* ===================== Elements ===================== */
const elThread    = document.getElementById('thread');
const elThreadS   = document.getElementById('thread-scroll');
const elChatTitle = document.getElementById('chat-title');
const elInput     = document.getElementById('input');
const elSend      = document.getElementById('send');
const elAttach    = document.getElementById('attach');
const elFile      = document.getElementById('file');
const elChips     = document.getElementById('file-chips');
const elPfId      = document.getElementById('pf-id');
const elPfEmail   = document.getElementById('pf-email');
const elPfMsg     = document.getElementById('pf-msg');
const elChatList  = document.getElementById('chatlist');
const elTreeBody  = document.getElementById('tree-body');
const elBranchInd = document.getElementById('branch-ind');
const elBranchText= document.getElementById('branch-ind-text');

elPfId.value = state.userId || '';
elPfEmail.value = state.email || '';

/* ===================== Toasts ===================== */
function toast(msg, kind='info'){
  const root = document.getElementById('toast-root');
  const el = document.createElement('div');
  el.className = 'toast ' + (kind || '');
  const icon = kind === 'error'
    ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
    : kind === 'success'
    ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
    : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';
  el.innerHTML = icon + '<span>' + msg + '</span>';
  root.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transform = 'translateY(6px)'; }, 2400);
  setTimeout(() => el.remove(), 2700);
}

/* ===================== Modal ===================== */
function modal({ title, desc, placeholder = '', value = '', okLabel = 'Confirm' } = {}){
  return new Promise(resolve => {
    const root = document.getElementById('modal-root');
    document.getElementById('modal-title').textContent = title || 'Confirm';
    document.getElementById('modal-desc').textContent  = desc || '';
    const input = document.getElementById('modal-input');
    input.placeholder = placeholder;
    input.value = value;
    document.getElementById('modal-ok').textContent = okLabel;
    root.classList.add('show');
    setTimeout(() => input.focus(), 50);

    function close(val){
      root.classList.remove('show');
      document.getElementById('modal-ok').onclick = null;
      document.getElementById('modal-cancel').onclick = null;
      input.onkeydown = null;
      resolve(val);
    }
    document.getElementById('modal-ok').onclick = () => close(input.value.trim());
    document.getElementById('modal-cancel').onclick = () => close(null);
    input.onkeydown = e => {
      if (e.key === 'Enter') close(input.value.trim());
      if (e.key === 'Escape') close(null);
    };
  });
}
function confirmDialog({ title, desc, okLabel = 'Confirm' } = {}){
  return new Promise(resolve => {
    const root = document.getElementById('modal-root');
    document.getElementById('modal-title').textContent = title || 'Confirm';
    document.getElementById('modal-desc').textContent  = desc || '';
    const input = document.getElementById('modal-input');
    input.style.display = 'none'; input.parentElement.style.display = 'none';
    const ok = document.getElementById('modal-ok');
    ok.textContent = okLabel;
    root.classList.add('show');

    function restore(){ input.style.display = ''; input.parentElement.style.display = '' }
    function close(val){
      root.classList.remove('show'); restore();
      ok.onclick = null;
      document.getElementById('modal-cancel').onclick = null;
      resolve(val);
    }
    ok.onclick = () => close(true);
    document.getElementById('modal-cancel').onclick = () => close(false);
  });
}

/* ===================== Fetch helpers ===================== */
function hdrs(){
  const h = {};
  if (state.userId) h['X-User-Id'] = state.userId;
  return h;
}
async function api(method, url, { body, form } = {}){
  const opts = { method, headers: { ...hdrs() } };
  if (form){ opts.body = form; }
  else if (body){
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(url, opts);
  if (!r.ok){
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch(_){}
    throw new Error(msg + ' (' + r.status + ')');
  }
  return r.json();
}

/* ===================== Formatting ===================== */
function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])) }

function mdToHtml(md){
  if (!md) return '';
  if (/<\w+[^>]*>/.test(md)) return md;
  let h = md.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  h = h.replace(/```([\s\S]*?)```/g, (_, c) => '<pre><code>' + c + '</code></pre>');
  h = h.replace(/`([^`]+?)`/g, '<code>$1</code>');
  h = h.replace(/^######\s+(.*)$/gm, '<h6>$1</h6>')
       .replace(/^#####\s+(.*)$/gm,  '<h5>$1</h5>')
       .replace(/^####\s+(.*)$/gm,   '<h4>$1</h4>')
       .replace(/^###\s+(.*)$/gm,    '<h3>$1</h3>')
       .replace(/^##\s+(.*)$/gm,     '<h2>$1</h2>')
       .replace(/^#\s+(.*)$/gm,      '<h1>$1</h1>');
  h = h.replace(/^>\s?(.*)$/gm, '<blockquote>$1</blockquote>');
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
       .replace(/__(.+?)__/g,     '<strong>$1</strong>')
       .replace(/\*(?!\s)(.+?)\*/g, '<em>$1</em>')
       .replace(/_(?!\s)(.+?)_/g,  '<em>$1</em>');
  h = h.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  h = h.replace(/\n{2,}/g, '</p><p>').replace(/^(?!<h\d|<ul|<ol|<pre|<hr|<p|<blockquote|<table)(.+)$/gm, '<p>$1</p>');
  return h;
}

function normalizeTrustDoc(html){
  let out = html;
  out = out.replace(/<p>\s*<strong>\s*([A-Z0-9][A-Z0-9\s\-&,.'()]+?)\s*<\/strong>\s*<\/p>/g, '<h2>$1</h2>');
  const labelMap = [
    [/\*\*\s*TRUST\s*NAME\s*:\s*\*\*/gi, 'Trust: '],
    [/\*\*\s*DATE\s*:\s*\*\*/gi,         'Date: '],
    [/\*\*\s*TAX\s*YEAR\s*:\s*\*\*/gi,   'Tax Year: '],
    [/\*\*\s*TRUSTEE\(S\)\s*:\s*\*\*/gi, 'Trustee(s): '],
    [/\*\*\s*LOCATION\s*:\s*\*\*/gi,     'Location: '],
  ];
  labelMap.forEach(([re, rep]) => { out = out.replace(re, rep) });
  return out;
}

/* ===================== Thread rendering ===================== */
function now(){ return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }

function pathFromLeaf(leafId){
  // walk up parent_ids from leaf to root, collecting messages
  if (!leafId) return [];
  const byId = {};
  state.treeNodes.forEach(n => byId[n.id] = n);
  const out = [];
  let cur = byId[leafId];
  const guard = new Set();
  while (cur && !guard.has(cur.id)){
    guard.add(cur.id);
    out.unshift(cur);
    cur = cur.parent_id ? byId[cur.parent_id] : null;
  }
  return out;
}

function leaves(){
  // ids that no other node claims as parent
  const hasChild = new Set();
  state.treeNodes.forEach(n => { if (n.parent_id) hasChild.add(n.parent_id) });
  return state.treeNodes.filter(n => !hasChild.has(n.id));
}

function plainText(html){
  const d = document.createElement('div');
  d.innerHTML = html || '';
  return (d.textContent || '').replace(/\s+/g, ' ').trim();
}

function renderThread(){
  elThread.innerHTML = '';
  const path = pathFromLeaf(state.activeLeafId);
  if (!path.length){
    elThread.innerHTML = welcomeHTML();
    bindSuggestions();
    return;
  }
  path.forEach(n => {
    const isUser = n.role === 'user';
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + (isUser ? 'user' : 'advisor');
    wrap.dataset.id = n.id;

    const avatar = isUser
      ? '<div class="avatar">You</div>'
      : '<div class="avatar">FA</div>';

    const siblings = state.treeNodes.filter(m => m.parent_id === n.parent_id);
    const hasBranches = siblings.length > 1;
    const branchMark = hasBranches
      ? `<span class="branch-mark" title="${siblings.length} alternatives">
           <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>
           ${siblings.length}
         </span>`
      : '';

    const time = new Date(n.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const meta = `<div class="msg-meta">${isUser ? 'You' : 'Advisor'} · ${time}${branchMark ? ' · ' + branchMark : ''}</div>`;

    let body = n.content_html || '';
    if (!/<\w+/.test(body)) body = mdToHtml(body);
    body = normalizeTrustDoc(body);

    // Branch semantics:
    //   - On a user message: "ask alternative" => new message is a sibling (same parent_id)
    //   - On an advisor message: "continue differently" => new message is a child (parent_id = advisor)
    const branchTitle  = isUser ? 'Ask an alternative (sibling branch)' : 'Continue with an alternative question';
    const branchAction = isUser ? 'alt' : 'branch';
    const actions = `
      <div class="bubble-actions">
        ${!isUser ? `
          <button class="icon-btn" data-act="copy" title="Copy response">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          </button>` : ''}
        <button class="icon-btn" data-act="${branchAction}" title="${branchTitle}">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>
        </button>
      </div>`;

    wrap.innerHTML = `
      <div class="bubble-wrap">
        ${isUser ? '' : avatar}
        <div class="bubble-content">
          ${meta}
          <div class="bubble">${body}</div>
          ${actions}
        </div>
        ${isUser ? avatar : ''}
      </div>`;

    elThread.appendChild(wrap);
  });

  // bind actions
  elThread.querySelectorAll('[data-act]').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const act = btn.dataset.act;
      const id  = btn.closest('.msg').dataset.id;
      if (act === 'copy')  { copyMessage(id) }
      if (act === 'branch'){ setBranchFrom(id, 'child') }       // next msg parent = this advisor
      if (act === 'alt')   { setBranchFrom(id, 'sibling') }     // next msg parent = this user's parent
    });
  });

  // scroll to bottom on fresh render
  requestAnimationFrame(() => { elThreadS.scrollTop = elThreadS.scrollHeight });
}

function welcomeHTML(){
  const suggestions = [
    { title: 'Draft a trustee resolution',
      desc: 'For a quarterly distribution under a non-grantor trust.',
      prompt: 'Draft a trustee resolution authorizing a quarterly distribution under a non-grantor irrevocable trust.' },
    { title: '§643(b) DNI computation',
      desc: 'Walk through DNI calculation for a complex trust.',
      prompt: 'Walk through §643(b) DNI computation for a complex non-grantor trust with capital gains retained in corpus.' },
    { title: '508(c)(1)(A) defense package',
      desc: 'Outline the 14-point IRS church test documentation.',
      prompt: 'Outline the 14-point IRS church test and the evidence needed for each element in a 508(c)(1)(A) defense package.' },
    { title: 'Grantor trust risk review',
      desc: 'Identify clauses that trigger §§671–679 treatment.',
      prompt: 'Review the typical clauses that risk triggering grantor trust treatment under IRC §§671–679 and how to draft around them.' },
  ];
  const cards = suggestions.map(s =>
    `<button class="suggestion" data-prompt="${escapeHtml(s.prompt)}">
       <div class="title">${s.title}</div>
       <div class="desc">${s.desc}</div>
     </button>`).join('');
  return `
    <div class="welcome">
      <div class="welcome-title">Private Trust Fiduciary Advisor</div>
      <div class="welcome-sub">Formal trust, fiduciary, and contractual analysis with strategic guidance, grounded in your private knowledge base.</div>
      <div class="suggestions">${cards}</div>
    </div>`;
}
function bindSuggestions(){
  document.querySelectorAll('.suggestion').forEach(b => {
    b.addEventListener('click', () => {
      elInput.textContent = b.dataset.prompt;
      elInput.focus();
      placeCaretAtEnd(elInput);
    });
  });
}

/* ===================== Tree rendering ===================== */
function renderTree(){
  if (!state.currentChatId){
    elTreeBody.innerHTML = '<div class="tree-empty">Select or start a conversation to see its tree.</div>';
    return;
  }
  if (!state.treeNodes.length){
    elTreeBody.innerHTML = '<div class="tree-empty">No messages yet. Send a question to begin.</div>';
    return;
  }

  // Build children map
  const byParent = {};
  state.treeNodes.forEach(n => {
    const p = n.parent_id || 'ROOT';
    (byParent[p] = byParent[p] || []).push(n);
  });
  Object.values(byParent).forEach(arr =>
    arr.sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
  );

  const path = new Set(pathFromLeaf(state.activeLeafId).map(n => n.id));

  const out = [];
  function walk(parent, depth){
    (byParent[parent] || []).forEach(n => {
      const isActive = n.id === state.activeLeafId;
      const inPath = path.has(n.id);
      const label = (() => {
        const t = plainText(n.content_html);
        if (!t) return n.role === 'user' ? 'User message' : 'Advisor response';
        return t.length > 80 ? t.slice(0, 80) + '…' : t;
      })();
      const classes = ['tree-node', n.role === 'user' ? 'user' : 'assistant'];
      if (isActive) classes.push('active');
      if (inPath)   classes.push('in-path');
      const role = n.role === 'user' ? 'U' : 'A';
      out.push(`
        <div class="${classes.join(' ')}" data-id="${n.id}" data-depth="${depth}">
          <span class="dot"></span>
          <div class="tree-row">
            <span class="role-tag">${role}</span>
            <span class="tree-label" title="${escapeHtml(plainText(n.content_html))}">${escapeHtml(label)}</span>
            <div class="tree-actions">
              <button class="icon-btn" data-tact="open" title="Open this path">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6"/><path d="M10 14L21 3"/><path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5"/></svg>
              </button>
              ${n.role === 'user' ? `
              <button class="icon-btn" data-tact="branch" title="Branch from here">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>
              </button>` : ''}
            </div>
          </div>
        </div>`);
      walk(n.id, depth + 1);
    });
  }
  walk('ROOT', 0);

  elTreeBody.innerHTML = out.join('');

  elTreeBody.querySelectorAll('.tree-node').forEach(el => {
    el.addEventListener('click', () => {
      state.activeLeafId = findDescendantLeaf(el.dataset.id);
      renderThread();
      renderTree();
    });
  });
  elTreeBody.querySelectorAll('[data-tact]').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const act = btn.dataset.tact;
      const id  = btn.closest('.tree-node').dataset.id;
      if (act === 'open'){
        state.activeLeafId = findDescendantLeaf(id);
        renderThread();
        renderTree();
      }
      if (act === 'branch'){ setBranchFrom(id) }
    });
  });
}

function findDescendantLeaf(startId){
  // Prefer the most recent descendant of startId; fallback to startId itself.
  const children = {};
  state.treeNodes.forEach(n => {
    const p = n.parent_id;
    if (p) (children[p] = children[p] || []).push(n);
  });
  let cur = startId;
  while (children[cur] && children[cur].length){
    children[cur].sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    cur = children[cur][0].id;
  }
  return cur;
}

/* ===================== Branching =====================
 * Two modes:
 *   'child'   -> next user message becomes a CHILD of the clicked node
 *                (typical: continue down an advisor's answer with a new
 *                 alternative follow-up question).
 *   'sibling' -> next user message becomes a SIBLING of the clicked node
 *                (typical: ask an alternative version of a user question
 *                 at the same conversation level — shares the same parent).
 */
function setBranchFrom(msgId, mode){
  const n = state.treeNodes.find(x => x.id === msgId);
  if (!n) return;
  mode = mode || 'child';

  let parentId = null;
  let labelPrefix = '';
  if (mode === 'sibling'){
    // Sibling: parent = clicked node's parent (may be null for root-level)
    parentId = n.parent_id || null;
    labelPrefix = 'Asking alternative to: ';
  } else {
    // Child: parent = clicked node
    parentId = n.id;
    labelPrefix = 'Branching from: ';
  }

  state.branchFromId = parentId;
  const txt = plainText(n.content_html);
  const snippet = txt.length > 80 ? txt.slice(0, 80) + '…' : (txt || '(root)');
  elBranchText.textContent = labelPrefix + snippet;
  elBranchInd.classList.add('show');
  elInput.focus();
}
function clearBranch(){
  state.branchFromId = null;
  elBranchInd.classList.remove('show');
}
document.getElementById('branch-clear').addEventListener('click', clearBranch);

/* ===================== Copy ===================== */
function copyMessage(id){
  const n = state.treeNodes.find(x => x.id === id);
  if (!n) return;
  const tmp = document.createElement('div');
  tmp.innerHTML = n.content_html || '';
  const text = tmp.textContent || '';
  navigator.clipboard.writeText(text).then(
    () => toast('Copied to clipboard', 'success'),
    () => toast('Copy failed', 'error')
  );
}

/* ===================== Profile ===================== */
document.getElementById('pf-save').addEventListener('click', () => {
  const uid = elPfId.value.trim();
  const em  = elPfEmail.value.trim();
  if (!uid){ elPfMsg.textContent = 'Client ID is required.'; elPfMsg.className = 'msg-line error'; return }
  state.userId = uid; state.email = em;
  localStorage.setItem(LS.USER, uid);
  localStorage.setItem(LS.EMAIL, em);
  elPfMsg.textContent = 'Profile saved.';
  elPfMsg.className = 'msg-line success';
  state.currentChatId = null;
  state.treeNodes = [];
  state.activeLeafId = null;
  renderThread();
  renderTree();
  loadChats();
});
document.getElementById('pf-logout').addEventListener('click', () => {
  localStorage.removeItem(LS.USER);
  localStorage.removeItem(LS.EMAIL);
  state.userId = ''; state.email = '';
  elPfId.value = ''; elPfEmail.value = '';
  elPfMsg.textContent = 'Logged out.';
  elPfMsg.className = 'msg-line';
  state.currentChatId = null;
  state.treeNodes = [];
  state.activeLeafId = null;
  elChatList.innerHTML = '<div class="empty-state">Sign in with your Client ID to begin.</div>';
  renderThread();
  renderTree();
});

/* ===================== Chats ===================== */
async function loadChats(){
  elChatList.innerHTML = '';
  if (!state.userId){
    elChatList.innerHTML = '<div class="empty-state">Sign in to view conversations.</div>';
    return;
  }
  try{
    const data = await api('GET', '/chats');
    if (!data.items || !data.items.length){
      elChatList.innerHTML = '<div class="empty-state">No conversations yet. Start a new one.</div>';
      return;
    }
    data.items.forEach(ch => {
      const row = document.createElement('div');
      row.className = 'chat-item' + (state.currentChatId === ch.id ? ' active' : '');
      row.dataset.id = ch.id;
      const when = new Date(ch.updated_at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
      row.innerHTML = `
        <div class="title">${escapeHtml(ch.title || 'Untitled')}</div>
        <div class="meta">${when}</div>
        <div class="actions">
          <button class="icon-btn" data-cact="rename" title="Rename">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
          </button>
          <button class="icon-btn" data-cact="delete" title="Delete">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"/></svg>
          </button>
        </div>`;
      row.addEventListener('click', () => openChat(ch.id));
      row.querySelectorAll('[data-cact]').forEach(btn => {
        btn.addEventListener('click', async e => {
          e.stopPropagation();
          const act = btn.dataset.cact;
          if (act === 'rename'){
            const title = await modal({ title: 'Rename conversation', desc: 'Enter a new title.', placeholder: 'Title', value: ch.title || '' });
            if (title){
              const fd = new FormData(); fd.append('title', title);
              try { await api('POST', `/chats/${ch.id}/title`, { form: fd }); loadChats() }
              catch(err){ toast('Rename failed: ' + err.message, 'error') }
            }
          }
          if (act === 'delete'){
            const ok = await confirmDialog({ title: 'Delete conversation?', desc: 'This cannot be undone.', okLabel: 'Delete' });
            if (ok){
              try {
                await api('DELETE', `/chats/${ch.id}`);
                if (state.currentChatId === ch.id){
                  state.currentChatId = null; state.treeNodes = []; state.activeLeafId = null;
                  elChatTitle.innerHTML = '<span class="muted">Advisor</span>';
                  renderThread(); renderTree();
                }
                loadChats();
              } catch(err){ toast('Delete failed: ' + err.message, 'error') }
            }
          }
        });
      });
      elChatList.appendChild(row);
    });
    if (!state.currentChatId && data.items.length){
      openChat(data.items[0].id);
    }
  } catch(e){
    elChatList.innerHTML = '<div class="empty-state" style="color:var(--error)">Failed to load: ' + escapeHtml(e.message) + '</div>';
  }
}

async function openChat(chatId){
  state.currentChatId = chatId;
  state.branchFromId = null;
  clearBranch();
  document.querySelectorAll('.chat-item').forEach(x => x.classList.toggle('active', x.dataset.id === chatId));
  try{
    const [chatData, treeData] = await Promise.all([
      api('GET', `/chats/${chatId}`),
      api('GET', `/chats/${chatId}/tree`),
    ]);
    state.currentChatTitle = chatData.chat.title || 'Untitled';
    elChatTitle.innerHTML = escapeHtml(state.currentChatTitle);
    state.treeNodes = treeData.nodes || [];
    // Default active leaf = most recent descendant path
    const lvs = leaves();
    lvs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    state.activeLeafId = lvs.length ? lvs[0].id : null;
    renderThread();
    renderTree();
  } catch(e){
    toast('Failed to open chat: ' + e.message, 'error');
  }
}

document.getElementById('btn-newchat').addEventListener('click', async () => {
  if (!state.userId){ elPfMsg.textContent = 'Sign in first.'; elPfMsg.className = 'msg-line error'; return }
  try{
    const res = await api('POST', '/chats', { form: new FormData() });
    state.currentChatId = res.chat_id;
    state.treeNodes = [];
    state.activeLeafId = null;
    elChatTitle.innerHTML = '<span class="muted">New chat</span>';
    renderThread(); renderTree();
    loadChats();
  } catch(e){
    toast('Failed to create chat: ' + e.message, 'error');
  }
});

/* ===================== Composer ===================== */
function placeCaretAtEnd(el){
  const range = document.createRange();
  range.selectNodeContents(el);
  range.collapse(false);
  const sel = window.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
}

function readInput(){
  const tmp = elInput.cloneNode(true);
  tmp.querySelectorAll('div, br').forEach(d => {
    if (d.tagName === 'BR' || d.innerHTML === '<br>') d.outerHTML = '\n';
  });
  return tmp.innerText.replace(/\u00A0/g, ' ').trim();
}

function renderFileChips(){
  elChips.innerHTML = '';
  const files = Array.from(elFile.files || []);
  files.forEach((f, i) => {
    const c = document.createElement('span');
    c.className = 'chip';
    c.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
      ${escapeHtml(f.name)}
      <button data-rm="${i}" title="Remove">&times;</button>`;
    elChips.appendChild(c);
  });
  elChips.querySelectorAll('[data-rm]').forEach(b =>
    b.addEventListener('click', () => {
      const dt = new DataTransfer();
      Array.from(elFile.files).forEach((f, i) => { if (String(i) !== b.dataset.rm) dt.items.add(f) });
      elFile.files = dt.files;
      renderFileChips();
    })
  );
}

elAttach.addEventListener('click', () => elFile.click());
elFile.addEventListener('change', renderFileChips);

async function handleSend(q){
  if (!state.userId){ toast('Sign in with your Client ID first.', 'error'); return }
  if (!q) return;

  // Ensure we have a chat
  if (!state.currentChatId){
    try{
      const res = await api('POST', '/chats', { form: new FormData() });
      state.currentChatId = res.chat_id;
      state.treeNodes = [];
      state.activeLeafId = null;
    } catch(e){
      toast('Failed to create chat: ' + e.message, 'error'); return;
    }
  }

  // Optimistic user message
  const tempUserId = 'tmp-u-' + Math.random().toString(36).slice(2);
  const tempAdvisorId = 'tmp-a-' + Math.random().toString(36).slice(2);
  const parent = state.branchFromId || (state.activeLeafId || null);
  state.treeNodes.push({
    id: tempUserId, role: 'user',
    content_html: '<p>' + escapeHtml(q).replace(/\n/g, '<br>') + '</p>',
    created_at: new Date().toISOString(), parent_id: parent,
  });
  state.treeNodes.push({
    id: tempAdvisorId, role: 'advisor',
    content_html: '<div class="bubble thinking">Consulting the knowledge base<span class="dots"><span></span><span></span><span></span></span></div>',
    created_at: new Date().toISOString(), parent_id: tempUserId, _thinking: true,
  });
  state.activeLeafId = tempAdvisorId;
  renderThread(); renderTree();

  // auto-title first prompt
  if (!state.treeNodes.filter(n => !n._thinking && !n.id.startsWith('tmp-')).length){
    const fd = new FormData(); fd.append('title', q.slice(0, 60));
    api('POST', `/chats/${state.currentChatId}/title`, { form: fd }).catch(() => {});
  }

  try{
    const files = Array.from(elFile.files || []);
    let data;
    if (files.length){
      const fd = new FormData();
      fd.append('question', q);
      fd.append('chat_id', state.currentChatId);
      if (parent) fd.append('parent_id', parent);
      files.forEach(f => fd.append('files', f));
      data = await api('POST', '/review', { form: fd });
      elFile.value = ''; renderFileChips();
    } else {
      const url = new URL('/rag', location.origin);
      url.searchParams.set('question', q);
      url.searchParams.set('chat_id', state.currentChatId);
      if (parent) url.searchParams.set('parent_id', parent);
      url.searchParams.set('top_k', '12');
      data = await api('GET', url.toString());
    }

    // Refresh tree authoritatively
    const treeData = await api('GET', `/chats/${state.currentChatId}/tree`);
    state.treeNodes = treeData.nodes || [];

    // New leaf = latest message descending from our branch parent
    const newestUser = [...state.treeNodes].reverse()
      .find(n => n.role === 'user' && (n.parent_id || null) === parent);
    if (newestUser){
      const newestReply = [...state.treeNodes].reverse()
        .find(n => n.role === 'advisor' && n.parent_id === newestUser.id);
      state.activeLeafId = (newestReply || newestUser).id;
    } else {
      const lvs = leaves();
      lvs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      state.activeLeafId = lvs.length ? lvs[0].id : null;
    }

    clearBranch();
    renderThread(); renderTree();
    loadChats();
  } catch(e){
    // Remove optimistic entries
    state.treeNodes = state.treeNodes.filter(n => n.id !== tempUserId && n.id !== tempAdvisorId);
    state.activeLeafId = pathFromLeaf(state.activeLeafId).length ? state.activeLeafId : null;
    renderThread(); renderTree();
    toast('Request failed: ' + e.message, 'error');
  }
}

elInput.addEventListener('keydown', ev => {
  if (ev.key === 'Enter' && !ev.shiftKey){
    ev.preventDefault();
    const q = readInput();
    if (!q) return;
    elInput.innerHTML = '';
    handleSend(q);
  }
});
elSend.addEventListener('click', () => {
  const q = readInput();
  if (!q) return;
  elInput.innerHTML = '';
  handleSend(q);
});

/* ===================== Layout / theme toggles ===================== */
document.getElementById('btn-rail-collapse').addEventListener('click', () => {
  ui.sidebarOpen = false; saveUI(); applyLayout();
});
document.getElementById('left-tab').addEventListener('click', () => {
  ui.sidebarOpen = true; saveUI(); applyLayout();
});
document.getElementById('btn-tree-collapse').addEventListener('click', () => {
  ui.treeOpen = false; saveUI(); applyLayout();
});
document.getElementById('right-tab').addEventListener('click', () => {
  ui.treeOpen = true; saveUI(); applyLayout();
});
document.getElementById('btn-toggle-rail-m').addEventListener('click', () => {
  ui.sidebarOpen = !ui.sidebarOpen; saveUI(); applyLayout();
});
document.getElementById('btn-toggle-tree-m').addEventListener('click', () => {
  ui.treeOpen = !ui.treeOpen; saveUI(); applyLayout();
});
document.getElementById('btn-theme').addEventListener('click', () => {
  state.theme = state.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem(LS.THEME, state.theme);
  applyTheme();
});

/* Keyboard shortcuts */
document.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'k'){ e.preventDefault(); elInput.focus() }
  if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'O'){ e.preventDefault(); document.getElementById('btn-newchat').click() }
  if ((e.metaKey || e.ctrlKey) && e.key === '\\'){ e.preventDefault(); ui.sidebarOpen = !ui.sidebarOpen; saveUI(); applyLayout() }
});

/* ===================== Init ===================== */
applyTheme();
applyLayout();

if (state.userId){ loadChats() }
else { elChatList.innerHTML = '<div class="empty-state">Sign in with your Client ID to begin.</div>' }

renderThread();
renderTree();
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
        "has_OPENAI_API_KEY":   bool(os.getenv("OPENAI_API_KEY")),
        "PINECONE_INDEX": index_name or None,
        "PINECONE_HOST":  host or None,
        "NO_PROXY": os.getenv("NO_PROXY"),
        "db_path": DB_PATH,
    }
    try:
        lst = pc.list_indexes()
        info["pinecone_ok"]  = True
        info["index_count"]  = len(lst or [])
    except Exception as e:
        info["pinecone_ok"] = False
        info["error"] = str(e)
    return info

# ========== /search (raw context) ==========
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
            meta = s.get("meta", {})
            rows.append({
                "title":   s["title"],
                "level":   s["level"],
                "page":    s["page"],
                "version": s.get("version", ""),
                "score":   s["score"],
                "snippet": _extract_snippet(meta) or "",
            })
        return {"question": question, "titles": titles, "matches": rows, "t_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ========== /rag (synthesis + persistence + branching) ==========
@app.get("/rag")
def rag_endpoint(
    question: str = Query(..., min_length=3),
    chat_id: Optional[str] = Query(None),
    parent_id: Optional[str] = Query(None),
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
        # If caller didn't specify a branch point, fall back to the latest message in the chat
        effective_parent = parent_id or _latest_leaf(conn, chat_id)
        user_msg_id = insert_message(
            conn, chat_id, user_id, "user",
            content_html=f"<p>{question}</p>",
            content_raw=question, meta={}, parent_id=effective_parent,
        )
        result = _run_rag(question, top_k=top_k, level=level)
        insert_message(
            conn, chat_id, None, "advisor",
            content_html=result["answer"], content_raw=None,
            meta={"t_ms": result["t_ms"], "sources": result["sources"]},
            parent_id=user_msg_id,
        )
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "titles": result["titles"],
            "t_ms": result["t_ms"],
            "chat_id": chat_id,
            "user_msg_id": user_msg_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# ========== /review (PDF/TXT/DOCX) + persistence + branching ==========
@app.post("/review")
def review_endpoint(
    authorization: Optional[str] = Header(None),
    chat_id: Optional[str] = Form(None),
    parent_id: Optional[str] = Form(None),
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
        effective_parent = parent_id or _latest_leaf(conn, chat_id)
        user_msg_id = insert_message(
            conn, chat_id, user_id, "user",
            content_html=f"<p>{question or '(Uploaded documents for review.)'}</p>",
            content_raw=question, meta={"upload": True},
            parent_id=effective_parent,
        )

        texts: List[str] = []
        for f in files:
            name = (f.filename or "").lower()
            raw  = f.file.read(UPLOAD_MAX_BYTES + 1)
            if len(raw) > UPLOAD_MAX_BYTES:
                raise HTTPException(status_code=413,
                                    detail=f"{f.filename} exceeds {UPLOAD_MAX_BYTES//1024//1024}MB limit.")
            if name.endswith(".pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(raw))
                    pages = []
                    for p in reader.pages:
                        try:    pages.append(p.extract_text() or "")
                        except Exception: pages.append("")
                    texts.append("\n".join(pages))
                except Exception as e:
                    raise HTTPException(status_code=415, detail=f"Failed to parse PDF: {f.filename} ({e})")
            elif name.endswith(".txt"):
                try:    texts.append(raw.decode("utf-8", errors="ignore"))
                except Exception: texts.append(raw.decode("latin-1", errors="ignore"))
            elif name.endswith(".docx"):
                try:
                    try:
                        import docx  # python-docx
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
                raise HTTPException(status_code=415,
                                    detail=f"Unsupported file type: {f.filename} (only PDF/TXT/DOCX)")

        merged = "\n---\n".join([t for t in texts if t.strip()])
        chunks  = [merged[i:i + 2000] for i in range(0, len(merged), 2000)][:MAX_SNIPPETS]
        pseudo  = [{"title": "Uploaded Document", "level": "L5", "page": "?",
                    "version": "", "score": 1.0, "meta": {}}]
        html    = synthesize_html(question or "Please analyze the attached materials.", pseudo, chunks)

        insert_message(
            conn, chat_id, None, "advisor",
            content_html=html, content_raw=None, meta={"upload": True},
            parent_id=user_msg_id,
        )
        return {"answer": html, "t_ms": 0, "chat_id": chat_id, "user_msg_id": user_msg_id}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# ==========================================================
# /draft — generate full HTML fiduciary resolution (internal RAG)
# ==========================================================
@app.post("/draft")
async def draft(request: Request,
                authorization: Optional[str] = Header(None)):
    """
    Produces a comprehensive, HTML-formatted trustee resolution using:
      (1) the internal RAG pipeline for citations, and
      (2) an OpenAI completion with the fiduciary system prompt.
    """
    require_auth(authorization)
    check_rate_limit()

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or missing JSON body.")

    question = (data.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' field.")

    # 1. Internal RAG — no self-HTTP call
    rag = _run_rag(question, top_k=12)
    sources = rag.get("sources", [])
    src_text = "\n".join([
        f"{s.get('title','')} (L{s.get('level','')}, p.{s.get('page','')})" for s in sources
    ])

    # 2. Compose prompt
    system_prompt = os.environ.get("FIDUCIARY_SYSTEM_PROMPT", "").strip()
    model = os.environ.get("SYNTH_MODEL", SYNTH_MODEL)
    max_tokens = int(os.environ.get("MAX_OUT_TOKENS", str(MAX_OUT_TOKENS)))

    draft_instructions = f"""{system_prompt}

Use the following citations where applicable:
{src_text}

The trustee request is:
{question}

Produce a comprehensive, legally formatted trustee resolution in rich HTML.

Requirements:
- Write as if by an experienced fiduciary attorney preparing an official trust record.
- Use clear HTML structure: <h1> title, <h2> section headings, <p> narrative paragraphs, and <ul>/<li> lists for details.
- Sections, in order:
   1. Title of the resolution
   2. Date and Location
   3. Trust Details -- name, type, situs/jurisdiction, trustee(s)
   4. Recitals (WHEREAS clauses)
   5. Resolution Section -- detailed authorization
   6. Legal Basis and References -- integrate citations from the sources above
   7. Execution Section -- closing paragraph and signature block
- Only include sections for which data is available.
- Do not insert underscores or placeholders for missing data.
- Output must be valid, styled HTML ready for web display.
""".strip()

    try:
        res = client.chat.completions.create(
            model=model,
            temperature=0.3,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt or "You are a private fiduciary advisor."},
                {"role": "user",   "content": draft_instructions},
            ],
        )
        answer = res.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")

    return {"answer": answer, "sources": sources, "model": model}

# ==========================================================
# /chat — conversational fiduciary advisor (internal RAG)
# ==========================================================
@app.post("/chat")
async def chat_endpoint(request: Request,
                        authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    check_rate_limit()

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or missing JSON body.")

    message = (data.get("message") or "").strip()
    thread_id = data.get("thread_id", "default")
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message' field.")

    rag = _run_rag(message, top_k=12)
    sources = rag.get("sources", [])
    src_text = "\n".join([
        f"{s.get('title','')} (L{s.get('level','')}, p.{s.get('page','')})" for s in sources
    ])

    system_prompt = os.environ.get("FIDUCIARY_SYSTEM_PROMPT", "You are the Private Fiduciary Advisor.")
    model = os.environ.get("SYNTH_MODEL", SYNTH_MODEL)
    max_tokens = int(os.environ.get("MAX_OUT_TOKENS", str(MAX_OUT_TOKENS)))

    composite_prompt = f"""{system_prompt}

Use these citations where relevant:
{src_text}

Question:
{message}

Respond as the Private Fiduciary Advisor: formal, clear, authoritative, but conversational.
Include citations inline where appropriate (e.g., IRC §642(c)(2); Treas. Reg. §1.642(c)-2).
Answer in full sentences, not bullet points unless listing authorities.
""".strip()

    try:
        res = client.chat.completions.create(
            model=model,
            temperature=0.3,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": composite_prompt},
            ],
        )
        answer = res.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")

    return {"answer": answer, "sources": sources, "thread_id": thread_id, "model": model}
