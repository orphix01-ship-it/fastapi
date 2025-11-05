# trust_rag_api.py
from fastapi import FastAPI, Query, Header, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pinecone import Pinecone
from openai import OpenAI
import httpx
import os, time, traceback, io
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

API_TOKEN = os.getenv("API_TOKEN", "")  # set in Railway to protect /rag and /review
SYNTH_MODEL = os.getenv("SYNTH_MODEL", "gpt-4o-mini")
MAX_SNIPPETS = int(os.getenv("MAX_SNIPPETS", "8"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))
UPLOAD_MAX_BYTES = 5 * 1024 * 1024  # 5MB cap per file for safety

# -------------------- APP --------------------
app = FastAPI(title="Private Trust Fiduciary Advisor API")

# CORS (relax now; you can lock to your domain later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics (optional; won't crash if package missing)
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
except Exception:
    pass

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

# -------------------- Utilities --------------------
def _extract_snippet(meta: dict) -> str:
    for k in ("text", "chunk", "content", "body", "passage"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _dedup_and_rank_sources(matches, top_k: int):
    """Unique sources with L1→L5 precedence and highest score per (title, level, page)."""
    rank = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}
    best = {}
    for m in matches:
        meta = m.get("metadata", {}) if isinstance(m, dict) else (getattr(m, "metadata", {}) or {})
        title = (meta.get("title") or meta.get("doc_parent") or "Unknown").strip()
        lvl   = (meta.get("doc_level") or meta.get("level") or "N/A").strip()
        page  = str(meta.get("page", "?"))
        score = float(m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0))
        key = (title, lvl, page)
        if key not in best or score > best[key]["score"]:
            best[key] = {"title": title, "level": lvl, "page": page, "score": score, "meta": meta}
    unique = list(best.values())
    unique.sort(key=lambda s: (rank.get(s["level"], 99), -s["score"]))
    return unique[:top_k]

def _make_citation_entry(s: dict) -> str:
    """Title (doc_id, L{level}, p.{page}, v.{version}) — include fields only if present."""
    meta = s.get("meta", {}) or {}
    doc_id = meta.get("doc_id")
    version = meta.get("version") or meta.get("v")
    bits = []
    if doc_id: bits.append(str(doc_id))
    bits.append(f"L{s.get('level','N/A')}")
    bits.append(f"p.{s.get('page','?')}")
    if version: bits.append(f"v.{version}")
    return f"{s.get('title','Unknown')} ({', '.join(bits)})"

def _citations_block(unique_sources):
    """Single deduplicated 'Citations' block; no repetitions."""
    seen = set()
    lines = []
    for s in unique_sources:
        key = (s["title"], s["level"], s["page"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(_make_citation_entry(s))
    return "\n".join(lines) if lines else "No relevant material found in the Trust-Law knowledge base."

def synthesize_answer(question: str, unique_sources: list[dict], snippets: list[str]) -> str:
    """Compose a legal-register answer strictly from provided context; append a single Citations block + disclaimer."""
    # If nothing usable, return the mandated fallback
    if not snippets and not unique_sources:
        return "No relevant material found in the Trust-Law knowledge base."

    # Build compact context
    context = ""
    used = kept = 0
    for s in snippets:
        t = (s or "").strip()
        if not t:
            continue
        if used + len(t) > MAX_CONTEXT_CHARS:
            break
        context += f"\n---\n{t}"
        used += len(t)
        kept += 1
        if kept >= MAX_SNIPPETS:
            break

    citations = _citations_block(unique_sources)

    system_msg = (
        "You are the 'Private Trust Fiduciary Advisor'—a comprehensive fiduciary structuring and compliance engine. "
        "Operate exclusively in legal register (legalese). Prioritize statutory and doctrinal fidelity; "
        "scholarly/practice commentary is subordinate. Every substantive proposition must be grounded in the provided context. "
        "Do NOT rely on internal knowledge. If context is insufficient, reply exactly: "
        "'No relevant material found in the Trust-Law knowledge base.'"
    )

    # Instruction block mirrors the GPT configuration (condensed for runtime)
    methodology = (
        "Apply layered methodology: statutory foundation → regulatory gloss → judicial interpretation → scholarly elaboration → practical application. "
        "Follow precedence L1 > L2 > L3 > L4 > L5; if conflict exists, follow L1 and explain briefly."
    )
    citation_rules = (
        "Use formal legal citation conventions in prose or parentheticals: "
        "Statutes '26 U.S.C. § 641' (or 'IRC § 641'); Regs 'Treas. Reg. § 1.641(a)-0'; "
        "Cases 'Gregory v. Helvering, 293 U.S. 465 (1935)'; "
        "Revenue Rulings 'Rev. Rul. 79-47, 1979-1 C.B. 312'; Restatements 'Restatement (Third) of Trusts § 78 (2007)'; "
        "Treatises 'Scott & Ascher on Trusts § 17.2 (5th ed. 2007)'."
    )

    disclaimer = (
        "This response is provided solely for educational and informational purposes. "
        "It does not constitute legal, tax, or financial advice, nor does it establish an attorney-client or fiduciary relationship. "
        "Users must consult qualified counsel or a CPA for application of law to specific facts."
    )

    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT (verbatim snippets from authoritative sources):\n{context}\n\n"
        f"{methodology}\n{citation_rules}\n\n"
        f"Write 3–8 sentences in formal legal register. Do NOT invent sources. "
        f"Append exactly one block titled 'Citations' with the lines below; do not duplicate items.\n\n"
        f"Citations:\n{citations}\n\n"
        f"Finally, append a single-line 'Disclaimer' sentence as provided."
    )

    try:
        chat = client.chat.completions.create(
            model=SYNTH_MODEL,
            temperature=0.1,
            max_tokens=900,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
        )
        text = (chat.choices[0].message.content or "").strip()
        if not text:
            return "No relevant material found in the Trust-Law knowledge base."
        # Ensure disclaimer present once
        if "Disclaimer" not in text:
            text += f"\n\nDisclaimer: {disclaimer}"
        return text
    except Exception as e:
        return f"(Synthesis unavailable: {e})"

# -------------------- SIMPLE WIDGET PAGE (no bottom 'Sources' box) --------------------
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
    form{display:flex;gap:8px;margin:12px 0;flex-wrap:wrap}
    input[type=text]{flex:1;min-width:280px;padding:12px;border:1px solid #d0d0d6;border-radius:8px}
    input[type=file]{padding:10px;border:1px dashed #c8ccd3;border-radius:8px;background:#fff}
    button{padding:12px 16px;border:none;border-radius:8px;background:#0B3B5C;color:#fff;cursor:pointer}
    .card{background:#fff;border:1px solid #e5e5ea;border-radius:12px;padding:14px;margin-top:12px}
    .muted{color:#777}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Private Trust Fiduciary Advisor</h1>
    <form id="f">
      <input id="q" type="text" placeholder="Ask a fiduciary/trust question…" required />
      <input id="file" type="file" multiple accept=".pdf,.txt" />
      <button type="submit">Ask</button>
    </form>
    <div id="out" class="muted">Ask something—or attach PDF/TXT—then press “Ask”.</div>
  </div>
  <script>
    const OUT = document.getElementById('out');
    const F = document.getElementById('f');
    const Q = document.getElementById('q');
    const FILES = document.getElementById('file');

    async function askViaRag(q) {
      const u = new URL('/rag', location.origin);
      u.searchParams.set('question', q);
      u.searchParams.set('top_k', '5');
      const res = await fetch(u, { headers: { /* 'Authorization':'Bearer YOUR_API_TOKEN' */ } });
      return res.json();
    }

    async function askViaReview(q, files) {
      const fd = new FormData();
      fd.append('question', q);
      for (const f of files) fd.append('files', f);
      const res = await fetch('/review', {
        method: 'POST',
        body: fd,
        headers: { /* 'Authorization':'Bearer YOUR_API_TOKEN' */ }
      });
      return res.json();
    }

    F.addEventListener('submit', async (e) => {
      e.preventDefault();
      OUT.innerHTML = '<div class="card">Working…</div>';
      try {
        let data;
        if (FILES.files && FILES.files.length > 0) {
          data = await askViaReview(Q.value.trim(), FILES.files);
        } else {
          data = await askViaRag(Q.value.trim());
        }
        const answer = data.answer || data.response || '(no answer)';
        OUT.innerHTML = '<div class="card">' + escapeHtml(answer).replaceAll('\\n','<br>') + '</div>';
      } catch (err) {
        OUT.innerHTML = '<div class="card">Error: ' + escapeHtml(String(err)) + '</div>';
      }
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
        # 1) Embed
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        ).data[0].embedding

        # 2) Query Pinecone (optional level filter)
        flt = {"doc_level": {"$eq": level}} if level else None
        results = idx.query(vector=emb, top_k=max(top_k, 5), include_metadata=True, filter=flt)
        matches = results["matches"] if isinstance(results, dict) else getattr(results, "matches", [])

        # 3) Unique & precedence
        unique = _dedup_and_rank_sources(matches, top_k=top_k)

        # 4) Harvest snippets for synthesis
        snippets = []
        for s in unique:
            sn = _extract_snippet(s["meta"])
            if sn:
                snippets.append(sn)

        # 5) Synthesize in legal register; answer contains a single 'Citations' block + disclaimer
        answer = synthesize_answer(question, unique, snippets)

        # 6) If nothing found, return exact mandated message
        if answer.strip() == "":
            answer = "No relevant material found in the Trust-Law knowledge base."

        return {"answer": answer, "t_ms": int((time.time()-t0)*1000)}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{e.__class__.__name__}: {e}")

@app.post("/review")
def review_endpoint(
    authorization: str | None = Header(default=None),
    question: str = Form("Please summarize and analyze the attached document for fiduciary/trust implications."),
    files: list[UploadFile] = File(default=[]),
):
    """
    Trustees can upload PDF/TXT for quick review.
    We extract text and synthesize an answer (no Pinecone write needed).
    """
    require_auth(authorization)
    check_rate_limit()

    t0 = time.time()
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        texts = []
        for uf in files:
            filename = (uf.filename or "").lower()
            content = uf.file.read(UPLOAD_MAX_BYTES + 1)
            if len(content) > UPLOAD_MAX_BYTES:
                raise HTTPException(status_code=413, detail=f"{uf.filename} exceeds {UPLOAD_MAX_BYTES//1024//1024}MB limit.")
            if filename.endswith(".pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(content))
                    pages = []
                    for p in reader.pages:
                        try:
                            pages.append(p.extract_text() or "")
                        except Exception:
                            pages.append("")
                    texts.append("\n".join(pages))
                except Exception as e:
                    raise HTTPException(status_code=415, detail=f"Failed to parse PDF: {uf.filename} ({e})")
            elif filename.endswith(".txt"):
                try:
                    texts.append(content.decode("utf-8", errors="ignore"))
                except Exception:
                    texts.append(content.decode("latin-1", errors="ignore"))
            else:
                raise HTTPException(status_code=415, detail=f"Unsupported type: {uf.filename} (only PDF/TXT)")

        merged = "\n---\n".join([t.strip() for t in texts if t.strip()])
        # Split into chunks for context cap
        chunks = []
        CHUNK_SIZE = 1500
        for i in range(0, len(merged), CHUNK_SIZE):
            chunks.append(merged[i:i+CHUNK_SIZE])
            if len(chunks) >= MAX_SNIPPETS:
                break

        pseudo_sources = [{"title": "Uploaded Document", "level": "L5", "page": "?", "score": 1.0}]
        answer = synthesize_answer(question, pseudo_sources, chunks)
        if answer.strip() == "":
            answer = "No relevant material found in the Trust-Law knowledge base."
        return {"answer": answer, "t_ms": int((time.time()-t0)*1000)}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{e.__class__.__name__}: {e}")
