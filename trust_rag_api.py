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

for k in (
    "HTTP_PROXY","HTTPS_PROXY","ALL_PROXY",
    "http_proxy","https_proxy","all_proxy",
    "OPENAI_PROXY","OPENAI_HTTP_PROXY","OPENAI_HTTPS_PROXY"
):
    os.environ.pop(k, None)
os.environ.setdefault("NO_PROXY", "*")

if os.getenv("OPENAI_BASE_URL", "").strip().lower() in ("", "none", "null"):
    os.environ.pop("OPENAI_BASE_URL", None)

API_TOKEN = os.getenv("API_TOKEN", "")
SYNTH_MODEL = os.getenv("SYNTH_MODEL", "gpt-4o-mini")
MAX_SNIPPETS = int(os.getenv("MAX_SNIPPETS", "20"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))
UPLOAD_MAX_BYTES = 12 * 1024 * 1024

app = FastAPI(title="Private Trust Fiduciary Advisor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

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
RATE_WINDOW, RATE_LIMIT = 10, 100

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
host = os.getenv("PINECONE_HOST", "").strip()
idx = pc.Index(host=host) if host else pc.Index(index_name)

def _clean_openai_key(raw: str) -> str:
    s = (raw or "").strip()
    if not s.startswith("sk-"):
        parts = [t.strip() for t in s.replace("=", " ").split() if t.strip().startswith("sk-")]
        if parts: s = parts[-1]
    if not s.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY malformed.")
    return s

_openai_key = _clean_openai_key(os.getenv("OPENAI_API_KEY", ""))
openai_http = httpx.Client(timeout=120.0, trust_env=False)
client = OpenAI(api_key=_openai_key, http_client=openai_http)

# -------------------- HELPERS --------------------
def _extract_snippet(meta: dict) -> str:
    for k in ("text","chunk","content","body","passage"):
        v = meta.get(k)
        if isinstance(v,str) and v.strip(): return v.strip()
    return ""

def _clean_title(title: str) -> str:
    t = title or "Unknown"
    t = re.sub(r'^[Ll]\d[_\-:\s]+','',t)
    t = re.sub(r'(?i)\bocr\b','',t)
    t = re.sub(r'[0-9a-f]{8,}','',t)
    if ' -- ' in t:
        first,*_ = t.split(' -- ')
        if len(first)>=6: t=first
    return re.sub(r'\s+',' ',t.replace('_',' ')).strip(' -–—')

def _dedup_and_rank_sources(matches, top_k:int):
    rank={"L1":1,"L2":2,"L3":3,"L4":4,"L5":5}
    best={}
    for m in matches:
        meta=m.get("metadata",{}) if isinstance(m,dict) else (getattr(m,"metadata",{}) or {})
        title=_clean_title(meta.get("title") or meta.get("doc_parent") or "Unknown")
        lvl=(meta.get("doc_level") or meta.get("level") or "N/A").strip()
        page=str(meta.get("page","?"))
        ver=str(meta.get("version",meta.get("v",""))) if meta.get("version",meta.get("v","")) else ""
        score=float(m.get("score") if isinstance(m,dict) else getattr(m,"score",0.0))
        key=(title,lvl,page,ver)
        if key not in best or score>best[key]["score"]:
            best[key]={"title":title,"level":lvl,"page":page,"version":ver,"score":score,"meta":meta}
    unique=list(best.values())
    unique.sort(key=lambda s:(rank.get(s["level"],99),-s["score"]))
    return unique[:top_k]

def _citations_titles_only(unique):
    seen,titles=set(),[]
    for s in unique:
        if s["title"] in seen: continue
        seen.add(s["title"])
        titles.append(s["title"])
    return titles

# -------------------- Synthesize (no system message) --------------------
def synthesize_answer_html(question:str, unique_sources:list[dict], snippets:list[str]) -> str:
    if not snippets and not unique_sources:
        return "<p>No relevant material found in the Trust-Law knowledge base.</p>"
    context,used="",0
    for s in snippets:
        s=s.strip()
        if not s: continue
        if used+len(s)>MAX_CONTEXT_CHARS: break
        context+=f"\n---\n{s}"
        used+=len(s)
    titles=_citations_titles_only(unique_sources)
    titles_html="<ul>"+"".join(f"<li>{t}</li>" for t in titles)+"</ul>" if titles else "<p>No relevant material found in the Trust-Law knowledge base.</p>"
    user_msg=(
        f"<h2>Question</h2><p>{question}</p>"
        f"<h3>Context</h3><pre>{(context or '').strip()}</pre>"
        f"<hr><h3>Citations</h3>{titles_html}"
        f"<p><em>This response is provided solely for educational and informational purposes. "
        f"It does not constitute legal, tax, or financial advice, nor does it establish an attorney-client or fiduciary relationship. "
        f"Users must consult qualified counsel or a CPA for application of law to specific facts.</em></p>"
    )
    try:
        chat=client.chat.completions.create(
            model=SYNTH_MODEL,
            temperature=0.15,
            max_tokens=2200,
            messages=[{"role":"user","content":user_msg}],
        )
        html=(chat.choices[0].message.content or "").strip()
        if not html: return "<p>No relevant material found in the Trust-Law knowledge base.</p>"
        if "<" not in html: html="<div><p>"+html.replace("\n","<br>")+"</p></div>"
        return html
    except Exception as e:
        return f"<p><em>(Synthesis unavailable: {e})</em></p>"

# -------------------- Widget --------------------
WIDGET_HTML="""<!doctype html><html><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width,initial-scale=1'/>
<title>Private Trust Fiduciary Advisor</title>
<style>
:root{--ink:#000;--bg:#f7f7f8;--card:#fff;--border:#e5e5ea;--brand:#0B3B5C;}
body{font-family:system-ui,Arial;margin:0;padding:20px;background:var(--bg);color:var(--ink)}
.wrap{max-width:1000px;margin:auto;display:flex;flex-direction:column;min-height:100vh}
textarea{flex:1;min-height:140px;max-height:50vh;resize:vertical;padding:12px;border:1px solid #d0d0d6;border-radius:10px;color:var(--ink)}
input[type=file]{padding:10px;border:1px dashed #c8ccd3;border-radius:10px;background:#fff;color:var(--ink)}
button{padding:12px 18px;border:none;border-radius:10px;background:var(--brand);color:#fff;cursor:pointer}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px;margin-top:14px;color:var(--ink)}
</style></head><body>
<div class='wrap'>
<h1>Private Trust Fiduciary Advisor</h1>
<form id='f'><textarea id='q' placeholder='Type your question or drafting request...' required></textarea>
<input id='file' type='file' multiple accept='.pdf,.txt,.docx'/><button type='submit'>Ask</button></form>
<div id='out'><div class='card'>Type or paste your question; attach files if needed; then click <strong>Ask</strong>.</div></div>
</div>
<script>
const OUT=document.getElementById('out'),F=document.getElementById('f'),Q=document.getElementById('q'),FILES=document.getElementById('file');
function autoresize(){Q.style.height='auto';Q.style.height=Math.min(Q.scrollHeight,window.innerHeight*0.5)+'px';}
Q.addEventListener('input',autoresize);window.addEventListener('resize',autoresize);
async function askRag(q){const u=new URL('/rag',location.origin);u.searchParams.set('question',q);u.searchParams.set('top_k','12');const r=await fetch(u);return r.json();}
async function askReview(q,files){const fd=new FormData();fd.append('question',q);for(const f of files)fd.append('files',f);const r=await fetch('/review',{method:'POST',body:fd});return r.json();}
F.addEventListener('submit',async e=>{e.preventDefault();OUT.innerHTML='<div class=card>Working…</div>';try{let d;if(FILES.files.length>0)d=await askReview(Q.value,FILES.files);else d=await askRag(Q.value);OUT.innerHTML='<div class=card>'+d.answer+'</div>';}catch(x){OUT.innerHTML='<div class=card>Error: '+x+'</div>';}});</script>
</body></html>"""

@app.get("/widget",response_class=HTMLResponse)
def widget(): return HTMLResponse(WIDGET_HTML)

# -------------------- Health/Diag --------------------
@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/diag")
def diag():
    info={"has_PINECONE_API_KEY":bool(os.getenv("PINECONE_API_KEY")),
          "has_OPENAI_API_KEY":bool(os.getenv("OPENAI_API_KEY")),
          "PINECONE_INDEX":index_name or None,"PINECONE_HOST":host or None}
    try: lst=pc.list_indexes();info["pinecone_ok"]=True;info["count"]=len(lst)
    except Exception as e: info["pinecone_ok"]=False;info["error"]=str(e)
    return info

# -------------------- /rag --------------------
@app.get("/rag")
def rag_endpoint(question:str=Query(...,min_length=3),top_k:int=Query(12,ge=1,le=30),level:str|None=Query(None),authorization:str|None=Header(default=None)):
    require_auth(authorization);check_rate_limit()
    t0=time.time()
    try:
        emb=client.embeddings.create(model="text-embedding-3-small",input=question).data[0].embedding
        flt={"doc_level":{"$eq":level}} if level else None
        results=idx.query(vector=emb,top_k=max(top_k,12),include_metadata=True,filter=flt)
        matches=results["matches"] if isinstance(results,dict) else getattr(results,"matches",[])
        unique=_dedup_and_rank_sources(matches,top_k=top_k)
        snippets=[_extract_snippet(s["meta"]) for s in unique if _extract_snippet(s["meta"])]
        html=synthesize_answer_html(question,unique,snippets)
        return {"answer":html,"t_ms":int((time.time()-t0)*1000)}
    except Exception as e:
        traceback.print_exc();raise HTTPException(status_code=500,detail=str(e))

# -------------------- /review --------------------
@app.post("/review")
def review_endpoint(authorization:str|None=Header(default=None),question:str=Form("Please analyze the attached document."),files:list[UploadFile]=File(default=[])):
    require_auth(authorization);check_rate_limit()
    t0=time.time()
    try:
        if not files: raise HTTPException(status_code=400,detail="No files uploaded.")
        texts=[]
        for uf in files:
            n=(uf.filename or "").lower();raw=uf.file.read(UPLOAD_MAX_BYTES+1)
            if len(raw)>UPLOAD_MAX_BYTES: raise HTTPException(status_code=413,detail=f"{uf.filename} too large.")
            if n.endswith(".pdf"):
                import pypdf;pages=[];r=pypdf.PdfReader(io.BytesIO(raw))
                for p in r.pages:
                    try: pages.append(p.extract_text() or "")
                    except: pages.append("")
                texts.append("\n".join(pages))
            elif n.endswith(".txt"):
                try:texts.append(raw.decode("utf-8",errors="ignore"))
                except:texts.append(raw.decode("latin-1",errors="ignore"))
            elif n.endswith(".docx"):
                try:
                    import docx
                    d=docx.Document(io.BytesIO(raw))
                    texts.append("\n".join(p.text for p in d.paragraphs if p.text))
                except Exception:
                    with zipfile.ZipFile(io.BytesIO(raw)) as z:
                        xml=z.read("word/document.xml").decode("utf-8",errors="ignore")
                        stripped=re.sub(r"<[^>]+>"," ",xml)
                        texts.append(re.sub(r"\s+"," ",stripped))
            else: raise HTTPException(status_code=415,detail=f"Unsupported type: {uf.filename}")
        merged="\n---\n".join([t for t in texts if t.strip()])
        chunks=[merged[i:i+2000] for i in range(0,len(merged),2000)][:MAX_SNIPPETS]
        pseudo=[{"title":"Uploaded Document","level":"L5","page":"?","version":"","score":1.0}]
        html=synthesize_answer_html(question,pseudo,chunks)
        return {"answer":html,"t_ms":int((time.time()-t0)*1000)}
    except Exception as e:
        traceback.print_exc();raise HTTPException(status_code=500,detail=str(e))
