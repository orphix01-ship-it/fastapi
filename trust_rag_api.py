# trust_rag_api.py
from fastapi import FastAPI, Query, Header, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from openai import OpenAI
import httpx, io, os, time, traceback
from collections import deque

# ======== BASIC SETUP ========
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

for k in (
    "HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy",
    "OPENAI_PROXY","OPENAI_HTTP_PROXY","OPENAI_HTTPS_PROXY"
):
    os.environ.pop(k, None)
os.environ.setdefault("NO_PROXY", "*")

API_TOKEN   = os.getenv("API_TOKEN", "")
MODEL       = os.getenv("SYNTH_MODEL", "gpt-4o-mini")

app = FastAPI(title="Private Trust Fiduciary Advisor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limit
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

# OpenAI client
openai_http = httpx.Client(timeout=120.0, trust_env=False)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), http_client=openai_http)

# ======== MINIMAL FRONTEND ========
WIDGET_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Private Trust Fiduciary Advisor</title>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#fff; --text:#000; --border:#e5e5e5; --ring:#d9d9d9;
  --user:#e8f1ff; --shadow:0 1px 2px rgba(0,0,0,.03),0 8px 24px rgba(0,0,0,.04);
  --font:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial;
  --title:"Cinzel",serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 var(--font)}
.app{display:flex;flex-direction:column;height:100vh;width:100%}
.header{background:#fff}
.header .inner{max-width:900px;margin:0 auto;padding:14px 16px;text-align:center}
.title{font-family:var(--title);font-weight:300;letter-spacing:.2px;font-size:20px}
.main{flex:1;overflow:auto;padding:24px 12px 140px}
.container{max-width:900px;margin:0 auto}
.thread{display:flex;flex-direction:column;gap:16px}
.msg .bubble{display:inline-block;max-width:80%}
.msg.user .bubble{
  background:var(--user);border:1px solid var(--border);
  border-radius:14px;padding:12px 14px;box-shadow:var(--shadow);
}
.msg.advisor .bubble{background:transparent;padding:0;max-width:100%}
.meta{font-size:12px;margin-bottom:6px;color:#000}
.composer{position:fixed;bottom:0;left:0;right:0;background:#fff;padding:18px 12px;border-top:none}
.composer .inner{max-width:900px;margin:0 auto}
.bar{display:flex;align-items:flex-end;gap:8px;background:#fff;
  border:1px solid var(--ring);border-radius:22px;padding:8px;box-shadow:var(--shadow)}
.input{flex:1;min-height:24px;max-height:160px;overflow:auto;outline:none;
  padding:8px 10px;font:16px/1.5 var(--font);color:#000}
.input:empty:before{content:attr(data-placeholder);color:#000}
.btn{cursor:pointer}
.send{padding:8px 12px;border-radius:12px;background:#000;color:#fff;border:1px solid #000}
.attach{padding:8px 10px;border-radius:12px;background:#fff;color:#000;border:1px solid var(--border)}
#file{display:none}
</style>
</head>
<body>
<div class="app">
  <div class="header"><div class="inner"><div class="title">Private Trust Fiduciary Advisor</div></div></div>
  <main class="main"><div class="container"><div id="thread" class="thread"></div></div></main>
  <div class="composer">
    <div class="inner">
      <div class="bar">
        <input id="file" type="file" multiple accept=".pdf,.txt,.docx"/>
        <div id="input" class="input" role="textbox" aria-multiline="true"
             contenteditable="true" data-placeholder="Message the Advisor… (Shift+Enter for newline)"></div>
        <button id="attach" class="attach btn" type="button" title="Add files">+</button>
        <button id="send" class="send btn" type="button" title="Send">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <path d="m5 12 14-7-4 14-3-5-7-2z" stroke="#fff"
                  stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>
      <div id="hint" style="margin-top:8px;color:#000;font-size:12px;">
        State your inquiry to receive a response.
      </div>
    </div>
  </div>
</div>
<script>
const thread=document.getElementById('thread');
const input=document.getElementById('input');
const send=document.getElementById('send');
const attach=document.getElementById('attach');
const file=document.getElementById('file');
const hint=document.getElementById('hint');
function now(){return new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});}
function add(role,html){
  const d=document.createElement('div');
  d.className='msg '+(role==='user'?'user':'advisor');
  d.innerHTML=`<div class="meta">${role==='user'?'You':'Advisor'} · ${now()}</div><div class="bubble">${html}</div>`;
  thread.appendChild(d); thread.scrollTop=thread.scrollHeight;
}
async function ask(q){
  if(!q)return;
  add('user',q.replace(/\\n/g,'<br>'));
  const work=document.createElement('div');
  work.className='msg advisor';
  work.innerHTML='<div class="meta">Advisor · thinking…</div><div class="bubble"><p>Working…</p></div>';
  thread.appendChild(work); thread.scrollTop=thread.scrollHeight;
  try{
    const files=Array.from(file.files||[]);
    const fd=new FormData();
    fd.append('question',q);
    for(const f of files) fd.append('files',f);
    const r=await fetch('/direct',{method:'POST',body:fd});
    const data=await r.json();
    work.querySelector('.meta').textContent='Advisor · '+now();
    work.querySelector('.bubble').innerHTML=data.answer||'(no response)';
  }catch(e){
    work.querySelector('.meta').textContent='Advisor · error';
    work.querySelector('.bubble').innerHTML='<p style="color:red">'+e+'</p>';
  }
}
input.addEventListener('keydown',ev=>{
  if(ev.key==='Enter'&&!ev.shiftKey){
    ev.preventDefault();
    const q=input.innerText.trim(); if(!q)return;
    input.innerText=''; ask(q);
  }
});
send.addEventListener('click',()=>{
  const q=input.innerText.trim(); if(!q)return;
  input.innerText=''; ask(q);
});
attach.addEventListener('click',()=>file.click());
file.addEventListener('change',()=>{
  hint.textContent=file.files.length?`${file.files.length} file(s) ready.`:'State your inquiry to receive a response.';
});
</script>
</body></html>
"""

@app.get("/widget", response_class=HTMLResponse)
def widget():
    return HTMLResponse(WIDGET_HTML)

# ======== DIRECT PASSTHROUGH ========
@app.post("/direct")
def direct_endpoint(
    question: str = Form(...),
    files: list[UploadFile] = File(default=[]),
):
    """Pure passthrough: sends user question and optional files straight to GPT."""
    check_rate_limit()
    try:
        # Merge file text if any
        attachments=[]
        for f in files:
            try:
                text=f.file.read().decode("utf-8","ignore")
            except Exception:
                text=""
            if text:
                attachments.append(f"\n\n[File: {f.filename}]\n{text}")
        full_prompt = question + "".join(attachments)
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":full_prompt}],
            max_tokens=2500,
        )
        reply = res.choices[0].message.content.strip()
        return {"answer": reply}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ======== HEALTH ========
@app.get("/health")
def health():
    return {"status": "ok"}
