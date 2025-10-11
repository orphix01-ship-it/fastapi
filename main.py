# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import os

app = FastAPI()

ALLOWED = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "https://app.motion.io,https://fctadvisor.com,https://www.fctadvisor.com"
).split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED,
    allow_credentials=True,
    allow_methods=["*"],   # includes OPTIONS for CORS
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return """
<!doctype html><meta charset="utf-8"><title>Trustee Chat</title>
<div style="font-family:system-ui;padding:16px;max-width:720px;margin:auto">
  <h3>Judicium Sapientiae™ — Trustee Chat</h3>
  <div id="log" style="border:1px solid #ddd;border-radius:10px;padding:12px;height:420px;overflow:auto;margin-top:8px"></div>
  <form id="f" style="margin-top:10px;display:flex;gap:8px">
    <input id="msg" placeholder="Ask a question..." style="flex:1;padding:10px;border:1px solid #ccc;border-radius:8px">
    <button>Send</button>
  </form>
</div>
<script>
const log = document.getElementById('log'); const f=document.getElementById('f'); const msg=document.getElementById('msg');
function line(who,t){const p=document.createElement('div');p.innerHTML='<b>'+who+':</b> '+t;log.appendChild(p);log.scrollTop=log.scrollHeight;}
f.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const q = (msg.value||'').trim(); if(!q) return; line('You', q); msg.value='';
  try{
    const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message:q})});
    const j = await r.json(); line('AI', j.reply || JSON.stringify(j));
  } catch(err){ line('Error', String(err)); }
});
</script>
"""

@app.post("/api/chat")
async def api_chat(request: Request):
    # Defensive parsing so empty/invalid JSON won’t crash
    try:
        data = await request.json()
    except Exception:
        data = {}
    q = (data.get("message") or "").strip()
    if not q:
        return JSONResponse({"reply": "Please type a message."})
    # TODO: replace with GPT+Pinecone; for now, respond safely
    return JSONResponse({"reply": f"Echo: {q}"})
