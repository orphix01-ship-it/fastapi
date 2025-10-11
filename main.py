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
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

# ✅ GET endpoint for iframe
@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Trustee Chat</title>
    <style>body{font-family:system-ui;margin:0;padding:16px}
    #log{border:1px solid #ddd;border-radius:10px;padding:12px;height:420px;overflow:auto;margin-top:8px}</style>
  </head>
  <body>
    <h3>Judicium Sapientiae™ — Trustee Chat</h3>
    <div id="log"></div>
    <form id="f" style="margin-top:10px;display:flex;gap:8px">
      <input id="msg" placeholder="Ask a question..." style="flex:1;padding:10px;border:1px solid #ccc;border-radius:8px">
      <button>Send</button>
    </form>
    <script>
      const log = document.getElementById('log'); const f = document.getElementById('f'); const msg = document.getElementById('msg');
      function append(who,text){const p=document.createElement('div');p.innerHTML='<b>'+who+':</b> '+text;log.appendChild(p);log.scrollTop=log.scrollHeight;}
      f.addEventListener('submit', async (e)=>{
        e.preventDefault();
        const q = msg.value.trim(); if(!q) return;
        append('You', q); msg.value='';
        try{
          const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message:q})});
          const j = await r.json();
          append('AI', j.reply || JSON.stringify(j));
        }catch(err){ append('Error', String(err)); }
      });
    </script>
  </body>
</html>
    """

# ✅ POST API the page calls
@app.post("/api/chat")
async def api_chat(request: Request):
    body = await request.json()
    q = (body.get("message") or "").strip()
    # TODO: replace with your GPT + Pinecone logic
    return JSONResponse({"reply": f"Echo: {q}"})
