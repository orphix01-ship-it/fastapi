from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import os, time

# === AI deps ===
from openai import OpenAI
from pinecone import Pinecone

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "fct-trust-knowledge")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
pc_index = pc.Index(PINECONE_INDEX) if pc else None

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

# -------- Simple GET page for your iframe (unchanged) --------
@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Trustee Chat</title>
    <style>body{font-family:system-ui;margin:0;padding:16px}
    #log{border:1px solid #ddd;border-radius:10px;padding:12px;height:420px;overflow:auto;margin-top:8px}
    input,button{font-size:14px}</style>
  </head>
  <body>
    <h3>Judicium Sapientiae™ — Trustee Chat</h3>
    <div style="color:#666;margin-bottom:6px">Tip: append <code>?project_id=FCT123&trust_id=SAINTBASIL</code> to URL.</div>
    <div id="log"></div>
    <form id="f" style="margin-top:10px;display:flex;gap:8px">
      <input id="msg" placeholder="Ask a question..." style="flex:1;padding:10px;border:1px solid #ccc;border-radius:8px">
      <button>Send</button>
    </form>
    <script>
      function qs(k){return new URLSearchParams(location.search).get(k)||""}
      const pid = qs('project_id') || 'default_project';
      const tid = qs('trust_id') || 'default_trust';
      const log = document.getElementById('log'); const f = document.getElementById('f'); const msg = document.getElementById('msg');
      function append(who,text){const p=document.createElement('div');p.innerHTML='<b>'+who+':</b> '+text;log.appendChild(p);log.scrollTop=log.scrollHeight;}
      f.addEventListener('submit', async (e)=>{
        e.preventDefault();
        const q = msg.value.trim(); if(!q) return;
        append('You', q); msg.value='';
        try{
          const r = await fetch('/api/chat', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ message:q, project_id: pid, trust_id: tid })
          });
          const j = await r.json();
          if(j.citations && j.citations.length){
            append('AI', (j.reply||'') + "<br><small style='color:#666'>Sources: "+j.citations.map(c=>c.id).join(", ")+"</small>");
          } else {
            append('AI', j.reply || JSON.stringify(j));
          }
        }catch(err){ append('Error', String(err)); }
      });
    </script>
  </body>
</html>
    """

# -------- Helper: embeddings + retrieval --------
def embed_text(text: str):
    # text-embedding-3-small is cheaper; use -large for best recall
    res = openai_client.embeddings.create(model="text-embedding-3-small", input=text)
    return res.data[0].embedding

def pinecone_query(embedding, namespace: str, top_k: int = 5):
    if not pc_index:
        return []
    matches = pc_index.query(
        vector=embedding, top_k=top_k, include_metadata=True, namespace=namespace
    )
    out = []
    for m in matches.get("matches", []):
        meta = m.get("metadata") or {}
        out.append({
            "id": meta.get("doc_id") or meta.get("source") or m.get("id"),
            "score": m.get("score"),
            "text": meta.get("text") or ""
        })
    return out

def build_namespace(project_id: str, trust_id: str):
    # Keep namespaces short & consistent
    return f"{project_id}:{trust_id}"

# -------- GPT + Retrieval chat endpoint --------
@app.post("/api/chat")
async def api_chat(request: Request):
    data = await request.json()
    question = (data.get("message") or "").strip()
    project_id = (data.get("project_id") or "default_project").strip()
    trust_id = (data.get("trust_id") or "default_trust").strip()

    if not question:
        return JSONResponse({"reply": "Please ask a question."})

    # 1) Embed the question
    try:
        q_emb = embed_text(question)
    except Exception as e:
        return JSONResponse({"reply": f"Embedding error: {e}"}, status_code=500)

    # 2) Retrieve from Pinecone
    ns = build_namespace(project_id, trust_id)
    retrieved = []
    try:
        retrieved = pinecone_query(q_emb, namespace=ns, top_k=5)
    except Exception as e:
        # Retrieval is optional; still answer with GPT if Pinecone fails
        retrieved = []

    # 3) Build context for GPT
    context_chunks = "\n\n".join(
        [f"[{i+1}] {m['text']}" for i, m in enumerate(retrieved) if m.get("text")]
    )
    system_msg = (
        "You are Judicium Sapientiae, a fiduciary AI advisor for private trusts. "
        "Answer concisely and cite sources as [1], [2], ... matching the context list if used."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Context (optional):\n{context_chunks if context_chunks else '(no context found)'}\n\n"
        f"Instructions: If context is relevant, cite it as [n]. If not, answer from general knowledge clearly."
    )

    # 4) Ask GPT
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        return JSONResponse({"reply": f"Model error: {e}"}, status_code=500)

    # 5) Return with simple citations list
    citations = [{"id": m["id"], "score": m["score"]} for m in retrieved[:3]]
    return JSONResponse({"reply": answer, "citations": citations})
