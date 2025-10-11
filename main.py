from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, hmac, hashlib, time, jwt, csv, io

APP_SECRET = os.getenv("APP_SECRET", "dev_secret")
ALLOWED = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="FCT Advisor GPT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

def verify_sig(pid:str, cid:str, ts:str, sig:str):
    msg = f"{pid}|{cid}|{ts}".encode()
    good = hmac.new(APP_SECRET.encode(), msg, hashlib.sha256).hexdigest()
    return hmac.compare_digest(good, sig)

def issue_jwt(subject:str, plan:str, project_id:str):
    now = int(time.time())
    payload = {"sub": subject, "plan": plan, "project_id": project_id, "iat": now, "exp": now + 600}
    return jwt.encode(payload, APP_SECRET, algorithm="HS256")

@app.get("/auth/handshake")
async def handshake(pid:str, cid:str, tier:str, ts:str, sig:str):
    if abs(int(time.time()) - int(ts)) > 300:
        raise HTTPException(401, "Signature expired")
    if not verify_sig(pid, cid, ts, sig):
        raise HTTPException(401, "Bad signature")
    token = issue_jwt(subject=cid, plan=tier, project_id=pid)
    return {"token": token}

class ChatBody(BaseModel):
    message: str

@app.post("/chat")
async def chat(body: ChatBody):
    # TODO: Replace this with your GPT + Pinecone logic
    return {"reply": f"Echo: {body.message}", "citations": []}

@app.post("/tools/ledger-categorize")
async def ledger(file: UploadFile = File(...)):
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode()))
    rows = list(reader)
    output = []
    for r in rows:
        amt = float(r.get("amount","0") or 0)
        tag = "Reinvestment" if "equipment" in (r.get("memo","").lower()) else ("Income" if amt>0 else "Expense")
        output.append({"date": r.get("date"), "amount": amt, "memo": r.get("memo"), "category": tag, "irc": "§212"})
    return {"rows": output, "summary": {"count": len(output)}}

@app.post("/tools/generate-doc")
async def generate_doc(request: Request):
    data = await request.json()
    # Placeholder document generator
    return {"docx_url": "s3://fct-trust-files/resolution.docx", "pdf_url": "s3://fct-trust-files/resolution.pdf"}

@app.post("/webhooks/motionio")
async def motionio_webhook(request: Request):
    body = await request.json()
    print("Motion.io event:", body)
    return {"ok": True}

@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    body = await request.body()
    print("Stripe webhook:", body[:200])
    return {"ok": True}
