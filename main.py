# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import os

app = FastAPI()

ALLOWED = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "https://app.motion.io,https://www.fctadvisor.com"
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

@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return "<h3>Chat OK</h3>"

@app.post("/api/chat")
async def api_chat(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    q = (data.get("message") or "").strip()
    return JSONResponse({"reply": f"Echo: {q or '…'}"})
