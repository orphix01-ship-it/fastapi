from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

ALLOWED = os.getenv("ALLOWED_ORIGINS", "https://app.motion.io,https://fctadvisor.com").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    msg = body.get("message", "")
    # TODO: call your GPT + Pinecone here instead of echo:
    return {"reply": f"Echo: {msg}"}
