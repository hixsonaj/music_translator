"""
Simple REST API
--------------
Exposes the lyric translation pipeline as an HTTP endpoint
so other parts of the project can plug in easily.

POST /translate
  Input:  { "text": "Never gonna give you up", "target_lang": "es" }
  Output: { "translation": "Nunca te voy a soltar", "attempts": 1, "score": "100%" }

Run with:
  python api.py
"""
import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uagents.communication import send_sync_message
from models import LyricLine, AcceptedLine
import re
import uvicorn

app = FastAPI(title="Lyric Translator API")

# Allow requests from anywhere (useful for hackathon)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ORCHESTRATOR_ADDR = os.getenv("ORCHESTRATOR_ADDR", "")


class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "es"  # default to Spanish


class TranslateResponse(BaseModel):
    original: str
    translation: str
    target_lang: str
    attempts: int


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    syllables = max(1, len(re.findall(r'[aeiouAEIOU]+', req.text)))
    line_id = abs(hash(req.text + req.target_lang)) % 100000

    result = await send_sync_message(
        destination=ORCHESTRATOR_ADDR,
        message=LyricLine(
            line_id=line_id,
            original=req.text,
            translated="",
            target_lang=req.target_lang,
            target_syllables=syllables,
        ),
        response_type=AcceptedLine,
        timeout=60,
    )

    return TranslateResponse(
        original=req.text,
        translation=result.final_translation,
        target_lang=req.target_lang,
        attempts=result.attempts,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "orchestrator": ORCHESTRATOR_ADDR}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
