import os
from typing import Optional
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

print(f"Loading .env from: {ENV_PATH}")

GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
MURF_API_KEY: Optional[str] = os.getenv("MURF_API_KEY")

if not GROQ_API_KEY or not MURF_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY/GROK_API_KEY or MURF_API_KEY.")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    audio_base64: str

app = FastAPI(title="Groq + Murf Voice Agent (Fast)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Smaller, faster model
GROQ_MODEL = "llama-3.1-8b-instant"


def call_groq_llm(user_message: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a friendly, fast voice assistant. Reply in 1–2 short sentences.",
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "stream": False,
        "temperature": 0.5,
        # ✅ Limit length to reduce latency
        "max_tokens": 96,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_murf_tts(text: str) -> str:
    url = "https://api.murf.ai/v1/speech/generate"
    headers = {
        "api-key": MURF_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "voiceId": "en-US-natalie",
        "format": "MP3",
        "encodeAsBase64": True,
        "modelVersion": "GEN2",
        "channelType": "MONO",
        "sampleRate": 44100,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    encoded_audio = data.get("encodedAudio")
    if not encoded_audio:
        raise RuntimeError("Murf response did not contain encodedAudio")
    return "data:audio/mp3;base64," + encoded_audio


@app.get("/")
def root():
    return {"status": "ok", "message": "Groq + Murf voice agent backend running (fast model)"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply_text = call_groq_llm(req.message)
    audio_data_url = call_murf_tts(reply_text)
    return ChatResponse(reply=reply_text, audio_base64=audio_data_url)
