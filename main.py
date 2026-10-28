import os
import json
import base64
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import edge_tts
from google import genai
from google.genai import types

app = FastAPI(title="GyaanLens Visual Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

class SolveRequest(BaseModel):
    prompt: str
    voice_name: str = "hi-IN-MadhurNeural"

async def generate_audio_base64(text: str, voice: str) -> str:
    communicate = edge_tts.Communicate(text, voice, rate="+2%", pitch="+0Hz")
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return base64.b64encode(audio_data).decode("utf-8")

@app.post("/api/solve")
async def solve_question(req: SolveRequest):
    if not ai_client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY server environment mein set nahi hai.")

    system_instruction = """
    Aap ek adhyapak hain jo vigyan aur ganit ke kathin prashnon ko step-by-step visual roop mein samjhate hain.
    Nirdesh:
    1. Bhasha: Shuddh Hindi (Sanskrit mool). Urdu, Arabic ya Angrezi shabdon ka prayog na karein (sawal -> prashn, answer -> uttar, formula -> sutra).
    2. Har step ko JSON array ke roop mein dein.
    3. Output format:
    [
      {
        "step_no": 1,
        "title": "Charan ka shirshak",
        "narration": "Adhyapak dwara bola jane wala shuddh Hindi vakya.",
        "canvas_command": {
          "type": "triangle",
          "points": [[100, 300], [300, 300], [100, 150]],
          "labels": ["A", "B", "C"]
        }
      }
    ]
    Kewal valid JSON return karein, koi extra text nahi.
    """

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=req.prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json"
            )
        )
        steps = json.loads(response.text)

        for step in steps:
            audio_b64 = await generate_audio_base64(step["narration"], req.voice_name)
            step["audio_base64"] = audio_b64

        return {"status": "success", "steps": steps}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "GyaanLens Engine Running"}