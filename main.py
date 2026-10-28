import os
import json
import base64
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import edge_tts
from google import genai
from google.genai import types

app = FastAPI(title="GyaanLens AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

class SolveRequest(BaseModel):
    prompt: Optional[str] = ""
    image_base64: Optional[str] = None
    voice_name: str = "hi-IN-MadhurNeural" # Ya "hi-IN-SwaraNeural"

async def text_to_speech_base64(text: str, voice: str) -> str:
    communicate = edge_tts.Communicate(text, voice, rate="+2%", pitch="+0Hz")
    audio = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return base64.b64encode(audio).decode("utf-8")

@app.post("/api/solve")
async def solve(req: SolveRequest):
    if not ai_client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    system_instruction = """
    Aap ek adhyapak hain jo prashnon ko step-by-step whiteboard video roop me samjhate hain.
    Nirdesh:
    1. Bhasha: Shuddh Hindi (Devanagari). Urdu ya Persian shabdon se bachein (prashn, uttar, sutra, charan).
    2. Ganit ke sanketon ko bolne ke liye shabdon me likhein (= ko barabar, + ko jama).
    3. Output KEWAL valid JSON hona chahiye:
    {
      "scenes": [
        {
          "title": "Charan ka shirshak",
          "board": ["Board par likhi jane wali 2-3 lines", "Sutra ya sankhya"],
          "say": "Adhyapak dwara bola jane wala shuddh aur aakarshak vakya.",
          "emoji": "✏️",
          "svg": "<svg viewBox='0 0 300 160' xmlns='http://www.w3.org/2000/svg'><polygon points='50,130 220,130 50,40' fill='none' stroke='#ffd36b' stroke-width='3'/></svg>"
        }
      ]
    }
    """

    contents = []
    if req.image_base64:
        image_bytes = base64.b64decode(req.image_base64.split(",")[-1])
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
    
    user_text = req.prompt if req.prompt else "Is tasveer me diye gaye prashn ko hal karein aur board par samjhayein."
    contents.append(user_text)

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text)
        scenes = data.get("scenes", [])

        # Har scene ke liye Indian Neural Voice banana
        for scene in scenes:
            scene["audio_base64"] = await text_to_speech_base64(scene["say"], req.voice_name)

        return {"status": "success", "scenes": scenes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def ping():
    return {"status": "GyaanLens Engine Live"}
