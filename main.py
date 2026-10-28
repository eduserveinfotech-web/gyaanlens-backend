import os, json, base64
import edge_tts, httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
VOICE = os.getenv("VOICE", "hi-IN-SwaraNeural")
DID = os.getenv("DID_API_KEY")          # optional: real teacher video
TEACHER = os.getenv("TEACHER_IMAGE_URL")  # optional: teacher photo URL (apni/licensed)

SYSTEM = """तुम एक अनुभवी भारतीय शिक्षक हो। छात्र का प्रश्न (पाठ या फोटो) चरण-दर-चरण समझाओ।
केवल JSON लौटाओ: {"title":str,"scenes":[{"heading":str,"say":str,"lines":[str],"formula":str}]}
नियम: 4 से 7 दृश्य। "say" बोला जाने वाला शुद्ध हिंदी पाठ है: छोटे वाक्य, उर्दू/फ़ारसी/अरबी शब्द नहीं,
नुक्ते वाले अक्षर नहीं, कोई markdown या चिह्न नहीं। "lines" में अधिकतम 3 छोटे बोर्ड-बिंदु।
"formula" में सरल पाठ-सूत्र (जैसे v = u + at), न हो तो खाली। अंतिम दृश्य में उत्तर और सार दो।
फोटो हो तो पहले प्रश्न पढ़ो, फिर हल करो।"""

class Ask(BaseModel):
    text: str = ""
    image_b64: str | None = None
    mime: str = "image/jpeg"

class Say(BaseModel):
    text: str

@app.post("/api/lesson")
async def lesson(a: Ask):
    parts = []
    if a.image_b64:
        parts.append(types.Part.from_bytes(data=base64.b64decode(a.image_b64), mime_type=a.mime))
    parts.append(a.text or "इस फोटो का प्रश्न हल करके समझाओ।")
    try:
        r = await client.aio.models.generate_content(
            model=MODEL, contents=parts,
            config=types.GenerateContentConfig(system_instruction=SYSTEM,
                response_mime_type="application/json", temperature=0.4))
        return json.loads(r.text)
    except Exception as e:
        raise HTTPException(502, f"AI error: {str(e)[:200]}")

cache = {}
@app.post("/api/tts")
async def tts(s: Say):
    if s.text in cache:
        return Response(cache[s.text], media_type="audio/mpeg")
    buf = bytearray()
    async for c in edge_tts.Communicate(s.text, VOICE, rate="-8%").stream():
        if c["type"] == "audio":
            buf += c["data"]
    if not buf:
        raise HTTPException(502, "voice nahi bani")
    if len(cache) > 200:
        cache.clear()
    cache[s.text] = bytes(buf)
    return Response(cache[s.text], media_type="audio/mpeg")

@app.get("/api/config")
def config():
    local = os.path.exists("static/teacher.jpg")
    return {"avatar": bool(DID and TEACHER), "teacher": TEACHER or ("/teacher.jpg" if local else None)}

# --- Optional real-teacher video (D-ID). Endpoint/fields D-ID docs se ek bar verify kar lena ---
@app.post("/api/avatar")
async def avatar(s: Say):
    if not (DID and TEACHER):
        raise HTTPException(404, "avatar off")
    async with httpx.AsyncClient(timeout=30) as h:
        r = await h.post("https://api.d-id.com/talks",
            headers={"Authorization": f"Basic {DID}"},
            json={"source_url": TEACHER, "script": {"type": "text", "input": s.text,
                  "provider": {"type": "microsoft", "voice_id": VOICE}}})
    if r.status_code >= 300:
        raise HTTPException(502, r.text[:200])
    return {"id": r.json()["id"]}

@app.get("/api/avatar/{tid}")
async def avatar_status(tid: str):
    async with httpx.AsyncClient(timeout=30) as h:
        r = await h.get(f"https://api.d-id.com/talks/{tid}", headers={"Authorization": f"Basic {DID}"})
    j = r.json()
    return {"status": j.get("status"), "url": j.get("result_url")}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
