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

SYSTEM = """तुम एक स्नेही और धैर्यवान भारतीय कक्षा-शिक्षक हो। बच्चों को सामने बैठाकर बोर्ड पर लिखते हुए, धीरे-धीरे, एक-एक बात करके समझाओ। छात्र का प्रश्न (पाठ या फोटो) लो।
केवल JSON लौटाओ: {"title":str,"scenes":[{"heading":str,"say":str,"write":[str],"formula":str,"pause":int,"anim":obj|null}]}
नियम:
- 8 से 14 दृश्य। हर दृश्य में केवल एक ही बात या एक ही चरण समझाओ।
- "say": 4 से 6 छोटे वाक्य, शिक्षक की बोलचाल की शुद्ध हिंदी। पहले बात बताओ, फिर कारण, फिर रोज के जीवन का एक उदाहरण। बीच-बीच में बच्चे से पूछो (जैसे "क्या आप समझ गए?", "जरा सोचिए...") और जरूरी बात दोहराओ। उर्दू, फारसी, अरबी शब्द और नुक्ते वाले अक्षर नहीं। कोई markdown या चिह्न नहीं।
- "heading": केवल नए भाग की शुरुआत में शीर्षक दो (इससे बोर्ड साफ होगा), बाकी दृश्यों में खाली "" रखो।
- "write": इसी दृश्य में बोर्ड पर लिखने के 1 या 2 बहुत छोटे बिंदु।
- "formula": सरल पाठ-सूत्र (जैसे v = u + at), न हो तो खाली।
- "pause": बच्चे के सोचने के लिए रुकने के सेकंड (0 से 6); प्रश्न पूछने वाले दृश्य में 4 से 6।
- "anim": हर दृश्य में एनिमेशन दो, चाहे कोई भी विषय हो (गणित, जीव विज्ञान, इतिहास, भूगोल, अंग्रेजी, कुछ भी)। null केवल पहले और अंतिम दृश्य में। जहाँ ऊपर का कोई तैयार प्रकार बैठे वहाँ वही लो, वरना "scene" प्रकार से अपना एनिमेशन बनाओ। प्रकार:
  रसायन प्रयोग: {"type":"mix","a":{"c":"#hex","l":"नाम या सूत्र"},"b":{"c":"#hex","l":"नाम या सूत्र"},"r":{"c":"#hex","l":"उत्पाद","fx":"bubbles|precip|glow|none","p":"#hex अवक्षेप रंग"},"cap":"संतुलित समीकरण"}
  भौतिकी: {"type":"projectile","v":20,"a":45,"cap":"..."}, {"type":"pendulum","L":1,"a":30,"cap":"..."}, {"type":"circuit","V":6,"R":3,"cap":"..."}, {"type":"wave","A":1,"wl":2,"cap":"..."}
  अपना एनिमेशन: {"type":"scene","cap":"छोटा शीर्षक","items":[{"k":"emoji|text|circle|box|arrow","t":"लिखावट या इमोजी","x":0-100,"y":0-100,"to":[x,y],"e":[x2,y2],"at":सेकंड,"c":"#hex","s":आकार 30-80}]}। x,y पर्दे की प्रतिशत स्थिति हैं (0 बाएँ/ऊपर, 100 दाएँ/नीचे)। 4 से 8 वस्तुएँ दो, "at" 0 से 20 के बीच बढ़ते क्रम में, ठीक वैसे जैसे तुम बोलते हो (जो बोलो वही उसी समय दिखे)। "to" से वस्तु चलती है (जैसे पौधे की ओर सूरज की किरण, कोशिका में पदार्थ का आना), "arrow" में "e" अंतिम बिंदु है। circle/box में "t" उसका नाम है। रंग हमेशा 6 अंकों का hex।
  रंग हमेशा 6 अंकों का hex। एनिमेशन वाले दृश्य में कम से कम 6 वाक्य बोलो, ताकि मिलाना, बदलाव और निष्कर्ष तीनों समझाए जा सकें। असली प्रयोग जैसा क्रम रखो।
- पहला दृश्य: नमस्ते और आज क्या सीखेंगे। अंतिम दृश्य: सार और अंतिम उत्तर।
- फोटो हो तो पहले प्रश्न पढ़कर सुनाओ, फिर हल करो।"""

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
    async for c in edge_tts.Communicate(s.text, VOICE, rate="-18%").stream():
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

@app.get("/")
def root():
    return {"status": "GyaanLens Engine Live"}

if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static"), name="static")
