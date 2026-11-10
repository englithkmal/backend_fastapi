"""
Backend (FastAPI) between the Flutter app and AnythingLLM.
Run:   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Docs:  http://localhost:8000/docs   (try /ask and /explain from the browser)
"""
import asyncio
import glob
import json
import logging
import os
import re
import time
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, model_validator

load_dotenv()
logging.basicConfig(level=logging.INFO)
BASE_URL = os.getenv("ANYTHINGLLM_URL", "http://localhost:3001").rstrip("/")
API_KEY = os.getenv("ANYTHINGLLM_API_KEY", "")
SLUG = os.getenv("WORKSPACE_SLUG", "")
APP_KEY = os.getenv("APP_KEY", "")   # مفتاح حماية بين Flutter والباك اند (اتركه فارغًا للتطوير)
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))   # كرت واحد 4GB
QUEUE_WAIT = float(os.getenv("QUEUE_WAIT_SECONDS", "60"))
TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
MAX_AUDIO_MB = 15
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "chunks")   # اختياري: مجلد ملفات c_xxxx.txt لاستخراج الصفحة
JSON_MARK = "[[JSON]]"   # تضاف لرسالة /explain؛ تعليمات الـ Workspace تفهمها وترد بـ JSON

log = logging.getLogger("wajeh")
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.S)
HEADER_RE = re.compile(r"\[(الكتاب[^\[\]]*)\]")
HEADERS: dict[str, dict] = {}

app = FastAPI(title="Smart Education Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # dev only
slots = asyncio.Semaphore(MAX_CONCURRENT)   # one GPU: limit simultaneous questions
gemini = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials= True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    """يطبع سبب الـ 422 بالتحديد في الطرفية ويرجعه لـ Flutter."""
    body = await request.body()
    log.warning("422 على %s | الجسم: %r | الأخطاء: %s",
                request.url.path, body[:300], exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": 'طلب غير صالح: أرسل JSON بالشكل {"question": "نص السؤال"}',
                 "errors": jsonable_encoder(exc.errors())},
    )


async def check_key(x_app_key: str = Header(default="")):
    if APP_KEY and x_app_key != APP_KEY:
        raise HTTPException(401, "غير مصرح")


def log_q(kind: str, q: str, found: bool):
    """يسجل كل سؤال (لقياس الـ RAG ونقاط الضعف لاحقًا)."""
    try:
        with open("questions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "kind": kind, "q": q, "found": found},
                               ensure_ascii=False) + "\n")
    except OSError:
        log.warning("تعذّر تسجيل السؤال")


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)

    @model_validator(mode="before")
    @classmethod
    def accept_aliases(cls, data):
        """يقبل question أو message أو q أو text أو query أو prompt."""
        if isinstance(data, dict):
            for k in ("question", "message", "q", "text", "query", "prompt"):
                if isinstance(data.get(k), str):
                    return {"question": data[k].strip()}
        return data


class Source(BaseModel):
    pages: str | None = None
    unit: str | None = None
    lesson: str | None = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class VoiceResponse(AskResponse):
    text: str   # النص المفرَّغ من الصوت


class Block(BaseModel):
    """كتلة واحدة في الشرح التفاعلي: explain | question | summary"""
    type: Literal["explain", "question", "summary"]
    text: str = Field(min_length=1)
    options: list[str] | None = None      # للسؤال فقط (2-4 خيارات)
    answer_index: int | None = None       # للسؤال فقط (يبدأ من 0)
    feedback: str | None = None           # للسؤال فقط: سبب الجواب الصحيح

    @model_validator(mode="after")
    def check(self):
        if self.type == "question":
            if (not self.options or len(self.options) < 2 or self.answer_index is None
                    or not (0 <= self.answer_index < len(self.options))):
                raise ValueError("invalid question block")
        else:
            self.options = self.answer_index = self.feedback = None
        return self


class ExplainResponse(BaseModel):
    title: str
    blocks: list[Block]
    sources: list[Source]
    structured: bool   # False = النموذج لم يرجع JSON صالحًا، والكتلة الوحيدة نص عادي


def parse_header(text: str) -> tuple[dict | None, str]:
    """يقرأ [الكتاب: ... | الصفحة: ...] ويرجع (القيم، النص بدون الترويسة)."""
    m = HEADER_RE.search(text)
    if not m:
        return None, text
    info = {}
    for part in m.group(1).split("|"):
        k, _, v = part.partition(":")
        info[k.strip()] = v.strip()
    return info, text.replace(m.group(0), "", 1)


def load_headers():
    for f in glob.glob(os.path.join(CHUNKS_DIR, "*.txt")):
        try:
            with open(f, encoding="utf-8") as fh:
                info, _ = parse_header(fh.readline())
        except OSError:
            continue
        if info:
            HEADERS[os.path.basename(f)] = info


def parse_source(item: dict) -> Source:
    text = re.sub(r"<document_metadata>.*?</document_metadata>", "",
                  (item.get("text") or ""), flags=re.S).strip()
    info, text = parse_header(text)
    if not info:
        info = HEADERS.get(item.get("title") or "", {})
    return Source(
        pages=info.get("الصفحة") or info.get("الصفحات"),
        unit=info.get("الوحدة"),
        lesson=info.get("الدرس"),
        snippet=" ".join(text.split())[:160],
    )


load_headers()


def clean_answer(text: str) -> str:
    text = THINK_RE.sub("", text or "")
    return re.sub(r"\[?CONTEXT\s*\d+\]?\s*", "", text).strip()   # AnythingLLM internal label


def parse_blocks(raw: str) -> tuple[str, list[Block]] | None:
    """يستخرج JSON من رد النموذج (حتى لو جاء داخل ``` أو بعد <think>) ويتحقق منه كتلةً كتلة."""
    text = THINK_RE.sub("", raw or "")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("blocks"), list):
        return None
    blocks = []
    for b in data["blocks"][:8]:
        if not isinstance(b, dict):
            continue
        if isinstance(b.get("options"), list):
            b["options"] = [str(o).strip() for o in b["options"] if str(o).strip()][:4]
        try:
            blocks.append(Block.model_validate(b))
        except ValidationError:
            continue   # كتلة غير صالحة تُهمل، والبقية تُعرض
    if not blocks:
        return None
    return str(data.get("title") or "").strip()[:120], blocks


async def call_llm(message: str) -> tuple[str, list[Source]]:
    """يرسل الرسالة لـ AnythingLLM ويرجع (النص الخام، المصادر). مشتركة بين كل النقاط."""
    if not API_KEY or not SLUG:
        raise HTTPException(500, "ANYTHINGLLM_API_KEY / WORKSPACE_SLUG غير مضبوطين في ملف .env")
    try:
        await asyncio.wait_for(slots.acquire(), timeout=QUEUE_WAIT)
    except asyncio.TimeoutError:
        raise HTTPException(503, "الخادم مشغول حاليًا، حاول بعد قليل")
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(
                f"{BASE_URL}/api/v1/workspace/{SLUG}/chat",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"message": message, "mode": "query"},
            )
        r.raise_for_status()
        data = r.json()
    except httpx.ConnectError:
        raise HTTPException(502, "تعذّر الاتصال بـ AnythingLLM، تأكد أنه يعمل")
    except httpx.TimeoutException:
        raise HTTPException(504, "تأخر النموذج في الرد")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"AnythingLLM أعاد خطأ {e.response.status_code}")
    except httpx.HTTPError:
        raise HTTPException(502, "خطأ في الاتصال بـ AnythingLLM")
    except ValueError:
        raise HTTPException(502, "رد غير مفهوم من AnythingLLM")
    finally:
        slots.release()

    seen, sources = set(), []
    for item in data.get("sources", []):
        s = parse_source(item)
        key = (s.pages, s.snippet)
        if key not in seen:
            seen.add(key)
            sources.append(s)
    return data.get("textResponse") or "", sources


async def query_llm(question: str) -> AskResponse:
    """جواب نصي عادي (يستخدمه /ask و /voice-chat)."""
    raw, sources = await call_llm(question)
    return AskResponse(answer=clean_answer(raw), sources=sources)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(check_key)])
async def ask(req: AskRequest):
    result = await query_llm(req.question)
    log_q("ask", req.question, bool(result.sources))
    return result


@app.post("/explain", response_model=ExplainResponse, dependencies=[Depends(check_key)])
async def explain(req: AskRequest):
    """شرح تفاعلي: كتل (شرح / سؤال / ملخص) يرسمها تطبيق Flutter."""
    raw, sources = await call_llm(f"{req.question}\n{JSON_MARK}")
    log_q("explain", req.question, bool(sources))
    parsed = parse_blocks(raw)
    if parsed:
        title, blocks = parsed
        return ExplainResponse(title=title, blocks=blocks, sources=sources, structured=True)
    log.warning("explain: JSON غير صالح، رد النموذج: %r", (raw or "")[:400])
    fallback = Block(type="explain", text=clean_answer(raw) or "لم أجد هذا في المنهج")
    return ExplainResponse(title="", blocks=[fallback], sources=sources, structured=False)


@app.post("/voice-chat", response_model=VoiceResponse, dependencies=[Depends(check_key)])
async def voice_chat(file: UploadFile = File(...)):
    if not gemini:
        raise HTTPException(500, "GEMINI_API_KEY غير مضبوط في ملف .env")

    audio = await file.read()
    if not audio:
        raise HTTPException(400, "الملف الصوتي فارغ")
    if len(audio) > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(413, "الملف الصوتي كبير جدًا")

    mime = file.content_type if (file.content_type or "").startswith("audio/") else "audio/mp4"
    try:
        resp = await gemini.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                "فرّغ هذا الصوت إلى نص عربي كما قيل حرفيًا. أرجع النص فقط بدون أي إضافة. "
                "إذا لم يوجد كلام واضح أرجع كلمة EMPTY.",
                types.Part.from_bytes(data=audio, mime_type=mime),
            ],
        )
        text = (resp.text or "").strip()
    except Exception as e:
        raise HTTPException(502, f"فشل تفريغ الصوت: {type(e).__name__}")

    text = text[:500]
    if len(text) < 3 or text == "EMPTY":
        raise HTTPException(422, "لم أتمكن من فهم الصوت، حاول مرة أخرى")

    result = await query_llm(text)
    log_q("voice", text, bool(result.sources))
    return VoiceResponse(text=text, answer=result.answer, sources=result.sources)