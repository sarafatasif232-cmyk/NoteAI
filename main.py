"""
LectureNotes AI — Backend
--------------------------------------------------
FastAPI service that:
  1. Accepts a recorded lecture audio file from the frontend
  2. Uploads it to Gemini's File API (Gemini can read audio natively —
     no separate Whisper transcription step is required)
  3. Asks Gemini to transcribe + analyze the lecture in one call,
     constrained to a strict JSON schema
  4. Returns structured study notes + predicted exam questions

Run locally:
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000
"""

import os
import tempfile
import time
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lecture-notes")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set — requests to /api/generate-notes will fail.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="LectureNotes AI API")

# Allow the local frontend (and any dev server) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB safety cap
ALLOWED_CONTENT_TYPES = {
    "audio/webm", "audio/wav", "audio/x-wav", "audio/mpeg",
    "audio/mp3", "audio/ogg", "audio/mp4", "audio/m4a",
}


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class Subtopic(BaseModel):
    heading: str
    points: List[str]


class Topic(BaseModel):
    heading: str
    summary: str
    subtopics: List[Subtopic] = Field(default_factory=list)


class ExamQuestion(BaseModel):
    type: str  # "Short", "Broad", or "MCQ"
    question: str
    options: Optional[List[str]] = None  # populated only for MCQ
    answer_hint: str


class LectureNotes(BaseModel):
    lecture_title: str
    topics: List[Topic]
    key_takeaways: List[str]
    exam_questions: List[ExamQuestion]


# Gemini structured-output schema (mirrors the Pydantic model above).
# Using response_schema forces Gemini to return valid, parseable JSON.
GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "lecture_title": {"type": "string"},
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "summary": {"type": "string"},
                    "subtopics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "heading": {"type": "string"},
                                "points": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["heading", "points"],
                        },
                    },
                },
                "required": ["heading", "summary", "subtopics"],
            },
        },
        "key_takeaways": {"type": "array", "items": {"type": "string"}},
        "exam_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["Short", "Broad", "MCQ"]},
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "answer_hint": {"type": "string"},
                },
                "required": ["type", "question", "answer_hint"],
            },
        },
    },
    "required": ["lecture_title", "topics", "key_takeaways", "exam_questions"],
}

SYSTEM_PROMPT = """You are an expert academic teaching assistant helping a university
student review a recorded lecture. You will be given the raw audio of a classroom
lecture. Listen to it carefully and produce structured study notes.

Rules:
- Base everything strictly on what is actually said in the lecture audio. Never invent
  facts, dates, or figures that were not mentioned.
- Organize content the way a diligent student would write it in a notebook: clear
  topic headings, short subtopics, and concise bullet points (not long paragraphs).
- "key_takeaways" should capture what the professor visibly emphasized (repeated,
  said "this is important", "this will be on the exam", spoke slowly about, etc.),
  not just a generic summary.
- Produce exactly 3 to 5 "exam_questions", using a mix of Short, Broad, and MCQ types
  where the content supports it. Each question must be answerable from the lecture
  content alone, and each needs a one- to two-sentence answer_hint pointing the
  student to the right concept (not a full answer key).
- For MCQ questions, include 4 "options" with exactly one correct answer reflected
  in the answer_hint.
- If the audio is very short, unclear, or contains little academic content, still
  return valid JSON with your best-effort notes and note the limitation inside
  "key_takeaways".
- Respond with JSON only, matching the provided schema exactly.
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY)}


@app.post("/api/generate-notes", response_model=LectureNotes)
async def generate_notes(audio: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")

    if audio.content_type not in ALLOWED_CONTENT_TYPES:
        logger.info("Unrecognized content-type %s — attempting to process anyway.", audio.content_type)

    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    tmp_path = None

    try:
        # Stream the upload to a temp file rather than loading it fully into memory.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            size = 0
            while chunk := await audio.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Audio file too large (max 200MB).")
                tmp.write(chunk)

        logger.info("Uploading %s (%d bytes) to Gemini File API...", audio.filename, size)
        gemini_file = genai.upload_file(path=tmp_path, mime_type=audio.content_type or "audio/webm")

        # Gemini processes uploaded files asynchronously; poll until it's ready.
        while gemini_file.state.name == "PROCESSING":
            time.sleep(1.5)
            gemini_file = genai.get_file(gemini_file.name)

        if gemini_file.state.name != "ACTIVE":
            raise HTTPException(status_code=502, detail="Gemini could not process the audio file.")

        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )

        response = model.generate_content(
            [gemini_file, "Analyze this lecture recording and return the structured study notes JSON."],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": GEMINI_RESPONSE_SCHEMA,
                "temperature": 0.3,
            },
        )

        # Clean up the remote file now that we have the result.
        try:
            genai.delete_file(gemini_file.name)
        except Exception:
            pass  # non-fatal — Gemini also auto-expires files after 48h

        notes = LectureNotes.model_validate_json(response.text)
        return notes

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to generate notes")
        raise HTTPException(status_code=500, detail=f"Failed to generate notes: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Serve the frontend (must be mounted AFTER the /api/* routes above so those
# take priority; this mount catches every other path, e.g. "/" and "/index.html").
# Visiting http://localhost:8000/ now serves the whole app from one origin,
# so there's no cross-origin / file:// fetch failure to worry about.
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning("Frontend directory not found at %s — only the API will be served.", FRONTEND_DIR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
