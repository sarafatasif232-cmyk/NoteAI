# LectureNotes AI

Record a lecture in the browser, send the audio to Gemini, and get back
structured study notes and predicted exam questions.

## Architecture

```
┌─────────────────┐        1. record audio (MediaRecorder API)
│   Browser (UI)   │───────────────────────────────┐
│  index.html      │                                │
│  - Start/Stop     │  2. POST /api/generate-notes  ▼
│  - Live visualizer│     multipart/form-data   ┌─────────────────┐
│  - Notes display  │◄───────────────────────────│  FastAPI backend │
└─────────────────┘   6. structured JSON notes  │    main.py       │
                                                  └────────┬─────────┘
                                                           │ 3. upload audio file
                                                           ▼
                                                  ┌─────────────────┐
                                                  │  Gemini File API │
                                                  └────────┬─────────┘
                                                           │ 4. generate_content()
                                                           │    system prompt +
                                                           │    JSON schema
                                                           ▼
                                                  ┌─────────────────┐
                                                  │  Gemini model     │
                                                  │  (transcribes +   │
                                                  │   analyzes audio  │
                                                  │   in one call)    │
                                                  └────────┬─────────┘
                                                           │ 5. JSON: topics,
                                                           │    takeaways,
                                                           │    exam questions
                                                           └──────────►(back to backend)
```

**Why no separate Whisper step?** Gemini 2.0/2.5 models accept audio directly
as input, so a single API call handles transcription *and* analysis. This
keeps the pipeline simpler and avoids paying for/hosting a separate STT
service. If you'd rather use OpenAI Whisper (e.g. for cost control, offline
transcription, or to keep raw audio off Gemini), see "Swapping in Whisper"
below — the JSON-analysis step stays the same, you just feed it text instead
of audio.

## Tech stack

| Layer          | Choice                                   | Why |
|----------------|-------------------------------------------|-----|
| Frontend       | Plain HTML + Tailwind (CDN) + vanilla JS   | Zero build step, `MediaRecorder`/`Web Audio` APIs are native, easy to swap for React later without changing the backend contract |
| Backend        | Python + FastAPI                          | Async file uploads, automatic OpenAPI docs, thin layer around the Gemini SDK |
| AI             | Google Gemini API (`gemini-2.0-flash`)     | Native audio understanding + structured JSON output (`response_schema`) in one call |
| Transport      | `multipart/form-data` over REST            | Simple, debuggable, no websocket complexity needed since this is single-shot (not streaming) analysis |

## Project structure

```
lecture-notes-app/
├── backend/
│   ├── main.py            FastAPI app + Gemini integration
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── index.html         Recording UI + notes display (open directly, or serve statically)
```

## Setup

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your key from https://aistudio.google.com/apikey

uvicorn main:app --reload --port 8000
```

Verify it's running: open `http://localhost:8000/api/health` — you should see
`{"status": "ok", "gemini_configured": true}`.

### 2. Frontend

No separate step needed — `main.py` serves `frontend/index.html` itself, so
the whole app runs from the one `uvicorn` command above.

**Open `http://localhost:8000/` in your browser** (not `file:///...index.html`
— opening the file directly triggers "Failed to fetch" in most browsers
because a `file://` page can't call `fetch()` against `localhost`).

`MediaRecorder` requires a secure context: `localhost` is fine, but a real
deployment needs HTTPS.

> Prefer to run the frontend on its own dev server instead (e.g. Vite, Live
> Server)? Set `const API_BASE = "http://localhost:8000";` back in
> `index.html` and keep the CORS middleware in `main.py` enabled.

### 3. Try it

1. Click the record button, grant microphone permission, talk through a mock
   "lecture" for 30–60 seconds.
2. Click stop.
3. Click **Generate class notes** — the audio uploads to the backend, which
   forwards it to Gemini and returns structured notes within roughly 10–30
   seconds depending on length.

## API

### `POST /api/generate-notes`

- **Body:** `multipart/form-data` with a single field `audio` (webm/wav/mp3/m4a)
- **Response:**

```json
{
  "lecture_title": "Introduction to Cellular Respiration",
  "topics": [
    {
      "heading": "Glycolysis",
      "summary": "Breakdown of glucose into pyruvate in the cytoplasm.",
      "subtopics": [
        { "heading": "Net ATP yield", "points": ["2 ATP produced net", "Occurs without oxygen"] }
      ]
    }
  ],
  "key_takeaways": ["Professor emphasized glycolysis is anaerobic and happens in the cytoplasm, not the mitochondria."],
  "exam_questions": [
    {
      "type": "MCQ",
      "question": "Where does glycolysis take place in the cell?",
      "options": ["Mitochondrial matrix", "Cytoplasm", "Nucleus", "Golgi apparatus"],
      "answer_hint": "Glycolysis occurs in the cytoplasm — the professor stressed this is a common exam mix-up with the mitochondria."
    }
  ]
}
```

## Production considerations (not included in this blueprint)

- **Auth & persistence:** add user accounts and a database (Postgres) to save
  past lectures/notes instead of holding state only in the browser tab.
- **Long lectures:** a 90-minute lecture audio file can be large; consider
  chunking uploads or compressing to a lower bitrate client-side before
  sending.
- **Async processing:** for long-running Gemini calls, move `/api/generate-notes`
  to a background job (e.g. Celery/RQ) and let the frontend poll or use a
  websocket, rather than holding an HTTP request open for 30+ seconds.
- **Rate limiting & auth on the API** before exposing this beyond localhost.
- **CORS:** tighten `allow_origins` in `main.py` to your real frontend domain.

## Swapping in Whisper instead of Gemini's native audio input

If you'd rather transcribe with OpenAI Whisper (or a local `faster-whisper`
model) and only send text to Gemini:

1. Add `openai` to `requirements.txt`.
2. Replace the `genai.upload_file(...)` block in `main.py` with:
   ```python
   from openai import OpenAI
   client = OpenAI()
   transcript = client.audio.transcriptions.create(
       model="whisper-1",
       file=open(tmp_path, "rb"),
   ).text
   ```
3. Pass `transcript` (a plain string) to `model.generate_content([transcript, ...])`
   instead of `gemini_file`. The system prompt and JSON schema stay identical.
