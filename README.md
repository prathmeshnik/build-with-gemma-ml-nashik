# Privacy-First On-Premise AI Interviewer

**Kaggle "Build with Gemma — ML Nashik" — Privacy-First Interviewer Track**

> **THIS FILE IS THE SINGLE SOURCE OF TRUTH.** Everything else in the repo was
> deleted. Every file (`lib/*`, `server.py`, `static/index.html`,
> `test_pipeline.py`, `requirements.txt`, `setup.sh`, `AGENTS.md`, `.gitignore`,
> `.claude/`, the Kaggle notebook) is regenerated from the specs in this
> document. Read it top to bottom before writing a single line of code.

---

## 1. TL;DR

A fully-local AI interviewer: **speech-to-text → LLM → text-to-speech**, three
models co-resident on a 6 GB GPU.

| Stage | Model | VRAM | Role |
|---|---|---|---|
| STT | `faster-whisper` small (CTranslate2, int8) | ~0.5 GB | candidate audio → transcript |
| LLM | Gemma (GGUF) via `llama.cpp` server | ~2.5 GB | transcript → JSON eval + follow-up |
| TTS | Kokoro 82M | ~0.4 GB | question/follow-up text → speech |

**Total ~3.4 GB of 6 GB.** The pipeline runs the three models *sequentially* per
turn, so VRAM contention is never an issue — they just share residency.

Privacy promise: **zero outbound network connections during a turn.** Audio never
leaves the machine, transcription never leaves the machine, and — because TTS is
Kokoro (fully local) — *even the interviewer's spoken questions never leave the
machine.* No cloud STT, no cloud TTS, no cloud LLM.

---

## 2. Why We Pivoted (the failure that shaped this stack)

### 2.1 What we tried first

The original design used **Gemma 4's any-to-any pipeline** via `transformers`:
one in-process model that ingested raw audio + text and emitted JSON. Model:
`google/gemma-4-E2B-it-qat-mobile-transformers` (Google QAT Mobile, wNa8o8
quant, ~2.5 GB on disk).

### 2.2 Why it failed (exact issues, not hand-waving)

1. **It would not load/run on the RTX 3050 6 GB.** Even though the weights are
   2.5 GB, the QAT mobile checkpoint dequantizes its *full* 262144×1536 vocab
   tables (main embedding + `lm_head`) on **every forward call** — ~2.3 GiB
   transient spikes per decode step. On top of resident weights, a 6 GB card
   thrashes and OOMs.
2. **fp32 PLE upcast.** The quantized per-layer embedding was built with
   `output_dtype=fp32`, which silently upcast the *whole* graph to fp32 — the
   original slowness and a dtype crash. "Fixing" it required surgical
   `module.output_dtype = torch.bfloat16` patches per layer.
3. **Materials-workaround code.** Getting it to run at all meant materializing
   dequantized plain `nn.Embedding`/`nn.Linear` tables before `model.to(device)`
   (see the old `chat.py`). This is fighting the library, not using it.
4. **Version drift.** Installed `transformers==5.14.1` vs. the 4.54+ path the QAT
   code assumed. Fragile matrix of Gemma 4 × QAT × transformers versions.
5. **Cloud-TTS caveat.** gTTS/edge-tts sent interviewer *text* to a cloud TTS —
   a documented privacy leak that weakened the competition pitch.
6. **Constrained decoding complexity.** `lm-format-enforcer` for guaranteed JSON
   is another moving part on top of an already-fragile stack.
7. **Doc phantoms.** Docs referenced `app.py`, `lib/interview_flow.py`,
   `lib/privacy_audit.py`, `lib/conversation_manager.py` — which never existed.

### 2.3 Why the pivot works

- **The app becomes a thin HTTP client.** No model load in-process, no OOM, no
  transformers version matrix. `llama.cpp` server does the heavy lifting.
- **Proven components.** llama.cpp + GGUF, faster-whisper, and Kokoro are each
  battle-tested and GPU-light. Whisper wants 16 kHz mono — the exact audio format
  we already produce.
- **TTS is now fully local** → the "zero outbound connections" claim becomes
  100% true. Stronger competition pitch.
- **Each stage is independently testable** and independently swappable (see
  §9 and §7 fallback knobs).

### 2.4 What we kept / deleted

**Kept and reused:** `torchaudio` for resampling, the EVAL schema, the
retry+regex JSON fallback, `QuestionBank` (`lib/question_loader.py`),
`lib/logger.py`, the Piper TTS binary (as silent fallback), the question-bank
data (`questions/ml_entry_level.json`).

**Deleted:** in-process transformers loading, `lm-format-enforcer`, cloud TTS, the
30-second audio padding for the any-to-any window, dead chat REPLs (`chat.py`,
`vllm_chat.py`), and the 5.8 GB of QAT weights (`models/`) that llama.cpp can't
use directly.

**Restored (not Gradio):** the UI is **FastAPI + a hand-written static HTML page**
(`server.py` + `static/index.html`) — the same dependency weight as Gradio
(Gradio pulls fastapi/uvicorn in transitively anyway), but with an already-written
browser UI (MediaRecorder mic, JS WAV encoder, phase indicators, base64 audio
playback) and same-origin requests. Gradio stays **notebook-only**.

---

## 3. Architecture & Data Flow

### 3.1 System diagram

```
┌────────────────────────────── Local Machine (6 GB GPU) ──────────────────────────────┐
│                                                                                       │
│  Browser (static/index.html served by uvicorn :8000)                                  │
│    │  MediaRecorder mic → JS encodeWav() → POST /process (raw WAV bytes)   ────┐     │
│    ▼                                                                             │     │
│  server.py  (FastAPI, orchestration lives here — no separate flow module)       │     │
│    │                                                                             │     │
│    ├──► lib/audio_capture.py     to_mono_16k(wav_bytes) → 16k mono float32      │     │
│    │        ▲                                                                │       │
│    │        └── torchaudio Resample (16 kHz, never scipy)                    │       │
│    │                                                                          │       │
│    ├──► lib/stt.py               faster-whisper small/int8  (~0.5 GB)        │       │
│    │        └── transcript string                                             │       │
│    │                                                                          │       │
│    ├──► lib/interview_model.py   httpx POST http://localhost:8080/v1/chat/   │       │
│    │        └── transcript + eval prompt ──► llama.cpp server (Gemma GGUF)   │       │
│    │                                   │    ~2.5 GB, JSON out + retry/regex   │       │
│    │                                   │                                      │       │
│    │        ◄── scores / gaps / follow-up JSON                               │       │
│    │                                                                          │       │
│    ├──► lib/tts.py                Kokoro 82M (~0.4 GB) → question audio      │       │
│    │        └── WAV bytes (base64) → browser <audio> playback                 │       │
│    │                                                                          │       │
│    └──► lib/logger.py             save transcript → interviews/YYYYMMDD_HHMMSS.json │
│                                                                                       │
│  lib/privacy_audit.py — psutil snapshot, asserts 0 new outbound connections   │       │
└───────────────────────────────────────────────────────────────────────────────────────┘

   NOTE: llama.cpp server runs as a SEPARATE process on localhost:8080,
   started by the user (see §5.3). server.py never loads an LLM in-process.
```

### 3.2 Turn lifecycle

**Phase A — warm-up (never scored).** The interview never starts with a scored
question. `POST /start` returns a friendly greeting that asks the candidate's
name. Each warm-up answer is transcribed and the candidate's actual words are
passed to the LLM so the interviewer can respond to what was said, then ask the
next topic: name → what they're currently doing → a bit about themselves →
their interest in machine learning/this field. After the last topic the model
transitions ("warm-up done, the real interview starts now") and asks the first
scored question in a warm, conversational tone.

**Phase B — scored interview (one candidate answer):**

1. User records a response in the browser mic via MediaRecorder (up to 2 min).
2. JS `encodeWav()` converts to WAV bytes → `POST /process` (raw body).
   `lib.audio_capture.to_mono_16k(wav_bytes)` → 16k mono float32, **no padding**,
   truncated at 2 min.
3. `lib.stt.transcribe(audio)` → faster-whisper → transcript string.
4. `transcribe_and_eval(transcript, question, keywords)` → JSON (retry + regex).
5. UI shows transcript + scores + a **Coach** note: what the candidate covered
   well and one concrete improvement pointer.
6. If follow-up is warranted (avg < 7, gaps present, < 2 follow-ups so far):
   a spoken coach line acknowledges + delivers the tip, then asks the
   follow-up → record again (loop to step 1, `followup_count += 1`).
7. Otherwise a spoken coach line acknowledges + delivers the tip, then asks the
   next question in a warm, conversational tone (or, when the bank is
   exhausted, thanks the candidate, says they'll be in touch via email/their
   contact method, wishes them luck, and says goodbye).
8. **End** → `POST /end` → summary + `logger.save_transcript()` writes
   `interviews/<ts>.json`.

Phase indicators shown to the candidate: `Transcribing → Evaluating → Follow-up`.

### 3.3 Why co-residency is safe

The models never run concurrently — a turn is strictly STT-then-LLM-then-TTS.
3.4 GB peak is measured when all three are *loaded and idle* (verified in §9.3).
If a card ever OOMs, the §7 fallback knobs drop that further.

---

## 4. Module Rebuild Contract

Regenerate every file from these specs. Signatures are the contract; small
implementation details left to the writer must match the documented behavior.

### 4.0 `requirements.txt`

```
fastapi>=0.115
uvicorn[standard]>=0.34
httpx>=0.28
numpy>=1.24
torch>=2.5.0
torchaudio>=2.5.0
faster-whisper>=1.0.0
kokoro>=0.8.0
psutil>=5.9.0
```

**Explicitly NOT included:** `gradio`, `transformers`, `torchvision`, `Pillow`,
`lm-format-enforcer`, `python-multipart`. The local stack needs `torch`/`torchaudio`
(Kokoro + resampling), `faster-whisper` (CTranslate2 is pulled in as a dependency),
and `fastapi`/`uvicorn` (the UI server). `python-multipart` is unnecessary —
`/process` takes the raw WAV body, no form parsing. Gradio and `transformers`
are Kaggle-notebook-only.

### 4.1 `lib/__init__.py`

Empty file (makes `lib/` a package).

### 4.2 `lib/audio_capture.py` — audio plumbing

**Audio contract (project-wide):** 16 kHz, mono, `np.float32`, normalized to
[-1, 1], max 2 min, in-memory only. Never written to disk.

```python
TARGET_SR = 16000
MAX_SAMPLES = int(TARGET_SR * 120.0)   # 1_920_000 — hard cap (2 min), no padding

def to_mono_16k(audio, sample_rate=None) -> np.ndarray:
    """Normalize ANY input to the audio contract. Returns np.float32 16k mono.

    - If input is bytes -> parse WAV (handles 1/2/4-byte sample widths).
    - Downmix stereo/multi-channel to mono (mean across channels).
    - Resample to 16 kHz with torchaudio.transforms.Resample — NEVER
      scipy.signal.resample (FFT artifacts).
    - Convert to float32, normalize to [-1, 1] (div by abs-max if > 0).
    - Truncate to MAX_SAMPLES (2 min). Do NOT zero-pad — the STT path
      transcribes real audio only.
    """
```

Keep the private `_from_wav(bytes)` helper. **The old `preprocess()` (which
padded to a fixed 30 s for the any-to-any window) is deleted** — nothing needs
it anymore. `to_mono_16k` is the only entry point.

### 4.3 `lib/stt.py` — speech to text (NEW)

```python
from faster_whisper import WhisperModel

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel("small", device="cuda", compute_type="int8")
    return _model

def transcribe(audio: np.ndarray) -> str:
    """audio = 16k mono float32 (see audio contract). Returns transcript string.

    Use beam_size=5; join segment texts with spaces. Empty/muted audio should
    return "" (or close) — never raise.
    """
```

- Lazy-loads on first call (warm it up at app startup to absorb the load hit).
- `compute_type="int8"` → ~0.5 GB VRAM. If accuracy disappoints, §7 fallback to
  `float16`.
- The CTranslate2 model downloads (~500 MB) to the HF cache on first load — see
  §5.5 for the cache policy override.

**Self-check (in `__main__`):** build a 2 s 440 Hz sine tone at 16k, assert
`transcribe` returns *something* (not empty) — a tone will be transcribed as
*something*, not silence. See §9.1.

### 4.4 `lib/interview_model.py` — LLM HTTP client (REWRITTEN from transformers)

```python
import httpx

EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "transcript": {"type": "string"},
        "relevance_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "completeness_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "clarity_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "feedback": {"type": "string"},
    },
    "required": ["transcript", "relevance_score", "completeness_score", "clarity_score", "gaps", "feedback"],
}

DEFAULT_BASE_URL = "http://localhost:8080"   # matches llama-server's default port

def clean_transcript(text): ...
# Strips filler words ("uh", "um", "like", "you know", "I mean", ...) and
# immediate word repeats ("the the model" -> "the model") from an ASR
# transcript. Real content always survives. Self-corrections ("no wait, I
# mean X") are NOT repaired here — the eval prompt below tells the model to
# judge the intended final answer instead.

# FIXED interviewer persona — identical for every subject. Warm, human, teaching.
# The subject-specific part is appended by server.py from the question bank
# (title + description), so the tone never changes between topics.
SYSTEM_PROMPT_CORE = (...)
```

`InterviewModel`:

- `__init__(self, base_url=None)` — `base_url = base_url or os.environ.get("LLAMA_SERVER", DEFAULT_BASE_URL)`. Build one `httpx.Client(base_url=base_url, timeout=180)`.
- `_chat(self, prompt, system=None, max_tokens=512, temperature=0.7, top_p=0.9) -> str`
  — POST `/v1/chat/completions` with `messages = [system?, user]`. Return
  `resp["choices"][0]["message"]["content"]`. Raise a clear error if the server
  is unreachable (tell the user to start llama.cpp — see §5.3).
- `chat_reply(self, prompt, system) -> str` — free-form conversational reply
  (warm-up turns, transitions, coaching lines). `max_tokens=256`.
- `transcribe_and_eval(self, transcript, question, keywords, system) -> dict` —
  **text in, not audio in.** `transcript` is passed through `clean_transcript()`
  first. Prompt below, call `_chat`, return `_parse_json(text)`.

  ```
  You are an AI interviewer. Evaluate the candidate's answer to the question
  below against the expected keywords. Return ONLY valid JSON matching:
  {EVAL_SCHEMA as JSON}.

  Scoring rules:
  - Score each dimension 1-10. The transcript may contain spoken hesitations
    ("uh", "um", "like"), repeated words, or self-corrections — judge the
    candidate's intended final answer and ignore them.
  - In the "feedback" field: if the answer is fully correct and complete, praise
    it genuinely and do not invent nitpicks. Otherwise acknowledge what was good,
    then point out the specific technical mistake or key omission. Never mention
    word choice, grammar, or sentence structure.

  Question: {question}
  Expected keywords: {", ".join(keywords)}
  Candidate's answer: {clean_transcript(transcript)}
  ```

- `generate_followup(self, context, gaps, system) -> str` — prompt asking for JSON
  `{"follow_up_question": "..."}`; parse, fall back to raw text on failure.
- `generate_question(self, config_title, skills, previous_qs, system) -> str` —
  prompt asking for JSON `{"question": "..."}`; parse, fall back to raw text.
  (May be unused by `server.py`, which drives questions from `QuestionBank` —
  keep it for the notebook/consistency.)
- `_parse_json(self, text) -> dict` — **unchanged behavior from the old code:**
  1. try `json.loads(text)`;
  2. else regex-scan for `"transcript"`, `"feedback"`, `"relevance_score"`,
     `"completeness_score"`, `"clarity_score"` (string or integer forms, coerce
     scores to `int`), and `"gaps": [...]` (split on commas, strip quotes).
  3. Return whatever was recovered (possibly partial). **Max 2 retries** of the
     whole eval round (re-call `_chat`) when the parse comes back empty —
     see §7.

### 4.5 `lib/tts.py` — text to speech (REWRITTEN: Kokoro primary, Piper fallback)

```python
import numpy as np

KOKORO_VOICE = "af_heart"        # American English female; alt: "am_michael"
KOKORO_SR = 24000

class TTS:
    """Kokoro 82M primary, Piper subprocess fallback (silent)."""

    def speak(self, text: str):
        """Returns WAV bytes (16-bit PCM) for the browser, or None if unavailable.

        Tries Kokoro; on any import/load/runtime failure falls back to Piper;
        if neither works returns None (UI degrades to text-only).
        """
```

- **Kokoro** (`from kokoro import KPipeline`): `KPipeline(lang_code="a")` for
  American English; voice `af_heart`. The pipeline yields `(graphemes, phonemes,
  audio_tensor)` chunks — concatenate the audio tensors, cast to `np.float32`,
  encode to a WAV (PCM 16-bit, 24000 Hz) in memory. Weights (~330 MB)
  auto-download on first use (§5.5).
- **Piper fallback** — keep the existing `PiperTTS` class (subprocess call to
  `piper/piper` with `piper/en_US-lessac-medium.onnx`, output to a temp WAV,
  read bytes) — already returns WAV bytes (22050 Hz). Silent if binaries missing.
- Lazy-load both on first use; warm up at app startup.

### 4.6 `lib/question_loader.py` — KEPT VERBATIM

`QuestionBank(path)` reading a JSON bank `{title, description,
min_score_to_explain, questions:[{id, question, topic, difficulty, explanation,
answer_template, expected_keywords, ...}]}`. API:

- `next()` → random unasked question (marks it asked) or `None` when exhausted
- `remaining` / `done` / `total` properties
- `reset()`

The app drives the interview from this — **not** from `config.json`
(`lib/conversation.py` is deleted; it was superseded by `QuestionBank`).

### 4.7 `lib/logger.py` — KEPT VERBATIM

`save_transcript(config_title, history) -> path` writing
`interviews/YYYYMMDD_HHMMSS.json` with `{config, timestamp, turns}`.

### 4.8 `lib/privacy_audit.py` — network verification (NEW — referenced by docs but never existed)

```python
import psutil

def snapshot():
    """Return set of (dst_ip, dst_port) for all established OUTBOUND conns."""
    return {(c.raddr.ip, c.raddr.port) for c in psutil.net_connections()
            if c.raddr and c.status == "ESTABLISHED"}

def audit(callable, *args, **kwargs):
    """Run callable; assert zero NEW outbound destinations appeared while it ran.

    Loopback (127.0.0.1 / ::1) and RFC1918/private ranges are allowed.
    Raises AssertionError with the offending destinations on violation.
    """
```

The "audit" is a thin wrapper that runs a turn under a before/after snapshot.
Used by `test_pipeline.py` and manually during a live turn.

### 4.9 `server.py` — FastAPI entry + `static/index.html` (NEW)

The server **is** the orchestrator — there is no `lib/interview_flow.py`
(YAGNI; the notebook does the same in one function, and so can server.py).

**`server.py`** — FastAPI app:

- Module globals (one interview session at a time, mirroring the old design):
  `qbank = QuestionBank("questions/ml_entry_level.json")`, `history = []`,
  `current_q = None`, `followup_count = 0`, `stage` (`None` | `"intro"` |
  `"interview"`), `intro_idx = 0`, `intro_topics`.
- **Lazy imports + warmup** at startup: load STT model + TTS once (absorbs the
  load latency); probe the llama.cpp server (`InterviewModel._chat("ping")` or a
  HEAD to `LLAMA_SERVER`) and print a clear "start llama.cpp (README §5.3)"
  message if unreachable — do **not** crash.
- `GET /` → serve `static/index.html` (read file, `HTMLResponse`).
- `app.mount("/static", StaticFiles(directory="static"))` — same-origin.
- `_system()` → `SYSTEM_PROMPT_CORE + "\n\nInterview subject: {qbank.title}.\n{qbank.description}"`.
  The persona is fixed; only the subject changes per call.
- `POST /start` → `qbank.reset()`, `history = []`, `followup_count = 0`,
  `intro_idx = 0`, `stage = "intro"`. Return `{"intro": true, "question":
  <greeting + first warm-up question (name)>, "audio": base64(wav)}` (audio from
  `tts.speak(...)`).
- `POST /process` — body is **raw WAV bytes** (`await request.body()`):
  1. `audio = to_mono_16k(wav_bytes)` (bytes branch of §4.2)
  2. `transcript = stt.transcribe(audio)`
  3. **Intro stage:** `reply = chat_reply(...)` with the candidate's actual
     words included in the prompt (`The candidate just said: "..."`) so the
     interviewer responds to what was said, then asks the next topic
     (`INTRO_TOPICS`: name → what doing → about self → interest in the field).
     Return `{"intro": true, "transcript": ..., "reply": ..., "reply_audio":
     base64(wav)}`. No scoring.
  4. **Transition** (after the 4th topic): acknowledge the final answer, tell
     them the warm-up is done and the real interview starts now, ask the first
     scored question in a warm, conversational tone. `stage = "interview"`.
  5. **Interview stage:** `result = model.transcribe_and_eval(transcript,
     question, keywords, _system())` (max 2 retries on empty parse).
  6. follow-up or next question per §3.2 steps 6–7; every coach line is a
     `chat_reply` that includes the candidate's actual answer +
     `feedback`, so acknowledgments and tips reference what was really said.
  7. `history.append({question, transcript, scores, feedback, follow_up})`
  8. Return `{"transcript", "scores", "feedback", "gaps", "follow_up",
     "reply" (coach line, spoken), "reply_audio", "next_question",
     "next_audio", "done", "intro": false}`. When the bank is exhausted,
     `done = true` and the reply thanks the candidate, says they'll be in
     touch via email/contact, and wishes them luck.
- `POST /end` → summary (`total_turns`, `average_scores`) +
  `logger.save_transcript(qbank.title, history)` (mkdirs `interviews/`) →
  return `{**summary, "transcript_path": path}`.
- No `python-multipart`, no Pydantic models — the only input is the raw body.

**`static/index.html`** — the browser UI. Already written once; regenerate it as
a single self-contained page (no build step) with:

- **Mic capture:** `navigator.mediaDevices.getUserMedia({audio: true})` →
  `MediaRecorder` → on stop, decode via `AudioContext`, `encodeWav()` to a
  mono 16-bit WAV blob (JS helper, ~20 lines).
- **Playback:** server returns base64 WAV → `atob` → `Blob` → `URL.createObjectURL`
  → `new Audio(url).play()`. Play the question on Start, the reply/coach line on
  result.
- **Coach box (`#coach`):** after a scored turn the server's `feedback` string
  renders here — what the candidate covered well + one concrete pointer.
  `#scores` reset every turn; `#transcript` labeled "You said:".
- **Phase indicators:** three spans (`Transcribing / Evaluating / Follow-up`)
  toggled `pending → active → done` via classes; set to `active` while the
  corresponding `fetch("/process")` stage runs. The JS marks phase 1 done only
  after the response arrives (the server does all three stages before
  responding, so the client drives the indicator progression).
- **Intro turns:** when `data.intro` is true the app keeps recording enabled
  and just plays `data.reply` — no scores shown, mic stays hot.
- **Controls:** Start / Record (toggle, turns into Stop while recording) / End.
- **Result panel:** score tiles (relevance/completeness/clarity, each `/10`),
  transcript box, coach box, follow-up box, next-question text.
- Dark theme, no external CDNs, no frameworks — plain HTML/CSS/JS (the old
  page is the template; keep its look).

Run: `.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000`
(see §5.4).

### 4.10 `test_pipeline.py` — assert-based checks (NEW)

Plain `assert` self-checks, runnable via `.venv/bin/python test_pipeline.py`
(no pytest dependency required; keep it pytest-compatible). At minimum:

1. `test_to_mono_16k` — stereo 48k int16 → mono 16k float32 in [-1,1], length
   ≤ `MAX_SAMPLES`, no padding (short input keeps its short length).
2. `test_stt_transcribes_tone` — 2 s 440 Hz tone at 16k → `transcribe` returns
   a non-empty string.
3. `test_tts_returns_wav` — `TTS().speak("Hello")` returns non-empty WAV
   bytes (starts with `RIFF....WAVE`), or `None` if no voice engine is installed.
4. `test_parse_json_valid` — `_parse_json` handles a clean JSON blob.
5. `test_parse_json_malformed` — regex fallback recovers scores from a
   JSON-with-prose string.
6. `test_parse_json_valid_feedback` — parse requires the `feedback` field
   (missing → treated as malformed, regex path handles it).
7. `test_question_bank_no_repeats` — `QuestionBank.next()` over an exhausted
   bank returns `None`; no ID repeats.
8. `test_clean_transcript` — fillers ("um", "like", "you know", "I mean") and
   repeats ("the the") are stripped; real content survives.
9. `test_server_start_intro` — `POST /start` returns `{"intro": true, ...}`
   with a spoken greeting, and a full intro→scored-turn→`/end` loop can be
   driven through TestClient (LLM mocked) without error.

Plus the `lib/` self-checks: `python lib/audio_capture.py`,
`python lib/interview_model.py`, `python lib/logger.py` (each has a `__main__`
block).

### 4.11 `setup.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
uv venv --python 3.12
# activate per-OS (.venv/bin/activate | Scripts/activate)
uv pip install -r requirements.txt
mkdir -p models piper interviews questions
# Piper fallback download (unchanged from before):
#   piper binary for linux-x86_64 / macos + en_US-lessac-medium.onnx(.json)
#   -> piper/piper, piper/en_US-lessac-medium.onnx
echo "Done. Next: start llama.cpp server (see README §5.3), then:"
echo "  .venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000"
```

llama.cpp server launch is **documented, not auto-run** — the user starts it.

### 4.12 `.gitignore`

```
.venv/
models/
piper/
interviews/
__pycache__/
*.pyc
.DS_Store
```

### 4.13 `.claude/CLAUDE.md` and `.claude/settings.json`

Regenerate `CLAUDE.md` to point at this README (`@AGENTS.md` → `@README.md`)
and to restate the conventions in §8. Keep `settings.json` permissions as
before (allow `.venv/bin/python`, `uv run`, pytest, `netstat`, `ps aux`, `ls`,
`mkdir`, `touch`; deny everything else; deny read/write to `~/.cache/huggingface`
and `~/.cache/torch`).

### 4.14 `AGENTS.md`

Regenerate as a condensed operator's guide pointing to this README, containing:
the file tree, the "no HF-cache blobs" policy (§5.5), the environment rules
(`uv`/`.venv/bin/python` only), conventions (§8), and the test commands (§9).

---

## 5. Setup & Runbook

### 5.1 Prereqs

- Linux, Python 3.12 (pinned by `setup.sh`)
- [uv](https://astral.sh/uv) package manager
- CUDA GPU with ≥ 6 GB VRAM (RTX 3050 = dev target; Kaggle T4/P100 = notebook)
- A prebuilt **Gemma GGUF** for llama.cpp (see §5.3)

### 5.2 Install

```bash
./setup.sh                 # creates .venv (3.12), installs requirements.txt,
                           # downloads Piper fallback
.venv/bin/python test_pipeline.py   # quick sanity before models are involved
```

**Never** use system `python`/`pip` — always `.venv/bin/python` or `uv run python`.

### 5.3 llama.cpp server (separate process, user-started)

1. Get llama.cpp (build from source or use a release build with the server, e.g.
   `llama-server`).
2. Obtain a prebuilt **Gemma 4 GGUF** (e.g. a community `gemma-4-*-it` GGUF in
   Q4_K_M) and place it under `models/` (e.g. `models/gemma-4-E2B-it.Q4_K_M.gguf`).
3. Launch:

   ```bash
   llama-server -m models/gemma-4-E2B-it.Q4_K_M.gguf \
                -ngl 999 --host 127.0.0.1 --ctx-size 4096
   ```

   - `-ngl 999` = offload all layers to GPU (~2.5 GB for a 2B Q4).
   - **`--port 8080` is llama-server's default** — it can be omitted; the app
     expects `localhost:8080`. If you do change it, set the app's `LLAMA_SERVER`
     env var to match (§4.4).
   - `--host 127.0.0.1` is **mandatory for privacy** — never bind `0.0.0.0`;
     the interview text must not be reachable on the network.
4. Smoke test:

   ```bash
   curl -s http://localhost:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":20}'
   ```

### 5.4 Run the app

```bash
LLAMA_SERVER=http://localhost:8080 .venv/bin/python -m uvicorn server:app \
    --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` (same-origin page + API; the browser fetches
`/start`, `/process`, `/end` — no cross-origin anywhere). Speak into the browser
mic; the turn flow in §3.2 runs end to end.

### 5.5 Model cache policy (no blobs in `~/.cache`)

- **Gemma GGUF**: lives in `models/` — no cache involvement.
- **Whisper (CTranslate2) and Kokoro weights** auto-download to the HF cache on
  first use. To keep *everything* project-local (privacy + reproducibility),
  export before first run:

  ```bash
  export HF_HOME="$PWD/models/hf-cache"
  export XDG_CACHE_HOME="$PWD/models/cache"
  ```

  The `.gitignore` already excludes `models/`, so nothing leaks into git.

### 5.6 Kaggle notebook (build-with-gemma-nashik-interviewer.ipynb)

The notebook is **self-contained** and targets the T4/P100 (16 GB) — it does
**not** use the any-to-any path. Current working state:

- Cell 1: `pip install vllm gradio transformers accelerate bitsandbytes`
- Cell 2: HF login (gated Gemma)
- Cell 3: transformers `whisper-small` (fp16, `cuda`) — works on T4
- Cell 4: vLLM serving `cyankiwi/gemma-4-E2B-it-AWQ-INT4`
  (`guided_json=EVAL_SCHEMA` for structured output)
- Cells 5–10: `QuestionBank`, `transcribe()`, `gen()`, `parse_json()`, schema,
  `run_interview_turn()`
- Cell 11: Gradio UI

**Keep vLLM** — it is the better fit for a T4 and already works; the notebook
was never the failing path. Optional alignment (only if you standardize):
swap Cell 3's transformers Whisper for `faster-whisper`. TTS (Kokoro) is
**optional on Kaggle** — the notebook renders question text; add Kokoro only if
the demo needs voice. If it is added, `install kokoro` and lazy-load it.

---

## 6. Privacy Story & Audit

### 6.1 Claim

**Zero outbound network connections during an interview turn.** Candidate audio
never leaves the machine. Transcript never leaves the machine. Interviewer text
never leaves the machine (Kokoro is local). The only listener is the llama.cpp
server on `127.0.0.1`.

### 6.2 Verification procedure

1. Ensure llama.cpp is bound to `127.0.0.1` (not `0.0.0.0`) — check with
   `netstat -tnp | grep 8080`.
2. `netstat -tnp | grep python` before starting the turn — snapshot.
3. Run a full turn (open `:8000`, speak into mic, get scores).
4. `netstat -tnp | grep python` after — assert **no new outbound** `ESTABLISHED`
   connections appeared. Loopback/private only.
5. Programmatic: `lib/privacy_audit.audit()` wraps a turn and raises on any new
   outbound destination — included in `test_pipeline.py` (see §9.1).
6. Screenshot the netstat output for the competition writeup.

### 6.3 Inbound vs outbound

The audit permits loopback/private destinations (localhost services, LAN). It
forbids **new** outbound connects to the public internet. If any library ever
phones home (telemetry), the assertion fails loudly.

---

## 7. Known Issues & Mitigations

### 7.1 Issues we already hit (see §2.2)

The entire pivot exists to remove them: in-process transformers load on a 6 GB
card, QAT vocab dequant transients, fp32 upcast, transformers version drift,
cloud-TTS caveat, constrained-decoding complexity, doc phantoms.

### 7.2 Issues we may hit next — and the plan

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| 1 | **No Gemma 4 GGUF found** at pivot time | Medium | Assumption is a prebuilt GGUF exists (§5.3). If none, convert from a Gemma HF checkpoint with llama.cpp's `convert_hf_to_gguf.py` — documented path, but check llama.cpp release notes for Gemma 4 arch support first. |
| 2 | **Whisper int8 accuracy** on heavy accents | Medium | `compute_type` knob: `int8` (~0.5 GB) → `float16` (~1 GB). Still fits budget. Also `beam_size` up to 5. |
| 3 | **Kokoro output quality/voice** mismatch | Low | Switch `KOKORO_VOICE` (`af_heart` ↔ `am_michael`); Piper fallback already covers hard breakage. |
| 4 | **LLM emits prose around JSON** (no grammar) | High | `_parse_json` retry + regex fallback, max 2 retries. If drift persists, llama.cpp server supports `grammar=`/`json_schema=` — add only if measured to be needed (last resort, it's extra complexity). |
| 5 | **llama.cpp server not running** when app starts | High | Startup probe prints "start llama.cpp (README §5.3)"; httpx call raises a clear, actionable error. |
| 6 | **CORS** if anyone calls :8080 from browser JS | n/a | Non-issue: httpx calls run **server-side** from `server.py`, and the UI is same-origin (`:8000` serves both the page and the `/start|/process|/end` API). Never wire the browser directly to :8080. |
| 7 | **Port conflict** (8080 busy) | Low | `LLAMA_SERVER=http://host:port` env override on the app side. |
| 8 | **Mic sample-rate mismatch** (48k stereo from browser) | High | `to_mono_16k` handles any `(sr, int16)` — mono downmix + torchaudio Resample. |
| 9 | **Cold-start latency** (first turn slow) | Medium | Lazy loads + `warmup()` at launch; subsequent turns are warm. Phase indicators keep UX responsive. |
| 10 | **VRAM creep / OOM** on another card | Low | Budget 3.4 GB < 6 GB with margin. Knobs: whisper int8, Kokoro CPU (`device="cpu"`), drop `-ngl` layers. Verify with `nvidia-smi` (§9.3). |
| 11 | **transformers removed** breaks something | Low | Local stack never imports transformers. Notebook installs its own. If Kokoro internally wants it, `pip show kokoro` and pin `transformers` as a transitive dep only. |
| 12 | **Audio > 2 min** recorded | Low | `to_mono_16k` truncates at `MAX_SAMPLES` (120 s). Bounds transcribe time and context. |
| 13 | **Whisper/Kokoro weights land in `~/.cache`** | Medium | §5.5 `HF_HOME`/`XDG_CACHE_HOME` export keeps everything under `models/`. |

---

## 8. Conventions & Constraints (regenerate AGENTS.md from these)

1. **Environment**: uv-managed `.venv/`, Python 3.12. All commands via
   `.venv/bin/python` or `uv run python`. Never system python/pip.
2. **Audio**: 16 kHz mono float32, [-1, 1], max 2 min, in-memory only.
   Resample with `torchaudio.transforms.Resample` — **never**
   `scipy.signal.resample`.
3. **Privacy**: zero outbound connections during inference; llama.cpp bound to
   127.0.0.1; `lib/privacy_audit.py` verifies.
4. **JSON reliability**: retry + regex fallback (max 2 retries) via
   `_parse_json`. No grammar/constraint lib by default (§7.2 #4).
5. **Testing**: `assert`-based self-checks in `test_pipeline.py`, runnable
   standalone (`python test_pipeline.py`) or via pytest. Non-trivial logic
   leaves one runnable check behind.
6. **Laziness rules**: shortest working diff; stdlib/native first; no
   speculative abstractions; mark deliberate shortcuts with
   `# ponytail:` comments naming the ceiling and upgrade path.
7. **No speculative modules**: no `interview_flow.py` (server.py inlines
   orchestration), no `conversation_manager.py`, no `config.json`/`conversation.py`
   (QuestionBank wins), no cloud STT/TTS.
8. **Model storage**: GGUF + any HF downloads under `models/` at project root,
   no `~/.cache` blobs (§5.5).

---

## 9. Testing & Verification

### 9.1 Unit (no models needed beyond the self-checks)

```bash
.venv/bin/python test_pipeline.py
```

Coverage: `to_mono_16k` format/no-pad, STT transcribes a tone, TTS returns
24000 Hz WAV, `_parse_json` valid + malformed, QuestionBank no-repeat/exhaust.

### 9.2 Integration (full local stack)

1. Start llama.cpp server (§5.3) → curl smoke test passes.
2. Start the server (§5.4) → open `http://127.0.0.1:8000`.
3. Click **Start** → question spoken (Kokoro) + displayed.
4. Record in browser mic → transcript appears → scores + gaps render →
   follow-up (if needed) spoken → next question or done.
5. **End** → summary + `interviews/<ts>.json` written.

### 9.3 VRAM

During a full turn run `watch -n1 nvidia-smi`. Expect the three models
co-resident at **~3.4 GB** of 6 GB (whisper small int8 + Gemma 2B Q4 + Kokoro
82M). Sequential execution means transient peaks stay well below budget.

### 9.4 Privacy audit

```bash
netstat -tnp | grep python      # before
# run one turn
netstat -tnp | grep python      # after — no new ESTABLISHED outbound
.venv/bin/python -c "from lib import privacy_audit as p; p.audit(run_one_turn)"
```

### 9.5 Kaggle notebook

Re-run the notebook end to end on a T4/P100: question loads, mic answer
transcribes, vLLM returns schema-valid JSON, next question advances. TTS
optional.

---

## 10. Decisions & Non-Goals (in case anyone asks "why not…")

| Question | Answer |
|---|---|
| Why not vLLM locally? | Doesn't fit 6 GB. It's the notebook's tool (T4/P100). |
| Why not transformers any-to-any? | It's the thing that failed (§2.2). |
| Why not cloud STT/TTS/LLM? | The whole pitch is privacy-first; zero outbound is the claim. |
| Why not grammar-constrained JSON? | retry+regex already exists and is simpler; add grammar only if drift is measured (§7.2 #4). |
| Why no `lib/interview_flow.py`? | YAGNI — server.py (like the notebook) orchestrates in one function. |
| Why FastAPI+static over Gradio locally? | Gradio pulls fastapi/uvicorn in transitively anyway — same dep weight, but the hand-written static UI (MediaRecorder + JS WAV encoder + phase chips) already existed and is same-origin. Gradio stays notebook-only. |
| Why QuestionBank over `config.json`/`Conversation`? | Notebook already uses it; it has `next()`/`done`/`remaining`; one path, not two. |
| Why keep Piper? | Already downloaded, silent fallback if Kokoro breaks. |
| Why drop transformers from requirements? | Local stack never imports it; smaller venv, fewer version conflicts. |
| Why 2 min cap with no padding? | Long answers to behavioral questions need room; transcribe time stays bounded. Padding only ever served the dead any-to-any window. |
| Why the scripted intro/warm-up? | Interviewers never open cold — a scored question on turn one feels robotic. A 4-topic warm-up (name → what you do → about yourself → interest in the field) lets the persona speak before the grade appears; answers are ungraded so candidates relax. |
| How are "uh/um" and self-corrections handled? | Two layers: `clean_transcript()` strips fillers + word repeats before evaluation, and the eval prompt tells the model to judge the intended final answer and ignore hesitations entirely. Regex does the cheap 90%, the prompt covers the rest. |
| What counts as feedback? | Only real technical mistakes or key omissions. Fully-correct answers get genuine praise and no invented nitpicks; word choice, grammar, and sentence structure are explicitly excluded from criticism. |

---

## 11. File Tree (target state after rebuild)

```
build-with-gemma/
├── README.md                        # THIS FILE — single source of truth
├── AGENTS.md                        # regenerated from §8
├── server.py                        # FastAPI entry + orchestration (§4.9)
├── static/
│   └── index.html                   # browser UI — mic, playback, phases (§4.9)
├── test_pipeline.py                 # assert self-checks (§4.10)
├── requirements.txt                 # (§4.0)
├── setup.sh                         # (§4.11)
├── build-with-gemma-nashik-interviewer.ipynb   # Kaggle, self-contained (§5.6)
├── .gitignore                       # (§4.12)
├── .claude/                         # (§4.13)
├── lib/
│   ├── __init__.py
│   ├── audio_capture.py             # to_mono_16k (§4.2)
│   ├── stt.py                       # faster-whisper wrapper (§4.3)
│   ├── interview_model.py           # httpx → llama.cpp (§4.4)
│   ├── tts.py                       # Kokoro + Piper fallback (§4.5)
│   ├── question_loader.py           # QuestionBank (§4.6)
│   ├── logger.py                    # save_transcript (§4.7)
│   └── privacy_audit.py             # psutil zero-outbound check (§4.8)
├── questions/
│   └── ml_entry_level.json          # question bank (survivor, 36 KB)
├── piper/                           # fallback TTS binary + voice (survivor)
├── models/                          # GGUF + hf-cache (gitignored)
│   └── .gitkeep
└── interviews/                      # saved transcripts (gitignored)
```
