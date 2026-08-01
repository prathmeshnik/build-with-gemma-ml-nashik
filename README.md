# Mira — your interview practice partner

![Mira](static/banner.webp)

An AI interviewer that lives on your laptop, hears you, grades you, and coaches you — and never sends a word of it to the internet.

Practicing a technical interview usually means one of two things: finding a friend willing to quiz you on ML fundamentals, or paying a coach. The AI interview tools that exist stream your voice to a cloud API for speech-to-text and text-to-speech, so every answer you give — including the weak ones under time pressure — ends up on someone else's server.

Mira is the third option. It runs entirely on one machine: a 6 GB RTX 3050 laptop. Speak, get scored, get coached, get a transcript at the end. Nothing leaves the machine.

Built with a local Gemma 4 E2B for the [Build with Gemma — ML Nashik](https://www.kaggle.com/competitions/build-with-gemma-ml-nashik) competition.

## The story

I tried the obvious thing first: load Gemma 4's any-to-any model — the one that ingests raw audio and talks back — directly into Python. One model, whole pipeline. It OOM'd on my 6 GB card. The QAT checkpoint dequantizes its full vocab tables on every forward call, ~2.3 GiB of transient spikes per decode step, and silently upcasts the whole graph to fp32. Fighting the library per-layer to keep it alive wasn't going to ship.

So I stopped fighting and split the job into three proven pieces that each fit:

- faster-whisper small hears the candidate (~0.5 GB).
- Gemma 4 E2B as a GGUF, served by llama.cpp, does the thinking (~2.5 GB).
- Kokoro 82M speaks back (~0.4 GB).

All three co-resident at ~3.4 GB of 6 GB, running one after another per turn. The app became a thin HTTP client to a llama.cpp server on localhost — no in-process model load, no OOM, no version-matrix whack-a-mole. And because every piece is local, the privacy claim isn't a marketing line: it's checkable.

The interviewer is named **Mira**. She's defined by a system prompt: warm, human, teaching. No stiff bullet-point speech.

## What an interview feels like

Click Start. Mira introduces herself once, asks what you'd like to be called, and eases you in with a short warm-up — your name, what you're doing, what drew you to machine learning. Three questions, never scored.

Then the real thing: ML fundamentals, one question at a time. The mic is hands-free — it opens when Mira stops talking and closes when you go quiet, so you never touch a button mid-answer. You talk the way people talk: um, like, start a sentence and correct yourself. It doesn't matter; the grading ignores all of it.

Every answer comes back with three scores (relevance, completeness, clarity), the key points you missed, and one concrete coaching pointer — what you covered, what would make it stronger. Fully-correct answers get genuine praise, not invented nitpicks.

When you don't know something — and you will, that's the point — say "I don't know" or go quiet for 15 seconds and Mira explains the answer on the spot, then moves on. You learn the missing piece in the moment instead of carrying a silent gap into the next question.

End the interview and you get a transcript and average scores, saved to disk. Practice on your own, get the feedback, come back stronger.

## Try it

Requires: Linux, Python 3.12, a CUDA GPU with ≥ 6 GB VRAM, and [uv](https://astral.sh/uv).

```bash
./setup.sh                              # venv + deps + Piper fallback voice
.venv/bin/python test_pipeline.py       # sanity checks
```

Start llama.cpp bound to localhost (mandatory for privacy):

```bash
llama-server -m models/gemma-4-E2B-it.Q4_K_M.gguf -ngl 999 --host 127.0.0.1
```

Then the app:

```bash
.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and practice.

### Customizing the interview to a different topic

The question bank lives in `questions/ml_entry_level.json` — one JSON file with a header block and a list of questions. Every field the interviewer uses is in there, so changing the topic is just editing this file:

```json
{
  "title": "Entry-Level ML Engineer Interview",
  "description": "Interview questions for entry-level ML engineering positions.",
  "difficulty": "entry-level",
  "min_score_to_explain": 4,
  "questions": [
    {
      "id": 1,
      "topic": "Machine Learning Fundamentals",
      "question": "What is the difference between supervised and unsupervised learning?",
      "difficulty": "easy",
      "expected_key_points": ["Supervised learning uses labeled data", "Unsupervised learning uses unlabeled data"],
      "answer_template": "Supervised learning trains on labeled data...",
      "explanation": "Think of it like a student learning from a textbook with answer keys..."
    }
  ]
}
```

To make it *your* interview — say, backend engineering or product management:

1. Copy `questions/ml_entry_level.json` to a new name, e.g. `questions/backend.json`, and rewrite it for your topic. Per question, supply the `question` (what Mira asks), `expected_key_points` (what she grades your answer against), `answer_template` (the model's skeleton for a correct answer), and `explanation` (what Mira teaches when you say "I don't know"). `difficulty`, `topic`, and `id` are free-form; just keep `id` unique.
2. Point the server at it — in `server.py`, change `QuestionBank("questions/ml_entry_level.json")` to `QuestionBank("questions/backend.json")`.
3. Restart uvicorn. That's it — the intro, scoring, and coaching all work unchanged.

`min_score_to_explain` (a question whose score drops below this gets a full walkthrough) and the `title`/`description` shown in the UI are optional; the defaults kick in if you omit them.

Never use system python/pip — always `.venv/bin/python`.

## The Kaggle notebook

Same interviewer, cloud version. `build-with-gemma-nashik-interviewer.ipynb` runs Gemma 4 E2B's native any-to-any path on a Kaggle T4 (16 GB) — one model that transcribes, scores, and replies, with gTTS speaking back. Same question bank and scoring, no shared code. The local stack is the shipped product; the notebook is what you demo where the 16 GB GPU is free.

## Privacy

Zero outbound network connections during a turn. Audio, transcript, and interviewer voice all stay in memory and on loopback. llama.cpp is bound to `127.0.0.1` — never the network. `lib/privacy_audit.py` snapshots connections before and after a turn and raises if anything new dialed out; `netstat -tnp | grep python` before and after shows the same thing by hand.

## Layout

`server.py` (FastAPI orchestration), `static/index.html` (the hands-free browser UI), `lib/` (STT, interview model, TTS, question bank, logger, privacy audit), `questions/ml_entry_level.json` (the bank), `test_pipeline.py`.
