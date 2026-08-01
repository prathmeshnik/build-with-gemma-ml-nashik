import base64
import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from lib.audio_capture import to_mono_16k
from lib.interview_model import (
    INTERVIEWER_NAME,
    SYSTEM_PROMPT_CORE,
    InterviewModel,
    is_dont_know,
)
from lib.logger import save_transcript
from lib.question_loader import QuestionBank
from lib.tts import TTS

_T0 = time.monotonic()


def _log(msg):
    print(f"[{time.monotonic() - _T0:6.1f}s] {msg}", flush=True)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

qbank = QuestionBank("questions/ml_entry_level.json")
model = InterviewModel()
tts = TTS()

history = []
current_q = None
followup_count = 0
stage = None  # "intro" | "interview"
intro_idx = 0
MAX_FOLLOWUPS = 2

# Short warm-up: greeting+name, current work, interest in ML — then the real
# interview. Kept to 3 so the warm-up doesn't drag into the session.
INTRO_TOPICS = [
    "their name and what they like to be called",
    "what they're currently doing (studying, working, or something in between)",
    "what sparked their interest in machine learning and this field",
]


def _system():
    return (
        SYSTEM_PROMPT_CORE
        + f"\n\nInterview subject: {qbank.title}.\n{qbank.description}".strip()
    )


def _keywords(q):
    return q.get("expected_key_points") or q.get("expected_keywords") or []


def _should_ask_followup(scores, gaps):
    vals = [scores.get(k) for k in ("relevance", "completeness", "clarity")]
    vals = [v for v in vals if v]
    if not vals:
        return False
    return sum(vals) / len(vals) < 7 and bool(gaps) and followup_count < MAX_FOLLOWUPS


def _warmup():
    import lib.stt as stt

    print("Warming up STT + TTS...", flush=True)
    try:
        t0 = time.monotonic()
        stt._get_model()
        _log(f"STT: whisper-small loaded in {time.monotonic() - t0:.1f}s")
    except Exception as exc:
        print(f"(STT warmup failed: {exc})", flush=True)
    try:
        tts._get_kokoro()
        _log("TTS: Kokoro pipeline loaded")
        probe = tts.speak("Hi there, welcome.")
        _log(f"TTS probe -> {'ok, ' + str(len(probe)) + ' bytes' if probe else 'FAILED: no audio!'}")
    except Exception as exc:
        print(f"(TTS warmup failed: {exc})", flush=True)
    try:
        model._chat("ping", max_tokens=1)
        print("llama.cpp server: reachable", flush=True)
    except RuntimeError as exc:
        print(f"WARNING: {exc}", flush=True)


@app.on_event("startup")
async def _startup():
    _warmup()


@app.get("/")
async def index():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())


@app.post("/start")
async def start():
    global history, current_q, followup_count, stage, intro_idx
    qbank.reset()
    history = []
    followup_count = 0
    intro_idx = 0
    stage = "intro"
    line = model.chat_reply(
        "You are starting a practice interview. FIRST greet the candidate warmly "
        "and casually — like \"Hi, how are you doing today?\" — and introduce "
        "yourself by your name, " + INTERVIEWER_NAME + ", as their interviewer. "
        "THEN, in the same turn, naturally ask what they'd like to be called. "
        "Keep it conversational and short (2-3 sentences). Do not ask any other "
        "questions.",
        system=_system(),
    )
    line = _say(line, f"Hi there! How are you doing today? I'm {INTERVIEWER_NAME}, "
                      "your interviewer. What would you like me to call you?")
    _log(f"/start: greeting ({len(line)} chars): {line[:90]!r}")
    audio = tts.speak(line)
    _log(f"/start: tts -> {len(audio)} bytes" if audio else "/start: tts FAILED (None)")
    return JSONResponse({
        "question": line,
        "audio": base64.b64encode(audio).decode() if audio else None,
        "intro": True,
    })


@app.post("/process")
async def process(request: Request):
    global current_q, followup_count, stage, intro_idx
    if stage is None:
        return JSONResponse({"error": "No active interview"}, status_code=400)

    import lib.stt as stt

    audio = to_mono_16k(await request.body())
    transcript = stt.transcribe(audio, vad_filter=True)
    _log(f"/process: {len(audio) / 16000:.1f}s audio -> transcript {transcript[:80]!r}")

    if stage == "intro":
        return _intro_turn(transcript)

    if current_q is None:
        return JSONResponse({"error": "No active question"}, status_code=400)

    # Verbal "I don't know" — explain the answer and move on, don't score it.
    if is_dont_know(transcript):
        _log(f"/process: candidate can't answer ({transcript[:60]!r}) -> explain+advance")
        reply, next_question, done = _explain_current(transcript)
        reply_audio = tts.speak(reply) if reply else None
        return JSONResponse({
            "intro": False,
            "transcript": transcript,
            "scores": {},
            "gaps": [],
            "feedback": "",
            "reply": reply,
            "reply_audio": base64.b64encode(reply_audio).decode() if reply_audio else None,
            "follow_up": None,
            "next_question": next_question,
            "done": done,
        })

    question = current_q["question"]
    keywords = _keywords(current_q)
    try:
        result = model.transcribe_and_eval(transcript, question, keywords, system=_system())
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)

    scores = {
        "relevance": result.get("relevance_score"),
        "completeness": result.get("completeness_score"),
        "clarity": result.get("clarity_score"),
    }
    gaps = result.get("gaps", [])
    feedback = result.get("feedback", "")

    follow_up = None
    next_question = None
    done = False
    conversation = context_str()

    if _should_ask_followup(scores, gaps):
        follow_up = model.generate_followup(conversation, gaps, system=_system())
        followup_count += 1
        reply = model.chat_reply(
            "Conversation so far:\n" + conversation + "\n\n"
            "The candidate just answered: \"" + transcript + "\"\n\n"
            "Continue the conversation naturally. Acknowledge what they actually "
            "said, weave in this coaching note if it's useful, then ask this "
            f"follow-up question: {follow_up}",
            system=_system(),
        )
    else:
        followup_count = 0
        nxt = qbank.next()
        if nxt:
            next_question = nxt["question"]
            current_q = nxt
            reply = model.chat_reply(
                "Conversation so far:\n" + conversation + "\n\n"
                "The candidate just answered: \"" + transcript + "\"\n\n"
                "Continue the conversation naturally — this is a conversation, not "
                "a script. Acknowledge what they actually said in your own words, "
                "and where it fits, reference something they mentioned earlier "
                "without re-summarizing it. Don't re-introduce yourself or repeat "
                "your name. Weave in this coaching note if there is one. Coaching "
                "note: "
                f"{feedback or 'no note'}\n\n"
                "Then move on to the next question naturally, keeping the question "
                f"text itself exactly: \"{next_question}\"",
                system=_system(),
            )
        else:
            done = True
            reply = model.chat_reply(
                "Conversation so far:\n" + conversation + "\n\n"
                "The candidate just answered: \"" + transcript + "\"\n\n"
                "All interview questions are complete. Thank the candidate "
                "sincerely for their time and effort, let them know we will be in "
                "touch through email or whichever contact method they provided, "
                "wish them good luck, and say goodbye.",
                system=_system(),
            )

    # Guarantee a spoken reply even if the model returned empty.
    if follow_up:
        reply = _say(reply, f"Let me ask you this one: {follow_up}")
    elif next_question:
        reply = _say(reply, f"Great, let's move on. {next_question}")
    else:
        reply = _say(reply, "That's all the questions I had — thank you so much for your time and best of luck!")
    reply_audio = tts.speak(reply) if reply else None
    _log(f"/process: reply ({len(reply)} chars), tts -> {'ok' if reply_audio else 'None'}")

    turn_scores = {
        "relevance_score": result.get("relevance_score"),
        "completeness_score": result.get("completeness_score"),
        "clarity_score": result.get("clarity_score"),
    }
    history.append({
        "question": question,
        "transcript": transcript,
        "scores": turn_scores,
        "gaps": gaps,
        "feedback": feedback,
        "follow_up": follow_up,
    })

    return JSONResponse({
        "intro": False,
        "transcript": transcript,
        "scores": scores,
        "gaps": gaps,
        "feedback": feedback,
        "reply": reply,
        "reply_audio": base64.b64encode(reply_audio).decode() if reply_audio else None,
        "follow_up": follow_up,
        "next_question": next_question,
        "done": done,
    })


def _advance_intro(transcript, moved_on):
    """Advance past the current warm-up topic. Called after a real answer
    (`moved_on=False`, respond to what was said) or after a silence timeout
    (`moved_on=True`, don't claim to have heard anything). Returns JSONResponse.
    """
    global stage, intro_idx, current_q, history
    _log(f"intro turn: transcript {transcript[:60]!r}, moved_on={moved_on}")
    answered_topic = INTRO_TOPICS[intro_idx]
    conversation = context_str()
    if intro_idx + 1 < len(INTRO_TOPICS):
        intro_idx += 1
        topic = INTRO_TOPICS[intro_idx]
        if moved_on:
            reply = model.chat_reply(
                "Conversation so far:\n" + conversation + "\n\n"
                "The candidate didn't say anything. Don't pressure them — say "
                "something gentle like \"no problem, let's keep going\", then ask "
                "the next warm-up question about: " + topic,
                system=_system(),
            )
        else:
            reply = model.chat_reply(
                "Conversation so far:\n" + conversation + "\n\n"
                "The candidate just said: \"" + transcript + "\"\n\n"
                "Respond warmly to what they actually said in one or two "
                "sentences, then ask the next warm-up question about: "
                + topic,
                system=_system(),
            )
        reply = _say(reply, f"Okay, let's keep going. Tell me about {topic}.")
        history.append({
            "question": f"Warm-up: {answered_topic}",
            "transcript": transcript,
            "scores": {},
            "gaps": [],
            "feedback": "",
            "follow_up": None,
        })
        return JSONResponse({
            "intro": True,
            "transcript": transcript,
            "reply": reply,
            "reply_audio": _audio64(reply),
        })
    # Last warm-up topic answered -> transition into the real interview.
    nxt = qbank.next()
    current_q = nxt
    stage = "interview"
    if moved_on:
        line = model.chat_reply(
            "Conversation so far:\n" + conversation + "\n\n"
            "The candidate didn't say anything. Don't pressure them. Transition "
            "warmly: tell them the warm-up is done and the real interview starts "
            "now, then ask the first question in a warm, conversational tone, "
            f"keeping the question itself exactly: \"{nxt['question']}\"",
            system=_system(),
        )
    else:
        line = model.chat_reply(
            "Conversation so far:\n" + conversation + "\n\n"
            "The candidate just said: \"" + transcript + "\"\n\n"
            "Warmly acknowledge what they shared, then transition: tell them the "
            "warm-up is done and the real interview starts now. Then ask the "
            "first question in a warm, conversational tone, keeping the question "
            f"itself exactly: \"{nxt['question']}\"",
            system=_system(),
        )
    line = _say(line, f"Thanks for sharing that. Let's get started with the real interview. {nxt['question']}")
    history.append({
        "question": f"Warm-up: {answered_topic}",
        "transcript": transcript,
        "scores": {},
        "gaps": [],
        "feedback": "",
        "follow_up": None,
    })
    return JSONResponse({
        "intro": False,
        "transcript": transcript,
        "reply": line,
        "reply_audio": _audio64(line),
        "question": nxt["question"],
        "keywords": _keywords(nxt),
        "scores": {},
        "gaps": [],
        "feedback": "",
        "follow_up": None,
        "next_question": nxt["question"],
        "done": False,
    })


def _intro_turn(transcript):
    return _advance_intro(transcript, moved_on=False)


def _explain_current(transcript):
    """Candidate didn't answer the current question (silence or verbal "I don't
    know"). Warmly explain the correct answer, then advance to the next
    question. Records the non-answer in history so the transcript is honest.
    Returns (reply, next_question, done)."""
    global current_q, followup_count
    question = current_q
    explanation = (
        question.get("explanation")
        or question.get("answer_template")
        or "I'm afraid we don't have a prepared explanation for this one."
    )
    conversation = context_str()
    _log(f"no answer on {question['question'][:60]!r} -> explain+advance")

    history.append({
        "question": question["question"],
        "transcript": transcript,
        "scores": {},
        "gaps": [],
        "feedback": "",
        "follow_up": None,
    })

    followup_count = 0
    nxt = qbank.next()
    done = False
    next_question = None
    if nxt:
        next_question = nxt["question"]
        current_q = nxt
        reply = model.chat_reply(
            "Conversation so far:\n" + conversation + "\n\n"
            "The candidate did not answer this question:\n"
            f"\"{question['question']}\"\n\n"
            "Warmly say it's okay that they didn't know it, and that you'll "
            "explain the answer. Then explain it clearly and encouragingly "
            "using this material:\n\n"
            f"{explanation}\n\n"
            "Then, still in the same turn, ask the next question naturally, "
            f"keeping the question itself exactly: \"{next_question}\"",
            system=_system(),
        )
    else:
        done = True
        reply = model.chat_reply(
            "Conversation so far:\n" + conversation + "\n\n"
            "All interview questions are complete. Thank the candidate sincerely "
            "for their time and effort, let them know we will be in touch through "
            "email or whichever contact method they provided, wish them good "
            "luck, and say goodbye.",
            system=_system(),
        )

    if next_question:
        reply = _say(reply, f"It's okay if you didn't know this one. {explanation} Now let's try: {next_question}")
    else:
        reply = _say(reply, "That's the end of the interview — thank you so much for your time and best of luck!")
    return reply, next_question, done


@app.post("/explain")
async def explain():
    """Candidate stayed silent -> no-answer handling.

    Interview stage: warm "you don't know it — let me explain", using the
    question's explanation material, then advance to the next question.
    Intro stage: gently move on to the next warm-up topic / first question.
    """
    global stage, current_q
    if stage is None:
        return JSONResponse({"error": "No active interview"}, status_code=400)

    if stage == "intro":
        return _advance_intro("", moved_on=True)

    if current_q is None:
        return JSONResponse({"error": "No active question"}, status_code=400)

    reply, next_question, done = _explain_current("[no answer]")
    reply_audio = tts.speak(reply) if reply else None
    _log(f"/explain: reply ({len(reply)} chars), tts -> {'ok' if reply_audio else 'None'}")

    return JSONResponse({
        "intro": False,
        "transcript": "[no answer]",
        "scores": {},
        "gaps": [],
        "feedback": "",
        "reply": reply,
        "reply_audio": base64.b64encode(reply_audio).decode() if reply_audio else None,
        "follow_up": None,
        "next_question": next_question,
        "done": done,
    })


def _audio64(text):
    audio = tts.speak(text) if text else None
    _log(f"tts -> {len(audio)} bytes" if audio else f"tts FAILED for {text[:50]!r}")
    return base64.b64encode(audio).decode() if audio else None


def _say(reply, fallback):
    """Never let a turn go unspoken — if the model returned empty, say the fallback."""
    reply = (reply or "").strip()
    return reply if reply else fallback


def context_str():
    lines = []
    for turn in history[-5:]:
        lines.append(f'Q: {turn["question"]}')
        lines.append(f'A: {turn["transcript"][:200]}')
        if turn.get("follow_up"):
            lines.append(f'Follow-up: {turn["follow_up"]}')
    return "\n".join(lines)


@app.post("/end")
async def end():
    global current_q, stage
    total = len(history)
    summary = {"total_turns": total}
    if total:
        avg = {}
        for key in ("relevance_score", "completeness_score", "clarity_score"):
            vals = [t["scores"].get(key, 0) or 0 for t in history if t.get("scores")]
            avg[key] = round(sum(vals) / len(vals), 1) if vals else 0
        summary["average_scores"] = avg
    path = save_transcript(qbank.title, history)
    current_q = None
    stage = None
    return JSONResponse({**summary, "transcript_path": path})
