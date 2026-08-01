import json
import os
import re

import httpx

# Spoken hesitations/fillers that carry no technical content. Stripped before
# evaluation so "um, like, dropout is, uh, regularization" reads cleanly.
_FILLERS = re.compile(
    r"(?:\buh[\s-]*huh\b|\bmm[\s-]*hmm\b|\buh\b|\bum\b|\buhm\b|\berm\b|\ber\b|"
    r"\bhmm\b|\bmm\b|\byou\s+know\b|\bi\s+mean\b|\blike\b|\bbasically\b|"
    r"\bactually\b|\bkind\s+of\b|\bsort\s+of\b)",
    re.IGNORECASE,
)
# Immediate word repeats: "the the model" -> "the model".
_REPEAT = re.compile(r"\b(\w+)\s+\1(?:\s+\1)?\b", re.IGNORECASE)


def clean_transcript(text):
    """Strip filler words and immediate repeats from an ASR transcript.

    Self-corrections ("no wait, I mean X") are not fully repaired here —
    the eval prompt tells the model to judge the intended final answer.
    """
    text = _FILLERS.sub(" ", text)
    for _ in range(3):
        new = _REPEAT.sub(r"\1", text)
        if new == text:
            break
        text = new
    return re.sub(r"\s+", " ", text).strip()

# Direct "I can't answer" signals — route these to the explain-and-advance
# flow instead of scoring them as a real answer. Kept deliberately narrow so
# "I'm not sure, but I think dropout..." still gets scored normally.
# `don'?t` handles "don't"/"dont"; `do\s+not` handles the spelled-out form
# ("I do not know answer to this"), which a real candidate is just as likely
# to say as the contraction.
_DONT_KNOW = re.compile(
    r"\bi\s+(?:don'?t|do\s+not)\s+know\b|"
    r"\bi\s+(?:don'?t|do\s+not)\s+have\s+(?:a\s+)?(?:clue|idea)\b|"
    r"\bi\s+have\s+no\s+idea\b|\bno\s+idea\b|"
    r"\bi\s+couldn'?t\s+(?:answer|say)\b|\bi\s+have\s+no\s+answer\b|"
    r"\bi\s+(?:don'?t|do\s+not)\s+want\s+to\s+(?:answer|say)\b|"
    r"\bidk\b",
    re.IGNORECASE,
)


def is_dont_know(text):
    """True when the candidate is declining to answer (I don't know / no idea /
    can't answer), as opposed to giving a real attempt."""
    if not text or not text.strip():
        return False
    return bool(_DONT_KNOW.search(clean_transcript(text).lower()))

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
    "required": [
        "transcript", "relevance_score", "completeness_score", "clarity_score",
        "gaps", "feedback",
    ],
}

DEFAULT_BASE_URL = "http://localhost:8080"  # matches llama-server's default port

# Fixed interviewer identity — the same for every subject. Human, warm, teaching.
INTERVIEWER_NAME = "Mira"

SYSTEM_PROMPT_CORE = (
    f"You are {INTERVIEWER_NAME}, a warm, friendly human interviewer — not a "
    "robotic assistant. Introduce yourself by name once, at the very start of "
    "the session. After that, never repeat your name and never re-introduce "
    "yourself. This "
    "is a practice interview that doubles as a coaching session: put the candidate "
    "at ease and genuinely help them improve.\n\n"
    "Rules you always follow:\n"
    "1. Tone: warm, conversational, encouraging. Address the candidate directly. "
    "Never use stiff, bullet-point speech.\n"
    "2. Friendly questions: phrase every question the way a warm human colleague "
    "would — conversationally, never read it mechanically or robotically.\n"
    "3. Flow: open with a short warm-up (greeting, their name, what they do, "
    "their interest in machine learning), then transition into the actual "
    "interview questions. Once the interview starts, the warm-up is over — "
    "don't return to it.\n"
    "4. After every answer: first acknowledge what they did well (e.g. \"that's a "
    "solid answer\"), then give one concrete, supportive pointer on what they "
    "could have added to make it stronger. Teach — don't just grade. Never mock "
    "or discourage.\n"
    "5. If the answer is fully correct and complete, praise it genuinely and stop "
    "— never invent a nitpick. Only ever call out real technical mistakes or key "
    "omissions. Never criticize word choice, grammar, or sentence structure.\n"
    "6. Candidates speak naturally: they say \"uh\", \"um\", \"like\", repeat "
    "words, or start a sentence and correct themselves. Judge the intended final "
    "answer and ignore these hesitations entirely.\n"
    "7. Never rush past what the candidate actually said — respond to it before "
    "moving on.\n"
    "8. Stay on-topic and professional beneath the warmth; this is still an "
    "interview.\n"
    "9. Ask questions one at a time. Keep each spoken turn to a few sentences.\n"
    "10. Vary how you open each turn. Never start every response the same way "
    "(\"That's a great answer\"), and don't follow a fixed script — let each "
    "acknowledgement flow naturally from what the candidate actually said.\n"
    "11. Never re-introduce yourself, never repeat your name, and never "
    "re-summarize earlier parts of the conversation — the candidate remembers "
    "them. Respond only to the current turn.\n"
    "12. Never output meta-instructions, bracketed notes, or placeholders like "
    "[mention something they said]. Always speak the actual words of the reply. "
    "If there is nothing specific to reference, respond warmly in general terms."
)

_SCORE_KEYS = ["transcript", "feedback", "relevance_score", "completeness_score", "clarity_score"]


class InterviewModel:
    def __init__(self, base_url=None):
        base_url = base_url or os.environ.get("LLAMA_SERVER", DEFAULT_BASE_URL)
        self.client = httpx.Client(base_url=base_url, timeout=180)

    def _chat(self, prompt, system=None, max_tokens=512, temperature=0.7, top_p=0.9):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self.client.post(
                "/v1/chat/completions",
                json={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    # This GGUF's tokenizer drops '</s>' from the EOG list (it sees
                    # '<|tool_response>' as special), so the model never stops on its
                    # own EOS — it spams EOS until max_tokens and returns empty text.
                    # Explicit stop sequences make it terminate at turn end.
                    "stop": ["</s>", "<end_of_turn>"],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if not content or not content.strip():
                reason = data["choices"][0].get("finish_reason", "?")
                print(
                    f"[llama] WARNING empty content (finish={reason}, status={resp.status_code}) "
                    f"prompt: {prompt[:120]!r}",
                    flush=True,
                )
            return content
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Cannot reach llama.cpp server at {self.client.base_url} — "
                "start it first (README §5.3)"
            ) from exc

    def chat_reply(self, prompt, system):
        """Free-form conversational reply (warm-up / transitions / coaching)."""
        # Was 256 — but that truncated mid-sentence on long warm-up/coach turns,
        # and the model would return empty content when cut off at the cap.
        for attempt in range(2):
            text = self._chat(prompt, system=system, max_tokens=512)
            if text and text.strip():
                return text
            print(f"[llama] empty chat_reply, retrying (attempt {attempt + 1})", flush=True)
        return ""

    def transcribe_and_eval(self, transcript, question, keywords, system):
        keywords_str = ", ".join(keywords)
        prompt = (
            "You are an AI interviewer. Evaluate the candidate's answer to the "
            "question below against the expected keywords. Return ONLY valid JSON "
            f"matching this schema: {json.dumps(EVAL_SCHEMA)}\n\n"
            "Scoring rules:\n"
            "- Score each dimension 1-10. The transcript may contain spoken "
            "hesitations (\"uh\", \"um\", \"like\"), repeated words, or "
            "self-corrections — judge the candidate's intended final answer and "
            "ignore them.\n"
            "- In the \"feedback\" field: if the answer is fully correct and "
            "complete, praise it genuinely and do not invent nitpicks. Otherwise "
            "acknowledge what was good, then point out the specific technical "
            "mistake or key omission. Never mention word choice, grammar, or "
            "sentence structure.\n\n"
            f"Question: {question}\n"
            f"Expected keywords: {keywords_str}\n"
            f"Candidate's answer: {clean_transcript(transcript)}"
        )
        result = {}
        for _ in range(3):  # up to 2 retries on an empty parse
            result = self._parse_json(self._chat(prompt, system=system))
            if result.get("transcript") or any(
                result.get(k) for k in ("relevance_score", "completeness_score", "clarity_score")
            ):
                break
        return result

    def generate_followup(self, context, gaps, system):
        gaps_str = ", ".join(gaps)
        prompt = (
            f"The candidate missed these key points: {gaps_str}.\n"
            f"Conversation history:\n{context}\n\n"
            f"Generate a single follow-up question that probes deeper on these gaps. "
            f'Output as JSON: {{"follow_up_question": "..."}}'
        )
        text = self._chat(prompt, system=system, max_tokens=256)
        try:
            return json.loads(text).get("follow_up_question", text)
        except json.JSONDecodeError:
            return text

    def generate_question(self, config_title, skills, previous_qs, system):
        prev = ", ".join(previous_qs) if previous_qs else "none yet"
        prompt = (
            f"Interview config: {config_title}\n"
            f"Skills: {skills}\n"
            f"Previous questions asked: {prev}\n\n"
            f"Generate a new interview question that probes a different area than "
            f'previous questions. Output as JSON: {{"question": "..."}}'
        )
        text = self._chat(prompt, system=system, max_tokens=256)
        try:
            return json.loads(text).get("question", text)
        except json.JSONDecodeError:
            return text

    def _parse_json(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        result = {}
        for key in _SCORE_KEYS:
            m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
            if not m:
                m = re.search(rf'"{key}"\s*:\s*(\d+)', text)
            if m:
                val = m.group(1)
                if key.endswith("_score"):
                    val = int(val)
                result[key] = val
        gaps_m = re.search(r'"gaps"\s*:\s*\[(.*?)\]', text)
        if gaps_m:
            result["gaps"] = [g.strip(' "') for g in gaps_m.group(1).split(",") if g.strip()]
        return result


if __name__ == "__main__":
    m = InterviewModel()
    parsed = m._parse_json('sure: {"relevance_score": 7, "gaps": ["a", "b"], '
                           '"transcript": "hello", "feedback": "add examples"} trailing junk')
    assert parsed.get("relevance_score") == 7
    assert parsed.get("transcript") == "hello"
    assert parsed.get("gaps") == ["a", "b"]
    assert parsed.get("feedback") == "add examples"
    cleaned = clean_transcript("um so like dropout is uh the the regularization technique I mean")
    assert cleaned == "so dropout is the regularization technique"
    assert is_dont_know("um I don't know") is True
    assert is_dont_know("I have no idea, sorry") is True
    assert is_dont_know("I do not know answer to this") is True
    assert is_dont_know("I do not want to answer this one") is True
    assert is_dont_know("Sorry, I don't think I can answer that") is True
    assert is_dont_know("Dropout randomly drops neurons during training") is False
    print("interview_model self-check ok")
