"""Assert-based checks for the pipeline. Run: .venv/bin/python test_pipeline.py

Note: test_stt_transcribes_tone and test_tts_returns_wav download their models
on first run (~0.5 GB whisper, ~0.3 GB kokoro) — see README §5.5 for keeping
them project-local. The rest need no models.
"""

import numpy as np

from lib.audio_capture import MAX_SAMPLES, TARGET_SR, to_mono_16k
from lib.interview_model import EVAL_SCHEMA, InterviewModel, clean_transcript
from lib.privacy_audit import audit
from lib.question_loader import QuestionBank


def _tone(seconds=2, freq=440.0, sr=TARGET_SR):
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_to_mono_16k():
    stereo_48k = (np.random.default_rng(0).random((2, 48000)) * 2 - 1).astype(np.float32)
    out = to_mono_16k(stereo_48k, 48000)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.shape[0] <= MAX_SAMPLES
    assert abs(out.max()) <= 1.0


def test_to_mono_16k_no_padding():
    short = np.zeros(100, dtype=np.float32)
    out = to_mono_16k(short, 48000)
    assert out.shape[0] < 1000, "short input must not be padded"


def test_parse_json_valid():
    m = InterviewModel()
    blob = '{"transcript": "yes", "relevance_score": 8, "completeness_score": 7, "clarity_score": 9, "gaps": ["x"], "feedback": "add an example"}'
    parsed = m._parse_json(blob)
    assert parsed["transcript"] == "yes"
    assert parsed["relevance_score"] == 8
    assert parsed["feedback"] == "add an example"


def test_parse_json_malformed():
    m = InterviewModel()
    raw = 'Sure! Here you go: {"transcript": "hi", "relevance_score": 7, "completeness_score": 6, "clarity_score": 8, "gaps": ["a","b"], "feedback": "be more specific"} hope that helps.'
    parsed = m._parse_json(raw)
    assert parsed.get("relevance_score") == 7
    assert parsed.get("completeness_score") == 6
    assert parsed.get("transcript") == "hi"
    assert parsed.get("gaps") == ["a", "b"]
    assert parsed.get("feedback") == "be more specific"


def test_question_bank_no_repeats():
    import json
    import tempfile

    bank = {
        "title": "t",
        "questions": [
            {"id": 1, "question": "a"},
            {"id": 2, "question": "b"},
            {"id": 3, "question": "c"},
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(bank, f)
        path = f.name
    try:
        qb = QuestionBank(path)
        ids = []
        while True:
            q = qb.next()
            if q is None:
                break
            ids.append(q["id"])
        assert len(ids) == 3 and len(set(ids)) == 3, "no repeats and full exhaust"
        assert qb.next() is None, "exhausted bank returns None"
    finally:
        import os

        os.unlink(path)


def test_privacy_audit_no_new_outbound():
    audit(lambda: None)


def test_stt_transcribes_tone():
    from lib.stt import transcribe

    text = transcribe(_tone())
    assert isinstance(text, str) and text, "tone should transcribe to something"


def test_tts_returns_wav():
    from lib.tts import TTS

    audio = TTS().speak("Hello, welcome to the interview.")
    if audio is None:
        print("  (skipped: no voice engine available)")
        return
    assert audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"
    assert len(audio) > 44


def test_eval_schema_required_fields():
    for k in ("transcript", "relevance_score", "completeness_score", "clarity_score", "gaps", "feedback"):
        assert k in EVAL_SCHEMA["required"]


def test_clean_transcript():
    messy = "um so like dropout is uh the the regularization technique, I mean it, you know"
    out = clean_transcript(messy)
    assert out == "so dropout is the regularization technique, it,"
    assert "um" not in out and "like" not in out and "the the" not in out
    assert "regularization" in out, "real content must survive cleaning"


if __name__ == "__main__":
    import sys

    only = sys.argv[1:] if len(sys.argv) > 1 else None
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and (only is None or name in only):
            try:
                fn()
                print(f"{name} ... ok")
            except Exception as exc:
                failures += 1
                print(f"{name} ... FAIL: {exc}")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("All tests passed.")
