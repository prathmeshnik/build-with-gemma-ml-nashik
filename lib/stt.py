import os
import time

import numpy as np
from faster_whisper import WhisperModel

_model = None


def _model_path():
    # Prefer a local HuggingFace faster-whisper checkout (models/whisper/small).
    if os.path.isdir("models/whisper/small"):
        return "models/whisper/small"
    return "small"


def _get_model():
    global _model
    if _model is None:
        t0 = time.monotonic()
        path = _model_path()
        print(f"[stt] loading faster-whisper from {path}...", flush=True)
        try:
            _model = WhisperModel(path, device="cuda", compute_type="int8")
        except Exception as exc:
            if path != "small":
                print(f"[stt] local load failed ({exc!r}), falling back to HF 'small'", flush=True)
                _model = WhisperModel("small", device="cuda", compute_type="int8")
            else:
                raise
        print(f"[stt] whisper-small loaded in {time.monotonic() - t0:.1f}s", flush=True)
    return _model


def transcribe(audio: np.ndarray, vad_filter: bool = False) -> str:
    """audio = 16k mono float32 (see audio contract). Returns transcript string.

    `vad_filter` drops silence/noise so whisper stops hallucinating
    "2.5kg"-style text from dead air. Defaults OFF so synthetic-tone callers
    (e.g. the self-check tone) still transcribe; the server enables it.
    """
    t0 = time.monotonic()
    segments, _ = _get_model().transcribe(audio, beam_size=5, vad_filter=vad_filter)
    text = " ".join(s.text for s in segments).strip()
    print(f"[stt] {len(audio) / 16000:.1f}s audio -> {text[:80]!r} in {time.monotonic() - t0:.1f}s", flush=True)
    return text


if __name__ == "__main__":
    sr = 16000
    t = np.arange(sr * 2) / sr
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    text = transcribe(tone)
    assert isinstance(text, str) and text, f"expected non-empty transcript, got {text!r}"
    print(f"stt self-check ok: {text!r}")
