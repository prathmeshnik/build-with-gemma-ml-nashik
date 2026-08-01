import io
import os
import subprocess
import tempfile
import time
import wave

import numpy as np

KOKORO_VOICE = "af_heart"  # American English female; alt: "am_michael"
KOKORO_SR = 24000
PIPER_PATH = "piper/piper"
PIPER_MODEL = "piper/en_US-lessac-medium.onnx"


def _wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    pcm = np.clip(samples, -1, 1)
    pcm = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class PiperTTS:
    def __init__(self):
        self._available = os.path.isfile(PIPER_PATH) and os.path.isfile(PIPER_MODEL)

    def speak(self, text):
        if not self._available:
            return None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        try:
            subprocess.run(
                [PIPER_PATH, "--model", PIPER_MODEL, "--output-file", out_path],
                input=text.encode(),
                capture_output=True,
                timeout=30,
            )
            with open(out_path, "rb") as f:
                return f.read()
        except Exception:
            return None
        finally:
            if os.path.isfile(out_path):
                os.unlink(out_path)


class TTS:
    """Kokoro 82M primary, Piper subprocess fallback (silent)."""

    def __init__(self):
        self._kokoro = None
        self._piper = PiperTTS()

    def _get_kokoro(self):
        if self._kokoro is None:
            from kokoro import KPipeline

            t0 = time.monotonic()
            print("[tts] loading Kokoro pipeline...", flush=True)
            self._kokoro = KPipeline(lang_code="a")
            print(f"[tts] Kokoro loaded in {time.monotonic() - t0:.1f}s", flush=True)
        return self._kokoro

    def speak(self, text: str):
        """Returns WAV bytes (16-bit PCM) for the browser, or None if unavailable."""
        t0 = time.monotonic()
        try:
            pipeline = self._get_kokoro()
            chunks = []
            for r in pipeline(text, voice=KOKORO_VOICE):
                if r.output is not None and r.output.audio is not None:
                    if chunks:  # natural pause between sentences; longer before a question
                        gap = 0.8 if getattr(r, "graphemes", "").strip().endswith("?") else 0.35
                        chunks.append(np.zeros(int(gap * KOKORO_SR), dtype=np.float32))
                    chunks.append(r.output.audio.detach().cpu().numpy().astype(np.float32))
            if not chunks:
                print(f"[tts] 0 audio chunks for {text[:40]!r} -> piper fallback", flush=True)
                return self._piper.speak(text)
            audio = _wav_bytes(np.concatenate(chunks), KOKORO_SR)
            print(f"[tts] {len(audio)} bytes ({len(chunks)} chunks) for {text[:40]!r} in {time.monotonic() - t0:.1f}s", flush=True)
            return audio
        except Exception as exc:
            print(f"[tts] kokoro failed ({exc!r}) -> piper fallback", flush=True)
            return self._piper.speak(text)


if __name__ == "__main__":
    audio = TTS().speak("Hello, welcome to the interview.")
    if audio:
        assert audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"
        print(f"tts self-check ok ({len(audio)} bytes)")
    else:
        print("tts self-check skipped: no voice engine available")
