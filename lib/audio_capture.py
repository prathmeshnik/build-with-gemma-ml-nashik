import io
import wave

import numpy as np
import torch
import torchaudio

TARGET_SR = 16000
MAX_SAMPLES = int(TARGET_SR * 120.0)  # 1_920_000 — hard cap (2 min), no padding


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
    if isinstance(audio, bytes):
        return _from_wav(audio)
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(np.ascontiguousarray(audio))
    if audio.ndim > 1 and audio.shape[0] > 1:
        audio = audio.mean(dim=0)
    elif audio.ndim > 1:
        audio = audio.squeeze(0)
    if sample_rate is not None and sample_rate != TARGET_SR:
        audio = torchaudio.transforms.Resample(sample_rate, TARGET_SR)(audio.float())
    if audio.dtype != torch.float32:
        audio = audio.float()
    max_val = audio.abs().max()
    if max_val > 0:
        audio = audio / max_val
    audio = audio[:MAX_SAMPLES]
    return audio.numpy()


def _from_wav(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        dtype = {1: np.uint8, 2: np.int16, 4: np.float32}[wf.getsampwidth()]
        audio = np.frombuffer(frames, dtype=dtype)
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels()).mean(axis=1)
    return to_mono_16k(audio, sr)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    stereo_48k = (rng.random((2, 48000)) * 2 - 1).astype(np.float32)
    out = to_mono_16k(stereo_48k, 48000)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert abs(out.max()) <= 1.0
    assert out.shape[0] <= MAX_SAMPLES
    print("audio_capture self-check ok")
