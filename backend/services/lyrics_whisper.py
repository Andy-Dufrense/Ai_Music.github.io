"""
Lyrics recognition using OpenAI Whisper.
"""
import numpy as np
import whisper
import re
import librosa

from services.common import to_simplified

_model = None
_MODEL_SIZE = "small"


def _post_process_text(text: str) -> str:
    text = re.sub(r"[，。！？、]{2,}", lambda m: m.group()[0], text)
    text = re.sub(r"\s+([，。！？、])", r"\1", text)
    text = re.sub(r"([，。！？、])\s+", r"\1", text)
    text = to_simplified(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_model():
    global _model
    if _model is not None:
        return _model
    import torch
    _model = whisper.load_model(_MODEL_SIZE, device="cpu")
    torch.set_grad_enabled(False)
    print(f"[lyrics] Whisper {_MODEL_SIZE} model loaded")
    return _model


def _bandpass_vocal(audio: np.ndarray, sr: int,
                    low_cut: float = 100.0, high_cut: float = 4000.0) -> np.ndarray:
    """Apply bandpass filter to isolate vocal frequency range."""
    from scipy.signal import butter, sosfiltfilt
    nyq = sr / 2
    sos = butter(4, [low_cut / nyq, high_cut / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def _spectral_subtraction(audio: np.ndarray, sr: int, noise_reduce_db: float = 12.0) -> np.ndarray:
    """Remove background noise using spectral subtraction.
    Estimates noise profile from quietest frames and subtracts from all frames."""
    # STFT
    D = librosa.stft(audio, n_fft=1024, hop_length=256)

    # Estimate noise from quietest 10% of frames
    mag = np.abs(D)
    frame_energy = np.mean(mag, axis=0)
    n_frames = len(frame_energy)
    n_noise_frames = max(1, n_frames // 10)
    noise_indices = np.argpartition(frame_energy, n_noise_frames)[:n_noise_frames]
    noise_profile = np.mean(mag[:, noise_indices], axis=1, keepdims=True)

    # Subtract noise with overshoot factor
    gain = np.maximum(1.0 - (noise_reduce_db / 20.0) * noise_profile / (mag + 1e-10), 0.01)
    D_clean = D * gain

    # Reconstruct
    cleaned = librosa.istft(D_clean, hop_length=256, length=len(audio))
    return cleaned.astype(np.float32)


def _apply_noise_gate(audio: np.ndarray, sr: int, threshold_db: float = -35.0,
                      attack_ms: float = 5.0, release_ms: float = 50.0) -> np.ndarray:
    """Apply a soft noise gate to suppress background noise before transcription."""
    # Compute RMS energy in short windows
    win_len = int(sr * attack_ms / 1000)
    if win_len < 64:
        win_len = 64
    hop = win_len // 4

    n_frames = (len(audio) - win_len) // hop + 1
    if n_frames <= 0:
        return audio

    # Per-frame RMS
    rms = np.array([np.sqrt(np.mean(audio[i * hop:i * hop + win_len] ** 2) + 1e-10)
                    for i in range(n_frames)])

    # Convert threshold to linear
    threshold_linear = 10 ** (threshold_db / 20)

    # Compute gain for each frame
    gain = np.ones(len(audio), dtype=np.float32)
    release_samples = int(sr * release_ms / 1000)

    for i in range(n_frames):
        if rms[i] < threshold_linear:
            # Apply soft attenuation proportional to how far below threshold
            ratio = (rms[i] / (threshold_linear + 1e-10)) ** 0.5
            start = i * hop
            end = min(start + win_len + release_samples, len(audio))
            # Linear ramp from attenuated to 1.0 over release
            ramp = np.linspace(ratio, 1.0, end - start)
            gain[start:end] = np.minimum(gain[start:end], ramp.astype(np.float32))

    return audio * gain


def transcribe_lyrics(audio_path: str) -> dict:
    """
    Transcribe singing vocals using OpenAI Whisper.
    Returns {"full_text": str, "words": [{word, start, end}], "language": str}.
    """
    import soundfile as sf

    model = _load_model()

    # Load audio ourselves to avoid ffmpeg dependency
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
    audio = audio.astype("float32")

    # Multi-stage vocal cleaning pipeline for clean lyrics input
    try:
        audio, _ = librosa.effects.trim(audio, top_db=30)
    except Exception:
        pass
    audio = _bandpass_vocal(audio, 16000, low_cut=100.0, high_cut=4000.0)
    audio = _spectral_subtraction(audio, 16000, noise_reduce_db=12.0)
    audio = _apply_noise_gate(audio, 16000, threshold_db=-28.0)

    # Whisper transcribe with anti-hallucination and VAD parameters
    try:
        result = model.transcribe(
            audio,
            language="zh",
            word_timestamps=True,
            condition_on_previous_text=False,
            compression_ratio_threshold=1.5,
            no_speech_threshold=0.5,
            logprob_threshold=-1.0,
            vad_filter=True,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            best_of=5,
            beam_size=5,
        )
    except TypeError:
        # Fallback for older Whisper without anti-hallucination params
        try:
            result = model.transcribe(
                audio,
                language="zh",
                word_timestamps=True,
                condition_on_previous_text=False,
            )
        except TypeError:
            result = model.transcribe(audio, language="zh", word_timestamps=True)

    segments = result.get("segments", [])
    full_text = _post_process_text(result.get("text", ""))

    words = []
    for seg in segments:
        for w in seg.get("words", []):
            word_text = w.get("word", "").strip()
            if word_text:
                words.append({
                    "word": to_simplified(word_text),
                    "start": round(w.get("start", 0), 3),
                    "end": round(w.get("end", 0), 3),
                })

    return {
        "full_text": full_text,
        "words": words,
        "language": result.get("language", "unknown"),
    }
