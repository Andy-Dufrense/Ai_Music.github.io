"""
Lyrics recognition using OpenAI Whisper.
"""
import numpy as np
import whisper
import re
import librosa

from services.common import to_simplified

_model = None
_MODEL_SIZE = "base"

# Chinese song lyrics prompt — guides Whisper toward real words vs hallucination
_INITIAL_PROMPT = (
    "以下是中文流行歌曲的歌词："
    "的 了 我 你 是 不 在 有 他 她 这 那 们 一 个 就 也 都 还 会 要 能 去 来 上 下 到 说 看 想 "
    "爱 心 人 天 地 时 年 月 日 星 花 风 雨 云 海 山 路 梦 光 火 夜 春 秋 冬 夏 "
    "好 大 小 多 少 高 远 近 长 短 快 慢 新 旧 美 冷 暖 明 暗 深 浅 白 黑 红 蓝 绿 "
    "手 眼 泪 笑 声 话 歌 步 影 舞 门 窗 灯 书 信 画 酒 杯 衣 鞋 伞 船 车 "
    "永远 世界 幸福 快乐 悲伤 寂寞 自由 温柔 勇敢 坚强 脆弱 孤单 思念 回忆 忘记 等待 拥抱 "
    "时光 梦想 方向 流浪 承诺 阳光 彩虹 绽放 翅膀 灿烂 瞬间 温暖 美丽 永恒 遥远 "
    "爱情 青春 生命 微笑 眼泪 天空 大海 月光 星光 远方 明天 昨天 现在 未来 曾经 如果 "
    "我们 他们 自己 一起 还是 不要 已经 没有 可以 因为 所以 但是 如果 虽然 然后 只是 "
    "喜欢 害怕 希望 开始 结束 离开 回来 看见 听见 知道 明白 相信 改变 守护 原谅 放弃 "
    "第一 一次 每个 所有 最后 继续 重新 终于 忽然 总是 一直 永远 再也 多么 如此 "
    "轻轻 慢慢 悄悄 静静 深深 远远 淡淡 微微 重重 "
)
# Whisper hallucination patterns — these are NOT real lyrics
_NOISE_PATTERNS = [
    "词曲", "作词", "作曲", "编曲", "演唱", "制作",
    "词、曲", "作词、", "作曲、", "、编曲",
    "谢谢大家", "谢谢", "再见", "拜拜",
    "请欣赏", "下面", "接下来",
]
# Single chars that should never appear as standalone words in Chinese lyrics
_STANDALONE_NOISE_CHARS = set("词曲奏编")
# Characters commonly repeated in real lyrics (vocal exclamations) — keep these as-is
_VALID_REPEAT_CHARS = set("啊啦喔哦嗯呀嘿嗨哟哇呵唉呐吧吗呢")

# Single characters that Whisper hallucinates in loops — never appear in real lyrics solo
_HALLUCINATION_CHARS = set("词曲奏演唱制作")


def _filter_noise_words(text: str) -> str:
    """Remove known noise phrases that Whisper hallucinates from background audio."""
    # Strip Chinese punctuation for pattern matching (e.g. "词、曲" → "词曲")
    text_clean = re.sub(r"[，。！？、：；]", "", text)
    for pattern in _NOISE_PATTERNS:
        while pattern in text_clean:
            text_clean = text_clean.replace(pattern, "")
            text = text.replace(pattern, "")
            # Also try removing with punctuation variants
            for sep in ["、", "，", ",", " "]:
                variant = sep.join(list(pattern))
                if variant in text:
                    text = text.replace(variant, "")
    # Remove repetitive substrings (hallucination loops): same 2-char sequence 3+ times
    for length in [2, 3]:
        for m in re.finditer(rf"([一-鿿]{{{length}}})", text):
            sub = m.group(1)
            count = text.count(sub)
            if count >= 3 and sub not in ("不要", "已经", "我们", "自己", "还是", "可以", "没有", "因为", "所以", "如果", "只是", "还是", "他们", "一起", "永远", "所有", "还是", "总是", "一直"):
                text = text.replace(sub, "")
    # Collapse runs of the same character: 4+ = definitely hallucination → collapse to 1
    text = re.sub(r"([一-鿿])\1{3,}", r"\1", text)
    # 3 consecutive only collapsed if NOT a legitimate vocal exclamation
    text = re.sub(r"([一-鿿])\1{2}", lambda m: m.group() if m.group(1) in _VALID_REPEAT_CHARS else m.group(1), text)
    # Remove hallucination single-char repetitions (e.g. "词词词词")
    for ch in _HALLUCINATION_CHARS:
        text = re.sub(rf"{ch}{{2,}}", "", text)
    return text


def _post_process_text(text: str) -> str:
    text = re.sub(r"[，。！？、]{2,}", lambda m: m.group()[0], text)
    text = re.sub(r"\s+([，。！？、])", r"\1", text)
    text = re.sub(r"([，。！？、])\s+", r"\1", text)
    text = to_simplified(text)
    text = _filter_noise_words(text)
    # Remove non-Chinese characters: Latin letters, digits, punctuation that Whisper hallucinates
    text = re.sub(r"[a-zA-Z0-9]", "", text)
    text = re.sub(r"[^一-鿿，。！？、\s]", "", text)
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

    # Light trim only — Demucs vocals are already clean; over-processing causes hallucination
    try:
        audio, _ = librosa.effects.trim(audio, top_db=20)
    except Exception:
        pass

    # Whisper transcribe — low temperature to suppress hallucination
    try:
        result = model.transcribe(
            audio,
            task="transcribe",
            language="zh",
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=_INITIAL_PROMPT,
            temperature=(0.0,),
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
        )
    except TypeError:
        result = model.transcribe(
            audio,
            task="transcribe",
            language="zh",
            word_timestamps=True,
            initial_prompt=_INITIAL_PROMPT,
        )

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

    # Filter hallucinated consecutive identical single-char words
    # (e.g. "紧紧紧紧紧紧" output as 6 separate "紧" tokens)
    filtered = []
    streak_char = None
    streak_count = 0
    for w in words:
        if len(w["word"]) == 1 and w["word"] == streak_char:
            streak_count += 1
        else:
            streak_char = w["word"] if len(w["word"]) == 1 else None
            streak_count = 1
        if streak_count <= 2:  # keep at most 2 consecutive identical single chars
            filtered.append(w)
    # Remove standalone noise chars (e.g. "词", "曲") — never appear alone in real lyrics
    filtered = [w for w in filtered
                if not (len(w["word"]) == 1 and w["word"] in _STANDALONE_NOISE_CHARS)]
    words = filtered

    return {
        "full_text": full_text,
        "words": words,
        "language": result.get("language", "unknown"),
    }
