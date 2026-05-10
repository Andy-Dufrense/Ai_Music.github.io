# AI Music Score Generator 🎵

Upload an MP3, automatically generate **piano**, **guitar TAB**, **bass TAB**, and **drum** sheet music — all rendered as vector SVG in the browser. No GPU required.

## Features

| Notation | Description |
|----------|-------------|
| **Piano** | Grand staff: right-hand melody + chord-tone fill, left-hand broken arpeggios, lyrics overlay |
| **Guitar TAB** | 6-line TAB with chord diagrams, jianpu (简谱), strumming/picking patterns, lyrics |
| **Bass TAB** | 4-line TAB with position-optimized fret numbers, section-aware patterns |
| **Drums** | Standard 5-line drum kit notation with kick/snare/hi-hat/cymbal patterns |

### Core Capabilities
- **Automatic BPM / key / time signature detection** (supports x/4 and x/8 meters)
- **Chord progression analysis** with roman numeral labeling
- **Song structure detection** (前奏/主歌/副歌/桥段/间奏/尾奏)
- **Lyrics transcription & alignment** (Chinese, via OpenAI Whisper)
- **Multi-stem separation** (vocals, drums, bass, guitar, piano, other — via Demucs)
- **Guitar chord diagrams** with auto-positioned fingerings, capo/transpose support
- **Pure SVG rendering** — no LilyPond, no music21, no external typesetting tools
- **CPU-friendly** — optimized for CPU-only inference, no GPU required

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| Stem Separation | Demucs (HTDemucs 6-stem) |
| Melody Extraction | torchcrepe (CREPE deep pitch model) + RMVPE fallback |
| Lyrics | OpenAI Whisper `base` (76M params, Chinese-optimized prompt) |
| Audio Analysis | librosa, scipy (BPM, key, chords, spectral processing) |
| Score Rendering | Custom hand-written SVG renderer (no LilyPond/music21/VexFlow) |
| Frontend | Vanilla HTML/CSS/JS, single-page app |

## Architecture

```
Browser (index.html)
    │  Upload MP3
    ▼
FastAPI Server (main.py)
    │
    ├─► Demucs → 6 separated stems
    ├─► Analyzer → BPM, key, chords, sections
    ├─► Melody (torchcrepe) → MIDI note sequence
    ├─► Whisper → Chinese lyrics with timestamps
    ├─► Notation generators → structured score data
    │       piano / guitar / bass / drums
    └─► SVG Renderer → vector sheet music
```

## Quick Start

### Requirements
- Python 3.10+
- 8GB+ RAM recommended
- Windows / macOS / Linux
- No GPU required

### 1. Clone

```bash
git clone https://github.com/Andy-Dufrense/Ai_Music.github.io.git
cd Ai_Music.github.io
```

### 2. Install

```bash
pip install -r backend/requirements.txt
```

Key dependencies: `torch`, `torchcrepe`, `whisper`, `librosa`, `soundfile`, `demucs`, `fastapi`, `uvicorn`, `scipy`

### 3. Run

```bash
python backend/main.py
```

Or double-click `start.bat` on Windows.

### 4. Use

Open **http://localhost:8020** → upload an MP3 → wait ~1.5-3 min for processing → view scores in 4 tabs.

## Project Structure

```
├── backend/
│   ├── main.py                    # FastAPI server, job pipeline, HTML generation
│   ├── config.py                  # Output paths, port config
│   ├── requirements.txt
│   └── services/
│       ├── analyzer.py            # BPM, key, chord, section detection
│       ├── common.py              # MIDI/note/duration/jianpu utilities
│       ├── lyrics.py              # Character splitting & lyrics-to-melody alignment
│       ├── lyrics_whisper.py      # Whisper transcription + multi-stage audio cleaning
│       ├── melody_rmvpe.py        # torchcrepe pitch extraction + bass note detection
│       ├── notation.py            # Score data generation (piano/guitar/bass/drums)
│       ├── separator.py           # Demucs stem separation
│       └── svg_renderer.py        # Hand-written SVG rendering engine
├── static/
│   ├── css/style.css
│   └── js/app.js
├── workflow/                      # Project planning & progress docs
├── index.html                     # Web UI (single-page app)
├── music_demo/                    # Demo MP3 files
├── start.bat                      # Windows quick-start
└── README.md
```

## Processing Pipeline

```
MP3 Upload
  → MP3→WAV conversion
  → Demucs 6-stem separation (vocals, drums, bass, guitar, piano, other)
  → Audio analysis (BPM, key, chord progression, song sections)
  → Melody extraction (torchcrepe on vocal stem)
  → Bass note extraction (torchcrepe on bass stem)
  → Lyrics transcription (Whisper base on cleaned vocal stem)
  → Score generation (4 notation types, section-aware patterns)
  → SVG rendering (custom hand-written SVG engine)
  → Display in browser with audio playback
```

## Time Signature Support

All score types support standard and compound meters:

- **x/4**: 2/4, 3/4, 4/4 — quarter-note beat units
- **x/8**: 6/8, 9/8, 12/8 — dotted-quarter beat units
- Measure duration, beat offsets, and rhythmic patterns adapt automatically

## Notes

- **First run** downloads Whisper `base` model (~140MB) and Demucs model (~300MB) — this happens automatically
- **Processing time**: ~1.5-3 min on CPU for a 3-minute song
- **Output files**: saved in `backend/outputs/<job_id>/` (WAV stems, JSON score data, HTML scores)
- **CPU optimizations**: Demucs shifts=0, 16kHz sample rate, 120s max duration — ~50% faster than defaults
- **No LilyPond**: all sheet music is rendered by a custom SVG engine (~2500 lines of hand-written rendering code)

## Known Limitations

### 歌词识别 ⚠️ 主要短板
- 使用 Whisper `base`（76M 参数），是 OpenAI 最小的模型，**中文识别准确率有限**
- **幻觉问题**：会出现重复字（"词词词词"、"紧紧紧紧紧"）、无意义虚词、漏句
- **时间戳偏移**：词级别的时间对齐有误差，导致歌词挂载位置不准
- **背景干扰**：Demucs 人声分离不干净时，伴奏残留会严重干扰识别
- **快节奏/说唱段落**：几乎无法正确识别
- 已有缓解措施（连续重复字过滤、虚词白名单），但不能根治
- 提升方向：换用更大 Whisper 模型（small/medium）、引入 FunASR/Paraformer、或对歌词做后编辑

### 其他不足
| 维度 | 问题 |
|------|------|
| 调性检测 | 大小调偶尔混淆（C 大调 vs A 小调），无中途转调支持 |
| 人声分离 | Demucs CPU 模式下分离质量一般，乐器声残留会影响 pitch/lyrics |
| 和弦识别 | 仅支持三和弦/七和弦，不含转位、挂留、增减等 |
| 段落检测 | 基于启发式规则，复杂编曲可能误判 |
| 处理上限 | 120 秒（可调），超长歌曲会被截断 |
| 吉他指法 | 生僻和弦（如 dim/aug/m7♭5）的指法自动生成不够准确 |

## License

This project is for personal/demo use.
