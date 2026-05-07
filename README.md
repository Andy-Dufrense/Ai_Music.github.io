# AI Music Score Generator 🎵

Upload an MP3, automatically generate **piano**, **guitar TAB**, **bass TAB**, and **drum** sheet music — all rendered as vector SVG scores.

## Features

| Notation | Description |
|----------|-------------|
| Piano | Grand staff: right-hand melody + chord tones, left-hand broken arpeggios |
| Guitar TAB | 6-line TAB with chord diagrams, jianpu (简谱), strumming/picking patterns |
| Bass TAB | 4-line TAB with position-optimized fret numbers |
| Drums | Standard drum kit notation |

- **Automatic key/BPM/time signature detection**
- **Chord progression analysis** with roman numeral labeling
- **Song structure detection** (intro/verse/chorus/bridge/outro)
- **Lyrics transcription** (Chinese) via Whisper

## Requirements

- **Python 3.10+**
- **Windows / macOS / Linux**
- **8GB+ RAM** recommended
- **CPU** works (no GPU required, but slower)

## Quick Start

### 1. Clone

```bash
git clone https://github.com/Andy-Dufrense/Ai_Music.github.io.git
cd Ai_Music.github.io
```

### 2. Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

Key dependencies:
- `torch`, `torchcrepe`, `whisper` — melody detection + lyrics
- `librosa`, `soundfile` — audio analysis
- `demucs` — stem separation (vocals/drums/bass/guitar/piano)
- `fastapi`, `uvicorn` — backend server
- `scipy` — audio filtering

### 3. Run

```bash
python backend/main.py
```

Or on Windows, double-click `start.bat`.

### 4. Open browser

Go to **http://localhost:8020**

Upload an MP3 file → wait ~2-4 minutes for processing → view the 4 score tabs.

## Project Structure

```
├── backend/
│   ├── main.py                  # FastAPI server entry point
│   ├── config.py                # Output paths, port config
│   ├── requirements.txt
│   └── services/
│       ├── analyzer.py          # BPM, key, chord, section detection
│       ├── common.py            # MIDI/note/duration utilities
│       ├── lyrics.py            # Lyrics-to-melody alignment
│       ├── lyrics_whisper.py    # Whisper transcription + audio cleaning
│       ├── melody_rmvpe.py      # torchcrepe pitch extraction
│       ├── notation.py          # Score data generation (piano/guitar/bass/drums)
│       ├── separator.py         # Demucs stem separation
│       └── svg_renderer.py      # Hand-written SVG rendering engine
├── frontend/
│   ├── index.html               # Web UI
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── music_demo/                  # Demo MP3 files
└── start.bat                    # Windows quick-start script
```

## Processing Pipeline

```
MP3 Upload
  → Demucs 6-stem separation (vocals, drums, bass, guitar, piano, other)
  → Audio analysis (BPM, key, chords, sections)
  → Melody extraction (torchcrepe on vocal stem)
  → Lyrics transcription (Whisper on cleaned vocal stem)
  → Score generation (4 notation types)
  → SVG rendering
  → Display in browser
```

## Notes

- **First run** downloads Whisper `small` model (~500MB) and Demucs model (~300MB) — this happens automatically
- Processing time: ~2-4 min on CPU for a 3-minute song
- Output files are saved in `backend/outputs/<job_id>/`
- The frontend hosted on GitHub Pages is **static only** — it won't work without the backend running locally

## License

This project is for personal/demo use.
