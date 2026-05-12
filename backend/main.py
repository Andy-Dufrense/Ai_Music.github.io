import sys
import os

# Fix site-packages path for current Python installation
_site_pkg = os.path.join(os.path.dirname(sys.executable), "lib", "site-packages")
if os.path.isdir(_site_pkg) and _site_pkg not in sys.path:
    sys.path.insert(0, _site_pkg)

# HuggingFace mirror for China
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# CPU thread optimization
_num_cores = os.cpu_count() or 4
os.environ.setdefault("OMP_NUM_THREADS", str(_num_cores))
os.environ.setdefault("MKL_NUM_THREADS", str(_num_cores))
try:
    import torch
    torch.set_num_threads(_num_cores)
    torch.set_num_interop_threads(_num_cores)
except Exception:
    pass

# Make ffmpeg available for audio processing
try:
    import imageio_ffmpeg
    _ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import uuid
import json
import shutil
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config import BASE_DIR, UPLOAD_DIR, OUTPUT_DIR, TEMPLATE_DIR, NOTATION_TYPES
from services.separator import separate_stems
from services.analyzer import analyze_full, detect_sections
from services.melody_rmvpe import extract_melody, extract_bass_notes
from services.lyrics_whisper import transcribe_lyrics
from services.lyrics import align_lyrics_to_notes
from services.notation import (
    MELODY_BASED_TYPES,
    generate_drum_score,
    generate_bass_score,
    generate_piano_score,
    generate_guitar_score,
)

app = FastAPI(title="AI Music Score Generator")

# Pre-load heavy models at startup
@app.on_event("startup")
async def startup_preload():
    # Pre-load Whisper model for lyrics recognition
    try:
        from services.lyrics_whisper import _load_model
        _load_model()
        print("[main] Startup pre-loading complete")
    except Exception as e:
        print(f"[main] Pre-load warning: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store
jobs: dict = {}


def process_job(job_id: str, file_path: str):
    """Background processing pipeline."""
    job = jobs[job_id]
    try:
        # Step 1: Separate stems
        job["status"] = "separating"
        job["message"] = "正在分离音轨..."
        stems = separate_stems(file_path, job_id)
        job["stems"] = stems
        job["progress"] = 25

        # Step 2: Analyze full mix (with bass stem for better chord detection)
        job["status"] = "analyzing"
        job["message"] = "正在分析BPM、调性、和弦..."
        analysis = analyze_full(file_path, bass_audio_path=stems.get("bass"))
        job["bpm"] = analysis["bpm"]
        job["key"] = analysis["key"]
        job["key_mode"] = analysis["key_mode"]
        job["time_signature"] = analysis["time_signature"]
        job["chords"] = analysis["chords"]
        job["progress"] = 50

        # Step 3: Detect song sections (intro/verse/chorus/bridge/outro)
        job["message"] = "正在分析歌曲结构..."
        sections = detect_sections(
            stems.get("vocals", file_path),
            analysis["chords"],
            analysis["bpm"],
            analysis["time_signature"],
            max(1, len(analysis["chords"])),
        )
        job["sections"] = sections
        job["progress"] = 55

        # Step 4: Extract melody from vocals
        job["message"] = "正在提取人声旋律..."
        melody_notes = extract_melody(stems.get("vocals", file_path))
        job["melody_notes"] = melody_notes
        job["progress"] = 65

        # Step 5: Pre-compute bass notes for faster score generation later
        job["message"] = "正在分析贝斯音轨..."
        bass_path = stems.get("bass")
        if bass_path:
            try:
                job["bass_notes"] = extract_bass_notes(bass_path)
            except Exception:
                job["bass_notes"] = []
        else:
            job["bass_notes"] = []
        job["progress"] = 75

        # Step 6: Transcribe lyrics from vocals
        job["message"] = "正在识别歌词..."
        lyrics_data = transcribe_lyrics(stems.get("vocals", file_path))
        job["lyrics"] = lyrics_data
        job["progress"] = 90

        # Step 7: Align lyrics to melody
        job["message"] = "正在对齐歌词与旋律..."
        aligned_notes = align_lyrics_to_notes(lyrics_data, melody_notes, analysis["bpm"])
        job["aligned_notes"] = aligned_notes
        job["progress"] = 100

        job["status"] = "done"
        job["message"] = "分析完成"

    except Exception as e:
        job["status"] = "error"
        job["message"] = f"处理失败: {str(e)}"
        import traceback
        traceback.print_exc()


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".mp3"):
        raise HTTPException(400, "仅支持MP3文件")

    job_id = uuid.uuid4().hex[:12]
    mp3_path = os.path.join(UPLOAD_DIR, f"{job_id}.mp3")

    with open(mp3_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Convert MP3 to WAV for processing (torchaudio needs WAV on Windows)
    wav_path = os.path.join(UPLOAD_DIR, f"{job_id}.wav")
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        processing_path = wav_path
    except Exception:
        # Fallback: try MP3 directly
        processing_path = mp3_path

    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    jobs[job_id] = {
        "id": job_id,
        "status": "processing",
        "message": "准备处理...",
        "progress": 0,
        "filename": file.filename,
        "stems": {},
        "bpm": None,
        "key": None,
        "key_mode": None,
        "time_signature": None,
        "chords": [],
        "melody_notes": [],
        "lyrics": {},
        "aligned_notes": [],
        "bass_notes": [],
        "sections": [],
    }

    # Start background processing
    thread = threading.Thread(target=process_job, args=(job_id, processing_path), daemon=True)
    thread.start()

    return JSONResponse({
        "job_id": job_id,
        "status": "processing",
    })


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")

    response = {
        "job_id": job_id,
        "status": job["status"],
        "message": job["message"],
        "progress": job["progress"],
        "filename": job.get("filename", ""),
    }

    if job["status"] == "done":
        response["stems"] = {
            name: f"/api/stems/{job_id}/{name}"
            for name, path in job.get("stems", {}).items()
        }
        response["bpm"] = job["bpm"]
        response["key"] = job["key"]
        response["key_mode"] = job["key_mode"]
        response["time_signature"] = job["time_signature"]
        response["chords"] = job["chords"]

    return JSONResponse(response)


@app.post("/api/generate")
async def generate_score(
    job_id: str = Form(...),
    notation_type: str = Form(...),
    audio_stem: str = Form("other"),
    fingering: str = Form("auto"),
):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if job["status"] != "done":
        raise HTTPException(400, "音频尚未分析完成")
    if notation_type not in NOTATION_TYPES:
        raise HTTPException(400, f"不支持的乐谱类型: {notation_type}")

    aligned_notes = job.get("aligned_notes", [])
    chords = job.get("chords", [])
    bpm = job.get("bpm", 120)
    key_name = job.get("key", "C")
    key_mode = job.get("key_mode", "major")
    time_sig = job.get("time_signature", [4, 4])
    sections = job.get("sections", [])
    lyrics_data = job.get("lyrics", {})

    # Generate score data
    try:
        if notation_type == "piano":
            score_data = generate_piano_score(
                aligned_notes, chords, bpm, key_name, key_mode, time_sig, sections
            )
        elif notation_type == "drums":
            drum_path = job.get("stems", {}).get("drums")
            if not drum_path or not os.path.exists(drum_path):
                raise HTTPException(400, "鼓音轨不可用")
            score_data = generate_drum_score(drum_path, bpm, time_sig, sections=sections)
        elif notation_type == "bass":
            bass_path = job.get("stems", {}).get("bass")
            if not bass_path or not os.path.exists(bass_path):
                raise HTTPException(400, "贝斯音轨不可用")
            bass_notes = job.get("bass_notes", [])
            score_data = generate_bass_score(bass_path, chords, bpm, key_name, key_mode, time_sig, bass_notes, sections=sections)
        else:
            func = MELODY_BASED_TYPES[notation_type]
            # For guitar: generate 2 fingering versions (auto + none) for client-side toggle
            if notation_type == "guitar":
                fingering_versions = {}
                for f in ["auto", "none"]:
                    fd = func(aligned_notes, chords, bpm, key_name, key_mode, time_sig,
                              sections, lyrics_data=lyrics_data, fingering=f)
                    fd["sections"] = sections
                    lyrics_data_job = job.get("lyrics", {})
                    if lyrics_data_job:
                        fd["lyrics"] = lyrics_data_job.get("full_text", "")
                    fd["_songName"] = job.get("filename", "").replace(".mp3", "").replace(".wav", "")
                    fingering_versions[f] = fd
                score_data = fingering_versions["auto"]
            else:
                score_data = func(aligned_notes, chords, bpm, key_name, key_mode, time_sig,
                                  sections, lyrics_data=lyrics_data)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(500, f"乐谱生成失败 [{notation_type}]: {str(e)}\n\n{tb[-500:]}")

    # Attach section info for renderer
    score_data["sections"] = sections

    # Include lyrics in score data
    lyrics_data = job.get("lyrics", {})
    if lyrics_data:
        score_data["lyrics"] = lyrics_data.get("full_text", "")

    # Include song name for title bar
    score_data["_songName"] = job.get("filename", "").replace(".mp3", "").replace(".wav", "")

    # Save score data
    score_file = os.path.join(OUTPUT_DIR, job_id, f"{notation_type}_score.json")
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(score_data, f, ensure_ascii=False, indent=2)

    # Check if vocals stem exists for accompaniment
    vocals_available = bool(job.get("stems", {}).get("vocals"))

    # Generate HTML page (with fingering versions for guitar)
    html = generate_score_html(score_data, audio_stem, job_id,
                               fingering_versions=fingering_versions if notation_type == "guitar" else None,
                               vocals_available=vocals_available)
    html_file = os.path.join(OUTPUT_DIR, job_id, f"{notation_type}_score.html")
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)

    return JSONResponse({
        "score_url": f"/api/score/{job_id}/{notation_type}",
        "score_data_url": f"/api/score_data/{job_id}/{notation_type}",
    })


@app.get("/api/score/{job_id}/{notation_type}")
async def get_score_html(job_id: str, notation_type: str):
    html_file = os.path.join(OUTPUT_DIR, job_id, f"{notation_type}_score.html")
    if not os.path.exists(html_file):
        raise HTTPException(404, "乐谱尚未生成")
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content)


@app.get("/api/score_data/{job_id}/{notation_type}")
async def get_score_data(job_id: str, notation_type: str):
    score_file = os.path.join(OUTPUT_DIR, job_id, f"{notation_type}_score.json")
    if not os.path.exists(score_file):
        raise HTTPException(404, "乐谱数据不存在")
    return FileResponse(score_file, media_type="application/json")


@app.get("/api/stems/{job_id}/{stem_name}")
async def get_stem(job_id: str, stem_name: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")

    stems = job.get("stems", {})
    stem_path = stems.get(stem_name)
    if not stem_path or not os.path.exists(stem_path):
        raise HTTPException(404, f"音轨不存在: {stem_name}")

    return FileResponse(stem_path, media_type="audio/wav")


def generate_score_html(score_data: dict, audio_stem: str, job_id: str,
                        fingering_versions: dict = None, vocals_available: bool = False) -> str:
    """Generate a self-contained HTML page with server-side rendered SVG score."""

    song_info = score_data.get("songInfo", {})
    notation_type = score_data.get("notationType", "piano")
    chords = score_data.get("chordProgression", [])
    measures = score_data.get("measures", [])

    # Get original filename
    job = jobs.get(job_id, {})
    song_name = job.get("filename", "Unknown").replace(".mp3", "").replace(".wav", "")

    display_key = song_info.get('originalKey') or song_info.get('key', 'C')
    key_display = f"{display_key} {'大调' if song_info.get('keyMode') == 'major' else '小调'}"
    chord_summary = " → ".join(
        [f"{c['degree']}({c['chord']})" for c in chords[:8]]
    ) if chords else "—"

    notation_names = {"piano": "钢琴谱", "guitar": "吉他谱", "bass": "贝斯谱", "drums": "架子鼓谱"}
    notation_name = notation_names.get(notation_type, notation_type)

    # Render score
    from services.svg_renderer import RENDERERS
    renderer = RENDERERS.get(notation_type)
    if renderer:
        try:
            score_content = renderer(score_data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            score_content = f'<p style="color:red;padding:20px;">乐谱渲染失败: {e}</p>'
    else:
        score_content = "<p>不支持的乐谱类型</p>"

    # Render fingering versions for guitar toggle
    fingering_scores = ""
    fingering_buttons = ""
    if fingering_versions and renderer:
        # Determine which shape auto selected — only show that + original
        auto_info = fingering_versions.get("auto", {}).get("songInfo", {})
        auto_fingering = auto_info.get("fingering")  # "C" or "G"
        show_keys = ["auto", "none"]
        fingering_labels = {"auto": "简化指法", "C": "C 调指法", "G": "G 调指法", "none": "原始调"}
        # Rename auto button to indicate which shape
        if auto_fingering in ("C", "G"):
            fingering_labels["auto"] = f"{auto_fingering} 调指法"
        score_divs = []
        buttons = []
        for fkey in show_keys:
            if fkey in fingering_versions:
                try:
                    fc = renderer(fingering_versions[fkey])
                except Exception:
                    fc = ""
                disp = "block" if fkey == "auto" else "none"
                score_divs.append(f'<div class="fingering-score" id="fingering-{fkey}" style="display:{disp}">{fc}</div>')
                active = " active" if fkey == "auto" else ""
                buttons.append(f'<button class="btn-fingering{active}" onclick="switchFingering(\'{fkey}\')">{fingering_labels[fkey]}</button>')
        fingering_scores = "\n".join(score_divs)
        fingering_buttons = "\n".join(buttons)

    # Get lyrics text
    lyrics_text = score_data.get("lyrics", "")

    # Get stem paths for audio
    audio_url = f"/api/stems/{job_id}/{audio_stem}"
    vocals_url = f"/api/stems/{job_id}/vocals" if vocals_available else ""
    stem_labels = {"piano": "钢琴", "guitar": "吉他", "bass": "贝斯", "drums": "架子鼓"}
    stem_label = stem_labels.get(notation_type, "伴奏")

    # Build compact audio bar — each track has its own play button + volume slider
    # Uses Web Audio API (GainNode) for volume boost beyond 100% (fixes quiet bass etc.)
    if vocals_available:
        audio_controls = f"""
<audio id="audio-inst" preload="auto" src="{audio_url}"></audio>
<audio id="audio-vocals" preload="auto" src="{vocals_url}"></audio>
<div class="audio-bar">
    <button id="btn-inst" class="btn-play" onclick="toggleTrack('inst')" title="播放/暂停{stem_label}">▶{stem_label}</button>
    <input type="range" id="vol-inst" class="vol-slider" min="0" max="200" value="100" oninput="setVol('inst')" title="{stem_label}音量">
    <button id="btn-vocals" class="btn-play btn-vocals" onclick="toggleTrack('vocals')" title="播放/暂停人声">▶人声</button>
    <input type="range" id="vol-vocals" class="vol-slider" min="0" max="200" value="100" oninput="setVol('vocals')" title="人声音量">
    <input type="range" id="seek-bar" class="seek-bar" min="0" max="100" value="0" oninput="seekAudio(this.value)" title="进度">
    <span class="time-display" id="time-display">0:00 / 0:00</span>
    <button class="btn-save" onclick="saveScore()">保存</button>
</div>
<script>
var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
var gainNodes = {{}};
function initGain(id) {{
    var a = document.getElementById('audio-' + id);
    var src = audioCtx.createMediaElementSource(a);
    var g = audioCtx.createGain();
    g.gain.value = 1.0;
    src.connect(g);
    g.connect(audioCtx.destination);
    gainNodes[id] = g;
    a.addEventListener('play', function() {{
        if (audioCtx.state === 'suspended') audioCtx.resume();
    }}, {{once: true}});
}}
initGain('inst');
initGain('vocals');
var tracks = {{
    inst: document.getElementById('audio-inst'),
    vocals: document.getElementById('audio-vocals')
}};
var playing = {{ inst: false, vocals: false }};
function toggleTrack(id) {{
    var a = tracks[id];
    var btn = document.getElementById('btn-' + id);
    if (playing[id]) {{ a.pause(); }}
    else {{ a.play(); }}
    playing[id] = !playing[id];
    btn.textContent = (playing[id] ? '⏸' : '▶') + btn.textContent.slice(1);
}}
function setVol(id) {{
    gainNodes[id].gain.value = document.getElementById('vol-' + id).value / 100;
}}
function seekAudio(val) {{
    var pct = parseFloat(val);
    for (var k in tracks) {{
        if (tracks[k].duration) tracks[k].currentTime = pct / 100 * tracks[k].duration;
    }}
}}
function formatTime(s) {{
    var m = Math.floor(s / 60);
    var sec = Math.floor(s % 60);
    return m + ':' + (sec < 10 ? '0' : '') + sec;
}}
tracks.inst.addEventListener('loadedmetadata', function() {{
    document.getElementById('time-display').textContent = '0:00 / ' + formatTime(tracks.inst.duration);
}});
tracks.inst.addEventListener('timeupdate', function() {{
    if (tracks.inst.duration) {{
        document.getElementById('time-display').textContent = formatTime(tracks.inst.currentTime) + ' / ' + formatTime(tracks.inst.duration);
        document.getElementById('seek-bar').value = (tracks.inst.currentTime / tracks.inst.duration) * 100;
    }}
}});
tracks.inst.onended = function() {{ playing.inst = false; document.getElementById('btn-inst').textContent = '▶{stem_label}'; }};
tracks.vocals.onended = function() {{ playing.vocals = false; document.getElementById('btn-vocals').textContent = '▶人声'; }};
</script>"""
    else:
        audio_controls = f"""
<audio id="audio-inst" preload="auto" src="{audio_url}"></audio>
<div class="audio-bar">
    <button id="btn-inst" class="btn-play" onclick="toggleTrack('inst')" title="播放/暂停{stem_label}">▶{stem_label}</button>
    <input type="range" id="vol-inst" class="vol-slider" min="0" max="200" value="100" oninput="setVol('inst')" title="{stem_label}音量">
    <input type="range" id="seek-bar" class="seek-bar" min="0" max="100" value="0" oninput="seekAudio(this.value)" title="进度">
    <span class="time-display" id="time-display">0:00 / 0:00</span>
    <button class="btn-save" onclick="saveScore()">保存</button>
</div>
<script>
var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
var gainNodes = {{}};
function initGain(id) {{
    var a = document.getElementById('audio-' + id);
    var src = audioCtx.createMediaElementSource(a);
    var g = audioCtx.createGain();
    g.gain.value = 1.0;
    src.connect(g);
    g.connect(audioCtx.destination);
    gainNodes[id] = g;
    a.addEventListener('play', function() {{
        if (audioCtx.state === 'suspended') audioCtx.resume();
    }}, {{once: true}});
}}
initGain('inst');
var tracks = {{ inst: document.getElementById('audio-inst') }};
var playing = {{ inst: false }};
function toggleTrack(id) {{
    var a = tracks[id];
    var btn = document.getElementById('btn-' + id);
    if (playing[id]) {{ a.pause(); }}
    else {{ a.play(); }}
    playing[id] = !playing[id];
    btn.textContent = (playing[id] ? '⏸' : '▶') + btn.textContent.slice(1);
}}
function setVol(id) {{
    gainNodes[id].gain.value = document.getElementById('vol-' + id).value / 100;
}}
function seekAudio(val) {{
    var pct = parseFloat(val);
    for (var k in tracks) {{
        if (tracks[k].duration) tracks[k].currentTime = pct / 100 * tracks[k].duration;
    }}
}}
function formatTime(s) {{
    var m = Math.floor(s / 60);
    var sec = Math.floor(s % 60);
    return m + ':' + (sec < 10 ? '0' : '') + sec;
}}
tracks.inst.addEventListener('loadedmetadata', function() {{
    document.getElementById('time-display').textContent = '0:00 / ' + formatTime(tracks.inst.duration);
}});
tracks.inst.addEventListener('timeupdate', function() {{
    if (tracks.inst.duration) {{
        document.getElementById('time-display').textContent = formatTime(tracks.inst.currentTime) + ' / ' + formatTime(tracks.inst.duration);
        document.getElementById('seek-bar').value = (tracks.inst.currentTime / tracks.inst.duration) * 100;
    }}
}});
tracks.inst.onended = function() {{ playing.inst = false; document.getElementById('btn-inst').textContent = '▶{stem_label}'; }};
</script>"""

    # Build fingering toggle bar and JS for guitar
    if fingering_scores:
        fingering_bar_html = f'<div class="fingering-bar">{fingering_buttons}</div>'
        score_content_block = fingering_scores
        fingering_js = """
function switchFingering(fkey) {
    document.querySelectorAll('.fingering-score').forEach(function(el) { el.style.display = 'none'; });
    document.querySelectorAll('.btn-fingering').forEach(function(el) { el.classList.remove('active'); });
    var scoreEl = document.getElementById('fingering-' + fkey);
    if (scoreEl) scoreEl.style.display = 'block';
    var btnEl = document.querySelector('.btn-fingering[onclick*=\"' + fkey + '\"]');
    if (btnEl) btnEl.classList.add('active');
}"""
    else:
        fingering_bar_html = ""
        score_content_block = score_content
        fingering_js = ""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>乐谱 - {notation_name}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    min-height: 100vh;
    padding-bottom: 120px;
}}
.header {{
    background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
    padding: 30px 40px;
    border-bottom: 3px solid #e94560;
}}
.header h1 {{
    font-size: 24px;
    color: #fff;
    margin-bottom: 16px;
}}
.analysis-info {{
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
}}
.info-card {{
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 12px 20px;
    min-width: 120px;
}}
.info-card .label {{
    font-size: 11px;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}}
.info-card .value {{
    font-size: 22px;
    font-weight: 700;
    color: #e94560;
}}
.chord-row {{
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 14px 24px;
    margin: 20px 40px 0;
    font-size: 14px;
    color: #ccc;
}}
.chord-row strong {{ color: #e94560; }}
.score-container {{
    margin: 20px auto;
    max-width: 1100px;
    padding: 0;
    background: transparent;
    overflow-x: auto;
}}
.score-container > div {{
    width: 100%;
    overflow-x: auto;
}}
.score-pages svg {{
    display: block;
    margin: 0 auto 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.15);
    border-radius: 2px;
}}
.nav-bar {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 24px;
    background: rgba(0,0,0,0.3);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 13px;
}}
.nav-bar a {{
    color: #aaa;
    text-decoration: none;
    transition: color 0.2s;
}}
.nav-bar a:hover {{ color: #e94560; }}
.nav-bar .sep {{ color: #444; }}
.nav-bar .current {{ color: #e94560; font-weight: 600; }}
@media print {{
    .nav-bar {{ display: none; }}
    .audio-bar {{ display: none; }}
    body {{ background: #fff; color: #000; }}
    .header {{ background: #f0f0f0; border-bottom-color: #333; }}
    .score-pages svg {{
        page-break-after: always;
        box-shadow: none;
        margin-bottom: 0;
    }}
    .score-pages svg:last-child {{
        page-break-after: avoid;
    }}
}}
.audio-bar {{
    position: fixed;
    bottom: 16px;
    left: 50%;
    transform: translateX(-50%);
    background: #0f3460;
    border: 1px solid #e94560;
    border-radius: 10px;
    padding: 8px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    z-index: 100;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
}}
.btn-play {{
    background: #e94560;
    color: #fff;
    border: none;
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    font-weight: 600;
    transition: background 0.2s;
    flex-shrink: 0;
    white-space: nowrap;
    font-family: inherit;
}}
.btn-play:hover {{ background: #c73a52; }}
.btn-vocals {{
    background: #6c5ce7;
}}
.btn-vocals:hover {{ background: #5a4bd1; }}
.vol-slider {{
    width: 48px;
    height: 3px;
    -webkit-appearance: none;
    appearance: none;
    background: rgba(255,255,255,0.15);
    border-radius: 2px;
    outline: none;
    cursor: pointer;
}}
.vol-slider::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #e94560;
    cursor: pointer;
}}
.seek-bar {{
    flex: 1;
    min-width: 80px;
    max-width: 300px;
    height: 4px;
    -webkit-appearance: none;
    appearance: none;
    background: rgba(255,255,255,0.1);
    border-radius: 2px;
    outline: none;
    cursor: pointer;
}}
.seek-bar::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: #e94560;
    cursor: pointer;
    box-shadow: 0 0 4px rgba(233,69,96,0.5);
}}
.time-display {{
    font-size: 11px;
    color: #999;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
    min-width: 88px;
    text-align: center;
}}
.btn-save {{
    background: rgba(255,255,255,0.08);
    color: #aaa;
    border: 1px solid rgba(255,255,255,0.1);
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.2s;
    white-space: nowrap;
    font-family: inherit;
}}
.btn-save:hover {{ background: #e94560; color: #fff; border-color: #e94560; }}
.fingering-bar {{
    display: flex;
    justify-content: center;
    gap: 10px;
    margin: 16px 40px 0;
    flex-wrap: wrap;
}}
.btn-fingering {{
    background: rgba(255,255,255,0.06);
    color: #aaa;
    border: 1px solid rgba(255,255,255,0.15);
    padding: 8px 18px;
    border-radius: 20px;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
    font-family: inherit;
}}
.btn-fingering:hover {{
    border-color: #e94560;
    color: #e94560;
}}
.btn-fingering.active {{
    background: #e94560;
    color: #fff;
    border-color: #e94560;
}}
@media print {{
    .fingering-bar {{ display: none; }}
}}
</style>
</head>
<body>

<div class="nav-bar">
    <a href="/">← 返回主页</a>
    <span class="sep">|</span>
    <span>AI Music</span>
    <span class="sep">|</span>
    <span class="current">{notation_name}</span>
    <span style="margin-left:auto;color:#666;font-size:12px;">{song_name}</span>
</div>

<div class="header">
    <h1>分析信息</h1>
    <p style="color:#888;font-size:13px;margin-top:4px;">歌曲: {song_name}</p>
    <div class="analysis-info">
        <div class="info-card">
            <div class="label">调性</div>
            <div class="value">{key_display}</div>
        </div>
        <div class="info-card">
            <div class="label">拍号</div>
            <div class="value">{song_info.get('timeSignature', [4, 4])[0]}/{song_info.get('timeSignature', [4, 4])[1]}</div>
        </div>
        <div class="info-card">
            <div class="label">BPM</div>
            <div class="value">{song_info.get('bpm', 120)}</div>
        </div>
        <div class="info-card">
            <div class="label">乐谱类型</div>
            <div class="value">{notation_name}</div>
        </div>
    </div>
</div>

<div class="chord-row">
    <strong>和弦进行:</strong> {chord_summary}
</div>

{fingering_bar_html}

<div class="score-container">
    {score_content_block}
</div>

{audio_controls}
<script>
function saveScore() {{
    var html = document.documentElement.outerHTML;
    var blob = new Blob(['<!DOCTYPE html>\\n' + html], {{ type: 'text/html;charset=utf-8' }});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'score_{job_id}_{notation_type}.html';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}}
{fingering_js}
</script>

</body>
</html>"""


@app.get("/")
async def serve_frontend():
    """Serve the main upload page."""
    project_root = os.path.dirname(BASE_DIR)
    index_path = os.path.join(project_root, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>Frontend not found</h1>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# Mount static files
frontend_static = os.path.join(os.path.dirname(BASE_DIR), "static")
if os.path.exists(frontend_static):
    app.mount("/static", StaticFiles(directory=frontend_static), name="static")

# Also mount output dir for direct file access
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
