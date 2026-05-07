import librosa
import numpy as np
import soundfile as sf

from services.common import CHROMATIC, note_to_chromatic_index
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
DEGREE_ROMAN_MAJOR = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
DEGREE_ROMAN_MINOR = ["i", "ii°", "III", "iv", "v", "VI", "VII"]

# Chord templates with harmonic weights (root, third, fifth emphasized)
CHORD_TEMPLATES = {
    "maj":   [1.0, 0.05, 0.1, 0.05, 0.8, 0.05, 0.1, 0.6, 0.1, 0.05, 0.05, 0.05],
    "min":   [1.0, 0.05, 0.1, 0.8, 0.05, 0.1, 0.05, 0.6, 0.1, 0.05, 0.1, 0.05],
    "dim":   [1.0, 0.05, 0.1, 0.8, 0.05, 0.1, 0.7, 0.05, 0.1, 0.05, 0.1, 0.05],
    "aug":   [1.0, 0.05, 0.1, 0.05, 0.9, 0.05, 0.1, 0.05, 0.8, 0.05, 0.1, 0.05],
    "7":     [1.0, 0.05, 0.1, 0.05, 0.8, 0.05, 0.1, 0.6, 0.1, 0.05, 0.7, 0.05],
    "maj7":  [1.0, 0.05, 0.1, 0.05, 0.8, 0.05, 0.1, 0.6, 0.1, 0.05, 0.1, 0.7],
    "min7":  [1.0, 0.05, 0.1, 0.8, 0.05, 0.1, 0.05, 0.6, 0.1, 0.05, 0.7, 0.05],
    "dim7":  [1.0, 0.05, 0.1, 0.8, 0.05, 0.1, 0.7, 0.05, 0.1, 0.8, 0.05, 0.05],
    "m7b5":  [1.0, 0.05, 0.1, 0.8, 0.05, 0.1, 0.7, 0.05, 0.1, 0.05, 0.7, 0.05],
    "sus4":  [1.0, 0.05, 0.1, 0.05, 0.05, 0.9, 0.1, 0.6, 0.1, 0.05, 0.05, 0.05],
    "sus2":  [1.0, 0.05, 0.9, 0.05, 0.05, 0.1, 0.05, 0.6, 0.1, 0.05, 0.05, 0.05],
    "add9":  [1.0, 0.05, 0.8, 0.05, 0.8, 0.05, 0.1, 0.6, 0.1, 0.05, 0.05, 0.05],
}


def load_audio(audio_path: str):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    return y, sr


def detect_bpm(y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])
    return round(float(tempo), 1)


# Preferred enharmonic spellings: practical keys favor flats over sharps
# Index: 1=C#→Db, 3=D#→Eb, 6=F#→Gb, 8=G#→Ab, 10=A#→Bb
_ENHARMONIC_FLAT = {1: "Db", 3: "Eb", 6: "F#", 8: "Ab", 10: "Bb"}


def detect_key(y: np.ndarray, sr: int) -> dict:
    """Krumhansl-Kessler key detection with NaN-safe chroma.
    Uses CQT chroma (better harmonic resolution); falls back to STFT if CQT fails."""
    # Primary: CQT chroma
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12, bins_per_octave=36)
        chroma = np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        chroma = None

    # Fallback: STFT chroma
    if chroma is None or chroma.size == 0:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        chroma = np.nan_to_num(chroma, nan=0.0)

    chroma_avg = chroma.mean(axis=1)

    # Sanity check: if chroma is effectively flat, use raw energy as fallback
    chroma_range = float(np.max(chroma_avg) - np.min(chroma_avg))
    if chroma_range < 1e-6:
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        chroma_stft = np.nan_to_num(chroma_stft, nan=0.0)
        chroma_avg = chroma_stft.mean(axis=1)

    best_major, best_minor = -99.0, -99.0
    major_key, minor_key = 0, 0

    for i in range(12):
        corr_maj = np.corrcoef(chroma_avg, np.roll(MAJOR_PROFILE, i))[0, 1]
        corr_min = np.corrcoef(chroma_avg, np.roll(MINOR_PROFILE, i))[0, 1]

        if not np.isfinite(corr_maj):
            corr_maj = 0.0
        if not np.isfinite(corr_min):
            corr_min = 0.0

        if corr_maj > best_major:
            best_major, major_key = corr_maj, i
        if corr_min > best_minor:
            best_minor, minor_key = corr_min, i

    # Low-confidence fallback: dominant pitch class from chroma energy
    if best_major <= 0.0 and best_minor <= 0.0:
        major_key = int(np.argmax(chroma_avg))
        minor_key = (major_key + 9) % 12
        best_major = 0.001

    if best_major >= best_minor:
        key_name = _ENHARMONIC_FLAT.get(major_key, CHROMATIC[major_key])
        return {"key": key_name, "mode": "major"}
    else:
        key_name = _ENHARMONIC_FLAT.get(minor_key, CHROMATIC[minor_key])
        return {"key": key_name, "mode": "minor"}


def _detect_bass_note(y: np.ndarray, sr: int, start_sample: int, end_sample: int) -> int:
    """Detect the most prominent bass note (MIDI number) in a segment."""
    segment = y[start_sample:end_sample]
    if len(segment) < 512:
        return -1

    # Low-pass filter to isolate bass frequencies
    S = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), 1.0 / sr)

    # Focus on bass range: 30-300 Hz
    bass_mask = (freqs >= 30) & (freqs <= 300)
    if not bass_mask.any():
        return -1

    bass_spec = S[bass_mask]
    bass_freqs = freqs[bass_mask]

    # Find peaks
    peak_indices = []
    for i in range(1, len(bass_spec) - 1):
        if bass_spec[i] > bass_spec[i - 1] and bass_spec[i] > bass_spec[i + 1]:
            peak_indices.append(i)

    if not peak_indices:
        return -1

    # Get the strongest peak
    best_idx = max(peak_indices, key=lambda i: bass_spec[i])
    best_freq = bass_freqs[best_idx]

    try:
        midi = int(round(librosa.hz_to_midi(best_freq)))
        return midi
    except (ValueError, ZeroDivisionError):
        return -1


def _get_build_chord_map(key_name: str, key_mode: str) -> dict:
    """Build a map of {chord_name: degree} for diatonic chords in the key."""
    key_root = note_to_chromatic_index(key_name)
    intervals = MAJOR_INTERVALS if key_mode == "major" else MINOR_INTERVALS
    scale = [CHROMATIC[(key_root + iv) % 12] for iv in intervals]

    degree_map = {}
    for deg_idx in range(7):
        root = scale[deg_idx]
        if key_mode == "major":
            quality = {0: "maj", 1: "min", 2: "min", 3: "maj", 4: "maj", 5: "min", 6: "dim"}[deg_idx]
            degree = DEGREE_ROMAN_MAJOR[deg_idx]
        else:
            quality = {0: "min", 1: "dim", 2: "maj", 3: "min", 4: "min", 5: "maj", 6: "maj"}[deg_idx]
            degree = DEGREE_ROMAN_MINOR[deg_idx]

        chord_name = root + ("" if quality == "maj" else "m" if quality == "min" else "dim")
        degree_map[chord_name] = degree

        # Also add 7th chord variants
        if key_mode == "major":
            seventh_quality = {0: "maj7", 1: "m7", 2: "m7", 3: "maj7", 4: "7", 5: "m7", 6: "m7b5"}[deg_idx]
        else:
            seventh_quality = {0: "m7", 1: "m7b5", 2: "maj7", 3: "m7", 4: "m7", 5: "maj7", 6: "7"}[deg_idx]
        seventh_name = root + seventh_quality if seventh_quality in ("7", "m7b5") else \
                       root + "m7" if seventh_quality == "m7" else \
                       root + "maj7"
        if seventh_quality == "7":
            seventh_name = root + "7"
        elif seventh_quality == "m7":
            seventh_name = root + "m7"
        elif seventh_quality == "maj7":
            seventh_name = root + "maj7"
        elif seventh_quality == "m7b5":
            seventh_name = root + "m7b5"
        degree_map[seventh_name] = degree + "7"

        # sus4 and sus2 variants
        degree_map[root + "sus4"] = degree + "sus4"
        degree_map[root + "sus2"] = degree + "sus2"

    return degree_map


def detect_chords(y: np.ndarray, sr: int, key_info: dict,
                  beats_per_measure: int = 4, bass_audio_path: str = None) -> list:
    """
    Detect chord progression from full mix audio, with optional bass stem guidance.
    Uses chromagram template matching with harmonic context weighting.
    """
    key_name = key_info["key"]
    key_mode = key_info["mode"]
    degree_map = _get_build_chord_map(key_name, key_mode)

    # Load bass audio for root detection if available
    bass_y = None
    if bass_audio_path:
        try:
            bass_y, bass_sr = load_audio(bass_audio_path)
        except Exception:
            bass_y = None

    # Harmonic CQT for better chromagram
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12, bins_per_octave=36)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if len(beats) == 0:
        return []

    beat_times = librosa.frames_to_time(beats, sr=sr)
    hop_length = 512
    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)

    chord_progression = []
    measure_idx = 0
    prev_root = None
    recent_roots = []  # Track last 4 chord roots for diversity

    for i in range(0, len(beat_frames) - 1, beats_per_measure):
        start_f = beat_frames[i]
        end_f = beat_frames[min(i + beats_per_measure, len(beat_frames) - 1)]
        if start_f >= chroma.shape[1]:
            break
        end_f = min(end_f, chroma.shape[1])
        if end_f <= start_f:
            continue

        segment = chroma[:, start_f:end_f].mean(axis=1)
        if segment.sum() < 1e-8:
            continue

        # Detect bass note from bass stem if available
        bass_midi = -1
        if bass_y is not None:
            start_sample = int(beat_times[i] * sr)
            end_sample = int(beat_times[min(i + beats_per_measure, len(beat_times) - 1)] * sr)
            bass_midi = _detect_bass_note(bass_y, sr, start_sample, end_sample)

        best_chord = "C"
        best_score = -99
        best_quality = "maj"

        for root_idx in range(12):
            for q_name, template in CHORD_TEMPLATES.items():
                tmpl = np.array(template, dtype=float)
                rolled = np.roll(tmpl, root_idx)
                # Cosine similarity
                dot = np.dot(segment, rolled)
                norm = np.linalg.norm(segment) * np.linalg.norm(rolled) + 1e-8
                score = dot / norm

                # Bonus for matching bass note
                if bass_midi >= 0:
                    bass_chroma = bass_midi % 12
                    if bass_chroma == root_idx:
                        score += 0.3  # Strong bonus for matching bass root
                    elif bass_chroma == (root_idx + 7) % 12:
                        score += 0.15  # Bonus for bass playing fifth

                # Bonus for chord in key
                test_chord = CHROMATIC[root_idx]
                if q_name == "min":
                    test_chord += "m"
                elif q_name == "dim":
                    test_chord += "dim"
                if test_chord in degree_map:
                    score += 0.15  # Diatonic bonus

                # Smoothing: prefer chords related to previous (no lock-in)
                if prev_root is not None:
                    distance = abs((root_idx - prev_root) % 12)
                    distance = min(distance, 12 - distance)  # Circular
                    if distance == 0:
                        score -= 0.08  # Penalize same chord to avoid lock-in
                    elif distance == 7 or distance == 5:
                        score += 0.08  # Fifth/fourth relationship
                    elif distance in [2, 4, 9]:
                        score += 0.03  # Diatonic step

                # Diversity: penalize roots that appeared in recent measures
                if root_idx in recent_roots:
                    score -= 0.06 * recent_roots.count(root_idx)

                if score > best_score:
                    best_score = score
                    best_chord = CHROMATIC[root_idx]
                    best_quality = q_name

        # Build chord name
        chord_name = best_chord
        if best_quality == "min":
            chord_name += "m"
        elif best_quality == "dim":
            chord_name += "dim"
        elif best_quality == "aug":
            chord_name += "aug"
        elif best_quality == "7":
            chord_name += "7"
        elif best_quality == "maj7":
            chord_name += "maj7"
        elif best_quality == "min7":
            chord_name += "m7"
        elif best_quality == "dim7":
            chord_name += "dim7"
        elif best_quality == "m7b5":
            chord_name += "m7b5"
        elif best_quality == "sus4":
            chord_name += "sus4"
        elif best_quality == "sus2":
            chord_name += "sus2"
        elif best_quality == "add9":
            chord_name += "add9"

        prev_root = CHROMATIC.index(best_chord)
        recent_roots.append(prev_root)
        if len(recent_roots) > 4:
            recent_roots.pop(0)

        degree = degree_map.get(chord_name)
        if degree is None:
            # Try with just root+m to match the map
            root_only = best_chord
            min_variant = best_chord + "m"
            if min_variant in degree_map:
                degree = degree_map[min_variant]
            else:
                # Find closest diatonic degree via modal mixture
                root_idx = CHROMATIC.index(best_chord)
                key_root = note_to_chromatic_index(key_name)
                intervals = MAJOR_INTERVALS if key_mode == "major" else MINOR_INTERVALS
                scale_pos = (root_idx - key_root) % 12
                if scale_pos in intervals:
                    deg_idx = intervals.index(scale_pos)
                    roman = DEGREE_ROMAN_MAJOR if key_mode == "major" else DEGREE_ROMAN_MINOR
                    degree = roman[deg_idx]
                else:
                    # Modal mixture: find nearest diatonic degree and add accidental
                    best_dist = 12
                    best_deg_idx = 0
                    best_accidental = "b"
                    for di, interval in enumerate(intervals):
                        dist_forward = (scale_pos - interval) % 12
                        dist_backward = (interval - scale_pos) % 12
                        dist = min(dist_forward, dist_backward)
                        accidental = "#" if dist_forward < dist_backward else "b"
                        # Prefer flat notation on ties (bIII, bVI, bVII are conventional)
                        if dist < best_dist or (dist == best_dist and accidental == "b" and best_accidental != "b"):
                            best_dist = dist
                            best_deg_idx = di
                            best_accidental = accidental
                    roman_base = DEGREE_ROMAN_MAJOR[best_deg_idx] if key_mode == "major" else DEGREE_ROMAN_MINOR[best_deg_idx]
                    # For modal mixture, keep just the scale degree (strip quality), uppercase
                    roman_base = roman_base.replace("°", "").replace("dim", "").upper()
                    degree = best_accidental + roman_base

        chord_progression.append({
            "measure": measure_idx + 1,
            "chord": chord_name,
            "degree": degree or "?",
        })
        measure_idx += 1

    return chord_progression


def detect_chords_with_bass(y: np.ndarray, sr: int, key_info: dict,
                            bass_audio_path: str = None, beats_per_measure: int = 4) -> list:
    """Detect chords with optional bass stem guidance."""
    return detect_chords(y, sr, key_info, beats_per_measure, bass_audio_path)


def detect_time_signature(y: np.ndarray, sr: int) -> list:
    """Estimate time signature using autocorrelation + onset accent analysis.

    4/4 is the strong default (~90% of pop music). Only switch to 3/4 or 6/8
    when multiple independent signals agree. This prevents false 3/4 detection
    on 4/4 songs with strong snare backbeats."""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    n_beats = len(beats)
    if n_beats < 8:
        return [4, 4]

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    beat_frames = beats[beats < len(onset_env)]
    if len(beat_frames) < 8:
        return [4, 4]

    strengths = onset_env[beat_frames].astype(float)

    # Method 1: Autocorrelation of beat strengths
    # In 3/4, strong beats repeat every 3. In 4/4, every 4.
    strengths_norm = strengths - np.mean(strengths)
    std_s = np.std(strengths_norm)
    if std_s > 1e-8:
        strengths_norm = strengths_norm / std_s

    max_lag = min(8, len(strengths_norm) // 3)
    autocorr = np.correlate(strengths_norm, strengths_norm, mode='full')
    mid = len(autocorr) // 2
    ac_scores = {}
    for lag in [2, 3, 4, 6]:
        if lag <= max_lag:
            raw = float(autocorr[mid + lag])
            # Normalize by (n_beats - lag) to remove bias toward smaller lags
            ac_scores[lag] = raw / max(1, n_beats - lag)

    # Method 2: Z-score of downbeat within each grouping
    def score_grouping(n):
        usable = len(strengths) // n * n
        if usable < n * 2:
            return -999.0
        groups = strengths[:usable].reshape(-1, n)
        group_means = groups.mean(axis=1)
        group_stds = groups.std(axis=1)
        group_stds[group_stds < 1e-6] = 1.0
        z_scores = (groups[:, 0] - group_means) / group_stds
        return float(np.mean(z_scores))

    s2 = score_grouping(2)
    s3 = score_grouping(3)
    s4 = score_grouping(4)
    s6 = score_grouping(6)

    # Default to 4/4 unless 3/4 evidence is overwhelming
    # Require: s3 > 0.5 AND ac[3] > ac[4] AND s3 > s4 + 0.25
    ac3 = ac_scores.get(3, 0)
    ac4 = ac_scores.get(4, 0)
    ac6 = ac_scores.get(6, 0)

    s3_evidence = (s3 > 0.50 and
                   ac3 > ac4 * 1.15 and
                   s3 > s4 + 0.25)

    s6_evidence = (s6 > 0.40 and
                   ac6 > ac4 * 1.2 and
                   ac6 > ac3 * 1.1 and
                   s6 > s4 + 0.20)

    if s6_evidence:
        return [6, 8]
    elif s3_evidence:
        return [3, 4]
    else:
        return [4, 4]


def analyze_full(audio_path: str, bass_audio_path: str = None, max_duration: float = 180.0) -> dict:
    y, sr = load_audio(audio_path)

    # Truncate to first N seconds for faster analysis
    max_samples = int(max_duration * sr)
    if len(y) > max_samples:
        y = y[:max_samples]

    bpm = detect_bpm(y, sr)
    key_info = detect_key(y, sr)
    time_sig = detect_time_signature(y, sr)

    # For chord detection with bass, also truncate bass if loaded internally
    chords = detect_chords(y, sr, key_info, bass_audio_path=bass_audio_path)

    return {
        "bpm": bpm,
        "key": key_info["key"],
        "key_mode": key_info["mode"],
        "time_signature": time_sig,
        "chords": chords,
    }


def detect_sections(vocals_audio_path: str, chords: list, bpm: float,
                     time_sig: list, total_measures: int) -> list:
    """
    Detect song structure: intro, verse, chorus, bridge, interlude, outro.

    Uses vocal energy to find instrumental vs vocal sections, and chord
    pattern similarity to distinguish verse vs chorus.

    Returns list of {start_measure, end_measure, label}.
    """
    import numpy as np

    if not chords or total_measures < 4:
        return []

    beat_dur = 60.0 / bpm
    beats_per_measure = time_sig[0]
    measure_dur = beat_dur * beats_per_measure

    # Load vocals and compute per-measure energy
    try:
        y, sr = load_audio(vocals_audio_path)
        # Truncate to match analyzed measures
        max_samples = int(total_measures * measure_dur * sr)
        if len(y) > max_samples:
            y = y[:max_samples]
    except Exception:
        return []

    # RMS energy per measure
    measure_energy = []
    for mi in range(total_measures):
        start_s = int(mi * measure_dur * sr)
        end_s = int((mi + 1) * measure_dur * sr)
        if start_s >= len(y):
            break
        end_s = min(end_s, len(y))
        segment = y[start_s:end_s]
        rms = float(np.sqrt(np.mean(segment ** 2) + 1e-10))
        measure_energy.append(rms)

    if not measure_energy:
        return []

    # Determine threshold for vocal presence
    max_energy = max(measure_energy) if measure_energy else 1
    # At least 12% of max energy to count as vocal; floor at 5e-4 absolute
    threshold = max(max_energy * 0.12, 5e-4)

    # Classify each measure: vocal or instrumental
    is_vocal = [e > threshold for e in measure_energy]

    # Build chord fingerprints for similarity comparison
    chord_degrees = [c.get("degree", "?") for c in chords]
    chord_names = [c.get("chord", "") for c in chords]

    # Find contiguous blocks
    sections = []
    i = 0
    while i < len(is_vocal) and i < total_measures:
        block_start = i
        block_vocal = is_vocal[i]

        # Extend block while vocal state is consistent (allow brief changes)
        j = i + 1
        while j < len(is_vocal) and j < total_measures:
            # Small gaps (<2 measures) of different state are absorbed
            if is_vocal[j] != block_vocal:
                if j + 1 < len(is_vocal) and is_vocal[j + 1] == block_vocal:
                    j += 1  # Skip brief deviation
                    continue
                else:
                    break
            j += 1

        block_end = j
        block_len = block_end - block_start

        if block_start == 0 and not block_vocal and block_len >= 1:
            # Cap intro at 8 measures to prevent excessively long intros
            intro_end = min(block_end, 8)
            sections.append({"start_measure": block_start, "end_measure": intro_end, "label": "前奏"})
            # If intro was capped, push remaining measures to the next block
            if intro_end < block_end:
                i = intro_end
                continue
        elif block_vocal:
            # Within vocal blocks, try to identify verse vs chorus
            # by chord pattern repetition
            if block_len >= 3:
                # Get the chord degree pattern for this block
                block_degrees = chord_degrees[block_start:block_end] if block_start < len(chord_degrees) else []

                # Check if a similar pattern appeared before
                is_chorus = False
                for prev_sec in sections:
                    if prev_sec["label"] in ("主歌", "副歌"):
                        prev_start = prev_sec["start_measure"]
                        prev_end = min(prev_sec["end_measure"], len(chord_degrees))
                        prev_pattern = chord_degrees[prev_start:prev_end]
                        # Compare first 4 chords
                        if (len(block_degrees) >= 4 and len(prev_pattern) >= 4
                                and block_degrees[:4] == prev_pattern[:4]):
                            is_chorus = True
                            break

                if is_chorus:
                    sections.append({"start_measure": block_start, "end_measure": block_end, "label": "副歌"})
                else:
                    # First vocal section is usually verse, later ones may be chorus
                    has_verse_before = any(s["label"] == "主歌" for s in sections)
                    has_chorus_before = any(s["label"] == "副歌" for s in sections)

                    if not has_verse_before and not has_chorus_before:
                        sections.append({"start_measure": block_start, "end_measure": block_end, "label": "主歌"})
                    elif has_chorus_before and not has_verse_before:
                        sections.append({"start_measure": block_start, "end_measure": block_end, "label": "主歌"})
                    else:
                        # Try to match with known structure
                        # Higher energy + more stable chords → chorus
                        block_energy = np.mean(measure_energy[block_start:block_end]) if block_start < len(measure_energy) else 0
                        prev_vocal_energy = []
                        for ps in sections:
                            if ps["label"] in ("主歌", "副歌"):
                                ps_e = np.mean(measure_energy[ps["start_measure"]:min(ps["end_measure"], len(measure_energy))])
                                prev_vocal_energy.append(ps_e)

                        avg_prev_energy = np.mean(prev_vocal_energy) if prev_vocal_energy else block_energy
                        if block_energy > avg_prev_energy * 1.15:
                            sections.append({"start_measure": block_start, "end_measure": block_end, "label": "副歌"})
                        else:
                            sections.append({"start_measure": block_start, "end_measure": block_end, "label": "主歌"})
            else:
                sections.append({"start_measure": block_start, "end_measure": block_end, "label": "主歌"})
        elif not block_vocal and block_len >= 1:
            # Instrumental block between vocal sections
            if block_start > 0 and block_end < total_measures - 1:
                sections.append({"start_measure": block_start, "end_measure": block_end, "label": "间奏"})
            elif block_end >= total_measures - 1:
                sections.append({"start_measure": block_start, "end_measure": block_end, "label": "尾奏"})

        i = block_end

    # Post-process: identify bridge (unique chord pattern after verse+chorus)
    vocal_sections = [s for s in sections if s["label"] in ("主歌", "副歌")]
    if len(vocal_sections) >= 3:
        seen_patterns = set()
        for vi, vs in enumerate(vocal_sections):
            vs_start = min(vs["start_measure"], len(chord_degrees))
            vs_end = min(vs["end_measure"], len(chord_degrees))
            pattern = tuple(chord_degrees[vs_start:vs_end][:4]) if vs_end > vs_start else ()
            if pattern and pattern in seen_patterns:
                continue  # Already seen this pattern
            seen_patterns.add(pattern)
            # Third unique vocal pattern after verse and chorus → bridge
            if vi >= 2 and len(seen_patterns) >= 3:
                vs["label"] = "桥段"

    return sections
