"""
Generate structured score data for each notation type:
- piano: grand staff (treble + bass)
- guitar: tab staff + chord diagrams
- drums: percussion staff
- jianpu: numbered notation

Each function returns a JSON-serializable dict for rendering by VexFlow or custom renderer.
"""

from typing import Optional

from services.common import (CHROMATIC, midi_to_vexnote,
                              dur_to_beats, note_to_chromatic_index,
                              note_name_normalized)

# Guitar standard tuning (MIDI numbers)
GUITAR_STRINGS = [64, 59, 55, 50, 45, 40]  # E4, B3, G3, D3, A2, E2
GUITAR_STRING_NAMES = ["E", "B", "G", "D", "A", "E"]

# Capo fingering shapes
_FINGERING_SHAPES = {"C": 0, "G": 7}  # MIDI index of shape root


def transpose_chord(chord_name: str, semitones: int) -> str:
    """Transpose a chord name by N semitones (positive = up, negative = down)."""
    if not chord_name or semitones == 0:
        return chord_name
    # Split off bass note (slash chord)
    bass = ""
    if "/" in chord_name:
        chord_name, bass = chord_name.split("/", 1)
    # Extract root (1-2 chars: C, C#, Db, D, ...)
    if len(chord_name) >= 2 and chord_name[1] in ("#", "b"):
        root = chord_name[:2]
        quality = chord_name[2:]
    else:
        root = chord_name[:1]
        quality = chord_name[1:]
    root_idx = note_to_chromatic_index(root)
    new_root = CHROMATIC[(root_idx + semitones) % 12]
    result = new_root + quality
    if bass:
        if len(bass) >= 2 and bass[1] in ("#", "b"):
            b_root, b_rest = bass[:2], bass[2:]
        else:
            b_root, b_rest = bass[:1], bass[1:]
        b_idx = note_to_chromatic_index(b_root)
        result += "/" + CHROMATIC[(b_idx + semitones) % 12] + b_rest
    return result


def _calc_capo_fret(key_name: str, fingering: str) -> int:
    """Calculate capo fret for a given key and fingering shape.
    Returns capo fret (0-11), or 0 if fingering not applicable."""
    if not key_name or fingering not in _FINGERING_SHAPES:
        return 0
    target = _FINGERING_SHAPES[fingering]
    orig = note_to_chromatic_index(key_name)
    return (orig - target + 12) % 12


def midi_to_jianpu(midi_val: float, key_root: int) -> tuple:
    """
    Convert MIDI to jianpu scale degree, accidental, and octave dot position.
    Returns (degree 1-7, accidental "", "#", "b", octave_shift -1/0/1/2).
    """
    scale_intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale
    midi_int = int(round(midi_val))

    rel = (midi_int - key_root) % 12
    octave_offset = (midi_int - key_root) // 12

    best_deg = 1
    best_iv = 0
    best_dist = 99
    for i, iv in enumerate(scale_intervals):
        dist = abs(rel - iv)
        if dist < best_dist:
            best_dist = dist
            best_deg = i + 1
            best_iv = iv

    # Determine accidental for chromatic notes
    accidental = ""
    if best_dist == 1:
        diff = rel - best_iv
        if diff == 1 or diff == -11:
            accidental = "#"
        elif diff == -1 or diff == 11:
            accidental = "b"
    elif best_dist >= 2:
        diff = rel - best_iv
        if diff > 0:
            accidental = "#"
        else:
            accidental = "b"

    return best_deg, accidental, octave_offset


def _add_jianpu_to_notes(notes: list, key_root: int, get_midi) -> None:
    """Add jianpu {degree, accidental, octaveShift} to each note that has pitch."""
    for note in notes:
        midi = get_midi(note)
        if midi is not None:
            degree, accidental, oct_shift = midi_to_jianpu(midi, key_root)
            note["jianpu"] = {"degree": degree, "accidental": accidental, "octaveShift": oct_shift}


def quantize_duration_dict(seconds: float, bpm: float) -> dict:
    """Convert duration to musical notation components."""
    beat_dur = 60.0 / bpm

    # Round to nearest 16th note
    sixteenth = beat_dur / 4
    num_sixteenths = max(1, round(seconds / sixteenth))

    # Build VexFlow duration
    if num_sixteenths <= 1:
        return {"duration": "16", "dots": 0, "beats": 0.25}
    elif num_sixteenths == 2:
        return {"duration": "8", "dots": 0, "beats": 0.5}
    elif num_sixteenths == 3:
        return {"duration": "8", "dots": 1, "beats": 0.75}
    elif num_sixteenths <= 4:
        return {"duration": "q", "dots": 0, "beats": 1.0}
    elif num_sixteenths <= 6:
        return {"duration": "q", "dots": 1, "beats": 1.5}
    elif num_sixteenths <= 8:
        return {"duration": "h", "dots": 0, "beats": 2.0}
    elif num_sixteenths <= 12:
        return {"duration": "h", "dots": 1, "beats": 3.0}
    else:
        return {"duration": "w", "dots": 0, "beats": 4.0}


def _extract_chord_root(chord_name: str) -> str:
    """Extract the root note name from a chord symbol like 'C', 'Am', 'F#m7', 'Bbmaj7'.
    Normalizes flat names to sharp equivalents for CHROMATIC compatibility."""
    root = "C"
    for i, ch in enumerate(chord_name):
        if ch in "ABCDEFG":
            if i + 1 < len(chord_name) and chord_name[i + 1] in "#b":
                root = chord_name[i:i + 2]
            else:
                root = ch
            break
    return note_name_normalized(root)


def simplify_chord_name(chord_name: str) -> str:
    """Strip color tones (sus, add) for guitar display — keep core triad + 7th quality."""
    import re
    name = re.sub(r"sus[24]", "", chord_name)
    name = re.sub(r"add\d+", "", name)
    return name


def _get_bass_pattern(chord_name: str, beats_per_measure: int,
                      section_label: str, measure_idx: int, denom: int = 4) -> list:
    """Return list of (midi, duration, beat_offset) for musically valid bass patterns."""
    tones = _bass_midi_notes(chord_name)
    root, third, fifth, octv = tones[0], tones[1], tones[2], tones[3]
    n = beats_per_measure

    patterns_4 = {
        "前奏": [(root, "h", 0.0), (root, "h", 2.0)],
        "主歌": [(root, "q", 0.0), (fifth, "8", 1.0), (fifth, "8", 1.5),
                 (root, "q", 2.0), (fifth, "q", 3.0)],
        "副歌": [(root, "q", 0.0), (fifth, "q", 1.0),
                 (root, "q", 2.0), (fifth, "q", 3.0)],
        "桥段": [(root, "8", 0.0), (fifth, "8", 0.5), (root, "8", 1.0), (fifth, "8", 1.5),
                 (third, "8", 2.0), (fifth, "8", 2.5), (root, "8", 3.0), (fifth, "8", 3.5)],
        "间奏": [(root, "h", 0.0), (fifth, "h", 2.0)],
        "尾奏": [(root, "w", 0.0)],
    }
    patterns_3 = {
        "前奏": [(root, "h.", 0.0)],
        "主歌": [(root, "q", 0.0), (fifth, "q", 1.0), (root, "q", 2.0)],
        "副歌": [(root, "q", 0.0), (fifth, "q", 1.0), (fifth, "q", 2.0)],
        "桥段": [(root, "8", 0.0), (fifth, "8", 0.5), (root, "8", 1.0),
                 (fifth, "8", 1.5), (fifth, "8", 2.0), (root, "8", 2.5)],
        "间奏": [(root, "h.", 0.0)],
        "尾奏": [(root, "h.", 0.0)],
    }
    patterns_68 = {
        "前奏": [(root, "h.", 0.0)],
        "主歌": [(root, "8", 0.0), (fifth, "8", 0.5), (root, "8", 1.0),
                 (fifth, "8", 1.5), (root, "8", 2.0), (fifth, "8", 2.5)],
        "副歌": [(root, "q", 0.0), (fifth, "q", 1.5)],
        "桥段": [(root, "8", 0.0), (fifth, "8", 0.5), (third, "8", 1.0),
                 (fifth, "8", 1.5), (root, "8", 2.0), (fifth, "8", 2.5)],
        "间奏": [(root, "h.", 0.0)],
        "尾奏": [(root, "h.", 0.0)],
    }

    if denom == 8:
        patterns = patterns_68
        default = [(root, "8", 0.0), (fifth, "8", 0.5), (root, "8", 1.0),
                   (fifth, "8", 1.5), (root, "8", 2.0), (fifth, "8", 2.5)]
    elif n == 3:
        patterns = patterns_3
        default = [(root, "q", 0.0), (fifth, "q", 1.0), (root, "q", 2.0)]
    elif n == 2:
        patterns = {
            "前奏": [(root, "h", 0.0)],
            "主歌": [(root, "q", 0.0), (fifth, "q", 1.0)],
            "副歌": [(root, "q", 0.0), (fifth, "q", 1.0)],
            "桥段": [(root, "8", 0.0), (fifth, "8", 0.5), (root, "8", 1.0), (fifth, "8", 1.5)],
            "间奏": [(root, "h", 0.0)],
            "尾奏": [(root, "h", 0.0)],
        }
        default = [(root, "q", 0.0), (fifth, "q", 1.0)]
    else:
        patterns = patterns_4
        default = [(root, "q", 0.0), (fifth, "q", 1.0), (root, "q", 2.0), (fifth, "q", 3.0)]
    return patterns.get(section_label, default)


def _get_left_hand_voicing(chord_name: str, beats_per_measure: int,
                            section_label: str, measure_idx: int, denom: int = 4) -> list:
    """Return broken-chord arpeggios for piano left hand (bass clef).

    Each note is a single key — arpeggiated, not block chords.
    Section-aware patterns.
    """
    tones = _bass_midi_notes(chord_name)
    root, third, fifth, octv = tones[0], tones[1], tones[2], tones[3]
    n = beats_per_measure

    vn = lambda m: [midi_to_vexnote(m)]

    # x/8 time signatures: 6/8, 9/8, 12/8 — eighth-note based patterns
    if denom == 8:
        if section_label in ("前奏", "尾奏"):
            return [{"keys": vn(root), "duration": "h", "dots": 1}]  # dotted half
        elif section_label == "副歌":
            return [
                {"keys": vn(root), "duration": "8", "dots": 0},
                {"keys": vn(third), "duration": "8", "dots": 0},
                {"keys": vn(fifth), "duration": "8", "dots": 0},
                {"keys": vn(third), "duration": "8", "dots": 0},
                {"keys": vn(fifth), "duration": "8", "dots": 0},
                {"keys": vn(octv), "duration": "8", "dots": 0},
            ]
        elif section_label == "桥段":
            return [
                {"keys": vn(root), "duration": "8", "dots": 0},
                {"keys": vn(fifth), "duration": "8", "dots": 0},
                {"keys": vn(third), "duration": "8", "dots": 0},
                {"keys": vn(fifth), "duration": "8", "dots": 0},
                {"keys": vn(root), "duration": "8", "dots": 0},
                {"keys": vn(fifth), "duration": "8", "dots": 0},
            ]
        elif section_label == "间奏":
            return [{"keys": vn(root), "duration": "h", "dots": 1}]
        else:
            return [
                {"keys": vn(root), "duration": "8", "dots": 0},
                {"keys": vn(fifth), "duration": "8", "dots": 0},
                {"keys": vn(third), "duration": "8", "dots": 0},
                {"keys": vn(fifth), "duration": "8", "dots": 0},
                {"keys": vn(third), "duration": "8", "dots": 0},
                {"keys": vn(fifth), "duration": "8", "dots": 0},
            ]
    elif n == 3:
        # Waltz: root → 3rd-5th together → 3rd-5th together
        return [
            {"keys": vn(root), "duration": "q", "dots": 0},
            {"keys": vn(third), "duration": "q", "dots": 0},
            {"keys": vn(fifth), "duration": "q", "dots": 0},
        ]
    elif n == 2:
        return [
            {"keys": vn(root), "duration": "q", "dots": 0},
            {"keys": vn(fifth), "duration": "q", "dots": 0},
        ]
    else:
        # 4/4: Alberti-style broken chord
        return [
            {"keys": vn(root), "duration": "8", "dots": 0},
            {"keys": vn(fifth), "duration": "8", "dots": 0},
            {"keys": vn(third), "duration": "8", "dots": 0},
            {"keys": vn(fifth), "duration": "8", "dots": 0},
            {"keys": vn(root), "duration": "8", "dots": 0},
            {"keys": vn(fifth), "duration": "8", "dots": 0},
            {"keys": vn(third), "duration": "8", "dots": 0},
            {"keys": vn(fifth), "duration": "8", "dots": 0},
        ]


def _bass_midi_notes(chord_name: str) -> list:
    """Get MIDI notes for bass (root, third, fifth, octave) in lower octaves."""
    root_name = _extract_chord_root(chord_name)

    if root_name not in CHROMATIC:
        root_name = "C"

    root_idx = CHROMATIC.index(root_name)
    is_minor = "m" in chord_name and "maj" not in chord_name
    third = root_idx + (3 if is_minor else 4)
    fifth = root_idx + 7

    return [
        root_idx + 36,  # Octave 2
        third + 36,
        fifth + 36,
        root_idx + 48,  # Octave 3
    ]


def _get_all_chord_tones(chord_name: str) -> list:
    """Get all possible chord tones across octaves 3-5 for right-hand voicing.
    Returns sorted unique MIDI notes."""
    root_name = _extract_chord_root(chord_name)
    if root_name not in CHROMATIC:
        root_name = "C"
    root_idx = CHROMATIC.index(root_name)

    is_minor = "m" in chord_name and "maj" not in chord_name
    is_dim = "dim" in chord_name
    is_seventh = "7" in chord_name and "maj7" not in chord_name

    third_iv = 3 if (is_minor or is_dim) else 4
    fifth_iv = 7

    tones = set()
    for octave in [3, 4]:  # C3 to C5 range for RH voicing
        base = (octave + 1) * 12
        tones.add(root_idx + base)
        tones.add(root_idx + third_iv + base)
        tones.add(root_idx + fifth_iv + base)
        if is_seventh:
            tones.add(root_idx + 10 + base)

    return sorted(tones)


def _chord_tone_below_melody(chord_name: str, melody_midi: int) -> Optional[int]:
    """Pick one chord tone (3rd, 5th, or root) below the melody for a 2-note texture.
    Returns the MIDI note, or None if no suitable tone exists."""
    root_name = _extract_chord_root(chord_name)
    root_idx = CHROMATIC.index(root_name) if root_name in CHROMATIC else 0
    is_minor = "m" in chord_name and "maj" not in chord_name
    is_dim = "dim" in chord_name
    third_iv = 3 if (is_minor or is_dim) else 4

    melody_class = melody_midi % 12
    root_class = root_idx % 12
    third_class = (root_idx + third_iv) % 12
    fifth_class = (root_idx + 7) % 12

    # Candidate intervals: prefer closer voicings (smaller interval below melody)
    candidates = []
    for pc in [third_class, fifth_class, root_class]:
        if pc == melody_class:
            continue  # Skip unison
        diff = (melody_class - pc) % 12
        if diff == 0:
            diff = 12  # One octave below if same class
        midi_below = melody_midi - diff
        if 48 <= midi_below < melody_midi:
            candidates.append((diff, midi_below))

    if not candidates:
        midi_below = melody_midi - 12
        if midi_below >= 48:
            candidates.append((12, midi_below))

    # Sort by interval distance — prefer closer voicings (3rds/4ths over 6ths/7ths)
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1] if candidates else None


def _right_hand_fill(chord_name: str, beats_per_measure: int, section_label: str, key_root: int, denom: int = 4) -> list:
    """Generate chord-tone fill for measures without vocal melody.
    Mostly single notes with occasional double/triple notes for emphasis."""
    tones = _bass_midi_notes(chord_name)
    # Shift to treble clef range (C4-C6): +24 puts root at C5
    r, th, f, o = tones[0] + 24, tones[1] + 24, tones[2] + 24, tones[3] + 24
    vn = midi_to_vexnote
    n = beats_per_measure

    def _single(midi_val, dur="q", dots=0):
        degree, accidental, oct_shift = midi_to_jianpu(midi_val, key_root)
        return {"keys": [vn(midi_val)], "duration": dur, "dots": dots, "lyric": "",
                "jianpu": {"degree": degree, "accidental": accidental, "octaveShift": oct_shift}}

    def _pair(midi_a, midi_b, dur="q", dots=0):
        keys = sorted([vn(midi_a), vn(midi_b)])
        top = max(midi_a, midi_b)
        degree, accidental, oct_shift = midi_to_jianpu(top, key_root)
        return {"keys": keys, "duration": dur, "dots": dots, "lyric": "",
                "jianpu": {"degree": degree, "accidental": accidental, "octaveShift": oct_shift}}

    def _triad(midi_a, midi_b, midi_c, dur="q", dots=0):
        keys = sorted([vn(midi_a), vn(midi_b), vn(midi_c)])
        top = max(midi_a, midi_b, midi_c)
        degree, accidental, oct_shift = midi_to_jianpu(top, key_root)
        return {"keys": keys, "duration": dur, "dots": dots, "lyric": "",
                "jianpu": {"degree": degree, "accidental": accidental, "octaveShift": oct_shift}}

    if denom == 8:
        # 6/8: 6 eighth notes — mostly single, pair on strong beats
        if section_label == "副歌":
            return [
                _pair(r, th, "8"), _single(f, "8"), _single(th, "8"),
                _pair(r, th, "8"), _single(f, "8"), _single(th, "8"),
            ]
        elif section_label in ("前奏", "尾奏"):
            return [
                _single(r, "8"), _single(th, "8"), _single(f, "8"),
                _single(r, "8"), _single(th, "8"), _single(f, "8"),
            ]
        else:
            return [
                _pair(r, th, "8"), _single(f, "8"), _single(th, "8"),
                _single(r, "8"), _single(f, "8"), _single(th, "8"),
            ]
    elif n == 3:
        # 3/4: 3 quarter notes — single with 1 pair on beat 1
        if section_label == "副歌":
            return [_pair(r, th, "q"), _single(f, "q"), _single(th, "q")]
        elif section_label in ("前奏", "尾奏", "间奏"):
            return [_single(r, "q"), _single(f, "q"), _single(th, "q")]
        else:
            return [_pair(r, th, "q"), _single(f, "q"), _single(th, "q")]
    elif n == 2:
        # 2/4: 2 quarter notes
        if section_label == "副歌":
            return [_pair(r, th, "q"), _pair(f, r + 12, "q")]
        else:
            return [_single(r, "q"), _single(f, "q")]
    else:
        # 4/4: 4 quarter notes — mostly single, pairs on strong beats
        if section_label in ("前奏", "尾奏"):
            return [
                _single(r, "q"), _single(f, "q"),
                _single(th, "q"), _single(r, "q"),
            ]
        elif section_label == "间奏":
            return [
                _single(r, "q"), _single(f, "q"),
                _pair(th, f, "q"), _single(r, "q"),
            ]
        elif section_label == "副歌":
            return [
                _pair(r, th, "q"), _single(f, "q"),
                _pair(r, th, "q"), _single(f, "q"),
            ]
        else:
            return [
                _pair(r, th, "q"), _single(f, "q"),
                _single(th, "q"), _single(r, "q"),
            ]


def generate_piano_score(melody_notes: list, chords: list, bpm: float, key_name: str,
                         key_mode: str, time_sig: list, sections: list = None) -> dict:
    """Generate piano solo grand staff.

    Right hand (treble): all melody notes (or chord-tone fill when no melody).
    Left hand (bass): broken chord arpeggios with section-aware rhythms.
    Every measure has notes — plays from intro to end.
    """
    if sections is None:
        sections = []

    sec_map = {}
    for sec in sections:
        for m in range(sec["start_measure"], sec["end_measure"]):
            sec_map[m] = sec["label"]

    key_root = note_to_chromatic_index(key_name)
    beats_per_measure = time_sig[0]
    denom = time_sig[1]
    beat_dur = 60.0 / bpm
    # For x/8, BPM is in dotted quarters; measure has (numerator/3) dotted quarters
    if denom == 8:
        measure_dur = beat_dur * (beats_per_measure // 3)
    else:
        measure_dur = beat_dur * beats_per_measure

    measures = []
    note_idx = 0

    for measure_idx in range(len(chords)):
        m_start = measure_idx * measure_dur
        m_end = m_start + measure_dur

        meas_melody = []
        while note_idx < len(melody_notes) and melody_notes[note_idx]["time"] < m_end:
            note = melody_notes[note_idx]
            if note["time"] >= m_start:
                meas_melody.append(note)
            note_idx += 1

        chord_info = chords[measure_idx]
        chord_name = chord_info.get("chord", "C")
        sec_label = sec_map.get(measure_idx, "")

        # === Treble clef: always filled, beat-by-beat ===
        # Strategy: generate a full measure of fill notes, then overlay melody
        # on top where it exists. This guarantees every beat is filled.
        treble_notes = _right_hand_fill(chord_name, beats_per_measure, sec_label, key_root, denom)

        if meas_melody:
            # Map melody notes by approximate beat position, then overlay
            # onto the fill backdrop: replace the nearest fill note's pitch
            # with melody pitch, keeping the fill's duration structure
            beat_dur = 60.0 / bpm
            m_start = measure_idx * measure_dur
            fill_beats = [0.0]
            for fn in treble_notes:
                fb = dur_to_beats(fn["duration"])
                fill_beats.append(fill_beats[-1] + fb)

            claimed_slots = set()  # Track which fill slots already have a melody note
            original_fill_count = len(treble_notes)  # freeze: only match original fill slots
            for note in meas_melody:
                beat_pos = (note["time"] - m_start) / beat_dur
                # Find the fill note whose beat position is closest
                best_i = -1
                best_d = float("inf")
                for fi in range(original_fill_count):
                    if fi in claimed_slots:
                        continue
                    fb = fill_beats[fi]
                    d = abs(fb - beat_pos)
                    if d < best_d and d < 0.75:  # Within 3/4 beat
                        best_d = d
                        best_i = fi
                if best_i >= 0:
                    claimed_slots.add(best_i)
                    midi_val = int(round(note["midi"]))
                    chord_tone = _chord_tone_below_melody(chord_name, midi_val)
                    keys = [midi_to_vexnote(midi_val)]
                    if chord_tone:
                        keys.insert(0, midi_to_vexnote(chord_tone))
                    jianpu = note.get("jianpu", {})
                    if not jianpu:
                        degree, accidental, oct_shift = midi_to_jianpu(midi_val, key_root)
                        jianpu = {"degree": degree, "accidental": accidental, "octaveShift": oct_shift}
                    treble_notes[best_i]["keys"] = keys
                    treble_notes[best_i]["jianpu"] = jianpu
                    treble_notes[best_i]["lyric"] = note.get("lyric", "")
                else:
                    # Melody note falls between fill beats: add as extra note
                    dur_info = quantize_duration_dict(note["duration"], bpm)
                    midi_val = int(round(note["midi"]))
                    chord_tone = _chord_tone_below_melody(chord_name, midi_val)
                    keys = [midi_to_vexnote(midi_val)]
                    if chord_tone:
                        keys.insert(0, midi_to_vexnote(chord_tone))
                    jianpu = note.get("jianpu", {})
                    if not jianpu:
                        degree, accidental, oct_shift = midi_to_jianpu(midi_val, key_root)
                        jianpu = {"degree": degree, "accidental": accidental, "octaveShift": oct_shift}
                    treble_notes.append({
                        "keys": keys,
                        "duration": dur_info["duration"],
                        "dots": dur_info["dots"],
                        "lyric": note.get("lyric", ""),
                        "jianpu": jianpu,
                    })

        # Failsafe
        if not treble_notes:
            treble_notes = _right_hand_fill(chord_name, beats_per_measure, sec_label, key_root, denom)

        # === Bass clef: chord voicings ===
        bass_notes = _get_left_hand_voicing(chord_name, beats_per_measure,
                                             sec_label, measure_idx, denom)
        measures.append({
            "index": measure_idx,
            "trebleNotes": treble_notes,
            "bassNotes": bass_notes,
            "chord": chord_info,
        })

    return {
        "notationType": "piano",
        "songInfo": {
            "key": key_name,
            "keyMode": key_mode,
            "bpm": bpm,
            "timeSignature": time_sig,
        },
        "chordProgression": chords,
        "measures": measures,
    }



def _get_chord_fingering(chord_name: str) -> list:
    """
    Return complete 6-string chord fingering positions.
    Each entry: {str: 6-1, fret: -1(muted), 0(open), or >0(fretted)}.
    Strings ordered 6→1 (low E → high E) matching standard chord diagrams
    where low E is on the left.
    """
    root_name = _extract_chord_root(chord_name)

    # Normalize flat root names to sharp for dict lookup (e.g. "Abm7" → "G#m7")
    for i, ch in enumerate(chord_name):
        if ch in "ABCDEFG":
            if i + 1 < len(chord_name) and chord_name[i + 1] in "#b":
                orig_root = chord_name[i:i + 2]
            else:
                orig_root = ch
            if orig_root != root_name:
                chord_name = root_name + chord_name[i + len(orig_root):]
            break

    # Format: 6 strings from low E (6) to high E (1), -1=muted, 0=open
    s = lambda s6, s5, s4, s3, s2, s1: [
        {"str": 6, "fret": s6}, {"str": 5, "fret": s5}, {"str": 4, "fret": s4},
        {"str": 3, "fret": s3}, {"str": 2, "fret": s2}, {"str": 1, "fret": s1},
    ]

    basic_chords = {
        # ===== 开放和弦 (Open Chords) =====
        #          6  5  4  3  2  1
        "C":   s( -1, 3, 2, 0, 1, 0),   # x32010
        "D":   s( -1,-1, 0, 2, 3, 2),   # xx0232
        "Dm":  s( -1,-1, 0, 2, 3, 1),   # xx0231
        "E":   s(  0, 2, 2, 1, 0, 0),   # 022100
        "Em":  s(  0, 2, 2, 0, 0, 0),   # 022000
        "F":   s(  1, 3, 3, 2, 1, 1),   # 133211 (E-shape barre at 1)
        "G":   s(  3, 2, 0, 0, 0, 3),   # 320003
        "A":   s( -1, 0, 2, 2, 2, 0),   # x02220
        "Am":  s( -1, 0, 2, 2, 1, 0),   # x02210

        # ===== E指型横按 大调 (root on 6th, pattern: N,N+2,N+2,N+1,N,N) =====
        "F#":  s(  2, 4, 4, 3, 2, 2),
        "G#":  s(  4, 6, 6, 5, 4, 4),

        # ===== A指型横按 大调 - lower barre than E-shape for these roots =====
        "Bb":  s( -1, 1, 3, 3, 3, 1),  # A-shape at 1 (was E-shape at 6)
        "B":   s( -1, 2, 4, 4, 4, 2),  # A-shape at 2 (was E-shape at 7)

        # ===== E指型横按 小调 (root on 6th, pattern: N,N+2,N+2,N,N,N) =====
        "Fm":  s(  1, 3, 3, 1, 1, 1),
        "F#m": s(  2, 4, 4, 2, 2, 2),
        "Gm":  s(  3, 5, 5, 3, 3, 3),
        "G#m": s(  4, 6, 6, 4, 4, 4),

        # ===== Am指型横按 小调 - lower barre than E-shape for these roots =====
        "Cm":  s( -1, 3, 5, 5, 4, 3),  # Am-shape at 3 (was E-shape at 8)
        "Bbm": s( -1, 1, 3, 3, 2, 1),  # Am-shape at 1 (was E-shape at 6)
        "Bm":  s( -1, 2, 4, 4, 3, 2),  # Am-shape at 2 (was E-shape at 7)

        # ===== A指型横按 大调 (root on 5th, pattern: x,N,N+2,N+2,N+2,N) =====
        "C#":  s( -1, 4, 6, 6, 6, 4),
        "D#":  s( -1, 6, 8, 8, 8, 6),
        "A#":  s( -1, 1, 3, 3, 3, 1),  # same as Bb

        # ===== A指型横按 小调 (root on 5th, pattern: x,N,N+2,N+2,N+1,N) =====
        "C#m": s( -1, 4, 6, 6, 5, 4),
        "D#m": s( -1, 6, 8, 8, 7, 6),
        "A#m": s( -1, 1, 3, 3, 2, 1),  # same as Bbm

        # ===== Additional open/minor chords =====
        "A7":  s( -1, 0, 2, 0, 2, 0),   # x02020
        "D7":  s( -1,-1, 0, 2, 1, 2),   # xx0212
        "E7":  s(  0, 2, 0, 1, 0, 0),   # 020100
        "G7":  s(  3, 2, 0, 0, 0, 1),   # 320001
        "C7":  s( -1, 3, 2, 3, 1, 0),   # x32310
        "B7":  s( -1, 2, 1, 2, 0, 2),   # x21202

        # ===== sus4 / sus2 =====
        "Dsus4": s(-1,-1, 0, 2, 3, 3),  # xx0233
        "Dsus2": s(-1,-1, 0, 2, 3, 0),  # xx0230
        "Asus4": s(-1, 0, 2, 2, 3, 0),  # x02230
        "Asus2": s(-1, 0, 2, 2, 0, 0),  # x02200
        "Esus4": s( 0, 2, 2, 2, 0, 0),  # 022200

        # ===== major 7th open chords =====
        "Cmaj7": s(-1, 3, 2, 0, 0, 0),  # x32000
        "Fmaj7": s(-1,-1, 3, 2, 1, 0),  # xx3210
        "Gmaj7": s( 3, 2, 0, 0, 0, 2),  # 320002
        "Amaj7": s(-1, 0, 2, 1, 2, 0),  # x02120
        "Dmaj7": s(-1,-1, 0, 2, 2, 2),  # xx0222
        "Emaj7": s( 0, 2, 1, 1, 0, 0),  # 021100

        # ===== minor 7th chords =====
        "Dm7":  s(-1,-1, 0, 2, 1, 1),   # xx0211
        "Em7":  s( 0, 2, 0, 0, 0, 0),   # 020000
        "Am7":  s(-1, 0, 2, 0, 1, 0),   # x02010
        "Bm7":  s(-1, 2, 0, 2, 0, 2),   # x20202

        # ===== barre m7 chords =====
        "Cm7":  s(-1, 3, 5, 3, 4, 3),   # Am7-shape at 3: x35343
        "Fm7":  s( 1, 3, 1, 1, 1, 1),   # Em7-shape at 1: 131111
        "Gm7":  s( 3, 5, 3, 3, 3, 3),   # Em7-shape at 3: 353333

        # ===== barre maj7 chords =====
        "Bmaj7": s(-1, 2, 4, 3, 4, 2),  # Amaj7-shape at 2: x24342

        # ===== Diminished =====
        "Bdim": s(-1, 2, 0, 1, 0, 1),   # x20101
        "Ddim": s(-1,-1, 0, 1, 0, 1),   # xx0101
    }

    # Try full chord name first; only fall back to root name for plain major chords
    result = basic_chords.get(chord_name)
    if not result and chord_name == root_name:
        result = basic_chords.get(root_name)
    if result:
        return result

    # Algorithmic fallback: try both E-shape (root on string 6) and A-shape (root on string 5)
    # barre patterns and pick the lower valid barre fret.
    try:
        root_chromatic = CHROMATIC.index(root_name)
    except ValueError:
        root_chromatic = 0

    is_minor = "m" in chord_name and "maj" not in chord_name
    is_seventh = "7" in chord_name
    is_maj7 = "maj7" in chord_name
    is_dim = "dim" in chord_name

    # Compute E-shape barre: root on string 6 (E = MIDI 40, 40%12 = 4)
    barre_e = (root_chromatic - 4) % 12  # fret on string 6 for this root
    barre_e = barre_e or 12  # open string → 12th fret (open handled by basic_chords)

    # Compute A-shape barre: root on string 5 (A = MIDI 45, 45%12 = 9)
    barre_a = (root_chromatic - 9) % 12  # fret on string 5 for this root
    barre_a = barre_a or 12

    # Pick the lower barre in practical range (1-12)
    candidates = []
    if 1 <= barre_e <= 12:
        candidates.append(("E", barre_e))
    if 1 <= barre_a <= 12:
        candidates.append(("A", barre_a))
    if not candidates:
        barre = max(1, min(12, barre_e))
        shape = "E"
    else:
        shape, barre = min(candidates, key=lambda x: x[1])

    if is_dim:
        if shape == "A":
            return [
                {"str": 6, "fret": -1}, {"str": 5, "fret": barre},      # root
                {"str": 4, "fret": barre + 1}, {"str": 3, "fret": barre + 2},  # ♭5, ♭3
                {"str": 2, "fret": barre + 1}, {"str": 1, "fret": barre},      # root, 5th
            ]
        else:
            return [
                {"str": 6, "fret": barre}, {"str": 5, "fret": barre + 1},  # root, ♭5
                {"str": 4, "fret": barre + 2}, {"str": 3, "fret": barre},  # ♭3, 5th
                {"str": 2, "fret": barre}, {"str": 1, "fret": barre},      # root, ♭3
            ]
    elif is_maj7:
        if shape == "A":
            return [
                {"str": 6, "fret": -1}, {"str": 5, "fret": barre},
                {"str": 4, "fret": barre + 2}, {"str": 3, "fret": barre + 1},
                {"str": 2, "fret": barre + 2}, {"str": 1, "fret": barre},
            ]
        else:
            return [
                {"str": 6, "fret": barre}, {"str": 5, "fret": barre + 2},
                {"str": 4, "fret": barre + 1}, {"str": 3, "fret": barre},
                {"str": 2, "fret": barre}, {"str": 1, "fret": barre},
            ]
    elif is_seventh:
        if shape == "A":
            if is_minor:
                return [
                    {"str": 6, "fret": -1}, {"str": 5, "fret": barre},
                    {"str": 4, "fret": barre + 2}, {"str": 3, "fret": barre},
                    {"str": 2, "fret": barre + 1}, {"str": 1, "fret": barre},
                ]
            else:
                return [
                    {"str": 6, "fret": -1}, {"str": 5, "fret": barre},
                    {"str": 4, "fret": barre + 2}, {"str": 3, "fret": barre},
                    {"str": 2, "fret": barre + 2}, {"str": 1, "fret": barre},
                ]
        else:
            if is_minor:
                return [
                    {"str": 6, "fret": barre}, {"str": 5, "fret": barre + 2},
                    {"str": 4, "fret": barre}, {"str": 3, "fret": barre},
                    {"str": 2, "fret": barre}, {"str": 1, "fret": barre},
                ]
            else:
                return [
                    {"str": 6, "fret": barre}, {"str": 5, "fret": barre + 2},
                    {"str": 4, "fret": barre}, {"str": 3, "fret": barre + 1},
                    {"str": 2, "fret": barre}, {"str": 1, "fret": barre},
                ]
    else:
        if shape == "A":
            if is_minor:
                return [
                    {"str": 6, "fret": -1}, {"str": 5, "fret": barre},
                    {"str": 4, "fret": barre + 2}, {"str": 3, "fret": barre + 2},
                    {"str": 2, "fret": barre + 1}, {"str": 1, "fret": barre},
                ]
            else:
                return [
                    {"str": 6, "fret": -1}, {"str": 5, "fret": barre},
                    {"str": 4, "fret": barre + 2}, {"str": 3, "fret": barre + 2},
                    {"str": 2, "fret": barre + 2}, {"str": 1, "fret": barre},
                ]
        else:
            if is_minor:
                return [
                    {"str": 6, "fret": barre}, {"str": 5, "fret": barre + 2},
                    {"str": 4, "fret": barre + 2}, {"str": 3, "fret": barre},
                    {"str": 2, "fret": barre}, {"str": 1, "fret": barre},
                ]
            else:
                return [
                    {"str": 6, "fret": barre}, {"str": 5, "fret": barre + 2},
                    {"str": 4, "fret": barre + 2}, {"str": 3, "fret": barre + 1},
                    {"str": 2, "fret": barre}, {"str": 1, "fret": barre},
                ]


def _get_picking_pattern(chord_name: str, beats_per_measure: int, denom: int) -> list:
    """Return picking pattern as list of (string_num, duration, beat_offset).
    User-specified patterns:
      4/4 → T-3-2-3-1-3-2-3 (8 eighth notes)
      3/4,6/8 → T-3-2-1-2-3 (6 eighth notes)
      2/4 → T-3-2-1 (4 eighth notes)
    """
    root_str = _get_chord_root_string(chord_name)
    n = beats_per_measure

    if denom == 8:
        # 6/8: T-3-2-1-2-3 (6 eighth notes across 2 dotted-quarter beats)
        return [
            (root_str, "8", 0.0), (3, "8", 1/3), (2, "8", 2/3),
            (1, "8", 1.0), (2, "8", 4/3), (3, "8", 5/3),
        ]
    elif n == 2:
        # 2/4: T-3-2-1 (4 eighth notes)
        return [
            (root_str, "8", 0.0), (3, "8", 0.5), (2, "8", 1.0), (1, "8", 1.5),
        ]
    elif n == 3:
        # 3/4: T-3-2-1-2-3 (6 eighth notes)
        return [
            (root_str, "8", 0.0), (3, "8", 0.5), (2, "8", 1.0),
            (1, "8", 1.5), (2, "8", 2.0), (3, "8", 2.5),
        ]
    else:
        # 4/4: T-3-2-3-1-3-2-3 (8 eighth notes)
        return [
            (root_str, "8", 0.0), (3, "8", 0.5), (2, "8", 1.0), (3, "8", 1.5),
            (1, "8", 2.0), (3, "8", 2.5), (2, "8", 3.0), (3, "8", 3.5),
        ]


# Unified strumming patterns per time signature.
# Durations calculated from gaps: q(quarter)=1beat, 8(eighth)=0.5, 16(sixteenth)=0.25
_GUITAR_STRUM_PATTERNS = {
    "4/4": [
        # Half-measure ×2: q↓ + e↓ + s↓↑
        (0.0,"down"), (1.0,"down"), (1.5,"down"), (1.75,"up"),
        (2.0,"down"), (3.0,"down"), (3.5,"down"), (3.75,"up"),
    ],
    "3/4": [
        # q↓ + e↓,e↑ + q↓
        (0.0,"down"), (1.0,"down"), (1.5,"up"),
        (2.0,"down"),
    ],
    "2/4": [
        # Half of 4/4 pattern: q↓ + e↓ + s↓↑
        (0.0,"down"), (1.0,"down"), (1.5,"down"), (1.75,"up"),
    ],
    "6/8": [
        # q↓ + e↓,e↑ + e↓,e↑ (quarter + two eighth-note pairs)
        (0.0,"down"), (2/3,"down"), (1.0,"up"),
        (4/3,"down"), (5/3,"up"),
    ],
}


def _get_strum_events(beats_per_measure: int, denom: int) -> list:
    """Get unified strum events as list of (beat_position, direction)."""
    if denom == 4 and beats_per_measure == 2:
        ts_key = "2/4"
    elif denom == 4 and beats_per_measure == 3:
        ts_key = "3/4"
    elif denom == 8:
        ts_key = "6/8"
    else:
        ts_key = "4/4"
    return _GUITAR_STRUM_PATTERNS.get(ts_key, _GUITAR_STRUM_PATTERNS["4/4"])


def _get_chord_root_string(chord_name: str) -> int:
    """Get root string for picking pattern 'T'.
    A/B/C → 5弦, D → 4弦, E/F/G → 6弦."""
    root_name = _extract_chord_root(chord_name).upper()
    first_char = root_name[0]  # Handle F#, Bb etc.
    if first_char in ("E", "F", "G"):
        return 6
    elif first_char == "D":
        return 4
    else:  # A, B, C (and fallback)
        return 5


def generate_guitar_score(melody_notes: list, chords: list, bpm: float, key_name: str,
                          key_mode: str, time_sig: list, sections: list = None,
                          lyrics_data: dict = None, fingering: str = "none") -> dict:
    """Generate guitar tab with chord accompaniment + lyrics + jianpu.

    Lyrics are placed by their Whisper timestamps — each character lands in the
    measure/beat where it actually occurs in the song. Melody notes are used
    only to look up pitch for jianpu numbers.

    fingering: "none" (original key), "C" (C-shape), "G" (G-shape), or "auto"
    """
    if sections is None:
        sections = []

    # ---- Fingering / capo ----
    original_key = key_name
    capo_fret = None
    tuning_semitones = None  # negative = tune down, positive = tune up (capo alternative)
    if fingering and fingering != "none" and key_name:
        if fingering == "auto":
            orig_idx = note_to_chromatic_index(key_name)
            # C-shape: keys C(0) through F(5), G-shape: F#(6) through B(11)
            if 0 <= orig_idx <= 5:
                fingering = "C"
            else:
                fingering = "G"
        raw_capo = _calc_capo_fret(key_name, fingering)
        if raw_capo > 0:
            orig_idx = note_to_chromatic_index(key_name)
            if raw_capo > 6:
                # Impractical capo — use tuning adjustment instead
                # Tune down by (12 - raw_capo) semitones, transpose chords UP to shape key
                tuning_semitones = -(12 - raw_capo)
                shift = -tuning_semitones  # positive: transpose up to shape key
                capo_fret = None
                transposed = []
                for c in chords:
                    tc = dict(c)
                    tc["chord"] = transpose_chord(c.get("chord", ""), shift)
                    transposed.append(tc)
                chords = transposed
                key_name = CHROMATIC[(orig_idx + shift) % 12]
            else:
                # Normal capo: transpose chords down, capo brings pitch back up
                capo_fret = raw_capo
                transposed = []
                for c in chords:
                    tc = dict(c)
                    tc["chord"] = transpose_chord(c.get("chord", ""), -capo_fret)
                    transposed.append(tc)
                chords = transposed
                key_name = CHROMATIC[(orig_idx - capo_fret + 12) % 12]
    # Simplify chord names for display (strip sus2, sus4, add9)
    chords = [dict(c, chord=simplify_chord_name(c.get("chord", ""))) for c in chords]

    sec_map = {}
    for sec in sections:
        for m in range(sec["start_measure"], sec["end_measure"]):
            sec_map[m] = sec["label"]

    key_root = note_to_chromatic_index(key_name)
    beats_per_measure = time_sig[0]
    denom = time_sig[1]
    beat_dur = 60.0 / bpm
    if denom == 8:
        measure_dur = beat_dur * (beats_per_measure // 3)
    else:
        measure_dur = beat_dur * beats_per_measure

    # Categorize time signature
    if denom == 4 and beats_per_measure == 2:
        ts_cat = "2/4"
    elif denom == 4 and beats_per_measure == 3:
        ts_cat = "3/4"
    elif denom == 8:
        ts_cat = "6/8"
    else:
        ts_cat = "4/4"

    # Build melody lookup: time → midi for pitch reference
    melody_by_time = sorted(melody_notes, key=lambda n: n["time"]) if melody_notes else []

    def _lookup_midi(target_time: float) -> int:
        """Find closest melody note pitch within a 2-beat window."""
        best_midi = 60
        best_dist = float("inf")
        for n in melody_by_time:
            d = abs(n["time"] - target_time)
            if d < best_dist and d < beat_dur * 2:
                best_dist = d
                best_midi = int(round(n.get("midi", 60)))
            elif n["time"] > target_time + beat_dur * 2:
                break
        return best_midi

    # Place lyrics by character timestamps from Whisper (not melody note times)
    meas_lyrics = {}  # measure_idx → [{lyric, midi, beatOffset}]
    raw_by_measure = {}  # measure_idx → [{lyric, midi, beatOffset}]

    if lyrics_data:
        from services.lyrics import _split_characters
        # Find where vocals start from section info (for filtering hallucinated early words)
        vocal_start_measure = 999
        if sections:
            for sec in sections:
                if sec["label"] in ("主歌", "副歌", "桥段"):
                    vocal_start_measure = min(vocal_start_measure, sec["start_measure"])
        chars = _split_characters(lyrics_data.get("words", []))
        for ch in chars:
            char_t = (ch["start"] + ch["end"]) / 2
            mi = int(char_t / measure_dur)
            if 0 <= mi < len(chords):
                # Skip instrumental sections — these are Whisper hallucinations from noise
                sl = sec_map.get(mi, "")
                if sl in ("前奏", "间奏", "尾奏"):
                    continue
                # Skip characters before the first vocal section
                if mi < vocal_start_measure:
                    continue
                beat_pos = (char_t - mi * measure_dur) / beat_dur
                if 0 <= beat_pos < beats_per_measure:
                    midi = _lookup_midi(char_t)
                    raw_by_measure.setdefault(mi, []).append({
                        "lyric": ch["word"],
                        "midi": midi,
                        "beatOffset": beat_pos,
                    })

    # Deduplicate within each measure on 16th-note grid
    for mi in range(len(chords)):
        raw = raw_by_measure.get(mi, [])
        meas_lyrics[mi] = []
        used_slots = set()
        for ln in sorted(raw, key=lambda x: x["beatOffset"]):
            slot = round(ln["beatOffset"] * 4) / 4
            while slot in used_slots and slot < beats_per_measure:
                slot += 0.25
            if slot < beats_per_measure:
                used_slots.add(slot)
                meas_lyrics[mi].append({
                    "lyric": ln["lyric"],
                    "midi": ln["midi"],
                    "beatOffset": round(slot, 3),
                })

    measures = []
    for measure_idx in range(len(chords)):
        chord_info = dict(chords[measure_idx])
        chord_info["chord"] = simplify_chord_name(chord_info.get("chord", "C"))
        chord_name = chord_info["chord"]
        sec_label = sec_map.get(measure_idx, "")

        root_str = _get_chord_root_string(chord_name)

        # ---- Build picking / strumming pattern by section ----
        # 主歌 = pure picking (分解), 副歌/桥段 = pure strumming (扫弦),
        # 前奏/间奏/尾奏 = hybrid (bass pick + sparse strum)
        tab_notes = []

        if sec_label == "副歌":
            # Chorus: strumming only
            strum_events = _get_strum_events(beats_per_measure, denom)
            measure_end = float(beats_per_measure // 3 if denom == 8 else beats_per_measure)
            for ei, (offset, direction) in enumerate(strum_events):
                next_offset = strum_events[ei + 1][0] if ei + 1 < len(strum_events) else measure_end
                gap = next_offset - offset
                if gap <= 0.25:
                    dur = "16"
                elif gap <= 0.5:
                    dur = "8"
                elif gap <= 1.0:
                    dur = "q"
                else:
                    dur = "h"
                tab_notes.append({
                    "isPicking": False,
                    "direction": direction,
                    "duration": dur,
                    "dots": 0,
                    "beatOffset": offset,
                })
        else:
            # All other sections: picking (分解)
            pick_pattern = _get_picking_pattern(chord_name, beats_per_measure, denom)
            for p_str, p_dur, p_offset in pick_pattern:
                tab_notes.append({
                    "isPicking": True,
                    "stringNum": p_str,
                    "duration": p_dur,
                    "dots": 0,
                    "beatOffset": p_offset,
                })

        # ---- Attach jianpu to lyric notes ----
        lyric_notes = meas_lyrics.get(measure_idx, [])
        for ln in lyric_notes:
            degree, accidental, oct_shift = midi_to_jianpu(ln["midi"], key_root)
            ln["jianpu"] = {"degree": degree, "accidental": accidental, "octaveShift": oct_shift}

        if sec_label == "副歌":
            ps = "strumming"
        else:
            ps = "picking"

        measures.append({
            "index": measure_idx,
            "tabNotes": tab_notes,
            "lyricNotes": lyric_notes,
            "chord": chord_info,
            "chordFingering": _get_chord_fingering(chord_name),
            "playStyle": ps,
            "rootString": root_str,
        })

    return {
        "notationType": "guitar",
        "songInfo": {
            "key": key_name,
            "keyMode": key_mode,
            "bpm": bpm,
            "timeSignature": time_sig,
            "capo": capo_fret,
            "originalKey": original_key if (capo_fret or tuning_semitones) else None,
            "fingering": fingering if (capo_fret or tuning_semitones) else None,
            "tuningSemitones": tuning_semitones,
        },
        "chordProgression": chords,
        "measures": measures,
    }


def generate_drum_score(drum_audio_path: str, bpm: float, time_sig: list, sections: list = None) -> dict:
    """
    Generate drum notation score from drum stem.
    Uses onset detection and frequency analysis to classify hits.
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(drum_audio_path, sr=22050, mono=True)
    hop_length = 256

    # Detect onsets with faster settings
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length,
        backtrack=True, units="frames",
        wait=2, pre_max=3, post_max=3,
        delta=0.15,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    beats_per_measure = time_sig[0]
    denom = time_sig[1]
    beat_dur = 60.0 / bpm
    if denom == 8:
        measure_dur = beat_dur * (beats_per_measure // 3)
    else:
        measure_dur = beat_dur * beats_per_measure

    # Classify each onset with 6-band spectral analysis
    drum_events = []
    for ft in onset_frames:
        t = librosa.frames_to_time(ft, sr=sr, hop_length=hop_length)

        frame_start = max(0, ft * hop_length - 1024)
        frame_end = min(len(y), ft * hop_length + 1024)
        frame = y[frame_start:frame_end]
        if len(frame) < 256:
            continue

        spec = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)

        bands = {
            "sub":    (20, 60),
            "low":    (60, 200),
            "low_mid": (200, 600),
            "high_mid": (600, 3000),
            "high":   (3000, 8000),
            "air":    (8000, 16000),
        }
        energy = {}
        for name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            energy[name] = spec[mask].sum() if mask.any() else 0

        total = sum(energy.values()) + 1e-8
        r = {k: v / total for k, v in energy.items()}

        if r["sub"] > 0.30:
            drum_type = "kick"
        elif r["low"] > 0.25 and r["high_mid"] > 0.10:
            drum_type = "snare"
        elif r["low_mid"] > 0.30 and r["high_mid"] < 0.10:
            drum_type = "tom_low"
        elif r["high_mid"] > 0.25 and r["high"] > 0.12:
            drum_type = "tom_high"
        elif r["high"] > 0.30 and r["air"] > 0.12:
            drum_type = "crash"
        elif r["high"] > 0.18 and r["air"] < 0.08:
            drum_type = "hihat_closed"
        elif r["high"] > 0.15 and r["air"] > 0.05:
            drum_type = "ride"
        else:
            drum_type = "hihat_closed"

        drum_events.append({
            "time": round(float(t), 4),
            "type": drum_type,
        })

    # Organize into measures
    measures = []
    event_idx = 0
    total_measures = max(1, int(drum_events[-1]["time"] / measure_dur + 1)) if drum_events else 1

    for measure_idx in range(total_measures):
        m_start = measure_idx * measure_dur
        m_end = m_start + measure_dur

        drum_notes = []
        while event_idx < len(drum_events) and drum_events[event_idx]["time"] < m_end:
            evt = drum_events[event_idx]
            if evt["time"] >= m_start:
                drum_notes.append({
                    "type": evt["type"],
                    "duration": "8",  # Most drum hits are eighth notes
                    "dots": 0,
                })
            event_idx += 1

        if not drum_notes:
            drum_notes.append({"type": "rest", "duration": "qr", "dots": 0})

        measures.append({
            "index": measure_idx,
            "drumNotes": drum_notes,
        })

    return {
        "notationType": "drums",
        "songInfo": {
            "key": "N/A",
            "keyMode": "",
            "bpm": bpm,
            "timeSignature": time_sig,
        },
        "chordProgression": [],
        "measures": measures,
    }



# Bass guitar standard tuning (4 strings, MIDI numbers)
BASS_STRINGS = [43, 38, 33, 28]  # G2, D2, A1, E1
BASS_STRING_NAMES = ["G", "D", "A", "E"]


def _midi_to_bass_fret(midi_val: float, prev: dict = None) -> Optional[dict]:
    """Map a MIDI note to a bass guitar string and fret.
    Prefers positions close to the previous fret to minimize jumps.
    Falls back to lowest-string-first when no prev context."""
    midi_int = int(round(midi_val))
    midi_int -= 12

    # Collect all playable positions
    positions = []
    # Strings: G=1(43), D=2(38), A=3(33), E=4(28)
    for string_idx in [3, 2, 1, 0]:
        open_midi = BASS_STRINGS[string_idx]
        fret = midi_int - open_midi
        if 0 <= fret <= 15:
            positions.append((string_idx + 1, fret))

    if not positions:
        return None

    # If we know the previous position, pick the closest one
    if prev and prev.get("str") and prev.get("fret") is not None:
        prev_str = prev["str"]
        prev_fret = prev["fret"]
        best = None
        best_dist = float("inf")
        for s, f in positions:
            # Distance: string change costs more than fret change
            str_diff = abs(s - prev_str) * 2  # each string away = 2 fret-distance penalty
            fret_diff = abs(f - prev_fret)
            dist = str_diff + fret_diff
            if dist < best_dist:
                best_dist = dist
                best = {"str": s, "fret": f}
        # Reject absurd jumps (>7 frets on same string or >4 strings + >5 frets)
        if best and best_dist < 8:
            return best

    # Fallback: lowest playable string (best bass tone)
    return {"str": positions[0][0], "fret": positions[0][1]}


def generate_bass_score(bass_audio_path: str, chords: list, bpm: float,
                        key_name: str, key_mode: str, time_sig: list,
                        precomputed_notes: list = None, sections: list = None) -> dict:
    """
    Generate bass guitar tab score from bass stem audio.
    Extracts pitch from bass stem and maps to 4-string tab.
    If precomputed_notes is provided, skips the expensive pYIN step.
    """
    if precomputed_notes:
        bass_notes = precomputed_notes
    else:
        import librosa
        import numpy as np

        y, sr = librosa.load(bass_audio_path, sr=22050, mono=True)
        hop_length = 256

        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C1"),
            fmax=librosa.note_to_hz("D3"),
            sr=sr,
            hop_length=hop_length,
            fill_na=0,
        )

        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

        bass_notes = []
        i = 0
        while i < len(f0):
            if voiced_flag[i] and f0[i] > 0:
                start_time = times[i]
                midi_vals = [librosa.hz_to_midi(f0[i])]
                j = i + 1
                while j < len(f0) and voiced_flag[j] and f0[j] > 0:
                    diff = abs(librosa.hz_to_midi(f0[j]) - librosa.hz_to_midi(f0[j - 1]))
                    if diff > 2.0:
                        break
                    midi_vals.append(librosa.hz_to_midi(f0[j]))
                    j += 1

                end_time = times[j - 1] if j < len(f0) else times[-1]
                avg_midi = float(np.mean(midi_vals))

                bass_notes.append({
                    "time": round(float(start_time), 4),
                    "end_time": round(float(end_time), 4),
                    "duration": round(float(end_time - start_time), 4),
                    "midi": round(avg_midi, 1),
                })
                i = j
            else:
                i += 1

    # Build section map for section-aware fallback
    sec_map = {}
    if sections:
        for sec in sections:
            for m in range(sec["start_measure"], sec["end_measure"]):
                sec_map[m] = sec["label"]

    # Organize into measures
    key_root = note_to_chromatic_index(key_name)
    beats_per_measure = time_sig[0]
    denom = time_sig[1]
    beat_dur = 60.0 / bpm
    if denom == 8:
        measure_dur = beat_dur * (beats_per_measure // 3)
    else:
        measure_dur = beat_dur * beats_per_measure

    measures = []
    note_idx = 0
    measure_idx = 0
    prev_pos = None  # Track last fret/string to minimize jumps

    while note_idx < len(bass_notes) or measure_idx < len(chords):
        m_start = measure_idx * measure_dur
        m_end = m_start + measure_dur

        tab_notes = []

        while note_idx < len(bass_notes) and bass_notes[note_idx]["time"] < m_end:
            note = bass_notes[note_idx]
            if note["time"] >= m_start:
                dur_info = quantize_duration_dict(note["duration"], bpm)
                fret_info = _midi_to_bass_fret(note["midi"], prev_pos)

                prev_pos = fret_info
                tab_entry = {
                    "positions": [fret_info] if fret_info else [{"str": 1, "fret": 0}],
                    "duration": dur_info["duration"],
                    "dots": dur_info["dots"],
                    "lyric": "",
                }
                tab_notes.append(tab_entry)
            note_idx += 1

        chord_info = None
        if measure_idx < len(chords):
            chord_info = chords[measure_idx]

        # Fill empty measures with section-aware bass patterns
        if not tab_notes and chord_info:
            sec_label = sec_map.get(measure_idx, "")
            pattern = _get_bass_pattern(chord_info["chord"], beats_per_measure,
                                        sec_label, measure_idx, denom)
            for p_midi, p_dur, p_offset in pattern:
                fret_info = _midi_to_bass_fret(p_midi, prev_pos)
                prev_pos = fret_info
                tab_notes.append({
                    "positions": [fret_info] if fret_info else [{"str": 2, "fret": 0}],
                    "duration": p_dur,
                    "dots": 0,
                    "lyric": "",
                })
        elif not tab_notes:
            tab_notes.append({
                "positions": [{"str": 2, "fret": 0}],
                "duration": "qr",
                "dots": 0,
                "lyric": "",
            })

        _add_jianpu_to_notes(tab_notes, key_root,
            lambda n: BASS_STRINGS[n["positions"][0]["str"] - 1] + n["positions"][0]["fret"] if n.get("positions") else None)

        measures.append({
            "index": measure_idx,
            "tabNotes": tab_notes,
            "chord": chord_info,
        })

        # Break after processing all bass notes and chords
        if note_idx >= len(bass_notes) and measure_idx >= len(chords) - 1:
            measure_idx += 1
            break
        measure_idx += 1

    # Detect slides: consecutive notes on same string with small fret difference
    for mi in range(len(measures)):
        tnotes = measures[mi]["tabNotes"]
        for ni in range(1, len(tnotes)):
            pos_a = tnotes[ni - 1].get("positions", [{}])[0]
            pos_b = tnotes[ni].get("positions", [{}])[0]
            if pos_a.get("str") == pos_b.get("str"):
                diff = abs(pos_b.get("fret", 0) - pos_a.get("fret", 0))
                if 1 <= diff <= 4:
                    tnotes[ni]["slide"] = diff

    return {
        "notationType": "bass",
        "songInfo": {
            "key": key_name,
            "keyMode": key_mode,
            "bpm": bpm,
            "timeSignature": time_sig,
        },
        "chordProgression": chords,
        "measures": measures,
    }


MELODY_BASED_TYPES = {"guitar": generate_guitar_score}
