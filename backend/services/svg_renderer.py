"""
Server-side SVG sheet music renderer.
Generates self-contained SVG for piano, guitar, drums, and bass notation.
"""

import math

from services.common import (CHROMATIC, parse_vex_key, DUR_BEATS, dur_to_beats,
                              midi_to_staff_pos, note_name_normalized, is_rest)

# Layout constants
S = 8            # Staff line spacing
NOTE_R = 5.5     # Note head radius
STEM_H = S * 2.8 # Stem length
PAD_X = 36
PAD_Y = 22
TITLE_H = 48

NOTATION_NAMES = {"piano": "钢琴谱", "guitar": "吉他谱", "bass": "贝斯谱", "drums": "架子鼓谱"}


def _title_bar(svg, w: float, notation_type: str, song_info: dict, song_name: str = ""):
    nt = NOTATION_NAMES.get(notation_type, notation_type)
    key = song_info.get("key", "C")
    km = song_info.get("keyMode", "major")
    bpm = song_info.get("bpm", 120)
    ts = song_info.get("timeSignature", [4, 4])
    capo = song_info.get("capo")
    original_key = song_info.get("originalKey")
    tuning_semitones = song_info.get("tuningSemitones")

    # If capo/tuning is active, show the original key (song's real key) as primary
    # rather than the transposed playing key
    display_key = original_key if original_key else key
    if display_key == "N/A" or not km:
        kd = ""
    else:
        kd = f"{display_key} {'大调' if km == 'major' else '小调'}"

    svg.append(f'<rect x="0" y="0" width="{w}" height="{TITLE_H}" fill="#f8f8f8" rx="4"/>')
    svg.append(f'<line x1="0" y1="{TITLE_H}" x2="{w}" y2="{TITLE_H}" stroke="#e94560" stroke-width="2.5"/>')
    svg.append(f'<text x="{PAD_X}" y="{TITLE_H * 0.42}" font-size="17" font-weight="bold" '
               f'fill="#1a1a2e" font-family="sans-serif">{nt}</text>')
    if song_name:
        svg.append(f'<text x="{PAD_X}" y="{TITLE_H * 0.82}" font-size="11" fill="#888" '
                   f'font-family="sans-serif">{song_name}</text>')
    info_parts = []
    if kd:
        info_parts.append(kd)
    info_parts.append(f"BPM {bpm}")
    info_parts.append(f"{ts[0]}/{ts[1]}拍")
    if capo:
        capo_text = f"Capo {capo}（{key} 调指法）"
        info_parts.append(capo_text)
    if tuning_semitones:
        steps = abs(tuning_semitones)
        if steps == 1:
            tune_label = "降半音调弦" if tuning_semitones < 0 else "升半音调弦"
        else:
            tune_label = f"降{steps}个半音调弦" if tuning_semitones < 0 else f"升{steps}个半音调弦"
        tune_text = f"{tune_label}（{key} 调指法）"
        info_parts.append(tune_text)
    info = "  ·  ".join(info_parts)
    svg.append(f'<text x="{w - PAD_X}" y="{TITLE_H * 0.52}" text-anchor="end" '
               f'font-size="13" fill="#555" font-family="sans-serif">{info}</text>')


def _staff_lines(svg, x: float, y: float, w: float, n: int = 5):
    for i in range(n):
        ly = y + i * S
        sw = 0.9 if i in (0, n - 1) else 0.5
        svg.append(f'<line x1="{x}" y1="{ly}" x2="{x + w}" y2="{ly}" stroke="#222" stroke-width="{sw}"/>')


def _bar_line(svg, x: float, y: float, n: int = 5, final: bool = False, spacing: float = None):
    sp = spacing if spacing is not None else S
    bot = y + (n - 1) * sp
    if final:
        svg.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{bot}" stroke="#222" stroke-width="1.2"/>')
        svg.append(f'<line x1="{x + 5}" y1="{y}" x2="{x + 5}" y2="{bot}" stroke="#222" stroke-width="2.8"/>')
    else:
        svg.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{bot}" stroke="#333" stroke-width="1.4"/>')


def _measure_num(svg, x: float, y: float, num: int):
    svg.append(f'<text x="{x + 2}" y="{y}" font-size="9" fill="#bbb" font-family="sans-serif">{num}</text>')


# Key signature data
_KEY_SIG = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
    "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7,
    "Am": 0, "Em": 1, "Bm": 2, "F#m": 3, "C#m": 4, "G#m": 5, "D#m": 6, "A#m": 7,
    "Dm": -1, "Gm": -2, "Cm": -3, "Fm": -4, "Bbm": -5, "Ebm": -6, "Abm": -7,
}
_SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
_FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]
_KS_TREBLE_MIDI = {"F": 77, "C": 72, "G": 79, "D": 74, "A": 69, "E": 76, "B": 71}
_KS_BASS_MIDI = {"F": 65, "C": 60, "G": 67, "D": 62, "A": 57, "E": 64, "B": 59}


def _draw_key_signature(svg, x: float, staff_y: float, key_name: str, key_mode: str, clef: str = "treble") -> float:
    """Draw key signature accidentals. Returns x-offset consumed (for spacing)."""
    if key_mode and key_mode.lower() == "minor":
        lookup = f"{key_name}m"
    else:
        lookup = key_name
    n_acc = _KEY_SIG.get(lookup, 0)
    if n_acc == 0:
        return 0
    ks_midi = _KS_TREBLE_MIDI if clef == "treble" else _KS_BASS_MIDI
    order = _SHARP_ORDER if n_acc > 0 else _FLAT_ORDER
    sym = "♯" if n_acc > 0 else "♭"
    spacing = 5.5
    for i in range(abs(n_acc)):
        note_name = order[i]
        midi_val = ks_midi.get(note_name, 60)
        pos = midi_to_staff_pos(midi_val, clef)
        cy = staff_y + 2 * S - pos * S / 2
        svg.append(f'<text x="{x + i * spacing}" y="{cy + 3}" text-anchor="middle" '
                   f'font-size="14" font-weight="bold" fill="#333" font-family="serif">{sym}</text>')
    return abs(n_acc) * spacing


def _needs_accidental(midi_val: int, key_name: str, key_mode: str) -> str:
    """Return accidental symbol (♯, ♭, ♮) for notes outside the key's diatonic scale."""
    major_intervals = [0, 2, 4, 5, 7, 9, 11]
    key_norm = note_name_normalized(key_name) if key_name else "C"
    root_idx = CHROMATIC.index(key_norm) if key_norm in CHROMATIC else 0
    if key_mode and key_mode.lower() == "minor":
        root_idx = (root_idx - 3) % 12
    scale_pcs = set((root_idx + iv) % 12 for iv in major_intervals)
    pc = midi_val % 12
    if pc in scale_pcs:
        return ""

    # Key signature accidentals: which notes are altered by the key sig?
    lookup = f"{key_name}m" if (key_mode and key_mode.lower() == "minor") else key_name
    n_acc = _KEY_SIG.get(lookup, 0)
    naturals = []
    if n_acc > 0:
        naturals = [CHROMATIC.index(n) for n in _SHARP_ORDER[:n_acc]]
    elif n_acc < 0:
        naturals = [CHROMATIC.index(n) for n in _FLAT_ORDER[:-n_acc]]

    if pc in naturals:
        return "♮"

    # Determine sharp vs flat by nearest scale degree.
    # In sharp keys prefer sharps, in flat keys prefer flats, in C/Am prefer flats.
    if n_acc > 0:
        # Sharp key: try sharp first
        for iv in major_intervals:
            scale_pc = (root_idx + iv) % 12
            if (pc - scale_pc) % 12 == 1:
                return "♯"
        for iv in major_intervals:
            scale_pc = (root_idx + iv) % 12
            if (scale_pc - pc) % 12 == 1:
                return "♭"
    else:
        # Flat key or C major: try flat first
        for iv in major_intervals:
            scale_pc = (root_idx + iv) % 12
            if (scale_pc - pc) % 12 == 1:
                return "♭"
        for iv in major_intervals:
            scale_pc = (root_idx + iv) % 12
            if (pc - scale_pc) % 12 == 1:
                return "♯"

    return "♮"


def _draw_ledger_lines(svg, cx: float, staff_y: float, cy: float):
    """Draw ledger lines for a single note-head y position."""
    st, sb = staff_y, staff_y + 4 * S
    na = max(0, math.ceil((st - (cy - NOTE_R)) / S))
    nb = max(0, math.ceil(((cy + NOTE_R) - sb) / S))
    for li in range(na):
        ly = st - (li + 1) * S
        svg.append(f'<line x1="{cx - NOTE_R * 1.5}" y1="{ly}" x2="{cx + NOTE_R * 1.5}" y2="{ly}" '
                   f'stroke="#333" stroke-width="0.8"/>')
    for li in range(nb):
        ly = sb + (li + 1) * S
        svg.append(f'<line x1="{cx - NOTE_R * 1.5}" y1="{ly}" x2="{cx + NOTE_R * 1.5}" y2="{ly}" '
                   f'stroke="#333" stroke-width="0.8"/>')


def _note_head(svg, cx: float, cy: float, duration: str):
    """Draw a single note head (open for whole/half, filled for quarter and shorter)."""
    is_open = duration in ("w", "h")
    if is_open:
        svg.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{NOTE_R * 0.85}" ry="{NOTE_R * 0.6}" '
                   f'fill="#fcfcfc" stroke="#222" stroke-width="1.2" transform="rotate(-15, {cx}, {cy})"/>')
    else:
        svg.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{NOTE_R * 0.85}" ry="{NOTE_R * 0.6}" '
                   f'fill="#222" stroke="#222" stroke-width="0.5" transform="rotate(-15, {cx}, {cy})"/>')


def _draw_beam(svg, start_x: float, end_x: float, stem_y: float, stem_dir: int, n_beams: int = 1):
    """Draw beam(s) connecting stems. stem_dir: -1 stems up, +1 stems down."""
    for bi in range(n_beams):
        offset = bi * 4 * stem_dir
        by = stem_y + offset
        svg.append(f'<rect x="{start_x}" y="{by - 1.5}" width="{end_x - start_x}" height="3" '
                   f'fill="#333"/>')


# =============================================================================
# Note drawing with beaming
# =============================================================================
def _draw_measure_notes(svg, notes: list, staff_y: float, x_start: float, measure_width: float,
                         beats_per_measure: float, clef: str = "treble", show_lyrics: bool = True,
                         key_name: str = "C", key_mode: str = "major"):
    """
    Draw all notes in a measure with proper rhythmic spacing and beaming.
    notes: list of {keys: [vexnote], duration: str, dots: int, lyric: str}
    Groups consecutive eighth/16th notes with beams.
    """
    if not notes:
        return

    # Calculate total beats in this measure
    total_beats = sum(dur_to_beats(n.get("duration", "q")) for n in notes)
    if total_beats <= 0:
        return

    # Effective drawing width (leave padding at edges)
    draw_w = measure_width - 16

    # Position notes proportionally by their beat value
    positions = []
    beat_pos = 0
    for note in notes:
        dur_str = note.get("duration", "q").rstrip(".")
        dur_beats = DUR_BEATS.get(dur_str, 1.0)
        # Dot adds 50% of the base duration
        if "." in note.get("duration", ""):
            dur_beats *= 1.5
        # x position: proportional to beat position
        nx = x_start + 8 + (beat_pos / total_beats) * draw_w
        positions.append((nx, dur_beats, note))
        beat_pos += dur_beats

    # Group consecutive eighth/16th notes for beaming
    beam_groups = []
    current_group = []
    for i, (nx, beats, note) in enumerate(positions):
        dur_str = note.get("duration", "q").rstrip(".")
        is_short = dur_str in ("8", "16")
        is_rest_note = is_rest(dur_str) or len(note.get("keys", [])) == 0

        if is_short and not is_rest_note:
            if current_group:
                # Check if adjacent (no non-short note between them)
                prev_note = positions[i - 1][2]
                prev_dur = prev_note.get("duration", "q").rstrip(".")
                prev_is_rest = is_rest(prev_dur) or len(prev_note.get("keys", [])) == 0
                if prev_dur in ("8", "16") and not prev_is_rest:
                    current_group.append((i, nx, beats, note))
                else:
                    if len(current_group) >= 2:
                        beam_groups.append(current_group)
                    current_group = [(i, nx, beats, note)]
            else:
                current_group = [(i, nx, beats, note)]
        else:
            if len(current_group) >= 2:
                beam_groups.append(current_group)
            current_group = []
    if len(current_group) >= 2:
        beam_groups.append(current_group)

    # Build set of indices that are in beam groups
    beam_indices = set()
    for bg in beam_groups:
        for idx, _, _, _ in bg:
            beam_indices.add(idx)

    # Draw each note
    for i, (nx, beats, note) in enumerate(positions):
        dur_str = note.get("duration", "q").rstrip(".")
        keys = note.get("keys", ["c/4"])
        is_rest_note = is_rest(dur_str) or len(keys) == 0
        lyric = note.get("lyric", "")
        chord_deg = note.get("chordDeg", "")

        if is_rest_note:
            # Draw rest symbol
            rest_y = staff_y + 2.5 * S
            rest_sym = {"qr": "𝄽", "hr": "𝄼", "wr": "𝄻"}.get(dur_str + "r" if "r" not in dur_str else dur_str, "𝄽")
            svg.append(f'<text x="{nx}" y="{rest_y}" text-anchor="middle" font-size="18" '
                       f'fill="#333">{rest_sym}</text>')
            continue

        midi_vals = [parse_vex_key(k) for k in keys]
        positions_y = [midi_to_staff_pos(m, clef) for m in midi_vals]
        cy_vals = [staff_y + 2 * S - p * S / 2 for p in positions_y]
        min_cy, max_cy = min(cy_vals), max(cy_vals)
        avg_pos = sum(positions_y) / len(positions_y)

        # Draw accidentals before note heads (for chromatic notes outside key signature)
        acc_x_offset = 0
        for midi, cy in zip(midi_vals, cy_vals):
            acc = _needs_accidental(midi, key_name, key_mode)
            if acc:
                svg.append(f'<text x="{nx}" y="{cy + 3.5}" text-anchor="middle" '
                           f'font-size="11" font-weight="bold" fill="#333" font-family="serif">{acc}</text>')
                acc_x_offset = max(acc_x_offset, NOTE_R * 1.8)

        # Shift note x to make room for accidentals
        note_x = nx + acc_x_offset

        # Draw ledger lines first (behind note heads)
        for cy in cy_vals:
            _draw_ledger_lines(svg, note_x, staff_y, cy)

        # Note heads
        for cy in cy_vals:
            _note_head(svg, note_x, cy, dur_str)

        # Stem (for non-whole notes, and only if not in a beam)
        if dur_str not in ("w", "wr") and i not in beam_indices:
            stem_up = avg_pos < 2
            stem_start = min_cy if stem_up else max_cy
            stem_end = stem_start - STEM_H if stem_up else stem_start + STEM_H
            sx = note_x + NOTE_R * 0.7
            svg.append(f'<line x1="{sx}" y1="{stem_start}" x2="{sx}" y2="{stem_end}" '
                       f'stroke="#333" stroke-width="1.0"/>')
            # Flag for 8th/16th
            if dur_str in ("8", "16"):
                n_flags = 1 if dur_str == "8" else 2
                for fi in range(n_flags):
                    fy = stem_end + fi * 3
                    # Draw flag curve
                    if stem_up:
                        svg.append(f'<path d="M{sx},{fy} Q{sx + 6},{fy + 5} {sx + 2},{fy + 9}" '
                                   f'fill="none" stroke="#333" stroke-width="1.0"/>')
                    else:
                        svg.append(f'<path d="M{sx},{fy} Q{sx + 6},{fy - 5} {sx + 2},{fy - 9}" '
                                   f'fill="none" stroke="#333" stroke-width="1.0"/>')

        # Dotted note
        if "." in note.get("duration", ""):
            dot_dy = -1.5 * S if (avg_pos < 2) else 1.5 * S
            dot_y = min_cy + dot_dy
            svg.append(f'<circle cx="{note_x + NOTE_R * 1.8}" cy="{dot_y}" r="1.8" fill="#333"/>')

        # Chord degree above
        if chord_deg:
            svg.append(f'<text x="{note_x}" y="{staff_y - 1.8 * S}" text-anchor="middle" '
                       f'font-size="12" font-weight="bold" fill="#e94560" font-family="sans-serif">{chord_deg}</text>')

        # Jianpu above lyrics (below stem extent)
        if show_lyrics and lyric:
            jp = note.get("jianpu", {})
            if jp.get("degree"):
                acc = jp.get("accidental", "")
                svg.append(f'<text x="{note_x}" y="{staff_y + 7.0 * S}" text-anchor="middle" '
                           f'font-size="10" font-weight="bold" fill="#c0392b" '
                           f'font-family="sans-serif">{acc}{jp["degree"]}</text>')
            svg.append(f'<text x="{note_x}" y="{staff_y + 8.5 * S}" text-anchor="middle" '
                       f'font-size="13" fill="#444" font-weight="500" font-family="sans-serif">{lyric}</text>')

    # Draw beams for grouped notes
    for bg in beam_groups:
        if len(bg) < 2:
            continue

        # Get stem direction from average position
        all_positions = []
        for idx, nx, beats, note in bg:
            keys = note.get("keys", ["c/4"])
            midi_vals = [parse_vex_key(k) for k in keys]
            positions_y = [midi_to_staff_pos(m, clef) for m in midi_vals]
            all_positions.extend(positions_y)
        avg_pos = sum(all_positions) / len(all_positions) if all_positions else 0
        stem_up = avg_pos < 2

        # Find overall beam height (unified across all notes in the group)
        all_cy = []
        for idx, nx, beats, note in bg:
            keys = note.get("keys", ["c/4"])
            midi_vals = [parse_vex_key(k) for k in keys]
            positions_y = [midi_to_staff_pos(m, clef) for m in midi_vals]
            for p in positions_y:
                all_cy.append(staff_y + 2 * S - p * S / 2)
        beam_min_cy = min(all_cy) if all_cy else staff_y + 2 * S
        beam_max_cy = max(all_cy) if all_cy else staff_y + 2 * S
        beam_stem_end = (beam_min_cy - STEM_H) if stem_up else (beam_max_cy + STEM_H)

        # Draw stems to unified beam height (note heads are drawn in main loop)
        for idx, nx, beats, note in bg:
            dur_str = note.get("duration", "q").rstrip(".")
            keys = note.get("keys", ["c/4"])
            midi_vals = [parse_vex_key(k) for k in keys]
            positions_y = [midi_to_staff_pos(m, clef) for m in midi_vals]
            cy_vals = [staff_y + 2 * S - p * S / 2 for p in positions_y]
            min_cy, max_cy = min(cy_vals), max(cy_vals)

            # Stem to unified beam height
            sx = nx + NOTE_R * 0.7
            stem_start_y = min_cy if stem_up else max_cy
            svg.append(f'<line x1="{sx}" y1="{stem_start_y}" x2="{sx}" y2="{beam_stem_end}" '
                       f'stroke="#333" stroke-width="1.0"/>')

        # Beam connecting all stems in the group
        first_nx = bg[0][1] + NOTE_R * 0.7
        last_nx = bg[-1][1] + NOTE_R * 0.7
        beam_y = beam_stem_end

        # Determine number of beams (eighth = 1 beam, sixteenth = 2 beams)
        dur_strings = [n.get("duration", "q").rstrip(".") for _, _, _, n in bg]
        n_beams = 2 if "16" in dur_strings else 1

        _draw_beam(svg, first_nx, last_nx, beam_y, -1 if stem_up else 1, n_beams)


# =============================================================================
# Chord diagram
# =============================================================================
def _chord_diagram(svg, dx: float, dy: float, chord_name: str, degree: str, fingering: list):
    """
    Standard guitar chord diagram. Strings 6→1 left to right.
    Uses relative fret positioning so all dots fit in a 5-fret window.
    """
    str_sp = 7
    fret_sp = 10
    n_strings = 6
    n_frets = 5
    grid_w = (n_strings - 1) * str_sp
    grid_h = (n_frets - 1) * fret_sp
    box_w = grid_w + 16
    box_h = grid_h + 24
    grid_left = dx + 8
    grid_top = dy + 28

    # Find base fret (lowest non-zero, non-muted fret)
    frets = [f["fret"] for f in fingering if f.get("fret", -1) > 0]
    base_fret = min(frets) if frets else 1
    # For open chords (base_fret = 1), show from the nut. Otherwise offset.
    show_position = base_fret > 1

    # Box
    svg.append(f'<rect x="{dx}" y="{dy}" width="{box_w}" height="{box_h}" '
               f'fill="#fafafa" stroke="#ddd" stroke-width="1" rx="3"/>')

    # Name and degree
    svg.append(f'<text x="{dx + box_w / 2}" y="{dy + 12}" text-anchor="middle" font-size="9" '
               f'font-weight="bold" fill="#e94560" font-family="sans-serif">{degree}</text>')
    svg.append(f'<text x="{dx + box_w / 2}" y="{dy + 24}" text-anchor="middle" font-size="15" '
               f'font-weight="bold" fill="#222" font-family="sans-serif">{chord_name}</text>')

    # Position marker (e.g., "5fr")
    if show_position:
        svg.append(f'<text x="{grid_left + grid_w + 4}" y="{grid_top + fret_sp * 0.5 + 4}" '
                   f'font-size="8" fill="#888" font-family="sans-serif">{base_fret}fr</text>')

    # Fret lines (nut/first line is thicker)
    for fr in range(n_frets):
        fy = grid_top + fr * fret_sp
        sw = 1.2 if fr == 0 and not show_position else 0.4
        svg.append(f'<line x1="{grid_left}" y1="{fy}" x2="{grid_left + grid_w}" y2="{fy}" '
                   f'stroke="#999" stroke-width="{sw}"/>')

    # String lines
    for s in range(n_strings):
        sx = grid_left + s * str_sp
        svg.append(f'<line x1="{sx}" y1="{grid_top}" x2="{sx}" y2="{grid_top + grid_h}" '
                   f'stroke="#999" stroke-width="0.6"/>')

    # Draw markers
    for entry in fingering:
        string = entry.get("str", 1)  # 1=high E, 6=low E
        fret = entry.get("fret", -1)
        # String 6 leftmost, string 1 rightmost
        sx = grid_left + (6 - string) * str_sp

        if fret < 0:
            # Muted: X above
            svg.append(f'<text x="{sx}" y="{grid_top - 5}" text-anchor="middle" '
                       f'font-size="9" fill="#aaa" font-family="sans-serif">✕</text>')
        elif fret == 0:
            # Open: O above
            svg.append(f'<circle cx="{sx}" cy="{grid_top - 5}" r="2.5" fill="none" stroke="#999" stroke-width="1"/>')
        else:
            # Relative fret position
            rel_fret = fret - base_fret + 1  # 1-indexed from nut/barre line
            if 1 <= rel_fret <= n_frets:
                dot_y = grid_top + (rel_fret - 0.5) * fret_sp
                svg.append(f'<circle cx="{sx}" cy="{dot_y}" r="3" fill="#e94560"/>')

    # Barre line for E-shape chords (when base_fret > 1 and all strings at base_fret)
    barre_strings = [f for f in fingering if f.get("fret", -1) == base_fret]
    if show_position and len(barre_strings) >= 4:
        min_s = min(f["str"] for f in barre_strings)
        max_s = max(f["str"] for f in barre_strings)
        bx1 = grid_left + (6 - max_s) * str_sp
        bx2 = grid_left + (6 - min_s) * str_sp
        barre_y = grid_top + 0.5 * fret_sp
        svg.append(f'<line x1="{bx1}" y1="{barre_y}" x2="{bx2}" y2="{barre_y}" '
                   f'stroke="#333" stroke-width="3" stroke-linecap="round" opacity="0.6"/>')


# =============================================================================
# Piano Grand Staff
# =============================================================================
def _render_piano_page(page_measures: list, song_info: dict, key_name: str, key_mode: str,
                        page: int, total_pages: int, mpr: int, song_name: str,
                        global_start_measure: int) -> str:
    """Render one page of the piano grand staff."""
    staff_gap = 7 * S
    sw = 960
    row_h = 2 * (4 * S) + staff_gap + 48
    page_rows = max(1, math.ceil(len(page_measures) / mpr))
    tw = sw + PAD_X * 2 + 64
    ph = PAD_Y + TITLE_H + 20 + page_rows * row_h + 40
    is_first_page = (page == 0)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {tw} {ph}" width="{tw}" style="max-width:100%;margin-bottom:12px;">']
    svg.append(f'<rect width="100%" height="100%" fill="#fcfcfc" rx="4"/>')

    # Title bar
    if is_first_page:
        _title_bar(svg, tw, "piano", song_info, song_name)
    else:
        svg.append(f'<rect x="0" y="0" width="{tw}" height="{TITLE_H}" fill="#f8f8f8" rx="4"/>')
        svg.append(f'<line x1="0" y1="{TITLE_H}" x2="{tw}" y2="{TITLE_H}" stroke="#e94560" stroke-width="2"/>')
        svg.append(f'<text x="{PAD_X}" y="{TITLE_H * 0.42}" font-size="14" font-weight="bold" '
                   f'fill="#333" font-family="sans-serif">{song_name}（续）</text>')

    # Page number (bottom-right, avoids overlapping title bar info)
    svg.append(f'<text x="{tw - PAD_X}" y="{ph - 10}" font-size="11" fill="#aaa" '
               f'text-anchor="end" font-family="sans-serif">第 {page + 1}/{total_pages} 页</text>')

    for row in range(page_rows):
        rm = page_measures[row * mpr:(row + 1) * mpr]
        nm_row = max(1, len(rm))
        mw = sw / nm_row

        by = PAD_Y + TITLE_H + 20 + row * row_h
        ty = by
        bass_y = ty + 4 * S + staff_gap

        # Brace on first row of each page
        if row == 0:
            brace_x = PAD_X - 12
            svg.append(f'<path d="M{brace_x},{ty + 2} Q{brace_x - 4},{ty + 2*S} {brace_x},{ty + 4*S} '
                       f'Q{brace_x - 4},{bass_y} {brace_x},{bass_y + 2*S}" '
                       f'fill="none" stroke="#888" stroke-width="1.5"/>')

        # Clefs on every system
        svg.append(f'<text x="{PAD_X + 4}" y="{ty + 3.4 * S}" font-size="{S * 4}" '
                   f'font-family="serif" fill="#222">𝄞</text>')
        svg.append(f'<text x="{PAD_X + 4}" y="{bass_y + 1.5 * S}" font-size="{S * 3.5}" '
                   f'font-family="serif" fill="#222">𝄢</text>')

        # Key signatures
        ks_x_treble = PAD_X + 18
        ks_x_bass = PAD_X + 18
        ks_w_t = _draw_key_signature(svg, ks_x_treble, ty, key_name, key_mode, "treble")
        ks_w_b = _draw_key_signature(svg, ks_x_bass, bass_y, key_name, key_mode, "bass")
        ks_offset = max(ks_w_t, ks_w_b)

        # Time signature (first row only)
        if is_first_page and row == 0:
            ts = song_info.get("timeSignature", [4, 4])
            ts_x = PAD_X + 18 + ks_offset + 4
            for sy in [ty, bass_y]:
                svg.append(f'<text x="{ts_x}" y="{sy + S}" font-size="15" '
                           f'font-family="serif" font-weight="bold" fill="#333">{ts[0]}</text>')
                svg.append(f'<text x="{ts_x}" y="{sy + 3 * S}" font-size="15" '
                           f'font-family="serif" font-weight="bold" fill="#333">{ts[1]}</text>')

        left_margin = 58 + ks_offset

        for i, measure in enumerate(rm):
            mi = global_start_measure + row * mpr + i
            mx = PAD_X + left_margin + i * mw
            mw2 = mw - 12
            _staff_lines(svg, mx, ty, mw2)
            _staff_lines(svg, mx, bass_y, mw2)
            _measure_num(svg, mx, bass_y + 5.5 * S, mi + 1)

            chord = measure.get("chord", {})
            cd = chord.get("degree", "") if chord else ""

            tn = measure.get("trebleNotes", [])
            if tn:
                tn_render = [dict(n) for n in tn]
                if cd:
                    tn_render[0] = {**tn_render[0], "chordDeg": cd}
                _draw_measure_notes(svg, tn_render, ty, mx, mw2,
                                    song_info.get("timeSignature", [4, 4])[0], "treble",
                                    show_lyrics=False, key_name=key_name, key_mode=key_mode)

            bn = measure.get("bassNotes", [])
            if bn:
                _draw_measure_notes(svg, bn, bass_y, mx, mw2,
                                    song_info.get("timeSignature", [4, 4])[0], "bass",
                                    show_lyrics=False, key_name=key_name, key_mode=key_mode)

            bx = mx + mw2
            is_last = (row == page_rows - 1 and i == nm_row - 1)
            _bar_line(svg, bx, ty, final=is_last)
            _bar_line(svg, bx, bass_y, final=is_last)

    svg.append('</svg>')
    return '\n'.join(svg)


def render_piano_svg(score_data: dict) -> str:
    measures = score_data.get("measures", [])
    song_info = score_data.get("songInfo", {})
    nm = len(measures)
    if nm == 0:
        return '<p style="color:#999;padding:20px;">暂无乐谱数据</p>'

    mpr = 2 if nm <= 16 else 3
    rows_per_page = 5
    measures_per_page = mpr * rows_per_page
    total_pages = max(1, math.ceil(nm / measures_per_page))

    key_name = song_info.get("key", "C")
    key_mode = song_info.get("keyMode", "major")
    song_name = score_data.get("_songName", "")

    pages = []
    for page in range(total_pages):
        page_measures = measures[page * measures_per_page:(page + 1) * measures_per_page]
        global_start = page * measures_per_page
        svg = _render_piano_page(page_measures, song_info, key_name, key_mode,
                                 page, total_pages, mpr, song_name, global_start)
        pages.append(svg)

    return '<div class="score-pages">' + '\n'.join(pages) + '</div>'


# =============================================================================
# Guitar Tab with strumming
# =============================================================================
STRING_NAMES = ["e", "B", "G", "D", "A", "E"]  # high-to-low (top-to-bottom on TAB: 1弦→6弦)


def _render_guitar_page(page_measures: list, song_info: dict, mpr: int,
                        page: int, total_pages: int, song_name: str,
                        global_start: int, chord_set: dict) -> str:
    """Render one page of guitar TAB."""
    sw = 960
    row_h = 185
    page_rows = max(1, math.ceil(len(page_measures) / mpr))
    tw = sw + PAD_X * 2 + 20

    is_first = (page == 0)
    chord_rows = math.ceil(len(chord_set) / 6) if is_first and chord_set else 0
    chord_h = chord_rows * 76 + 8 if chord_rows > 0 else 0
    content_top = PAD_Y + TITLE_H + 12
    ph = content_top + chord_h + 10 + page_rows * row_h + 28

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {tw} {ph}" width="{tw}" style="max-width:100%;margin-bottom:12px;">']
    svg.append(f'<rect width="100%" height="100%" fill="#fcfcfc" rx="4"/>')

    if is_first:
        _title_bar(svg, tw, "guitar", song_info, song_name)
    else:
        svg.append(f'<rect x="0" y="0" width="{tw}" height="{TITLE_H}" fill="#f8f8f8" rx="4"/>')
        svg.append(f'<line x1="0" y1="{TITLE_H}" x2="{tw}" y2="{TITLE_H}" stroke="#e94560" stroke-width="2"/>')
        svg.append(f'<text x="{PAD_X}" y="{TITLE_H * 0.42}" font-size="14" font-weight="bold" '
                   f'fill="#333" font-family="sans-serif">{song_name}（续）</text>')

    svg.append(f'<text x="{tw - PAD_X}" y="{ph - 10}" font-size="11" fill="#aaa" '
               f'text-anchor="end" font-family="sans-serif">第 {page + 1}/{total_pages} 页</text>')

    # Chord diagrams at top (first page only)
    if is_first and chord_set:
        cnames = list(chord_set.keys())
        for ci, cname in enumerate(cnames):
            info = chord_set[cname]
            col = ci % 6
            crow = ci // 6
            dx = PAD_X + 8 + col * 78
            dy = content_top + crow * 76
            _chord_diagram(svg, dx, dy, cname, info["degree"], info["fingering"])
        content_top += chord_h

    for row in range(page_rows):
        rm = page_measures[row * mpr:(row + 1) * mpr]
        nm = max(1, len(rm))
        mw = sw / nm
        tab_y = content_top + 10 + row * row_h

        for i, measure in enumerate(rm):
            mi = global_start + row * mpr + i
            mx = PAD_X + i * mw
            mw2 = mw - 8
            chord = measure.get("chord", {})
            play_style = measure.get("playStyle", "picking")
            tab_notes = measure.get("tabNotes", [])
            lyric_notes = measure.get("lyricNotes", [])

            chord_y = tab_y + 24
            if chord:
                svg.append(f'<text x="{mx + 4}" y="{chord_y - 6}" text-anchor="start" '
                           f'font-size="14" font-weight="bold" fill="#1a1a2e" font-family="sans-serif">{chord.get("chord", "")}</text>')

            line_top = chord_y + 10
            for s in range(6):
                ly = line_top + s * 9
                sw2_val = 0.7 if s in (0, 5) else 0.4
                svg.append(f'<line x1="{mx}" y1="{ly}" x2="{mx + mw2}" y2="{ly}" '
                           f'stroke="#333" stroke-width="{sw2_val}"/>')

            if is_first and row == 0 and i == 0:
                svg.append(f'<text x="{mx + 3}" y="{line_top - 6}" font-size="10" font-weight="bold" '
                           f'fill="#555" font-family="sans-serif">TAB</text>')
                for s, sn in enumerate(STRING_NAMES):
                    svg.append(f'<text x="{mx + 36}" y="{line_top + s * 9 + 5}" font-size="8" '
                               f'fill="#999" font-family="sans-serif">{sn}</text>')
                ts = song_info.get("timeSignature", [4, 4])
                svg.append(f'<text x="{mx + 54}" y="{line_top + 14}" font-size="13" font-weight="bold" '
                           f'fill="#333">{ts[0]}/{ts[1]}</text>')

            tab_bottom = line_top + 5 * 9
            if tab_notes:
                ts = song_info.get("timeSignature", [4, 4])
                total_beats = (ts[0] // 3) if ts[1] == 8 else ts[0]
                draw_w = mw2 - 10

                if play_style in ("picking", "hybrid"):
                    picking_notes = [(ni, n) for ni, n in enumerate(tab_notes) if n.get("isPicking")]
                    beam_groups, current_group = [], []
                    for pni, (ni, note) in enumerate(picking_notes):
                        if current_group:
                            prev_ni = current_group[-1][0]
                            prev_dur = current_group[-1][1].get("duration", "8")
                            this_dur = note.get("duration", "8")
                            if ni - prev_ni == 1 and prev_dur in ("8", "16") and this_dur in ("8", "16"):
                                current_group.append((ni, note))
                            else:
                                if len(current_group) >= 2:
                                    beam_groups.append(current_group)
                                current_group = [(ni, note)]
                        else:
                            current_group = [(ni, note)]
                    if len(current_group) >= 2:
                        beam_groups.append(current_group)

                    beam_indices = set()
                    for bg in beam_groups:
                        for ni, _ in bg:
                            beam_indices.add(ni)

                    for ni, note in enumerate(tab_notes):
                        beat_offset = note.get("beatOffset", ni * 0.5)
                        nx = mx + 6 + (beat_offset / total_beats) * draw_w
                        if note.get("isPicking"):
                            str_num = note.get("stringNum", 3)
                            str_y = line_top + (str_num - 1) * 9
                            svg.append(f'<text x="{nx}" y="{str_y + 5}" text-anchor="middle" '
                                       f'font-size="12" font-weight="bold" fill="#111" '
                                       f'font-family="sans-serif">×</text>')
                            if ni not in beam_indices:
                                stem_top = line_top - 8
                                stem_end = stem_top - 22
                                svg.append(f'<line x1="{nx}" y1="{stem_top}" x2="{nx}" y2="{stem_end}" '
                                           f'stroke="#333" stroke-width="1.2"/>')
                                dur_str = note.get("duration", "8")
                                if dur_str in ("8", "16"):
                                    n_flags = 1 if dur_str == "8" else 2
                                    for fi in range(n_flags):
                                        fy = stem_end + fi * 3
                                        svg.append(f'<path d="M{nx},{fy} Q{nx+5},{fy+4} {nx+2},{fy+8}" '
                                                   f'fill="none" stroke="#333" stroke-width="1.0"/>')
                        else:
                            direction = note.get("direction", "down")
                            dur = note.get("duration", "8")
                            arr_top, arr_bot = line_top - 4, tab_bottom + 3
                            svg.append(f'<line x1="{nx}" y1="{arr_top}" x2="{nx}" y2="{arr_bot}" '
                                       f'stroke="#333" stroke-width="1.3"/>')
                            if direction == "down":
                                svg.append(f'<polygon points="{nx-4},{arr_top+6} {nx+4},{arr_top+6} {nx},{arr_top}" '
                                           f'fill="#333"/>')
                            else:
                                svg.append(f'<polygon points="{nx-4},{arr_bot-6} {nx+4},{arr_bot-6} {nx},{arr_bot}" '
                                           f'fill="#333"/>')
                            dur_label = {"q": "♩", "8": "♪", "16": "♬"}.get(dur, "")
                            if dur_label:
                                svg.append(f'<text x="{nx+7}" y="{(arr_top+arr_bot)/2+4}" '
                                           f'font-size="10" fill="#555" font-family="serif">{dur_label}</text>')

                    for bg in beam_groups:
                        if len(bg) < 2:
                            continue
                        stem_top = line_top - 8
                        stem_end = stem_top - 22
                        first_nx = mx + 6 + (bg[0][1].get("beatOffset", 0) / total_beats) * draw_w
                        last_nx = mx + 6 + (bg[-1][1].get("beatOffset", 0) / total_beats) * draw_w
                        for ni, note in bg:
                            nx = mx + 6 + (note.get("beatOffset", 0) / total_beats) * draw_w
                            svg.append(f'<line x1="{nx}" y1="{stem_top}" x2="{nx}" y2="{stem_end}" '
                                       f'stroke="#333" stroke-width="1.2"/>')
                        dur_strings = [n.get("duration", "8") for _, n in bg]
                        n_beams = 2 if "16" in dur_strings else 1
                        for bi in range(n_beams):
                            by = stem_end + bi * 4
                            svg.append(f'<rect x="{first_nx}" y="{by - 1.5}" width="{last_nx - first_nx}" height="3" '
                                       f'fill="#333"/>')
                else:
                    for ni, note in enumerate(tab_notes):
                        beat_offset = note.get("beatOffset", 0)
                        nx = mx + 6 + (beat_offset / total_beats) * draw_w
                        direction = note.get("direction", "down")
                        dur = note.get("duration", "8")
                        arr_top, arr_bot = line_top - 4, tab_bottom + 3
                        svg.append(f'<line x1="{nx}" y1="{arr_top}" x2="{nx}" y2="{arr_bot}" '
                                   f'stroke="#333" stroke-width="1.3"/>')
                        if direction == "down":
                            svg.append(f'<polygon points="{nx-4},{arr_top+6} {nx+4},{arr_top+6} {nx},{arr_top}" '
                                       f'fill="#333"/>')
                        else:
                            svg.append(f'<polygon points="{nx-4},{arr_bot-6} {nx+4},{arr_bot-6} {nx},{arr_bot}" '
                                       f'fill="#333"/>')
                        dur_label = {"q": "♩", "8": "♪", "16": "♬"}.get(dur, "")
                        if dur_label:
                            svg.append(f'<text x="{nx+7}" y="{(arr_top+arr_bot)/2+4}" '
                                       f'font-size="10" fill="#555" font-family="serif">{dur_label}</text>')

            jianpu_y = tab_bottom + 16
            lyric_y = jianpu_y + 14

            if lyric_notes:
                ts = song_info.get("timeSignature", [4, 4])
                total_beats = (ts[0] // 3) if ts[1] == 8 else ts[0]
                draw_w = mw2 - 10
                for ln in lyric_notes:
                    beat_offset = ln.get("beatOffset", 0)
                    nx = mx + 6 + (beat_offset / total_beats) * draw_w
                    lyric_text = ln.get("lyric", "")
                    jp = ln.get("jianpu", {})
                    if jp.get("degree"):
                        acc = jp.get("accidental", "")
                        svg.append(f'<text x="{nx}" y="{jianpu_y}" text-anchor="middle" '
                                   f'font-size="12" font-weight="bold" fill="#c0392b" '
                                   f'font-family="sans-serif">{acc}{jp["degree"]}</text>')
                    if lyric_text:
                        svg.append(f'<text x="{nx}" y="{lyric_y}" text-anchor="middle" '
                                   f'font-size="14" fill="#1a1a2e" font-family="sans-serif">{lyric_text}</text>')

            _measure_num(svg, mx, lyric_y + 12, mi + 1)
            bar_x = mx + mw2
            is_last = (row == page_rows - 1 and i == nm - 1)
            _bar_line(svg, bar_x, line_top, n=6, final=is_last, spacing=9)

    svg.append('</svg>')
    return '\n'.join(svg)


def render_guitar_svg(score_data: dict) -> str:
    measures = score_data.get("measures", [])
    if not measures:
        return '<p style="color:#999;padding:20px;">暂无乐谱数据</p>'

    song_info = score_data.get("songInfo", {})
    chord_set = {}
    for m in measures:
        c = m.get("chord", {})
        if c:
            cn = c.get("chord", "")
            if cn and cn not in chord_set:
                chord_set[cn] = {"degree": c.get("degree", ""), "fingering": m.get("chordFingering", [])}

    mpr = 2
    rows_per_page = 4
    measures_per_page = mpr * rows_per_page
    total_pages = max(1, math.ceil(len(measures) / measures_per_page))
    song_name = score_data.get("_songName", "")

    pages = []
    for page in range(total_pages):
        page_measures = measures[page * measures_per_page:(page + 1) * measures_per_page]
        global_start = page * measures_per_page
        svg = _render_guitar_page(page_measures, song_info, mpr, page, total_pages,
                                  song_name, global_start, chord_set)
        pages.append(svg)

    return '<div class="score-pages">' + '\n'.join(pages) + '</div>'

BASS_STRING_NAMES = ["G", "D", "A", "E"]


def _render_bass_page(page_measures: list, song_info: dict, mpr: int,
                       page: int, total_pages: int, song_name: str, global_start: int) -> str:
    """Render one page of bass TAB."""
    sw = 960
    row_h = 105
    page_rows = max(1, math.ceil(len(page_measures) / mpr))
    tw = sw + PAD_X * 2 + 20
    ph = PAD_Y + TITLE_H + 20 + page_rows * row_h + 24

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {tw} {ph}" width="{tw}" style="max-width:100%;margin-bottom:12px;">']
    svg.append(f'<rect width="100%" height="100%" fill="#fcfcfc" rx="4"/>')

    if page == 0:
        _title_bar(svg, tw, "bass", song_info, song_name)
    else:
        svg.append(f'<rect x="0" y="0" width="{tw}" height="{TITLE_H}" fill="#f8f8f8" rx="4"/>')
        svg.append(f'<line x1="0" y1="{TITLE_H}" x2="{tw}" y2="{TITLE_H}" stroke="#e94560" stroke-width="2"/>')
        svg.append(f'<text x="{PAD_X}" y="{TITLE_H * 0.42}" font-size="14" font-weight="bold" '
                   f'fill="#333" font-family="sans-serif">{song_name}（续）</text>')

    svg.append(f'<text x="{tw - PAD_X}" y="{ph - 10}" font-size="11" fill="#aaa" '
               f'text-anchor="end" font-family="sans-serif">第 {page + 1}/{total_pages} 页</text>')

    ct = PAD_Y + TITLE_H + 14

    for row in range(page_rows):
        rm = page_measures[row * mpr:(row + 1) * mpr]
        nm = max(1, len(rm))
        mw = sw / nm
        tab_y = ct + row * row_h

        for i, measure in enumerate(rm):
            mi = global_start + row * mpr + i
            mx = PAD_X + i * mw
            mw2 = mw - 8

            for s in range(4):
                ly = tab_y + 22 + s * 10
                sw2 = 0.8 if s == 0 else 0.4
                svg.append(f'<line x1="{mx}" y1="{ly}" x2="{mx + mw2}" y2="{ly}" '
                           f'stroke="#333" stroke-width="{sw2}"/>')

            if page == 0 and row == 0 and i == 0:
                for s, sn in enumerate(BASS_STRING_NAMES):
                    svg.append(f'<text x="{mx + 3}" y="{tab_y + 22 + s * 10 + 5}" font-size="8" '
                               f'fill="#aaa" font-family="sans-serif">{sn}</text>')

            _measure_num(svg, mx, tab_y + 68, mi + 1)

            tnotes = measure.get("tabNotes", [])
            if tnotes:
                total_beats = sum(dur_to_beats(n.get("duration", "q")) for n in tnotes)
                if total_beats <= 0:
                    ts = song_info.get("timeSignature", [4, 4])
                    total_beats = (ts[0] // 3) if ts[1] == 8 else ts[0]
                draw_w = mw2 - 14
                beat_pos = 0
                prev_x = None
                prev_str = None
                for ni, note in enumerate(tnotes):
                    dur_beats = dur_to_beats(note.get("duration", "q"))
                    nx = mx + 7 + (beat_pos / total_beats) * draw_w
                    beat_pos += dur_beats

                    positions = note.get("positions", [{"str": 1, "fret": 0}])
                    for pos in positions:
                        fret = pos.get("fret", 0)
                        string = pos.get("str", 1)
                        ny = tab_y + 22 + (string - 1) * 10 + 5

                        slide_dist = note.get("slide", 0)
                        if slide_dist and prev_x is not None and string == prev_str:
                            mid_y = ny - 5
                            svg.append(f'<path d="M{prev_x + 5},{mid_y} Q{(prev_x + nx) / 2},{mid_y - 6} {nx - 3},{mid_y}" '
                                       f'fill="none" stroke="#e94560" stroke-width="1.2"/>')
                            svg.append(f'<text x="{(prev_x + nx) / 2}" y="{mid_y - 8}" text-anchor="middle" '
                                       f'font-size="7" fill="#e94560" font-family="sans-serif">sl.</text>')

                        svg.append(f'<text x="{nx}" y="{ny}" text-anchor="middle" font-size="12" '
                                   f'font-weight="bold" fill="#1a1a2e" font-family="sans-serif">{fret}</text>')
                        prev_x = nx
                        prev_str = string

                    jianpu_info = note.get("jianpu", {})
                    jianpu_deg = jianpu_info.get("degree", 0)
                    if jianpu_deg:
                        accidental = jianpu_info.get("accidental", "")
                        jianpu_str = f"{accidental}{jianpu_deg}"
                        jy = tab_y + 8
                        svg.append(f'<text x="{nx}" y="{jy}" text-anchor="middle" '
                                   f'font-size="11" font-weight="bold" fill="#c0392b" '
                                   f'font-family="sans-serif">{jianpu_str}</text>')

            bx = mx + mw2
            is_last = (row == page_rows - 1 and i == nm - 1)
            _bar_line(svg, bx, tab_y + 22, n=4, final=is_last, spacing=10)

    svg.append('</svg>')
    return '\n'.join(svg)


def render_bass_svg(score_data: dict) -> str:
    measures = score_data.get("measures", [])
    if not measures:
        return '<p style="color:#999;padding:20px;">暂无乐谱数据</p>'

    song_info = score_data.get("songInfo", {})
    mpr = 4
    rows_per_page = 6
    measures_per_page = mpr * rows_per_page
    total_pages = max(1, math.ceil(len(measures) / measures_per_page))
    song_name = score_data.get("_songName", "")

    pages = []
    for page in range(total_pages):
        page_measures = measures[page * measures_per_page:(page + 1) * measures_per_page]
        global_start = page * measures_per_page
        svg = _render_bass_page(page_measures, song_info, mpr, page, total_pages, song_name, global_start)
        pages.append(svg)

    return '<div class="score-pages">' + '\n'.join(pages) + '</div>'


# =============================================================================
# Drum Notation
# =============================================================================
def _render_drums_page(page_measures: list, song_info: dict, mpr: int,
                        page: int, total_pages: int, song_name: str, global_start: int) -> str:
    """Render one page of drum notation."""
    sw = 960
    row_h = 110
    page_rows = max(1, math.ceil(len(page_measures) / mpr))
    tw = sw + PAD_X * 2 + 20
    ph = PAD_Y + TITLE_H + 20 + page_rows * row_h + 24

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {tw} {ph}" width="{tw}" style="max-width:100%;margin-bottom:12px;">']
    svg.append(f'<rect width="100%" height="100%" fill="#fcfcfc" rx="4"/>')

    if page == 0:
        _title_bar(svg, tw, "drums", song_info, song_name)
    else:
        svg.append(f'<rect x="0" y="0" width="{tw}" height="{TITLE_H}" fill="#f8f8f8" rx="4"/>')
        svg.append(f'<line x1="0" y1="{TITLE_H}" x2="{tw}" y2="{TITLE_H}" stroke="#e94560" stroke-width="2"/>')
        svg.append(f'<text x="{PAD_X}" y="{TITLE_H * 0.42}" font-size="14" font-weight="bold" '
                   f'fill="#333" font-family="sans-serif">{song_name}（续）</text>')

    svg.append(f'<text x="{tw - PAD_X}" y="{ph - 10}" font-size="11" fill="#aaa" '
               f'text-anchor="end" font-family="sans-serif">第 {page + 1}/{total_pages} 页</text>')

    ct = PAD_Y + TITLE_H + 14

    for row in range(page_rows):
        rm = page_measures[row * mpr:(row + 1) * mpr]
        nm = max(1, len(rm))
        mw = sw / nm
        sy = ct + row * row_h

        for i, measure in enumerate(rm):
            mi = global_start + row * mpr + i
            mx = PAD_X + i * mw
            mw2 = mw - 8
            _staff_lines(svg, mx, sy, mw2)

            if page == 0 and row == 0 and i == 0:
                svg.append(f'<text x="{mx + 5}" y="{sy + 2.2 * S}" font-size="11" '
                           f'font-weight="bold" fill="#666" font-family="sans-serif">DRUM</text>')
                ts = song_info.get("timeSignature", [4, 4])
                svg.append(f'<text x="{mx + 50}" y="{sy + S}" font-size="14" '
                           f'font-family="serif" font-weight="bold" fill="#333">{ts[0]}</text>')
                svg.append(f'<text x="{mx + 50}" y="{sy + 3 * S}" font-size="14" '
                           f'font-family="serif" font-weight="bold" fill="#333">{ts[1]}</text>')
                svg.append(f'<text x="{mx}" y="{sy + 5.8 * S}" font-size="9" fill="#999">'
                           f'● Kick  ● Snare  × HH  △ Crash</text>')

            _measure_num(svg, mx, sy + 5.5 * S, mi + 1)

            dnotes = measure.get("drumNotes", [])
            if dnotes:
                total_beats = sum(dur_to_beats(n.get("duration", "q")) for n in dnotes)
                if total_beats <= 0:
                    ts = song_info.get("timeSignature", [4, 4])
                    total_beats = (ts[0] // 3) if ts[1] == 8 else ts[0]
                draw_w = mw2 - 14
                beat_pos = 0
                for ni, note in enumerate(dnotes):
                    dur_beats = dur_to_beats(note.get("duration", "q"))
                    nx = mx + 7 + (beat_pos / total_beats) * draw_w
                    beat_pos += dur_beats

                    dt = note.get("type", "rest")
                    if dt == "kick":
                        ny = sy + 4 * S
                    elif dt == "snare":
                        ny = sy + 2 * S
                    elif dt in ("hihat", "hihat_closed"):
                        ny = sy - S
                    elif dt == "tom_low":
                        ny = sy + 3.2 * S
                    elif dt == "tom_high":
                        ny = sy + 1.2 * S
                    elif dt == "crash":
                        ny = sy - 1.5 * S
                    elif dt == "ride":
                        ny = sy + 0.5 * S
                    else:
                        continue

                    if dt in ("kick", "tom_low", "tom_high"):
                        svg.append(f'<ellipse cx="{nx}" cy="{ny}" rx="4.5" ry="3.2" fill="#333"/>')
                    elif dt == "snare":
                        svg.append(f'<ellipse cx="{nx}" cy="{ny}" rx="4" ry="3.2" fill="#333"/>')
                    elif dt in ("hihat", "hihat_closed"):
                        svg.append(f'<text x="{nx}" y="{ny + 4}" text-anchor="middle" '
                                   f'font-size="17" fill="#333">×</text>')
                        if dt == "hihat_closed":
                            svg.append(f'<circle cx="{nx}" cy="{ny + 1}" r="4.5" fill="none" stroke="#333" stroke-width="0.8"/>')
                    elif dt == "crash":
                        svg.append(f'<text x="{nx}" y="{ny + 4}" text-anchor="middle" '
                                   f'font-size="18" fill="#333">×</text>')
                        svg.append(f'<circle cx="{nx}" cy="{ny + 1}" r="5" fill="none" stroke="#333" stroke-width="1.0"/>')
                    elif dt == "ride":
                        svg.append(f'<ellipse cx="{nx}" cy="{ny}" rx="4" ry="2.8" fill="#333"/>')

                    stem_up = dt in ("hihat", "hihat_closed", "snare", "crash", "ride")
                    stem_end = ny - S * 3 if stem_up else ny + S * 3
                    svg.append(f'<line x1="{nx + 3.5}" y1="{ny}" x2="{nx + 3.5}" y2="{stem_end}" '
                               f'stroke="#333" stroke-width="1.2"/>')

            bx = mx + mw2
            is_last = (row == page_rows - 1 and i == nm - 1)
            _bar_line(svg, bx, sy, final=is_last)

    svg.append('</svg>')
    return '\n'.join(svg)


def render_drums_svg(score_data: dict) -> str:
    measures = score_data.get("measures", [])
    if not measures:
        return '<p style="color:#999;padding:20px;">暂无乐谱数据</p>'

    song_info = score_data.get("songInfo", {})
    mpr = 4
    rows_per_page = 6
    measures_per_page = mpr * rows_per_page
    total_pages = max(1, math.ceil(len(measures) / measures_per_page))
    song_name = score_data.get("_songName", "")

    pages = []
    for page in range(total_pages):
        page_measures = measures[page * measures_per_page:(page + 1) * measures_per_page]
        global_start = page * measures_per_page
        svg = _render_drums_page(page_measures, song_info, mpr, page, total_pages, song_name, global_start)
        pages.append(svg)

    return '<div class="score-pages">' + '\n'.join(pages) + '</div>'


RENDERERS = {
    "piano": render_piano_svg,
    "guitar": render_guitar_svg,
    "bass": render_bass_svg,
    "drums": render_drums_svg,
}
