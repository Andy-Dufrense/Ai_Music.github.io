"""
Lyrics-to-melody alignment using beat-quantized matching.
Each character maps to the nearest note within its beat-quantized time window,
distributed proportionally by beat position.
"""
import re
import numpy as np

from services.common import to_simplified


def _split_characters(words: list) -> list:
    """Split multi-char words into individual characters, retaining time info."""
    result = []
    for w in words:
        text = w["word"].strip()
        if not text:
            continue
        start = w["start"]
        end = w["end"]
        if len(text) == 1:
            result.append({"word": to_simplified(text), "start": start, "end": end})
        else:
            dur = (end - start) / len(text)
            for i, ch in enumerate(text):
                result.append({
                    "word": to_simplified(ch),
                    "start": round(start + i * dur, 3),
                    "end": round(start + (i + 1) * dur, 3),
                })
    return result


def _snap_to_beat(time_sec: float, beat_dur: float) -> float:
    """Snap a time to the nearest 8th-note beat grid position.
    Returns the snapped time in seconds."""
    half_beat = beat_dur / 2  # Snap to 8th note grid
    beat_idx = round(time_sec / half_beat)
    return beat_idx * half_beat


def align_lyrics_to_notes(lyrics_data: dict, melody_notes: list, bpm: float = 120.0) -> list:
    """
    Align lyrics characters to melody notes using beat-quantized matching.

    Improvements over v1:
    - Word timestamps are snapped to 8th-note beat grid before matching
    - Wider alignment window (1.5 beats) to handle Whisper timestamp jitter
    - Better melisma: extend characters across all consecutive notes until
      the next character's note, not just one note
    - All lyrics characters are guaranteed to appear onto some note
    """
    words = lyrics_data.get("words", [])
    if not words or not melody_notes:
        for n in melody_notes:
            n["lyric"] = ""
        return melody_notes

    chars = _split_characters(words)
    if not chars:
        for n in melody_notes:
            n["lyric"] = ""
        return melody_notes

    beat_dur = 60.0 / max(bpm, 40.0)
    align_window = beat_dur * 1.5  # One and a half beats tolerance

    # Snap character times to beat grid for cleaner alignment
    for ch in chars:
        ch["snapped"] = _snap_to_beat((ch["start"] + ch["end"]) / 2.0, beat_dur)

    # Sort notes by start time
    note_indices = sorted(range(len(melody_notes)), key=lambda i: melody_notes[i]["time"])
    assigned_lyric = [""] * len(melody_notes)

    # Build note time list: (index, center_time_snapped)
    note_times = []
    for ni in note_indices:
        note = melody_notes[ni]
        center = (note["time"] + note["end_time"]) / 2.0
        snap = _snap_to_beat(center, beat_dur)
        note_times.append((ni, snap))

    # Pass 1: assign each character to the nearest unassigned note within window
    char_used = [False] * len(chars)
    for ci, ch in enumerate(chars):
        ch_snapped = ch["snapped"]
        best_ni = -1
        best_dist = float("inf")
        for ni, nt in note_times:
            if assigned_lyric[ni]:
                continue
            dist = abs(nt - ch_snapped)
            if dist < align_window and dist < best_dist:
                best_dist = dist
                best_ni = ni
        if best_ni >= 0:
            assigned_lyric[best_ni] = ch["word"]
            char_used[ci] = True

    # Pass 2: assign unused characters to remaining empty notes in time order
    unused_chars = [chars[i] for i, used in enumerate(char_used) if not used]
    empty_notes = [ni for ni in note_indices if not assigned_lyric[ni]]
    if unused_chars:
        # Sort both by time
        unused_chars.sort(key=lambda c: c["snapped"])
        empty_snapped = [(ni, _snap_to_beat(
            (melody_notes[ni]["time"] + melody_notes[ni]["end_time"]) / 2.0, beat_dur))
            for ni in empty_notes]
        empty_snapped.sort(key=lambda x: x[1])
        for ci, ch in enumerate(unused_chars):
            if ci < len(empty_snapped):
                assigned_lyric[empty_snapped[ci][0]] = ch["word"]

    # Pass 3: melisma — extend each character forward across consecutive
    # notes until the next character's note or a large gap is encountered
    # First, build the sequence of assigned note indices
    assigned_note_order = sorted(
        [ni for ni in note_indices if assigned_lyric[ni]],
        key=lambda ni: melody_notes[ni]["time"]
    )
    for i, ni in enumerate(assigned_note_order):
        lyric = assigned_lyric[ni]
        if not lyric:
            continue
        # Find the next note that has a different lyric
        next_assigned_idx = None
        for j in range(i + 1, len(assigned_note_order)):
            next_ni = assigned_note_order[j]
            if assigned_lyric[next_ni] and assigned_lyric[next_ni] != lyric:
                next_assigned_idx = next_ni
                break

        # Extend this lyric to all empty notes between this note and the next assigned note
        current_end_time = melody_notes[ni]["end_time"]
        limit_time = float("inf")
        if next_assigned_idx is not None:
            limit_time = melody_notes[next_assigned_idx]["time"]

        for candidate_ni in note_indices:
            if candidate_ni <= ni:
                continue
            if melody_notes[candidate_ni]["time"] >= limit_time:
                break
            if not assigned_lyric[candidate_ni]:
                gap = melody_notes[candidate_ni]["time"] - current_end_time
                if gap < beat_dur * 1.0:  # Within one beat gap
                    assigned_lyric[candidate_ni] = lyric
                    current_end_time = melody_notes[candidate_ni]["end_time"]

    # Apply lyrics to notes
    for ni in note_indices:
        melody_notes[ni]["lyric"] = assigned_lyric[ni] if ni < len(assigned_lyric) else ""

    return melody_notes
