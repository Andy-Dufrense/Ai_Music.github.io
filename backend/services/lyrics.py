"""
Lyrics-to-melody alignment using melody-centric distribution.
Melody notes (from torchcrepe) have reliable timing; Whisper word timestamps
are approximate for singing, so we distribute characters proportionally
across notes based on note density and duration.
"""
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


def align_lyrics_to_notes(lyrics_data: dict, melody_notes: list, bpm: float = 120.0) -> list:
    """
    Align lyrics characters to melody notes using melody-centric distribution.

    Strategy (v3):
    - Melody note timing is more reliable than Whisper word timestamps
    - Distribute characters proportionally across notes by note duration
    - First character → first note, last character → last note
    - Longer notes get more characters (dense passages)
    - Melisma: extend characters across adjacent empty notes
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

    # Sort notes by time, filter out notes that are too short (likely noise)
    valid_notes = [(i, n) for i, n in enumerate(melody_notes)
                   if n.get("duration", 0) >= 0.04]
    if not valid_notes:
        for n in melody_notes:
            n["lyric"] = ""
        return melody_notes

    valid_notes.sort(key=lambda x: x[1]["time"])
    n_notes = len(valid_notes)
    n_chars = len(chars)

    # Compute note weights: longer notes get proportionally more characters
    durations = np.array([n["duration"] for _, n in valid_notes], dtype=float)
    total_dur = durations.sum()
    if total_dur <= 0:
        durations = np.ones(n_notes)
        total_dur = n_notes

    # Target number of characters per note (fractional)
    char_quota = durations * (n_chars / total_dur)

    # Greedy assignment: assign whole characters to notes
    assigned_lyric = [""] * len(melody_notes)
    char_idx = 0
    accumulated = 0.0

    for note_pos, (orig_idx, note) in enumerate(valid_notes):
        accumulated += char_quota[note_pos]
        n_assign = int(round(accumulated))
        if n_assign > 0 and char_idx < n_chars:
            # Assign characters to this note
            chars_for_note = chars[char_idx:char_idx + n_assign]
            # Take the first character as primary, rest can be secondary
            if chars_for_note:
                assigned_lyric[orig_idx] = chars_for_note[0]["word"]
            char_idx += n_assign
            accumulated -= n_assign
        # If no char assigned but we have leftover accumulation, carry forward
        if char_idx >= n_chars:
            break

    # Distribute any remaining characters to remaining notes
    remaining_chars = chars[char_idx:]
    remaining_notes = [(orig_idx, note) for orig_idx, note in valid_notes
                       if not assigned_lyric[orig_idx]]
    if remaining_chars and remaining_notes:
        # Distribute evenly
        chars_per_note = max(1, len(remaining_chars) // len(remaining_notes))
        ci = 0
        for orig_idx, note in remaining_notes:
            if ci < len(remaining_chars):
                assigned_lyric[orig_idx] = remaining_chars[ci]["word"]
                ci += chars_per_note

    # If we still have leftover chars, assign to the last notes
    leftover = [ci for ci in range(len(chars)) if ci not in
                [i for i in range(len(chars)) if any(
                    assigned_lyric[ni] == chars[i]["word"] for ni in range(len(melody_notes)))]]
    if leftover:
        # Assign leftover to longest empty notes
        empty_by_dur = sorted(
            [(i, melody_notes[i]["duration"]) for i in range(len(melody_notes))
             if not assigned_lyric[i]],
            key=lambda x: -x[1]
        )
        for li, char_i in enumerate(leftover):
            if li < len(empty_by_dur):
                assigned_lyric[empty_by_dur[li][0]] = chars[char_i]["word"]

    # Melisma: extend each character across consecutive empty notes
    # (notes without their own lyrics, within a small time gap)
    last_assigned_note = None
    for ni in sorted(range(len(melody_notes)), key=lambda i: melody_notes[i]["time"]):
        if assigned_lyric[ni]:
            last_assigned_note = ni
            continue
        if last_assigned_note is not None:
            gap = melody_notes[ni]["time"] - melody_notes[last_assigned_note]["end_time"]
            if gap < beat_dur * 0.8:
                assigned_lyric[ni] = assigned_lyric[last_assigned_note]

    # Apply lyrics to notes
    for i in range(len(melody_notes)):
        melody_notes[i]["lyric"] = assigned_lyric[i] if i < len(assigned_lyric) else ""

    return melody_notes
