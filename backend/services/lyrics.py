"""
Lyrics-to-melody alignment using time-based matching.
Each character's Whisper timestamp is matched to the nearest melody note,
so lyrics appear at the correct time in the song — not proportionally distributed.
"""
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
    Align lyrics characters to melody notes using time-based matching (v4).

    Each character matches the closest melody note by time.
    Conflicts (multiple chars → same note) are resolved by keeping the
    best match and pushing others to adjacent empty notes.
    Melisma extends characters across consecutive empty notes.
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
    assigned_lyric = [""] * len(melody_notes)
    assigned_char_indices = set()

    # Build note index sorted by time
    note_order = sorted(range(len(melody_notes)), key=lambda i: melody_notes[i]["time"])

    # Step 1: Assign each character to its closest note by time
    char_to_best_note = []  # (char_idx, note_idx, time_dist)
    for ci, char in enumerate(chars):
        char_t = (char["start"] + char["end"]) / 2  # Center time
        best_ni = -1
        best_dist = float("inf")
        for ni in note_order:
            note_t = melody_notes[ni]["time"]
            d = abs(note_t - char_t)
            if d < best_dist:
                best_dist = d
                best_ni = ni
            elif note_t > char_t + beat_dur * 4:
                break  # Too far ahead, stop searching
        if best_ni >= 0 and best_dist < beat_dur * 2:
            char_to_best_note.append((ci, best_ni, best_dist))

    # Step 2: Resolve conflicts — one char per note, closest match wins
    # Group by target note
    by_note = {}
    for ci, ni, dist in char_to_best_note:
        by_note.setdefault(ni, []).append((ci, dist))
        by_note[ni].sort(key=lambda x: x[1])  # closest first

    used_note_indices = set()
    for ni, candidates in sorted(by_note.items(), key=lambda x: x[1][0][1]):
        # Best match gets the note
        ci, dist = candidates[0]
        assigned_lyric[ni] = chars[ci]["word"]
        assigned_char_indices.add(ci)
        used_note_indices.add(ni)

        # Push remaining candidates to adjacent empty notes
        for ci2, _ in candidates[1:]:
            # Search forward/backward for an empty note
            placed = False
            for offset in range(1, 8):
                for sign in [1, -1]:
                    candidate_ni = ni + sign * offset
                    if 0 <= candidate_ni < len(melody_notes):
                        if (not assigned_lyric[candidate_ni] and
                            candidate_ni not in used_note_indices):
                            assigned_lyric[candidate_ni] = chars[ci2]["word"]
                            assigned_char_indices.add(ci2)
                            used_note_indices.add(candidate_ni)
                            placed = True
                            break
                if placed:
                    break

    # Step 3: Assign any remaining unassigned characters to empty notes
    remaining = [(ci, chars[ci]) for ci in range(len(chars))
                 if ci not in assigned_char_indices]
    if remaining:
        empty = [ni for ni in note_order
                 if not assigned_lyric[ni] and ni not in used_note_indices]
        empty.sort(key=lambda ni: -melody_notes[ni]["duration"])
        for ri, (char_i, char_obj) in enumerate(remaining):
            if ri < len(empty):
                assigned_lyric[empty[ri]] = char_obj["word"]

    # Step 4: Melisma — extend characters across consecutive empty notes
    last_assigned = None
    for ni in note_order:
        if assigned_lyric[ni]:
            last_assigned = ni
            continue
        if last_assigned is not None:
            gap = melody_notes[ni]["time"] - melody_notes[last_assigned]["end_time"]
            if gap < beat_dur * 0.8:
                assigned_lyric[ni] = assigned_lyric[last_assigned]

    # Apply lyrics to notes
    for i in range(len(melody_notes)):
        melody_notes[i]["lyric"] = assigned_lyric[i] if i < len(assigned_lyric) else ""

    return melody_notes
