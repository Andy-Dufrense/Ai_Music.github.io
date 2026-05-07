"""
Shared constants and utility functions used across all service modules.
Centralizes CHROMATIC, parse_vex_key, midi_to_note_name, duration helpers.
"""

# ── Constants ──────────────────────────────────────────────────────────

CHROMATIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Flat-to-sharp mapping for note name normalization
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
                  "db": "c#", "eb": "d#", "gb": "f#", "ab": "g#", "bb": "a#"}


def note_to_chromatic_index(note_name: str) -> int:
    """Convert any note name (sharp or flat) to its CHROMATIC index.
    E.g. 'Eb' → 3, 'Ab' → 8, 'C#' → 1."""
    normalized = _FLAT_TO_SHARP.get(note_name, note_name)
    try:
        return CHROMATIC.index(normalized)
    except ValueError:
        return 0  # Default to C


def note_name_normalized(note_name: str) -> str:
    """Convert flat note names to their CHROMATIC-compatible sharp equivalents."""
    return _FLAT_TO_SHARP.get(note_name, note_name)

NAME_MAP = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

DUR_TO_QL = {
    ("w", 0): 4.0,   ("w", 1): 6.0,
    ("h", 0): 2.0,   ("h", 1): 3.0,
    ("q", 0): 1.0,   ("q", 1): 1.5,
    ("8", 0): 0.5,   ("8", 1): 0.75,
    ("16", 0): 0.25, ("16", 1): 0.375,
}

DUR_BEATS = {
    "w": 4.0, "h": 2.0, "q": 1.0, "8": 0.5, "16": 0.25,
    "qr": 1.0, "hr": 2.0, "wr": 4.0,
}

REST_DURS = {"qr", "hr", "wr", "8r", "16r"}

DEFAULT_OCTAVE = 4
FALLBACK_MIDI = 60

# ── Note name / VexFlow key parsing ───────────────────────────────────

def parse_vex_key(key_str: str) -> int:
    """Convert VexFlow key string like 'c/4', 'f#/3', 'bb/2', 'c##/5' to MIDI number.

    Handles double sharps (##), double flats (bb), missing octave, missing separator.
    Returns FALLBACK_MIDI (60) on parse failure.
    """
    key_str = key_str.strip().lower()
    if not key_str:
        return FALLBACK_MIDI

    if "/" in key_str:
        name_part, octave_str = key_str.split("/", 1)
    else:
        name_part, octave_str = key_str, str(DEFAULT_OCTAVE)

    # Allow things like "c#" with implicit octave (e.g. "c#" alone)
    if not octave_str:
        octave_str = str(DEFAULT_OCTAVE)

    base = NAME_MAP.get(name_part[0], 0)

    acc = 0
    i = 1
    while i < len(name_part):
        ch = name_part[i]
        if ch == "#":
            acc += 1
        elif ch == "b":
            acc -= 1
        else:
            break
        i += 1

    try:
        octv = int(octave_str)
    except (ValueError, TypeError):
        octv = DEFAULT_OCTAVE

    return (octv + 1) * 12 + base + acc


def midi_to_vexnote(midi_val: float) -> str:
    """Convert MIDI number to VexFlow note format: 'c/4', 'f#/3'."""
    midi_int = int(round(midi_val))
    octave = midi_int // 12 - 1
    name = CHROMATIC[midi_int % 12].lower()
    return f"{name}/{octave}"


def midi_to_note_name(midi_val: float) -> str:
    """Convert MIDI number to note name like 'C4', 'F#3', 'Bb2'."""
    midi_int = int(round(midi_val))
    name = CHROMATIC[midi_int % 12]
    octave = midi_int // 12 - 1
    return f"{name}{octave}"


# Diatonic pitch class → staff step (C=0, D=1, E=2, F=3, G=4, A=5, B=6)
_DIATONIC_STEP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5, 11: 6}


def midi_to_staff_pos(midi_val: float, clef: str = "treble") -> float:
    """Convert MIDI number to staff position (0 = middle line B4/D3).
    Uses proper diatonic mapping: 7 staff steps per octave, not 12.
    Treble: C4=-6, E4=-4(bottom), B4=0(mid), F5=4(top).
    Bass:   G2=-4(bottom), D3=0(mid), A3=4(top)."""
    midi = int(round(midi_val))
    pc = midi % 12
    octave = midi // 12 - 1  # MIDI octave convention
    diatonic_total = _DIATONIC_STEP[pc] + 7 * octave
    if clef == "treble":
        return float(diatonic_total - 34)  # C4(28)→-6, E4(30)→-4, B4(34)→0
    else:
        return float(diatonic_total - 22)  # D3(22)→0, G2(18)→-4


# ── Duration helpers ──────────────────────────────────────────────────

def dur_to_quarter_length(dur: str, dots: int = 0) -> float:
    """Convert duration string + dots to music21 quarterLength."""
    key = (dur, dots)
    if key in DUR_TO_QL:
        return DUR_TO_QL[key]
    rest_map = {
        "qr": ("q", 0), "hr": ("h", 0), "wr": ("w", 0),
        "8r": ("8", 0), "16r": ("16", 0),
    }
    if dur in rest_map:
        return dur_to_quarter_length(*rest_map[dur])
    return 1.0


def is_rest(dur: str) -> bool:
    """Check if a duration string represents a rest."""
    return dur in REST_DURS or dur.endswith("r")


def dur_to_beats(dur: str) -> float:
    """Get beat count for a duration string (e.g. 'q' → 1.0, '8' → 0.5, 'h.' → 3.0)."""
    dur_clean = dur.rstrip(".")
    beats = DUR_BEATS.get(dur_clean, 1.0)
    if dur.endswith("."):
        beats *= 1.5
    return beats


# ── Chinese text processing ──────────────────────────────────────────

# Traditional-to-Simplified Chinese mapping
T2S_MAP = {
    "噹": "当", "們": "们", "爲": "为", "愛": "爱", "時": "时",
    "來": "来", "後": "后", "開": "开", "關": "关", "對": "对", "會": "会",
    "說": "说", "話": "话", "過": "过", "著": "着", "讓": "让", "個": "个",
    "進": "进", "學": "学", "實": "实", "體": "体", "現": "现",
    "裏": "里", "麵": "面", "這": "这", "還": "还", "沒": "没", "難": "难",
    "問": "问", "間": "间", "門": "门", "聽": "听", "聲": "声", "點": "点",
    "頭": "头", "為": "为", "總": "总", "從": "从", "見": "见", "覺": "觉",
    "樣": "样", "麼": "么", "嗎": "吗", "吧": "吧", "給": "给", "與": "与",
    "妳": "你", "才": "才", "臺": "台", "灣": "湾", "國": "国", "萬": "万",
    "電": "电", "風": "风", "飛": "飞", "馬": "马", "魚": "鱼",
    "鳥": "鸟", "龍": "龙", "動": "动", "華": "华", "經": "经", "種": "种",
    "節": "节", "處": "处", "號": "号", "園": "园", "場": "场", "塊": "块",
    "莊": "庄", "葉": "叶", "藥": "药", "術": "术", "衛": "卫", "製": "制",
    "複": "复", "儘": "尽", "僅": "仅", "傳": "传", "夠": "够", "夢": "梦",
    "遠": "远", "長": "长", "當": "当", "應": "应", "該": "该", "認": "认",
    "識": "识", "誰": "谁", "謝": "谢", "請": "请", "記": "记", "張": "张",
    "帶": "带", "氣": "气", "東": "东", "紅": "红", "綠": "绿", "藍": "蓝",
    "黃": "黄", "黑": "黑", "白": "白", "變": "变", "舊": "旧", "雙": "双",
    "數": "数", "歲": "岁", "幾": "几", "邊": "边", "許": "许", "錯": "错",
    "錢": "钱", "鐵": "铁", "鐘": "钟", "雲": "云", "靈": "灵", "靜": "静",
    "輕": "轻", "亂": "乱", "爾": "尔", "塵": "尘", "憂": "忧", "戲": "戏",
    "戰": "战", "據": "据", "斷": "断", "顯": "显", "顧": "顾", "題": "题",
    "驗": "验", "驚": "惊", "髮": "发", "衹": "只", "隻": "只",
    "餘": "余", "鬆": "松", "瞭解": "了解",
}


def to_simplified(text: str) -> str:
    """Convert traditional Chinese characters to simplified."""
    for old, new in sorted(T2S_MAP.items(), key=lambda x: -len(x[0])):
        if len(old) > 1:
            text = text.replace(old, new)
    return "".join(T2S_MAP.get(ch, ch) for ch in text)
