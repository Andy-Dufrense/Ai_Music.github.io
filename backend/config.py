import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TEMPLATE_DIR = os.path.join(os.path.dirname(BASE_DIR), "templates")

for d in [UPLOAD_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

NOTATION_TYPES = ["piano", "guitar", "bass", "drums"]

STEM_NAMES = {
    "vocals": "人声",
    "drums": "鼓",
    "bass": "贝斯",
    "guitar": "吉他",
    "other": "其他乐器",
}
