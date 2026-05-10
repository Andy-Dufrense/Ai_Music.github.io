import subprocess
import os
import shutil
from pathlib import Path

from config import OUTPUT_DIR


def separate_stems(input_path: str, job_id: str) -> dict:
    """
    Use demucs (6-stem model) to separate audio into:
    vocals, drums, bass, guitar, piano, other.
    Returns dict mapping stem name to file path.
    """
    job_output_dir = os.path.join(OUTPUT_DIR, job_id, "stems")
    os.makedirs(job_output_dir, exist_ok=True)

    cmd = [
        "python", "-m", "demucs",
        "-o", job_output_dir,
        "-n", "htdemucs_6s",
        "--device", "cpu",
        "--shifts", "0",
        "--overlap", "0.1",
        input_path,
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)

    base_name = Path(input_path).stem
    stems_source = os.path.join(job_output_dir, "htdemucs_6s", base_name)

    # 6-stem model outputs: vocals, drums, bass, guitar, piano, other
    stem_names = ["vocals", "drums", "bass", "guitar", "piano", "other"]
    stems = {}
    for stem_name in stem_names:
        src = os.path.join(stems_source, f"{stem_name}.wav")
        if os.path.exists(src):
            dst_dir = os.path.join(job_output_dir, stem_name)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, f"{stem_name}.wav")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            stems[stem_name] = dst

    return stems
