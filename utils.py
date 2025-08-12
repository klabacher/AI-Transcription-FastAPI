import io
import zipfile
from pathlib import Path
import datetime
import time


def extract_audios_from_zip(zip_bytes: bytes) -> list[dict]:
    audio_files_info = []
    audio_extensions = ['.ogg', '.mp3', '.m4a', '.wav', '.opus']
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for fi in z.infolist():
            if fi.is_dir():
                continue
            if fi.filename.startswith('__MACOSX'):
                continue
            if any(fi.filename.lower().endswith(ext) for ext in audio_extensions):
                audio_files_info.append({"internal_path": fi.filename, "file_bytes": z.read(fi.filename)})
    return audio_files_info


def format_dialogue(utterances: list, use_markdown: bool = True) -> str:
    if not utterances:
        return "Nenhuma fala detectada."
    lines = []
    for u in utterances:
        start = u.get('start', 0) or 0
        # safety: some libs give ms instead of s
        if start > 1_000_000:
            start = start / 1000
        timestamp = str(datetime.timedelta(seconds=int(start)))
        text = u.get('text', '').strip()
        speaker = f"Locutor {u.get('speaker')}" if u.get('speaker') else "Fala"
        if use_markdown:
            lines.append(f"**`[{timestamp}]` {speaker}:** {text}")
        else:
            lines.append(f"[{timestamp}] {speaker}: {text}")
    return "\n\n".join(lines)


def calculate_eta(job: dict) -> float | None:
    if job.get('status') != 'processing':
        return None
    progress = job.get('progress', 0)
    started = job.get('started_at')
    if not started or progress <= 5:
        return None
    now = time.time()
    elapsed = now - started
    total_est = elapsed / (progress / 100.0)
    remain = total_est - elapsed
    if remain <= 0:
        return None
    return now + remain
