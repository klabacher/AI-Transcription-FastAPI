import io
import zipfile
import datetime
import time
from typing import List, Dict, Any, Optional


def extract_audios_from_zip(zip_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extracts audio files from a zip archive provided as bytes.

    Args:
        zip_bytes: The byte content of the zip file.

    Returns:
        A list of dictionaries, where each dictionary contains the
        internal path and byte content of an audio file.
    """
    audio_files_info = []
    audio_extensions = [".ogg", ".mp3", ".m4a", ".wav", ".opus"]
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for file_info in z.infolist():
            if file_info.is_dir() or file_info.filename.startswith("__MACOSX"):
                continue

            if any(
                file_info.filename.lower().endswith(ext) for ext in audio_extensions
            ):
                audio_files_info.append(
                    {
                        "internal_path": file_info.filename,
                        "file_bytes": z.read(file_info.filename),
                    }
                )
    return audio_files_info


def format_dialogue(utterances: List[Dict[str, Any]], use_markdown: bool = True) -> str:
    """
    Formats a list of transcription utterances into a dialogue string.

    Args:
        utterances: A list of utterance dictionaries, each with 'start' and 'text'.
        use_markdown: If True, formats the output with Markdown for emphasis.

    Returns:
        A formatted string representing the dialogue.
    """
    if not utterances:
        return "No speech detected."

    lines = []
    for u in utterances:
        start_seconds = u.get("start", 0) or 0
        # Safety check: some libraries might return milliseconds instead of seconds
        if start_seconds > 1_000_000:
            start_seconds /= 1000

        timestamp = str(datetime.timedelta(seconds=int(start_seconds)))
        text = u.get("text", "").strip()
        speaker = f"Speaker {u.get('speaker')}" if u.get("speaker") else "Speech"

        if use_markdown:
            lines.append(f"**`[{timestamp}]` {speaker}:** {text}")
        else:
            lines.append(f"[{timestamp}] {speaker}: {text}")

    return "\n\n".join(lines)


def calculate_eta(job: Dict[str, Any]) -> Optional[float]:
    """
    Calculates the Estimated Time of Arrival (ETA) for a processing job.

    Args:
        job: A dictionary representing the job, containing status, progress, and start time.

    Returns:
        The estimated completion timestamp, or None if ETA cannot be determined.
    """
    if job.get("status") != "processing":
        return None

    progress = job.get("progress", 0)
    started_at = job.get("started_at")

    if not started_at or progress <= 5:
        return None

    now = time.time()
    elapsed = now - started_at
    total_estimated_time = elapsed / (progress / 100.0)
    remaining_time = total_estimated_time - elapsed

    if remaining_time <= 0:
        return None

    return now + remaining_time
