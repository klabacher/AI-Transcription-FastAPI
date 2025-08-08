import io
import time
import zipfile
import datetime
from pathlib import Path

def extract_audios_from_zip(zip_bytes: bytes) -> list[dict]:
    """Extrai arquivos de áudio de um arquivo ZIP, preservando o caminho interno."""
    audio_files_info = []
    audio_extensions = ['.ogg', '.mp3', '.m4a', '.wav', '.opus']
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for file_info in z.infolist():
                if not file_info.is_dir() and not file_info.filename.startswith('__MACOSX'):
                    if any(file_info.filename.lower().endswith(ext) for ext in audio_extensions):
                        file_bytes = z.read(file_info.filename)
                        audio_files_info.append({
                            "internal_path": file_info.filename,
                            "file_bytes": file_bytes
                        })
        return audio_files_info
    except zipfile.BadZipFile:
        raise ValueError("Arquivo .ZIP inválido ou corrompido.")


def format_dialogue(utterances: list, use_markdown: bool) -> str:
    """Formata os segmentos de transcrição em diálogos (simples e markdown)."""
    if not utterances:
        return "Nenhuma fala detectada."

    dialogue_lines = []
    for utterance in utterances:
        start_seconds = utterance.get('start', 0)
        if start_seconds is None: start_seconds = 0
        
        if start_seconds > 1000000:
            start_seconds = start_seconds / 1000

        timestamp = str(datetime.timedelta(seconds=int(start_seconds)))
        text = utterance.get('text', '').strip()
        speaker = f"Locutor {utterance.get('speaker')}" if utterance.get('speaker') else "Fala"

        if use_markdown:
            dialogue_lines.append(f"**`[{timestamp}]` {speaker}:** {text}")
        else:
            dialogue_lines.append(f"[{timestamp}] {speaker}: {text}")

    return "\n\n".join(dialogue_lines)

def calculate_eta(job: dict) -> float | None:
    """Calcula o timestamp estimado de conclusão de um job em andamento."""
    status = job.get('status')
    if status != 'processing':
        return None

    progress = job.get('progress', 0)
    started_at = job.get('started_at')
    
    if not started_at or progress <= 5:
        return None

    now = time.time()
    time_elapsed = now - started_at
    
    total_estimated_time = time_elapsed / (progress / 100.0)
    time_remaining = total_estimated_time - time_elapsed
    
    if time_remaining < 0:
        return None
        
    return now + time_remaining