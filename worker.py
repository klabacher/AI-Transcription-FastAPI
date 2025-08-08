import sys
import os
import io
import json
import soundfile as sf
from pathlib import Path
import tempfile

sys.path.append(os.getcwd())

from logging_config import setup_worker_logging, get_logger
from engine import run_local_transcription, run_assemblyai_transcription

def worker_task(json_job_data, audio_bytes):
    """
    Esta é a função que roda no processo isolado (o "bunker").
    Ela agora se comunica via JSON e envia logs para stderr.
    """
    setup_worker_logging()
    logger = get_logger("worker")

    try:
        job = json.loads(json_job_data)
        logger.info(f"Worker (PID: {os.getpid()}) iniciado para o job {job['id'][:8]}.")
        internal_path = job['internal_path']
        
        duration_seconds = 0
        try:
            logger.debug("Lendo metadados do áudio...")
            audio_info = sf.info(io.BytesIO(audio_bytes))
            duration_seconds = audio_info.duration
            logger.debug(f"Duração do áudio detectada: {duration_seconds:.2f}s")
        except Exception:
            logger.warning("Não foi possível ler a duração do áudio.")
            pass

        suffix = Path(internal_path).suffix
        tmp_audio_path = None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_audio_path = tmp.name
        
        logger.debug(f"Arquivo de áudio temporário criado em: {tmp_audio_path}")

        impl = job['config']['model_config']['impl']
        if impl == 'assemblyai':
            logger.info("Roteando para a engine da AssemblyAI.")
            result = run_assemblyai_transcription(job, tmp_audio_path)
        else:
            logger.info(f"Roteando para a engine local: {impl}")
            result = run_local_transcription(job, tmp_audio_path, duration_seconds)

        os.remove(tmp_audio_path)
        logger.debug("Arquivo de áudio temporário removido.")

        success_response = json.dumps({"status": "completed", "result": result})
        sys.stdout.write(success_response)
        sys.stdout.flush()
        logger.info("Resultado de sucesso enviado para o gerente.")

    except Exception as e:
        import traceback
        logger.error(f"Worker falhou com uma exceção: {e}")
        error_info = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        error_response = json.dumps(error_info)
        sys.stderr.write(error_response + '\n') # Adiciona newline para separar dos logs
        sys.stderr.flush()

if __name__ == "__main__":
    job_data_size_bytes = sys.stdin.buffer.read(4)
    job_data_size = int.from_bytes(job_data_size_bytes, 'big')
    json_job_data = sys.stdin.buffer.read(job_data_size).decode('utf-8')

    audio_data_size_bytes = sys.stdin.buffer.read(4)
    audio_data_size = int.from_bytes(audio_data_size_bytes, 'big')
    audio_bytes = sys.stdin.buffer.read(audio_data_size)
    
    worker_task(json_job_data, audio_bytes)