import sys
import os
import time
import traceback
import soundfile as sf

sys.path.append(os.getcwd())

from logging_config import setup_worker_logging_json, get_logger
from engine import load_model_for_worker, transcribe_audio

def send_message(queue, msg_type, job_id, payload):
    try:
        message = {"type": msg_type, "job_id": job_id, "payload": payload}
        queue.put(message)
    except Exception as e:
        logger = get_logger('worker_send_message')
        logger.warning(f"Falha ao enviar mensagem para o gerente: {e}")

def worker_process(task_queue, result_queue, model_id, model_config, device):
    setup_worker_logging_json()
    logger = get_logger(f"Worker-{model_id}")
    
    model = None
    try:
        logger.info(f"Iniciando e carregando modelo '{model_id}' no dispositivo '{device}'...")
        model = load_model_for_worker(model_id, model_config, device)
        logger.info(f"Modelo '{model_id}' carregado. Aguardando jobs...")
    except Exception as e:
        tb = traceback.format_exc()
        logger.critical(f"Falha fatal ao carregar o modelo '{model_id}': {e}")
        send_message(result_queue, "FATAL_ERROR", f"worker_init_failed_{model_id}", {"error": f"Falha ao carregar {model_id}: {e}", "traceback": tb})
        return

    while True:
        try:
            task = task_queue.get()
            if task is None:
                logger.info("Sinal de desligamento recebido. Encerrando o worker.")
                break

            job_id = task.get("job_id")
            audio_path = task.get("audio_path")
            
            logger.info(f"Iniciando processamento do job {job_id[:8]} para o áudio: {os.path.basename(audio_path)}")
            
            duration_seconds = 0
            try:
                info = sf.info(audio_path)
                duration_seconds = info.duration
            except Exception as e:
                logger.warning(f"Não foi possível ler a duração do áudio '{audio_path}': {e}")

            transcription_generator = transcribe_audio(model, model_config, audio_path, duration_seconds)
            
            final_result = None
            for progress_or_result in transcription_generator:
                if isinstance(progress_or_result, int):
                    send_message(result_queue, "PROGRESS", job_id, progress_or_result)
                else:
                    final_result = progress_or_result
            
            if final_result:
                send_message(result_queue, "RESULT", job_id, final_result)
                logger.info(f"Job {job_id[:8]} finalizado com sucesso.")

        except Exception as e:
            tb = traceback.format_exc()
            logger.critical(f"Erro fatal durante o processamento do job {job_id[:8]}: {e}")
            send_message(result_queue, "FATAL_ERROR", job_id, {"error": str(e), "traceback": tb})
        
        finally:
            if 'audio_path' in locals() and audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logger.debug(f"Arquivo de áudio temporário '{audio_path}' removido.")
                except OSError as e:
                    logger.error(f"Erro ao remover arquivo temporário '{audio_path}': {e}")