import os
import tempfile
import json
import logging
import soundfile as sf
from pathlib import Path
from logging_config import setup_worker_logging_json, get_logger
from engine import load_model, transcribe_with_faster, transcribe_with_hf_pipeline


def worker_main(job: dict, audio_path: str, conn):
    """
    job: dicionário com configuração
    audio_path: caminho do arquivo temporário
    conn: Connection (multiprocessing.Pipe) para enviar mensagens ao gerente
    Mensagens: {"type": "PROGRESS"/"RESULT"/"FATAL_ERROR", "payload": ...}
    """
    setup_worker_logging_json()
    logger = get_logger('worker')

    try:
        logger.info(f"Worker iniciando job {job['id'][:8]} | arquivo: {audio_path}")

        duration = 0.0
        try:
            info = sf.info(audio_path)
            duration = float(info.duration or 0.0)
            logger.debug(f"Duração detectada: {duration}s")
        except Exception as e:
            logger.warning(f"Não foi possível ler metadados do áudio: {e}")

        model_cfg = job['config']['model_config']
        device = job['config'].get('device') or job['config'].get('device_choice') or 'cpu'

        model = load_model(job['config']['model_id'], model_cfg, device)

        impl = model_cfg['impl']
        if impl == 'faster':
            gen = transcribe_with_faster(model, audio_path, job['config']['language'], duration)
        else:
            gen = transcribe_with_hf_pipeline(model, audio_path, job['config']['language'])

        collected_segments = []
        text_result = ''

        for item in gen:
            if isinstance(item, int):
                try:
                    conn.send({'type': 'PROGRESS', 'payload': int(item)})
                except Exception:
                    logger.warning('Falha ao enviar PROGRESS; conexao pode estar fechada')
            elif isinstance(item, dict) and 'segment' in item:
                seg = item['segment']
                collected_segments.append({'start': seg.get('start', 0), 'text': seg.get('text', '').strip()})
            elif isinstance(item, dict) and 'text' in item:
                text_result = item.get('text', '').strip()
                # If segments present in final item use them
                collected_segments = item.get('segments', collected_segments)

        # Resultado final
        conn.send({'type': 'RESULT', 'payload': {'text': text_result, 'segments': collected_segments, 'entities': []}})

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.critical(f"Worker fatal: {e}\n{tb}")
        try:
            conn.send({'type': 'FATAL_ERROR', 'payload': {'error': str(e), 'traceback': tb}})
        except Exception:
            logger.error('Não foi possível enviar FATAL_ERROR por pipe.')
    finally:
        try:
            conn.close()
        except Exception:
            pass
