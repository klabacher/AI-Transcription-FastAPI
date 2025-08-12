import os
import uuid
import tempfile
import time
import threading
import multiprocessing as mp
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, Response

from config import AVAILABLE_MODELS, DeviceChoice, Language, get_device
from utils import extract_audios_from_zip, format_dialogue, calculate_eta
from logging_config import setup_root_logging, get_logger
from worker_pool import start_pool

setup_root_logging()
logger = get_logger('main_pool')

app = FastAPI(title='Transcription API - pool (faster + distil-ptbr)')

JOBS = {}
JOBS_LOCK = threading.Lock()

# Start a pool per model id on startup (simple strategy)
POOLS = {}
POOLS_LOCK = threading.Lock()

def pool_result_reader(result_queue):
    while True:
        try:
            msg = result_queue.get()
            if msg is None:
                break
            job_id = msg.get('job_id')
            t = msg.get('type')
            payload = msg.get('payload')
            with JOBS_LOCK:
                job_ref = JOBS.get(job_id)
                if not job_ref:
                    continue
                if t == 'PROGRESS':
                    job_ref['progress'] = max(job_ref.get('progress', 0), int(payload))
                elif t == 'RESULT':
                    job_ref['result'] = {
                        'internal_path': job_ref['internal_path'],
                        'filename': Path(job_ref['internal_path']).name,
                        'transcription_raw': payload.get('text', ''),
                        'transcription_dialogue_simple': format_dialogue(payload.get('segments', []), use_markdown=False),
                        'transcription_dialogue_markdown': format_dialogue(payload.get('segments', []), use_markdown=True),
                        'entities': payload.get('entities', [])
                    }
                    job_ref['status'] = 'completed'
                    job_ref['progress'] = 100
                    job_ref['finished_at'] = time.time()
                elif t == 'FATAL_ERROR':
                    job_ref['status'] = 'failed'
                    job_ref.setdefault('debug_log', []).append(payload.get('error'))
                    job_ref['finished_at'] = time.time()
        except Exception as e:
            logger.error(f'Error in pool_result_reader: {e}')
            break

@app.on_event('startup')
def setup_pools():
    # create a simple pool for each model (1 or 2 workers)
    with POOLS_LOCK:
        for model_id, cfg in AVAILABLE_MODELS.items():
            device = 'cuda' if cfg.get('req_gpu', False) and mp.cpu_count() > 0 else 'cpu'
            task_q, result_q, workers = start_pool(max(1, min(2, mp.cpu_count()//2)), model_id, cfg, device)
            POOLS[model_id] = {'task_q': task_q, 'result_q': result_q, 'workers': workers, 'cfg': cfg}
            # start reader thread for results
            threading.Thread(target=pool_result_reader, args=(result_q,), daemon=True).start()

@app.post('/jobs', status_code=202)
async def create_jobs(background_tasks: BackgroundTasks,
                      model_id: str = Form(...),
                      session_id: str = Form(...),
                      language: Language = Form(...),
                      files: list[UploadFile] = File(...),
                      device_choice: DeviceChoice = Form(DeviceChoice.AUTO)):

    model_cfg = AVAILABLE_MODELS.get(model_id)
    if not model_cfg:
        raise HTTPException(status_code=400, detail='Model ID inválido')

    device = get_device(device_choice)

    audios = []
    for f in files:
        content = await f.read()
        if f.filename.lower().endswith('.zip'):
            audios.extend(extract_audios_from_zip(content))
        else:
            audios.append({'internal_path': f.filename, 'file_bytes': content})

    if not audios:
        raise HTTPException(status_code=400, detail='Nenhum áudio válido')

    created = []
    with JOBS_LOCK:
        for audio in audios:
            job_id = str(uuid.uuid4())
            JOBS[job_id] = {
                'id': job_id,
                'session_id': session_id,
                'internal_path': audio['internal_path'],
                'status': 'queued',
                'progress': 0,
                'config': {'model_id': model_id, 'model_config': model_cfg, 'language': language.value, 'device': device},
                'created_at': time.time(),
                'started_at': None,
                'finished_at': None,
                'worker_pid': None,
                'debug_log': [],
                'result': None
            }
            created.append({'job_id': job_id, 'filename': audio['internal_path']})

            # write temp audio to disk
            suffix = Path(audio['internal_path']).suffix or '.wav'
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(audio['file_bytes'])
            tmp.flush()
            tmp.close()

            # dispatch to pool
            with POOLS_LOCK:
                pool = POOLS.get(model_id)
                if not pool:
                    raise HTTPException(status_code=500, detail='Pool nao disponivel para esse modelo')
                JOBS[job_id]['status'] = 'processing'
                JOBS[job_id]['started_at'] = time.time()
                pool['task_q'].put({'job': JOBS[job_id], 'audio_path': tmp.name})

    return JSONResponse({'jobs_created': created})

@app.get('/jobs/{job_id}')
def get_job(job_id: str):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail='Job não encontrado')
        j_copy = j.copy()
        j_copy['eta_timestamp'] = calculate_eta(j_copy)
    return JSONResponse(j_copy)

@app.get('/jobs/{job_id}/download')
def download_result(job_id: str, text_type: str = 'transcription_dialogue_markdown'):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if not j or j.get('status') != 'completed' or not j.get('result'):
            raise HTTPException(status_code=404, detail='Resultado indisponível')
        result_text = j['result'].get(text_type) or 'Conteúdo não encontrado.'
        filename = Path(j['internal_path']).with_suffix('.txt').name
    return Response(content=result_text, media_type='text/plain', headers={'Content-Disposition': f'attachment; filename="{filename}"'})

@app.get('/models')
def list_models():
    return JSONResponse({'available_models': list(AVAILABLE_MODELS.keys())})
