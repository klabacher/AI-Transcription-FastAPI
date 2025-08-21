import uuid
import time
import io
import threading
import sys
import os
import signal
import tempfile
import multiprocessing as mp
import subprocess
import torch
import logging
import zipfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Query, Response
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

# Adiciona o diretório raiz ao path para encontrar os módulos locais
sys.path.append(os.getcwd())

from config import (
    AVAILABLE_MODELS, Language,
    JOB_RETENTION_TIME_SECONDS, JANITOR_SLEEP_INTERVAL_SECONDS, DEBUG
)
from utils import (extract_audios_from_zip, format_dialogue, calculate_eta)
from logging_config import setup_root_logging, get_logger
from worker import worker_process

# --- Configuração Inicial ---
setup_root_logging()
logger = get_logger("main_api")

# --- Estruturas de Dados Globais ---
JOBS = {}
JOBS_LOCK = threading.Lock()

POOLS = {}
POOLS_LOCK = threading.Lock()

janitor_stop_event = threading.Event()

# --- Lógica de Detecção de Hardware ---
_hardware_caps_cache = None

if os.environ.get("FORCE_CUDA", "0") == "1":
    logger.info("FORCE_CUDA está ativado. Ignorando detecção automática de hardware.")

def get_hardware_capabilities():
    """
    Verifica as capacidades de hardware (CUDA, FP16) uma única vez e armazena em cache.
    Retorna um dicionário com as informações.
    """
    global _hardware_caps_cache
    if _hardware_caps_cache is not None:
        return _hardware_caps_cache

    caps = {
        "has_cuda": False,
        "cuda_device_name": None,
        "compute_capability": None,
        "supports_fp16": False,
        "info_message": "Nenhum dispositivo CUDA compatível encontrado. Apenas modelos de CPU estarão disponíveis."
    }
    try:
        if torch.cuda.is_available() or os.environ.get("FORCE_CUDA", "0") == "1":
            caps["has_cuda"] = True
            device_id = 0
            caps["cuda_device_name"] = torch.cuda.get_device_name(device_id)
            caps["compute_capability"] = torch.cuda.get_device_capability(device_id)
            
            # FP16 é bem suportado em arquiteturas com Compute Capability 7.0+
            if caps["compute_capability"][0] >= 7:
                caps["supports_fp16"] = True
                caps["info_message"] = f"GPU detectada: {caps['cuda_device_name']} (Compute Capability: {caps['compute_capability']}). Modelos de GPU e FP16 estão ATIVOS."
            else:
                caps["info_message"] = f"GPU detectada: {caps['cuda_device_name']} (Compute Capability: {caps['compute_capability']}). Modelos que exigem FP16 podem não funcionar."
            logger.info(caps["info_message"])
        else:
            logger.warning(caps["info_message"])
    except Exception as e:
        caps["info_message"] = f"Erro ao verificar as capacidades de hardware: {e}"
        logger.error(caps["info_message"])

    _hardware_caps_cache = caps
    return _hardware_caps_cache

# --- Funções do Ciclo de Vida da API (Lifespan) ---

def result_reader_thread(result_queue: mp.Queue):
    """
    Thread dedicada a ler resultados da fila de um pool de workers
    e atualizar o dicionário global de JOBS.
    """
    thread_name = f"ResultReader-{id(result_queue)}"
    logger.info(f"Thread leitora de resultados '{thread_name}' iniciada.")
    while True:
        try:
            message = result_queue.get()
            if message is None:
                break
            job_id = message.get("job_id")
            if not job_id: continue

            msg_type = message.get("type")
            payload = message.get("payload")
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job: continue

                if msg_type == "PROGRESS":
                    job['progress'] = max(job.get('progress', 0), int(payload))
                elif msg_type == "RESULT":
                    job['status'] = 'completed'
                    job['progress'] = 100
                    job['finished_at'] = time.time()
                    job['result'] = {
                        "internal_path": job['internal_path'],
                        "filename": Path(job['internal_path']).name,
                        "transcription_raw": payload['text'],
                        "transcription_dialogue_simple": format_dialogue(payload['segments'], use_markdown=False),
                        "transcription_dialogue_markdown": format_dialogue(payload['segments'], use_markdown=True)
                    }
                    logger.info(f"Job {job_id[:8]} concluído com sucesso.")
                elif msg_type == "FATAL_ERROR":
                    # Evita sobrescrever o status se já foi cancelado
                    if job['status'] != 'cancelled':
                        job['status'] = 'failed'
                    job['finished_at'] = time.time()
                    job['debug_log'].append(f"ERRO FATAL no worker: {payload.get('error')}")
                    if DEBUG: job['debug_log'].append(f"Traceback do worker:\n{payload.get('traceback')}")
                    logger.error(f"Job {job_id[:8]} falhou com erro fatal no worker.")
        except (EOFError, BrokenPipeError):
            logger.warning(f"Fila de resultados da thread '{thread_name}' foi fechada.")
            break
        except Exception as e:
            logger.error(f"Erro inesperado na thread leitora '{thread_name}': {e}", exc_info=DEBUG)
    logger.info(f"Thread leitora de resultados '{thread_name}' finalizada.")

def start_worker_pools(hardware_caps: dict):
    """
    Inicia os pools de workers apenas para os modelos compatíveis com o hardware detectado.
    """
    ctx = mp.get_context('spawn')
    with POOLS_LOCK:
        for model_id, model_config in AVAILABLE_MODELS.items():
            device = 'cpu'
            if model_config.get('req_gpu', False):
                if not hardware_caps.get('has_cuda'):
                    logger.warning(f"PULANDO POOL: Modelo '{model_id}' requer GPU, mas nenhum dispositivo CUDA foi encontrado.")
                    continue
                if model_config.get('compute_type') == 'float16' and not hardware_caps.get('supports_fp16'):
                    logger.warning(f"PULANDO POOL: Modelo '{model_id}' requer suporte a FP16, mas a GPU '{hardware_caps.get('cuda_device_name')}' não possui capacidade de computação >= 7.0.")
                    continue
                device = 'cuda'

            num_workers = model_config.get("workers", 1)
            task_queue = ctx.Queue()
            result_queue = ctx.Queue()
            workers = []
            for i in range(num_workers):
                worker = ctx.Process(target=worker_process, args=(task_queue, result_queue, model_id, model_config, device), daemon=True, name=f"Worker-{model_id}-{i}")
                worker.start()
                workers.append(worker)
            
            POOLS[model_id] = {"task_queue": task_queue, "result_queue": result_queue, "workers": workers, "device": device}
            threading.Thread(target=result_reader_thread, args=(result_queue,), daemon=True).start()
            logger.info(f"POOL INICIADO: '{model_id}' com {num_workers} worker(s) no dispositivo '{device}'.")

def shutdown_worker_pools():
    logger.info("Iniciando desligamento dos pools de workers...")
    with POOLS_LOCK:
        for model_id, pool in POOLS.items():
            for _ in pool["workers"]:
                pool["task_queue"].put(None)
            pool["result_queue"].put(None)
        for model_id, pool in POOLS.items():
            for worker in pool["workers"]:
                worker.join(timeout=10)
                if worker.is_alive():
                    logger.warning(f"Worker {worker.name} não terminou, forçando o desligamento.")
                    worker.terminate()
            logger.info(f"Pool para '{model_id}' finalizado.")
    logger.info("Desligamento dos pools de workers concluído.")

def run_janitor_task():
    logger.info(f"Janitor iniciado. Verificando jobs a cada {JANITOR_SLEEP_INTERVAL_SECONDS}s.")
    while not janitor_stop_event.is_set():
        try:
            jobs_to_delete = []
            now = time.time()
            with JOBS_LOCK:
                for job_id, job in JOBS.items():
                    if job.get('status') in ('completed', 'failed', 'cancelled'):
                        finished_at = job.get('finished_at', time.time())
                        if (now - finished_at) > JOB_RETENTION_TIME_SECONDS:
                            jobs_to_delete.append(job_id)
                if jobs_to_delete:
                    logger.info(f"Janitor encontrou {len(jobs_to_delete)} job(s) antigos para limpar.")
                    for job_id in jobs_to_delete:
                        del JOBS[job_id]
        except Exception as e:
            logger.error(f"Erro no Janitor: {e}", exc_info=DEBUG)
        janitor_stop_event.wait(JANITOR_SLEEP_INTERVAL_SECONDS)

def check_and_run_initial_setup():
    flag_file = Path.home() / ".transcription_api_cache/setup.flag"
    if flag_file.exists():
        logger.info("Setup inicial de modelos já concluído. Iniciando API rapidamente.")
        return
    logger.warning("Primeira execução detectada. Preparando o ambiente (isso pode levar vários minutos)...")
    os.makedirs(flag_file.parent, exist_ok=True)
    try:
        process = subprocess.run([sys.executable, "setup_worker.py"], check=True, capture_output=True, text=True, encoding='utf-8')
        if DEBUG: logger.debug(f"Saída do setup_worker.py:\n{process.stdout}")
        logger.info("Setup de modelos concluído com sucesso! Criando flag para futuras execuções.")
        flag_file.touch()
    except subprocess.CalledProcessError as e:
        logger.critical("ERRO CRÍTICO DURANTE O SETUP INICIAL DOS MODELOS!")
        logger.error(f"Saída do processo de setup:\n{e.stdout}\n{e.stderr}")
        sys.exit("Falha no setup. Verifique os logs e as dependências (requirements.txt).")
    except FileNotFoundError:
        logger.critical("'setup_worker.py' não encontrado.")
        sys.exit("Arquivo de setup ausente.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("="*80)
    hardware_caps = get_hardware_capabilities()
    check_and_run_initial_setup()
    start_worker_pools(hardware_caps)
    janitor_thread = threading.Thread(target=run_janitor_task, daemon=True)
    janitor_thread.start()
    logger.info("🚀 API iniciada e pronta para receber jobs.")
    logger.info("="*80)
    yield
    logger.info("\nAPI desligando...")
    janitor_stop_event.set()
    janitor_thread.join()
    shutdown_worker_pools()
    logger.info("Limpeza concluída. Até mais!")

app = FastAPI(title="API de Transcrição Otimizada", description="Focada em `faster-whisper` e modelos destilados com workers persistentes.", version="2.2.0-zip-download", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# --- Endpoints da API ---

@app.get("/ui", response_class=HTMLResponse, tags=["Interface"])
async def read_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/", tags=["Status"])
def read_root():
    hardware_caps = get_hardware_capabilities()
    return {
        "message": "API de Transcrição no ar. Acesse /ui para a interface de testes ou /docs para a documentação.",
        "hardware_info": hardware_caps.get('info_message')
    }

@app.get("/models", tags=["Configuração"])
def get_available_models():
    """Retorna apenas os modelos para os quais um pool de workers está ATIVO."""
    with POOLS_LOCK:
        active_models = list(POOLS.keys())
    hardware_caps = get_hardware_capabilities()
    
    return {
        "hardware_info": hardware_caps.get('info_message'),
        "available_models": active_models
    }

@app.get("/queues", tags=["Monitoramento"])
async def get_queues(session_ids: Optional[str] = Query(None, description="IDs de sessão para filtrar, separados por vírgula")):
    target_sessions = session_ids.split(',') if session_ids else None
    queue_view = []
    with JOBS_LOCK:
        for job_id, job in JOBS.items():
            if target_sessions and job.get('session_id') not in target_sessions:
                continue
            sanitized_job = {"job_id": job_id, "session_id": job.get("session_id"), "status": job.get("status"), "progress": job.get("progress"), "model_id": job.get("config", {}).get("model_id"), "internal_path": job.get("internal_path"), "eta_timestamp": calculate_eta(job)}
            queue_view.append(sanitized_job)
    return JSONResponse(content=sorted(queue_view, key=lambda x: (x.get('status', 'z') != 'processing', x.get('status', 'z') != 'queued', x.get('internal_path', ''))))

@app.post("/jobs", status_code=202, tags=["Transcrição"])
async def create_transcription_jobs(model_id: str = Form(...), session_id: str = Form(...), language: Language = Form(...), files: List[UploadFile] = File(...)):
    """Cria e despacha jobs de transcrição para os pools de workers ativos."""
    with POOLS_LOCK:
        if model_id not in POOLS:
            raise HTTPException(status_code=404, detail=f"Modelo '{model_id}' não está disponível ou não é compatível com o hardware atual.")
        pool = POOLS[model_id]
    
    model_config = AVAILABLE_MODELS.get(model_id)
    job_config = {"model_id": model_id, "model_config": model_config, "language": language.value}
    
    audios_to_process = []
    for file in files:
        try:
            content = await file.read()
            if file.filename.lower().endswith('.zip'):
                audios_to_process.extend(extract_audios_from_zip(content))
            else:
                audios_to_process.append({"internal_path": file.filename, "file_bytes": content})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao processar o arquivo '{file.filename}': {e}")
    
    if not audios_to_process:
        raise HTTPException(status_code=400, detail="Nenhum arquivo de áudio válido encontrado nos envios.")
    
    jobs_created = []
    temp_files_to_clean = []
    
    try:
        with JOBS_LOCK:
            for audio in audios_to_process:
                suffix = Path(audio['internal_path']).suffix or ".tmp"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio['file_bytes'])
                    audio_path = tmp.name
                    temp_files_to_clean.append(audio_path)
                
                job_id = str(uuid.uuid4())
                JOBS[job_id] = {"id": job_id, "session_id": session_id, "internal_path": audio["internal_path"], "status": "queued", "progress": 0, "config": job_config, "created_at": time.time(), "started_at": None, "finished_at": None, "debug_log": [f"Job criado para '{audio['internal_path']}'..."], "result": None, "audio_path": audio_path}
                
                task = {"job_id": job_id, "audio_path": audio_path}
                pool["task_queue"].put(task)
                
                JOBS[job_id]['status'] = 'processing'
                JOBS[job_id]['started_at'] = time.time()
                
                jobs_created.append({"job_id": job_id, "filename": audio["internal_path"]})
        
        return JSONResponse(content={"jobs_created": jobs_created})

    except Exception as e:
        for path in temp_files_to_clean:
            try: os.remove(path)
            except OSError: pass
        raise HTTPException(status_code=500, detail=f"Erro ao criar jobs e despachar para a fila: {e}")

@app.post("/jobs/{job_id}/cancel", status_code=200, tags=["Transcrição"])
def cancel_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job: raise HTTPException(status_code=404, detail="Job não encontrado.")
        
        if job['status'] == 'queued' or job['status'] == 'processing':
            job['status'] = 'cancelled'
            job['finished_at'] = time.time()
            if job['status'] == 'queued':
                logger.info(f"Job {job_id[:8]} cancelado da fila.")
                return {"message": "Job cancelado da fila."}
            else:
                logger.warning(f"Job {job_id[:8]} marcado como 'cancelled' durante o processamento. O resultado será descartado.")
                return {"message": "Job marcado como cancelado. O resultado será descartado."}

        return {"message": f"Não foi possível cancelar. Status atual: {job['status']}."}

@app.get("/jobs/{job_id}", tags=["Transcrição"])
def get_job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job: raise HTTPException(status_code=404, detail="Job não encontrado. Pode ter expirado.")
        job_copy = job.copy()
        job_copy["eta_timestamp"] = calculate_eta(job)
    return JSONResponse(content=job_copy)

@app.get("/jobs/{job_id}/download", tags=["Resultados"])
def download_job_result(job_id: str, text_type: str = "transcription_dialogue_markdown"):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or job.get('status') != 'completed' or not job.get('result'):
            raise HTTPException(status_code=404, detail="Job não concluído, falhou ou o resultado está indisponível.")
        result_text = job['result'].get(text_type, "Formato de texto solicitado não encontrado.")
        safe_filename = Path(job['internal_path']).with_suffix('.txt').name
    return Response(content=result_text, media_type="text/plain", headers={'Content-Disposition': f'attachment; filename="{safe_filename}"'})

@app.get("/jobs/download/session/{session_id}", tags=["Resultados"])
def download_session_results(session_id: str, text_type: str = "transcription_dialogue_markdown"):
    
    zip_buffer = io.BytesIO()
    
    with JOBS_LOCK:
        completed_jobs = [job for job in JOBS.values() if job.get('session_id') == session_id and job.get('status') == 'completed']

    if not completed_jobs:
        raise HTTPException(status_code=404, detail="Nenhum job concluído encontrado para esta sessão.")

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for job in completed_jobs:
            result_text = job['result'].get(text_type, "Formato de texto solicitado não encontrado.")
            file_name_in_zip = Path(job['internal_path']).with_suffix('.txt').name
            zip_file.writestr(file_name_in_zip, result_text)
    
    zip_buffer.seek(0)
    
    safe_session_id = "".join(c for c in session_id if c.isalnum() or c in ('-', '_'))
    zip_filename = f"transcricoes_sessao_{safe_session_id[:8]}.zip"
    
    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )