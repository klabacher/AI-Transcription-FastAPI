import uuid
import time
import io
import threading
import sys
import os
import signal
import tempfile
import multiprocessing as mp
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Query, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

# Adiciona o diretório raiz ao path para encontrar os módulos locais
sys.path.append(os.getcwd())

from config import (
    AVAILABLE_MODELS, DeviceChoice, Language,
    JOB_RETENTION_TIME_SECONDS, JANITOR_SLEEP_INTERVAL_SECONDS, DEBUG, get_processing_device
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

# --- Funções do Ciclo de Vida da API (Lifespan) ---

def result_reader_thread(result_queue: mp.Queue):
    """
    Thread dedicada a ler resultados da fila de um pool de workers
    e atualizar o dicionário global de JOBS.
    """
    logger.info("Thread leitora de resultados iniciada.")
    while True:
        try:
            message = result_queue.get()
            if message is None:  # Sinal de parada
                break

            job_id = message.get("job_id")
            msg_type = message.get("type")
            payload = message.get("payload")

            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    continue

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
                    job['status'] = 'failed'
                    job['finished_at'] = time.time()
                    job['debug_log'].append(f"ERRO FATAL no worker: {payload.get('error')}")
                    if DEBUG:
                        job['debug_log'].append(f"Traceback do worker:\n{payload.get('traceback')}")
                    logger.error(f"Job {job_id[:8]} falhou com erro fatal no worker.")

        except (EOFError, BrokenPipeError):
            logger.warning("Fila de resultados foi fechada. Encerrando thread leitora.")
            break
        except Exception as e:
            logger.error(f"Erro inesperado na thread leitora de resultados: {e}")

    logger.info("Thread leitora de resultados finalizada.")


def start_worker_pools():
    """
    Inicia os pools de workers para cada modelo configurado.
    """
    ctx = mp.get_context('spawn')
    with POOLS_LOCK:
        for model_id, model_config in AVAILABLE_MODELS.items():
            num_workers = model_config.get("workers", 1) # Default para 1 worker por pool
            task_queue = ctx.Queue()
            result_queue = ctx.Queue()
            workers = []

            # Determina o dispositivo para este pool de workers
            device = 'cuda' if model_config.get('req_gpu', False) else 'cpu'

            for i in range(num_workers):
                worker = ctx.Process(
                    target=worker_process,
                    args=(task_queue, result_queue, model_id, model_config, device),
                    daemon=True,
                    name=f"Worker-{model_id}-{i}"
                )
                worker.start()
                workers.append(worker)
            
            POOLS[model_id] = {
                "task_queue": task_queue,
                "result_queue": result_queue,
                "workers": workers,
                "device": device
            }
            # Inicia uma thread para ler os resultados deste pool
            threading.Thread(target=result_reader_thread, args=(result_queue,), daemon=True).start()
            logger.info(f"Pool para '{model_id}' iniciado com {num_workers} worker(s) no dispositivo '{device}'.")


def shutdown_worker_pools():
    """
    Envia um sinal de parada para todos os workers e aguarda a finalização.
    """
    logger.info("Iniciando desligamento dos pools de workers...")
    with POOLS_LOCK:
        for model_id, pool in POOLS.items():
            # Envia sinal de parada para cada worker no pool
            for _ in pool["workers"]:
                pool["task_queue"].put(None)
            # Envia sinal de parada para a thread leitora de resultados
            pool["result_queue"].put(None)

        # Espera os processos terminarem
        for model_id, pool in POOLS.items():
            for worker in pool["workers"]:
                worker.join(timeout=10) # Timeout de 10s para terminação graciosa
                if worker.is_alive():
                    logger.warning(f"Worker {worker.name} não terminou, forçando o desligamento.")
                    worker.terminate()
            logger.info(f"Pool para '{model_id}' finalizado.")
    logger.info("Desligamento dos pools de workers concluído.")


def run_janitor_task():
    """
    Limpa jobs antigos (concluídos, falhos, cancelados) periodicamente.
    """
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
            logger.error(f"Erro no Janitor: {e}")
        janitor_stop_event.wait(JANITOR_SLEEP_INTERVAL_SECONDS)

def check_and_run_initial_setup():
    """
    Verifica se o setup inicial de download de modelos já foi executado.
    """
    flag_file = Path.home() / ".transcription_api_cache/setup.flag"
    if flag_file.exists():
        logger.info("Setup inicial já concluído. Iniciando API rapidamente.")
        return
    
    logger.warning("Primeira execução detectada. Preparando o ambiente (isso pode levar vários minutos)...")
    os.makedirs(flag_file.parent, exist_ok=True)
    try:
        # Usamos sys.executable para garantir que estamos usando o mesmo python
        process = subprocess.run([sys.executable, "setup_worker.py"], check=True, capture_output=True, text=True, encoding='utf-8')
        if DEBUG:
            logger.debug(f"Saída do setup_worker.py:\n{process.stdout}")
        logger.info("Setup concluído com sucesso! Criando flag para futuras execuções.")
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
    # --- Início da Aplicação ---
    logger.info("="*80)
    check_and_run_initial_setup()
    start_worker_pools()
    janitor_thread = threading.Thread(target=run_janitor_task, daemon=True)
    janitor_thread.start()
    logger.info("🚀 API iniciada e pronta para receber jobs.")
    logger.info("="*80)
    
    yield
    
    # --- Desligamento da Aplicação ---
    logger.info("\nAPI desligando...")
    janitor_stop_event.set()
    janitor_thread.join()
    shutdown_worker_pools()
    logger.info("Limpeza concluída. Até mais!")

# --- Configuração do FastAPI ---
app = FastAPI(
    title="API de Transcrição Otimizada",
    description="Focada em `faster-whisper` e modelos destilados com workers persistentes.",
    version="2.0.0",
    lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# --- Endpoints da API ---

@app.get("/ui", response_class=HTMLResponse, tags=["Interface"])
async def read_ui(request: Request):
    """Serve a interface web (HUD)."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/", tags=["Status"])
def read_root():
    """Endpoint raiz para verificar se a API está no ar."""
    return {"message": "API de Transcrição no ar. Acesse /ui para a interface de testes ou /docs para a documentação."}

@app.get("/models", tags=["Configuração"])
def get_available_models():
    """Retorna a lista de IDs dos modelos disponíveis."""
    return {"available_models": list(AVAILABLE_MODELS.keys())}

@app.get("/queues", tags=["Monitoramento"])
async def get_queues(session_ids: Optional[str] = Query(None, description="IDs de sessão para filtrar, separados por vírgula")):
    """Retorna uma visão sanitizada da fila de jobs."""
    target_sessions = session_ids.split(',') if session_ids else None
    queue_view = []
    with JOBS_LOCK:
        for job_id, job in JOBS.items():
            if target_sessions and job.get('session_id') not in target_sessions:
                continue
            
            sanitized_job = {
                "job_id": job_id,
                "session_id": job.get("session_id"),
                "status": job.get("status"),
                "progress": job.get("progress"),
                "model_id": job.get("config", {}).get("model_id"),
                "internal_path": job.get("internal_path"),
                "eta_timestamp": calculate_eta(job)
            }
            queue_view.append(sanitized_job)
    # Ordena para melhor visualização na UI
    return JSONResponse(content=sorted(queue_view, key=lambda x: (
        x.get('status', 'z') != 'processing',
        x.get('status', 'z') != 'queued',
        x.get('internal_path', '')
    )))

@app.post("/jobs", status_code=202, tags=["Transcrição"])
async def create_transcription_jobs(
    background_tasks: BackgroundTasks,
    model_id: str = Form(...),
    session_id: str = Form(...),
    language: Language = Form(...),
    files: List[UploadFile] = File(...),
    device_choice: DeviceChoice = Form(DeviceChoice.AUTO) # Mantido para consistência, mas o device é do pool
):
    model_config = AVAILABLE_MODELS.get(model_id)
    if not model_config:
        raise HTTPException(status_code=400, detail=f"Model ID '{model_id}' inválido.")

    with POOLS_LOCK:
        pool = POOLS.get(model_id)
        if not pool:
            raise HTTPException(status_code=503, detail=f"O pool de workers para o modelo '{model_id}' não está disponível.")

    job_config = {
        "model_id": model_id,
        "model_config": model_config,
        "language": language.value
    }
    
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
                # Salva o áudio em um arquivo temporário que o worker possa acessar
                suffix = Path(audio['internal_path']).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio['file_bytes'])
                    audio_path = tmp.name
                    temp_files_to_clean.append(audio_path)

                job_id = str(uuid.uuid4())
                JOBS[job_id] = {
                    "id": job_id,
                    "session_id": session_id,
                    "internal_path": audio["internal_path"],
                    "status": "queued",
                    "progress": 0,
                    "config": job_config,
                    "created_at": time.time(),
                    "started_at": None,
                    "finished_at": None,
                    "debug_log": [f"Job criado para '{audio['internal_path']}'..."],
                    "result": None,
                    "audio_path": audio_path # Caminho para o arquivo temporário
                }
                
                # Despacha a tarefa para a fila do pool correto
                task = {"job_id": job_id, "audio_path": audio_path}
                pool["task_queue"].put(task)
                
                JOBS[job_id]['status'] = 'processing'
                JOBS[job_id]['started_at'] = time.time()
                
                jobs_created.append({"job_id": job_id, "filename": audio["internal_path"]})

        return JSONResponse(content={"jobs_created": jobs_created})

    except Exception as e:
        # Em caso de erro, limpa os arquivos temporários criados
        for path in temp_files_to_clean:
            try:
                os.remove(path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Erro ao criar jobs e despachar para a fila: {e}")

@app.post("/jobs/{job_id}/cancel", status_code=200, tags=["Transcrição"])
def cancel_job(job_id: str):
    """
    Cancela um job. ATENÇÃO: Na nova arquitetura, não matamos o worker.
    O cancelamento remove o job da fila se ele ainda não foi pego, ou
    apenas o marca como 'cancelled' se já estiver em processamento,
    deixando o worker terminar para não corromper o estado do modelo.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job não encontrado.")
        
        if job['status'] == 'queued':
            # Idealmente, teríamos como remover da fila, mas é complexo.
            # A estratégia mais simples é marcar como cancelado. O worker vai ignorá-lo.
            job['status'] = 'cancelled'
            job['finished_at'] = time.time()
            logger.info(f"Job {job_id[:8]} cancelado da fila.")
            return {"message": "Job cancelado da fila."}
        
        if job['status'] == 'processing':
            job['status'] = 'cancelled' # O worker vai terminar, mas o resultado será ignorado.
            job['finished_at'] = time.time()
            logger.warning(f"Job {job_id[:8]} marcado como 'cancelled' durante o processamento. O worker terminará a tarefa atual.")
            return {"message": "Job marcado como cancelado. O resultado será descartado."}

        return {"message": f"Não foi possível cancelar. Status atual: {job['status']}."}


@app.get("/jobs/{job_id}", tags=["Transcrição"])
def get_job_status(job_id: str):
    """Retorna o status detalhado de um único job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job não encontrado. Pode ter expirado.")
        
        # Cria uma cópia para evitar race conditions na UI
        job_copy = job.copy()
        job_copy["eta_timestamp"] = calculate_eta(job)
    
    return JSONResponse(content=job_copy)

@app.get("/jobs/{job_id}/download", tags=["Resultados"])
def download_job_result(job_id: str, text_type: str = "transcription_dialogue_markdown"):
    """Permite o download do resultado de um job como um arquivo de texto."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or job.get('status') != 'completed' or not job.get('result'):
            raise HTTPException(status_code=404, detail="Job não concluído, falhou ou o resultado está indisponível.")
        
        result_text = job['result'].get(text_type, "Formato de texto solicitado não encontrado.")
        # Garante um nome de arquivo seguro
        safe_filename = Path(job['internal_path']).with_suffix('.txt').name
        
    return Response(
        content=result_text,
        media_type="text/plain",
        headers={'Content-Disposition': f'attachment; filename="{safe_filename}"'}
    )