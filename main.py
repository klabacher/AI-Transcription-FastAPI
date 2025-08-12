import uuid
import time
import io
import threading
import sys
import subprocess
import json
import os
import signal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Query, Response
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

from config import (AVAILABLE_MODELS, DeviceChoice, Language,
                    JOB_RETENTION_TIME_SECONDS, JANITOR_SLEEP_INTERVAL_SECONDS, DEBUG)
from utils import (extract_audios_from_zip, format_dialogue, calculate_eta)
from logging_config import setup_logging, get_logger

# Configuração inicial de logging para o processo principal
setup_logging()
logger = get_logger("main_api")

JOBS = {}
JOBS_LOCK = threading.Lock()
janitor_stop_event = threading.Event()

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
            logger.error(f"Erro no Janitor: {e}")
        janitor_stop_event.wait(JANITOR_SLEEP_INTERVAL_SECONDS)

def check_and_run_initial_setup():
    flag_dir = Path.home() / ".transcription_api_cache"
    flag_file = flag_dir / "setup.flag"

    if flag_file.exists():
        logger.info("Setup inicial já concluído. Iniciando API rapidamente.")
        return

    logger.warning("Primeira execução detectada. Preparando o ambiente...")
    os.makedirs(flag_dir, exist_ok=True)

    try:
        process = subprocess.run([sys.executable, "setup_worker.py"], check=True, capture_output=True, text=True)
        if DEBUG:
            logger.debug(f"Saída do setup_worker:\n{process.stdout}")
        logger.info("Setup concluído com sucesso! Criando flag para futuras execuções.")
        flag_file.touch()
    
    except subprocess.CalledProcessError as e:
        logger.critical("ERRO CRÍTICO DURANTE O SETUP INICIAL DOS MODELOS!")
        logger.error(f"Saída do processo de setup:\n{e.stdout}\n{e.stderr}")
        sys.exit("Falha no setup. Verifique os logs e se as dependências estão instaladas.")
    except FileNotFoundError:
        logger.critical("'setup_worker.py' não encontrado.")
        sys.exit("Arquivo de setup ausente.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("="*80)
    check_and_run_initial_setup()
    logger.info("🚀 API iniciada e pronta para receber jobs.")
    logger.info("="*80)
    
    janitor_thread = threading.Thread(target=run_janitor_task, daemon=True)
    janitor_thread.start()

    yield

    logger.info("\nAPI desligando. Parando o Janitor e limpando processos...")
    janitor_stop_event.set()
    janitor_thread.join()
    with JOBS_LOCK:
        for job_id, job in JOBS.items():
            if job.get('worker_pid') and job['status'] == 'processing':
                try:
                    os.kill(job['worker_pid'], signal.SIGKILL)
                    logger.warning(f"Processo worker remanescente (PID: {job['worker_pid']}) foi morto.")
                except ProcessLookupError:
                    pass
    logger.info("Limpeza concluída.")

app = FastAPI(
    title="API de Transcrição Híbrida",
    version="2.0-debuggable",
    lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

def log_worker_stream(stream, job_id):
    """Lê o stderr do worker linha por linha e loga no processo principal."""
    for line in iter(stream.readline, ''):
        if not line:
            break
        try:
            log_data = json.loads(line)
            # Verifica se é um log ou o resultado final do erro
            if "traceback" in log_data:
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                    if job:
                        job['status'] = 'failed'
                        job['debug_log'].append(f"ERRO no worker: {log_data.get('error')}")
                        job['debug_log'].append(f"Traceback do worker:\n{log_data.get('traceback')}")
            else:
                logger.debug(f"[Worker PID:{JOBS.get(job_id, {}).get('worker_pid')}] {log_data.get('message')}")
        except json.JSONDecodeError:
            logger.warning(f"[Worker Stream Incompreensível] {line.strip()}")
    stream.close()

def process_single_file_job(job_id: str, audio_bytes: bytes):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or job['status'] != 'queued': return
        job['status'], job['started_at'] = 'processing', time.time()
        logger.info(f"Iniciando job {job_id[:8]} para o arquivo '{job['internal_path']}'.")
    
    try:
        json_job_data = json.dumps(job).encode('utf-8')
        job_data_size = len(json_job_data).to_bytes(4, 'big')
        audio_data_size = len(audio_bytes).to_bytes(4, 'big')
        
        process = subprocess.Popen(
            [sys.executable, "-u", "worker.py"], # -u para unbuffered output
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text_encoding='utf-8'
        )
        with JOBS_LOCK: JOBS[job_id]['worker_pid'] = process.pid
        logger.debug(f"Worker para job {job_id[:8]} iniciado com PID: {process.pid}")

        # Inicia a thread para ouvir os logs do worker em tempo real
        stderr_thread = threading.Thread(target=log_worker_stream, args=(process.stderr, job_id))
        stderr_thread.start()

        process.stdin.write(job_data_size.decode('latin-1'))
        process.stdin.write(json_job_data.decode('latin-1'))
        process.stdin.write(audio_data_size.decode('latin-1'))
        process.stdin.write(audio_bytes.decode('latin-1'))
        process.stdin.close()
        
        stdout, _ = process.communicate()
        stderr_thread.join() # Garante que todos os logs foram lidos
        
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job or job['status'] == 'cancelled': return

            if process.returncode == 0 and stdout:
                output = json.loads(stdout)
                result_data = output['result']
                job['result'] = {
                    "internal_path": job['internal_path'], "filename": Path(job['internal_path']).name,
                    "transcription_raw": result_data['text'],
                    "transcription_dialogue_simple": format_dialogue(result_data['segments'], use_markdown=False),
                    "transcription_dialogue_markdown": format_dialogue(result_data['segments'], use_markdown=True),
                    "entities": result_data.get('entities', [])
                }
                job['status'] = 'completed'
                logger.info(f"Job {job_id[:8]} concluído com sucesso.")
            elif job['status'] != 'failed': # Se o status não foi setado pelo logger
                job['status'] = 'failed'
                job['debug_log'].append(f"Worker terminou com código de erro {process.returncode} mas sem log de erro explícito.")
                logger.error(f"Job {job_id[:8]} falhou. Código de saída: {process.returncode}")

    except Exception as e:
        import traceback
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job and job['status'] != 'cancelled':
                job['status'] = 'failed'
                job['debug_log'].append(f"ERRO do Gerente: {e}\n{traceback.format_exc()}")
                logger.error(f"Erro no gerenciamento do job {job_id[:8]}: {e}")
    finally:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job['finished_at'] = time.time()
                job['progress'] = 100

@app.post("/jobs", status_code=202, tags=["Transcrição"])
async def create_transcription_jobs(
    background_tasks: BackgroundTasks, model_id: str = Form(...), session_id: str = Form(...),
    language: Language = Form(...), files: List[UploadFile] = File(...),
    device_choice: DeviceChoice = Form(DeviceChoice.AUTO), assemblyai_api_key: Optional[str] = Form(None),
    speaker_labels: bool = Form(False), entity_detection: bool = Form(False)
):
    model_config = AVAILABLE_MODELS.get(model_id)
    if not model_config: raise HTTPException(status_code=400, detail=f"Model ID '{model_id}' inválido.")

    job_config = {"model_id": model_id, "model_config": model_config, "language": language.value}
    if model_config['impl'] == 'assemblyai':
        if not assemblyai_api_key: raise HTTPException(status_code=400, detail="Chave da API AssemblyAI é necessária.")
        job_config.update({"assemblyai_api_key": assemblyai_api_key, "speaker_labels": speaker_labels, "entity_detection": entity_detection, "device": "cloud"})
    else:
        job_config["device_choice"] = device_choice.value

    audios_to_process = []
    for file in files:
        if file.filename.lower().endswith('.zip'):
            audios_to_process.extend(extract_audios_from_zip(await file.read()))
        else:
            audios_to_process.append({"internal_path": file.filename, "file_bytes": await file.read()})
            
    if not audios_to_process: raise HTTPException(status_code=400, detail="Nenhum arquivo de áudio válido encontrado.")

    jobs_created = []
    with JOBS_LOCK:
        for audio in audios_to_process:
            job_id = str(uuid.uuid4())
            JOBS[job_id] = {
                "id": job_id, "session_id": session_id, "internal_path": audio["internal_path"],
                "status": "queued", "progress": 0, "config": job_config, "created_at": time.time(),
                "started_at": None, "finished_at": None, "worker_pid": None,
                "debug_log": [f"Job criado para '{audio['internal_path']}'..."], "result": None
            }
            background_tasks.add_task(process_single_file_job, job_id, audio['file_bytes'])
            jobs_created.append({"job_id": job_id, "filename": audio["internal_path"]})

    return JSONResponse(content={"jobs_created": jobs_created})

@app.post("/jobs/{job_id}/cancel", status_code=200, tags=["Transcrição"])
def cancel_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job: raise HTTPException(status_code=404, detail="Job não encontrado.")
        
        pid = job.get('worker_pid')
        if pid and job['status'] == 'processing':
            try:
                os.kill(pid, signal.SIGKILL)
                job['status'] = 'cancelled'; job['finished_at'] = time.time()
                logger.warning(f"Processo (PID: {pid}) do job {job_id[:8]} morto na paulada.")
                return {"message": "Processo morto com sucesso."}
            except ProcessLookupError:
                job['status'] = 'cancelled'; job['finished_at'] = time.time()
                return {"message": "Processo já havia terminado."}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Falha ao matar o processo: {e}")
        
        elif job['status'] == 'queued':
            job['status'] = 'cancelled'; job['finished_at'] = time.time()
            return {"message": "Job cancelado da fila."}

        return {"message": f"Não foi possível cancelar. Status: {job['status']}."}

@app.post("/queues/cancel-session", status_code=200, tags=["Monitoramento"])
def cancel_session_jobs(session_id: str = Form(...)):
    if not session_id: raise HTTPException(status_code=400, detail="ID da sessão é obrigatório.")
    
    cancelled_count = 0
    with JOBS_LOCK:
        for job in JOBS.values():
            if job.get('session_id') == session_id:
                if job.get('status') == 'processing' and job.get('worker_pid'):
                    try:
                        os.kill(job['worker_pid'], signal.SIGKILL)
                        job['status'] = 'cancelled'; job['finished_at'] = time.time()
                        cancelled_count += 1
                    except ProcessLookupError: pass
                elif job.get('status') == 'queued':
                    job['status'] = 'cancelled'; job['finished_at'] = time.time()
                    cancelled_count += 1
    
    return {"message": f"{cancelled_count} jobs da sessão {session_id[:8]}... foram cancelados."}

@app.get("/jobs/{job_id}", tags=["Transcrição"])
def get_job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job: raise HTTPException(status_code=404, detail="Job não encontrado. Pode ter expirado.")
        job_with_eta = job.copy()
        job_with_eta["eta_timestamp"] = calculate_eta(job)
    return JSONResponse(content=job_with_eta)

@app.get("/jobs/{job_id}/download", tags=["Resultados"])
def download_job_result(job_id: str, text_type: str = "transcription_dialogue_markdown"):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or job['status'] != 'completed' or not job.get('result'): 
            raise HTTPException(status_code=404, detail="Job não concluído ou resultado indisponível.")
        result_text = job['result'].get(text_type, "Conteúdo não encontrado.")
        filename = Path(job['internal_path']).with_suffix('.txt').name
    
    return Response(content=result_text, media_type="text/plain", headers={'Content-Disposition': f'attachment; filename="{filename}"'})