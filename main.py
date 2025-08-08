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
                    filter_models_by_device, JOB_RETENTION_TIME_SECONDS,
                    JANITOR_SLEEP_INTERVAL_SECONDS)
from utils import (extract_audios_from_zip, format_dialogue, calculate_eta)

JOBS = {}
JOBS_LOCK = threading.Lock()
janitor_stop_event = threading.Event()

def run_janitor_task():
    print(f"🧹 Janitor iniciado. Verificando jobs a cada {JANITOR_SLEEP_INTERVAL_SECONDS}s.")
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
                    print(f"🧹 Janitor encontrou {len(jobs_to_delete)} job(s) antigos para limpar.")
                    for job_id in jobs_to_delete:
                        del JOBS[job_id]
        except Exception as e:
            print(f"❌ Erro no Janitor: {e}")
        janitor_stop_event.wait(JANITOR_SLEEP_INTERVAL_SECONDS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("="*80)
    print("🚀 API iniciada. O carregamento de modelos ocorrerá no primeiro job.")
    print("="*80)
    
    janitor_thread = threading.Thread(target=run_janitor_task, daemon=True)
    janitor_thread.start()

    yield

    print("\nAPI desligando. Parando o Janitor e limpando processos...")
    janitor_stop_event.set()
    janitor_thread.join()
    with JOBS_LOCK:
        for job_id, job in JOBS.items():
            if job.get('worker_pid') and job['status'] == 'processing':
                try:
                    os.kill(job['worker_pid'], signal.SIGKILL)
                    print(f"Processo worker remanescente (PID: {job['worker_pid']}) foi morto.")
                except ProcessLookupError:
                    pass
    print("Limpeza concluída.")

app = FastAPI(
    title="API de Transcrição Híbrida",
    version="9.0.0-secure",
    lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

@app.get("/ui", response_class=HTMLResponse, tags=["Interface"])
async def read_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/", tags=["Status"])
def read_root():
    return {"message": "API no ar. Acesse /ui para a interface de testes."}

@app.get("/models", tags=["Configuração"])
def get_available_models():
    return {"available_models": list(AVAILABLE_MODELS.keys())}

@app.get("/queues", tags=["Monitoramento"])
async def get_queues(session_ids: Optional[str] = Query(None, description="IDs de sessão")):
    target_sessions = session_ids.split(',') if session_ids else None
    queue_view = []
    with JOBS_LOCK:
        for job_id, job in JOBS.items():
            if target_sessions and job.get('session_id') not in target_sessions:
                continue
            sanitized_job = {
                "job_id": job_id, "session_id": job.get("session_id"), "status": job.get("status"),
                "progress": job.get("progress"), "model_id": job.get("config", {}).get("model_id"),
                "internal_path": job.get("internal_path"), "eta_timestamp": calculate_eta(job)
            }
            queue_view.append(sanitized_job)
    return JSONResponse(content=sorted(queue_view, key=lambda x: (x.get('status', 'z') != 'processing', x.get('status', 'z') != 'queued', x.get('internal_path',''))))

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
                job['status'] = 'cancelled'
                job['finished_at'] = time.time()
                job['debug_log'].append(f"Processo (PID: {pid}) morto na paulada.")
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

def process_single_file_job(job_id: str, audio_bytes: bytes):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or job['status'] != 'queued': return
        job['status'], job['started_at'] = 'processing', time.time()
    
    try:
        json_job_data = json.dumps(job).encode('utf-8')
        job_data_size = len(json_job_data).to_bytes(4, 'big')
        audio_data_size = len(audio_bytes).to_bytes(4, 'big')
        
        process = subprocess.Popen(
            [sys.executable, "worker.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        with JOBS_LOCK: JOBS[job_id]['worker_pid'] = process.pid

        process.stdin.write(job_data_size)
        process.stdin.write(json_job_data)
        process.stdin.write(audio_data_size)
        process.stdin.write(audio_bytes)
        process.stdin.close()
        
        stdout, stderr = process.communicate()
        
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job or job['status'] == 'cancelled': return

            if process.returncode == 0:
                output = json.loads(stdout.decode('utf-8'))
                result_data = output['result']
                job['result'] = {
                    "internal_path": job['internal_path'], "filename": Path(job['internal_path']).name,
                    "transcription_raw": result_data['text'],
                    "transcription_dialogue_simple": format_dialogue(result_data['segments'], use_markdown=False),
                    "transcription_dialogue_markdown": format_dialogue(result_data['segments'], use_markdown=True),
                    "entities": result_data['entities']
                }
                job['status'] = 'completed'
            else:
                error_info = json.loads(stderr.decode('utf-8'))
                job['status'] = 'failed'
                job['debug_log'].append(f"ERRO no worker: {error_info.get('error')}")
                job['debug_log'].append(f"Traceback do worker:\n{error_info.get('traceback')}")

    except Exception as e:
        import traceback
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job and job['status'] != 'cancelled':
                job['status'] = 'failed'
                job['debug_log'].append(f"ERRO do Gerente: {e}\n{traceback.format_exc()}")
    finally:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job['finished_at'] = time.time()
                job['progress'] = 100