import uuid
import time
import io
import threading
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Query, Response
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
from contextlib import asynccontextmanager
from pathlib import Path
import soundfile as sf

from config import (AVAILABLE_MODELS, DeviceChoice, Language, get_processing_device,
                    filter_models_by_device, JOB_RETENTION_TIME_SECONDS,
                    JANITOR_SLEEP_INTERVAL_SECONDS)
from utils import (extract_audios_from_zip, format_dialogue,
                   prepare_download_package, calculate_eta)
from engine import run_transcription

try:
    import whisper as openai_whisper
    from faster_whisper import WhisperModel
    from transformers import pipeline as hf_pipeline
except ImportError as e:
    raise ImportError(f"Biblioteca de ML faltando: {e}. Rode 'pip install -r requirements.txt'")

JOBS = {}
janitor_stop_event = threading.Event()

def run_janitor_task():
    print(f"🧹 Janitor iniciado. Verificando jobs a cada {JANITOR_SLEEP_INTERVAL_SECONDS}s.")
    while not janitor_stop_event.is_set():
        try:
            jobs_to_delete = []
            now = time.time()
            with threading.Lock():
                for job_id, job in JOBS.items():
                    if job.get('status') in ('completed', 'failed', 'cancelled'):
                        finished_at = job.get('finished_at', 0)
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
    print("🚀 INICIANDO SETUP AUTOMÁTICO DOS MODELOS...")
    for model_id, config in AVAILABLE_MODELS.items():
        if config['impl'] == 'assemblyai':
            print(f"ℹ️ Modelo de nuvem '{model_id}' configurado.")
            continue
        try:
            print(f"\nVerificando modelo: '{model_id}'...")
            if config['impl'] == 'openai':
                openai_whisper.load_model(config['model_name'], device='cpu')
            elif config['impl'] == 'faster':
                WhisperModel(config['model_name'], device='cpu', compute_type='int8')
            elif config['impl'] == 'hf_pipeline':
                hf_pipeline("automatic-speech-recognition", model=config['model_name'])
            print(f"✅ Modelo '{model_id}' está pronto.")
        except Exception as e:
            print(f"⚠️ AVISO: Falha ao verificar/baixar '{model_id}'. Erro: {e}")
    print("\n" + "="*80)

    janitor_thread = threading.Thread(target=run_janitor_task, daemon=True)
    janitor_thread.start()

    yield

    print("\nAPI desligando. Parando o Janitor...")
    janitor_stop_event.set()
    janitor_thread.join()
    print("Janitor parado.")

app = FastAPI(
    title="API de Transcrição Híbrida",
    version="7.0.0",
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
def get_available_models(device_choice: DeviceChoice = DeviceChoice.AUTO):
    try:
        device = get_processing_device(device_choice)
        models = filter_models_by_device(device)
        return {"device_used": device, "available_models": list(models.keys())}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/queues", tags=["Monitoramento"])
async def get_queues(session_ids: Optional[str] = Query(None, description="IDs de sessão separados por vírgula")):
    target_sessions = session_ids.split(',') if session_ids else None
    queue_view = []
    with threading.Lock():
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
        try:
            device = get_processing_device(device_choice)
            if model_config.get('req_gpu') and device == 'cpu': raise HTTPException(status_code=400, detail=f"Modelo '{model_id}' requer GPU.")
            job_config["device"] = device
        except ValueError as e: raise HTTPException(status_code=400, detail=str(e))

    audios_to_process = []
    for file in files:
        if file.filename.lower().endswith('.zip'):
            audios_to_process.extend(extract_audios_from_zip(await file.read()))
        else:
            audios_to_process.append({"internal_path": file.filename, "file_bytes": await file.read()})
            
    if not audios_to_process: raise HTTPException(status_code=400, detail="Nenhum arquivo de áudio válido encontrado.")

    jobs_created = []
    for audio in audios_to_process:
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {
            "id": job_id, "session_id": session_id, "internal_path": audio["internal_path"],
            "status": "queued", "progress": 0, "config": job_config, "created_at": time.time(),
            "started_at": None, "finished_at": None,
            "debug_log": [f"Job criado para '{audio['internal_path']}'..."], "result": None
        }
        background_tasks.add_task(process_single_file_job, job_id, audio['file_bytes'])
        jobs_created.append({"job_id": job_id, "filename": audio["internal_path"]})

    return JSONResponse(content={"jobs_created": jobs_created})

@app.post("/jobs/{job_id}/cancel", status_code=200, tags=["Transcrição"])
def cancel_job(job_id: str):
    with threading.Lock():
        job = JOBS.get(job_id)
        if not job: raise HTTPException(status_code=404, detail="Job não encontrado.")
        if job['status'] in ['queued', 'processing']:
            job['status'] = 'cancelling'
            job['debug_log'].append("Sinal de cancelamento recebido.")
            return {"message": "Sinal de cancelamento enviado. O job irá parar em breve."}
        return {"message": f"Não é possível cancelar. Status atual do job: {job['status']}."}

@app.post("/queues/cancel-session", status_code=200, tags=["Monitoramento"])
def cancel_session_jobs(session_id: str = Form(...)):
    if not session_id: raise HTTPException(status_code=400, detail="ID da sessão é obrigatório.")
    
    cancelled_count = 0
    with threading.Lock():
        for job_id, job in JOBS.items():
            if job.get('session_id') == session_id and job.get('status') in ['queued', 'processing']:
                job['status'] = 'cancelling'
                job['debug_log'].append("Sinal de cancelamento em massa recebido.")
                cancelled_count += 1
    
    return {"message": f"{cancelled_count} jobs da sessão {session_id[:8]}... foram sinalizados para cancelamento."}


@app.get("/jobs/{job_id}", tags=["Transcrição"])
def get_job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job: raise HTTPException(status_code=404, detail="Job não encontrado. Pode ter expirado.")
    job_with_eta = job.copy()
    job_with_eta["eta_timestamp"] = calculate_eta(job)
    return JSONResponse(content=job_with_eta)

@app.get("/jobs/{job_id}/download", tags=["Resultados"])
def download_job_result(job_id: str, text_type: str = "transcription_dialogue_markdown"):
    job = JOBS.get(job_id)
    if not job or job['status'] != 'completed' or not job.get('result'): 
        raise HTTPException(status_code=404, detail="Job não concluído ou resultado indisponível.")
    
    result_text = job['result'].get(text_type, "Conteúdo não encontrado.")
    filename = Path(job['internal_path']).with_suffix('.txt').name
    
    return Response(content=result_text, media_type="text/plain", headers={'Content-Disposition': f'attachment; filename="{filename}"'})

def process_single_file_job(job_id: str, audio_bytes: bytes):
    job = JOBS.get(job_id)
    if not job: return

    if job.get('status') == 'cancelling':
        job['status'] = 'cancelled'
        job['finished_at'] = time.time()
        job['debug_log'].append("Job cancelado antes de iniciar.")
        return

    job['status'], job['started_at'] = 'processing', time.time()
    job['progress'] = 5
    
    try:
        internal_path = job['internal_path']
        duration_seconds = 0
        try:
            audio_info = sf.info(io.BytesIO(audio_bytes))
            duration_seconds = audio_info.duration
            job['debug_log'].append(f"Duração do áudio: {duration_seconds:.2f}s.")
        except Exception as e:
            job['debug_log'].append(f"Aviso: Não foi possível ler a duração do áudio. {e}")

        job['debug_log'].append(f"Iniciando transcrição com {job['config']['model_id']}...")
        result = run_transcription(job, audio_bytes, internal_path, duration_seconds)
        
        if job.get('status') == 'cancelling':
             raise InterruptedError("Job cancelado durante o processamento final.")

        job['result'] = {
            "internal_path": internal_path, "filename": Path(internal_path).name,
            "transcription_raw": result['text'],
            "transcription_dialogue_simple": format_dialogue(result['segments'], use_markdown=False),
            "transcription_dialogue_markdown": format_dialogue(result['segments'], use_markdown=True),
            "entities": result['entities']
        }
        job['debug_log'].append("Transcrição concluída com sucesso.")
        job['status'] = 'completed'
    
    except InterruptedError:
        job['status'] = 'cancelled'
        job['debug_log'].append("Processo de transcrição interrompido com sucesso.")

    except Exception as e:
        import traceback
        job['status'] = 'failed'
        job['debug_log'].append(f"ERRO: {e}\n{traceback.format_exc()}")
    finally:
        if JOBS.get(job_id): 
            job['finished_at'] = time.time()
            if job['status'] not in ['failed', 'cancelled', 'cancelling']:
                job['progress'] = 100