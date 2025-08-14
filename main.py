import uuid
import sys
import os
import hashlib
from typing import List

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request,
    Depends,
)
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add the root directory to the path to find local modules
sys.path.append(os.getcwd())

from core.config import Settings, Language
from core.dependencies import (
    get_settings,
    get_job_service,
    get_cache_service,
    verify_api_key,
)
from dispatch.base import AbstractJobDispatcher
from dispatch.factory import get_dispatcher
from logging_config import setup_root_logging, get_logger
from services.job_service import JobService
from services.cache_service import TranscriptionCacheService
from utils import extract_audios_from_zip


# --- Initial Setup ---
setup_root_logging()
logger = get_logger("main_api")

# --- Rate Limiting ---
limiter = Limiter(key_func=get_remote_address, default_limits=["100 per minute"])

app = FastAPI(
    title="AI Transcription FastAPI V3",
    description="A modular, environment-agnostic, and scalable transcription service.",
    version="3.0.0",
)

# --- Attach Middleware and Handlers ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Observability ---
Instrumentator().instrument(app).expose(app)

# Mount static files for the UI
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")


# --- API Endpoints ---
@app.get("/ui", response_class=HTMLResponse, tags=["Interface"])
async def read_ui(request: Request):
    """Serves the main user interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/", tags=["Status"])
def read_root(settings: Settings = Depends(get_settings)):
    """
    Root endpoint providing a welcome message and basic API information.
    """
    return {
        "message": "AI Transcription API is running.",
        "debug_mode": settings.DEBUG,
        "execution_backend": settings.EXECUTION_BACKEND,
        "description": "Access /ui for the testing interface or /docs for the API documentation.",
    }


@app.get("/models", tags=["Configuration"])
def get_available_models(settings: Settings = Depends(get_settings)):
    """
    Returns the list of available transcription models from the configuration.
    """
    return {"available_models": list(settings.AVAILABLE_MODELS.keys())}


@app.post(
    "/jobs",
    status_code=202,
    tags=["Transcription"],
    dependencies=[Depends(verify_api_key)],
)
async def create_transcription_jobs(
    model_id: str = Form(...),
    session_id: str = Form(...),  # Retained for potential client-side grouping
    language: Language = Form(...),
    files: List[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
    dispatcher: AbstractJobDispatcher = Depends(get_dispatcher),
    job_service: JobService = Depends(get_job_service),
    cache_service: TranscriptionCacheService = Depends(get_cache_service),
):
    """
    Accepts audio files, creates jobs, and dispatches them for processing.

    This endpoint implements a cache-aside pattern. It first checks if a
    transcription for the given file already exists in the cache.
    - If a cache hit occurs, it returns the cached result directly.
    - If a cache miss occurs, it creates a new job, dispatches it, and
      returns a 202 Accepted response.
    """
    model_config = settings.AVAILABLE_MODELS.get(model_id)
    if not model_config:
        raise HTTPException(
            status_code=404, detail=f"Model '{model_id}' is not available."
        )

    # For simplicity, this example handles only the first file for caching.
    # A multi-file upload could be handled by returning a mix of cached results and new jobs.
    if len(files) == 1:
        file_content = await files[0].read()
        file_hash = hashlib.sha256(file_content).hexdigest()

        cached_result = await cache_service.get(file_hash)
        if cached_result:
            return JSONResponse(
                status_code=200,  # OK, since we are returning the result directly
                content={
                    "message": "Result retrieved from cache.",
                    "result": cached_result,
                },
            )
        # Reset cursor to allow reading the file again
        await files[0].seek(0)

    audios_to_process = []
    for file in files:
        try:
            content = await file.read()
            if file.filename.lower().endswith(".zip"):
                audios_to_process.extend(extract_audios_from_zip(content))
            else:
                audios_to_process.append(
                    {"internal_path": file.filename, "file_bytes": content}
                )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error processing file '{file.filename}': {e}"
            )

    if not audios_to_process:
        raise HTTPException(
            status_code=400, detail="No valid audio files found in the upload."
        )

    jobs_created = []
    for audio in audios_to_process:
        job_id = str(uuid.uuid4())

        await job_service.create_job(
            job_id=job_id, filename=audio["internal_path"], model_id=model_id
        )

        await dispatcher.dispatch(
            file_content=audio["file_bytes"],
            internal_path=audio["internal_path"],
            job_id=job_id,
            language=language.value,
            model_config=model_config,
        )
        jobs_created.append({"job_id": job_id, "filename": audio["internal_path"]})
        logger.info(f"Dispatched job {job_id} for file {audio['internal_path']}")

    return JSONResponse(
        content={
            "message": "Jobs accepted for processing.",
            "jobs_created": jobs_created,
        }
    )


@app.get("/jobs/{job_id}", tags=["Transcription"])
async def get_job_status(
    job_id: str, job_service: JobService = Depends(get_job_service)
):
    """
    Retrieves the status and result of a specific transcription job from Redis.
    """
    logger.debug(f"Fetching status for job_id: {job_id}")
    job_data = await job_service.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found or has expired.")
    return JSONResponse(content=job_data)
