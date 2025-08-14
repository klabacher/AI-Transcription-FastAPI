import multiprocessing as mp
import tempfile
import os
import sys
from pathlib import Path

# Add the root directory to the path to find local modules
sys.path.append(os.getcwd())

from .base import AbstractJobDispatcher
from logging_config import get_logger
from engine import load_model_for_worker, transcribe_audio
import soundfile as sf
import traceback


logger = get_logger("local_dispatcher")


def local_worker_process(task: dict):
    """
    A simplified worker process for the 'local' execution mode.
    This function is designed to be the target of a multiprocessing.Process.
    It performs the transcription and logs the result. It does not report
    progress back to a central store, as it's designed for simple,
    fire-and-forget local processing.
    """
    job_id = task["job_id"]
    audio_path = task["audio_path"]
    model_config = task["model_config"]
    model_id = model_config.get("model_name", "default")

    logger.info(f"[LocalWorker] Starting job {job_id} for model {model_id}")
    try:
        # 1. Load model
        # This is inefficient as the model is loaded for every job.
        # A more advanced local implementation would use a persistent worker pool.
        device = "cuda" if os.environ.get("FORCE_CUDA", "0") == "1" else "cpu"
        model = load_model_for_worker(model_id, model_config, device=device)

        # 2. Transcribe
        duration_seconds = sf.info(audio_path).duration
        transcription_generator = transcribe_audio(
            model, model_config, audio_path, duration_seconds
        )

        final_result = None
        for item in transcription_generator:
            if isinstance(item, dict):
                final_result = item

        if final_result:
            logger.info(
                f"[LocalWorker] Job {job_id} completed. Result: {final_result['text'][:50]}..."
            )
        else:
            logger.error(f"[LocalWorker] Job {job_id} failed to produce a result.")

    except Exception:
        logger.error(
            f"[LocalWorker] Job {job_id} failed with an exception:\n{traceback.format_exc()}"
        )
    finally:
        # 3. Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.debug(f"[LocalWorker] Cleaned up temp file: {audio_path}")


class LocalDispatcher(AbstractJobDispatcher):
    """
    A job dispatcher that runs transcription jobs locally using multiprocessing.
    """

    def __init__(self):
        logger.info(
            "Initializing LocalDispatcher. New processes will be spawned per job."
        )
        # Using 'spawn' context is safer and avoids issues with CUDA and forks.
        self.mp_context = mp.get_context("spawn")

    async def dispatch(
        self,
        file_content: bytes,
        internal_path: str,
        job_id: str,
        language: str,
        model_config: dict,
    ) -> None:
        """
        Dispatches a job by saving the file locally and starting a new process.
        The process is detached ('fire and forget').
        """
        try:
            suffix = Path(internal_path).suffix or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_content)
                audio_path = tmp.name

            logger.info(
                f"Dispatching job {job_id} locally. Audio saved to {audio_path}"
            )

            task = {
                "job_id": job_id,
                "audio_path": audio_path,
                "language": language,
                "model_config": model_config,
            }

            # Start a new, detached process for the job
            process = self.mp_context.Process(target=local_worker_process, args=(task,))
            process.daemon = True  # Allows main process to exit even if worker is running
            process.start()

            logger.info(f"Started detached process {process.pid} for job {job_id}.")

        except Exception as e:
            logger.error(f"Failed to dispatch job {job_id} locally: {e}")
            # In a real scenario with state, we'd update the job status to 'failed'.
            # Here, we just log the error.
            pass
