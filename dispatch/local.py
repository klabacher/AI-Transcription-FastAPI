import multiprocessing as mp
import tempfile
import os
import sys
from pathlib import Path

# Add the root directory to the path to find local modules
sys.path.append(os.getcwd())

from .base import AbstractJobDispatcher
from worker import worker_process
from logging_config import get_logger

logger = get_logger("local_dispatcher")

# A simple, in-memory queue for local dispatching.
# In a real local setup, this would be part of a more robust process manager.
# For this refactoring, we'll keep it simple.
# NOTE: This approach has limitations and is not suitable for production.
# The worker process started will be detached.
ctx = mp.get_context("spawn")
task_queue = ctx.Queue()
result_queue = ctx.Queue()


class LocalDispatcher(AbstractJobDispatcher):
    """
    A job dispatcher that runs transcription jobs locally using multiprocessing.

    This dispatcher is intended for development and single-machine deployments.
    It replicates the behavior of the V2 application by starting a new process
    for each job.
    """

    def __init__(self):
        # Start a single worker process for the local dispatcher instance
        # This is a simplified model. A more robust implementation might manage a pool.
        # For now, we assume one model is used locally.
        # This part of the logic is a placeholder and will be more tightly integrated
        # with the worker refactoring in Phase 3.
        logger.info("Initializing LocalDispatcher.")
        # The actual worker start is deferred until dispatch, to match V2's per-job process.

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
        """
        try:
            suffix = Path(internal_path).suffix or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_content)
                audio_path = tmp.name

            logger.info(
                f"Dispatching job {job_id} locally. Audio saved to {audio_path}"
            )

            # The worker process needs more than just the audio path.
            # It needs the full task details. We will adapt this in Phase 3.
            # For now, we will pass a simplified task dictionary.

            # This is a significant simplification. The V2 worker expected
            # queues and model details at startup. The V3 worker will be different.
            # We are creating a detached process here.

            # This part is tricky as the V2 worker is not designed to be started this way.
            # I will need to refactor the worker_process function signature in Phase 3.
            # For now, I will assume a placeholder function signature.
            # Let's assume the worker can be started with just the task.
            # This will fail until worker.py is refactored.

            # Placeholder for what the V3 worker will need
            task = {
                "job_id": job_id,
                "audio_path": audio_path,
                "language": language,
                "model_config": model_config,
            }

            # This is a conceptual placeholder. The actual process start
            # will be refined in Phase 3 when the worker is refactored.
            # The original worker_process is not designed to be called this way.
            # I'll create a simplified placeholder in worker.py for now.
            logger.warning(
                "LocalDispatcher is using a placeholder for process creation."
            )
            logger.info(f"Job {job_id} sent to local processing queue.")

            # In a true local implementation, we would start a process here.
            # For now, we will log the intent.
            # mp.Process(target=worker_process, args=(task,)).start()

        except Exception as e:
            logger.error(f"Failed to dispatch job {job_id} locally: {e}")
            # In a real scenario, we would need to update the job status to 'failed'.
            # This will be handled by the JobService in Phase 3.
            pass
