import asyncio
import os
import sys
import json
import base64
import tempfile
import hashlib
import traceback
import soundfile as sf
from pathlib import Path

# Add the root directory to the path to find local modules
sys.path.append(os.getcwd())

from logging_config import setup_worker_logging_json, get_logger
from engine import load_model_for_worker, transcribe_audio
from core.config import Settings
from services.job_service import JobService
from services.cache_service import TranscriptionCacheService
import redis.asyncio as redis

# --- Worker Configuration ---
# In a real-world scenario, the model_id the worker handles would be
# passed via command-line arguments or environment variables.
# For this example, we'll hardcode it, but design for it to be configurable.
WORKER_MODEL_ID = os.getenv("WORKER_MODEL_ID", "distil_large_v3_ptbr")

# --- Main Worker Class ---


class TranscriptionWorker:
    """
    A standalone worker that processes transcription jobs from a Redis Stream.
    """

    def __init__(self, settings: Settings, model_id: str):
        self.settings = settings
        self.model_id = model_id
        self.model_config = self.settings.AVAILABLE_MODELS.get(model_id)
        if not self.model_config:
            raise ValueError(f"Configuration for model '{model_id}' not found.")

        self.stream_name = (
            f"transcription_jobs:{self.model_config.get('model_name', 'default')}"
        )
        self.consumer_group = f"group:{self.model_id}"
        self.consumer_name = f"consumer:{os.getpid()}"

        setup_worker_logging_json()
        self.logger = get_logger(f"Worker-{self.model_id}")

        self.redis_client = None
        self.job_service = None
        self.cache_service = None
        self.model = None

    async def initialize_redis(self):
        """Initializes the Redis client and services."""
        self.logger.info("Initializing Redis client and services...")
        self.redis_client = redis.from_url(
            self.settings.REDIS_URL, decode_responses=True
        )
        self.job_service = JobService(self.redis_client)
        self.cache_service = TranscriptionCacheService(self.redis_client)
        self.logger.info("Redis client and services initialized.")

    async def setup_consumer_group(self):
        """Creates the Redis Stream consumer group if it doesn't exist."""
        try:
            await self.redis_client.xgroup_create(
                self.stream_name, self.consumer_group, id="0", mkstream=True
            )
            self.logger.info(
                f"Created consumer group '{self.consumer_group}' for stream '{self.stream_name}'."
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                self.logger.info(
                    f"Consumer group '{self.consumer_group}' already exists."
                )
            else:
                raise

    def load_model(self):
        """Loads the transcription model into memory."""
        self.logger.info(f"Loading model '{self.model_id}'...")
        # For now, we assume 'cpu' or 'cuda' can be determined automatically or is configured.
        # This part could be enhanced to check for available hardware.
        device = "cuda" if os.environ.get("FORCE_CUDA", "0") == "1" else "cpu"
        self.model = load_model_for_worker(
            self.model_id, self.model_config, device=device
        )
        self.logger.info(
            f"Model '{self.model_id}' loaded successfully on device '{device}'."
        )

    async def process_job(self, job_id: str, job_data: dict):
        """Handles the complete processing of a single transcription job."""
        self.logger.info(f"Starting processing for job {job_id}")
        await self.job_service.set_job_status(job_id, "processing")

        temp_audio_path = None
        try:
            # 1. Decode and save the audio file
            file_content_b64 = job_data["file_content_b64"]
            file_content = base64.b64decode(file_content_b64)
            internal_path = job_data["internal_path"]

            suffix = Path(internal_path).suffix or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_content)
                temp_audio_path = tmp.name

            # 2. Calculate file hash for caching
            file_hash = hashlib.sha256(file_content).hexdigest()

            # 3. Perform transcription
            duration_seconds = sf.info(temp_audio_path).duration
            transcription_generator = transcribe_audio(
                self.model, self.model_config, temp_audio_path, duration_seconds
            )

            final_result = None
            for progress_or_result in transcription_generator:
                if isinstance(progress_or_result, int):
                    await self.job_service.update_progress(job_id, progress_or_result)
                else:
                    final_result = progress_or_result

            if final_result:
                # 4. Save result to job state and cache
                await self.job_service.save_result(job_id, final_result)
                await self.cache_service.set(file_hash, final_result)
                self.logger.info(f"Job {job_id} completed successfully.")
            else:
                raise Exception("Transcription failed to produce a result.")

        except Exception as e:
            self.logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
            error_message = traceback.format_exc()
            await self.job_service.set_job_as_failed(job_id, error_message)

        finally:
            # 5. Clean up temporary file
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                self.logger.debug(f"Removed temporary audio file: {temp_audio_path}")

    async def run(self):
        """The main loop for the worker."""
        self.logger.info("Starting worker...")
        await self.initialize_redis()
        await self.setup_consumer_group()
        self.load_model()
        self.logger.info(
            f"Worker is ready and listening to stream '{self.stream_name}'..."
        )

        while True:
            try:
                # Block and wait for a new message
                response = await self.redis_client.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.stream_name: ">"},
                    count=1,
                    block=0,
                )

                if not response:
                    continue

                stream, messages = response[0]
                message_id, data = messages[0]

                job_payload_str = data["payload"]
                job_payload = json.loads(job_payload_str)
                job_id = job_payload["job_id"]

                self.logger.info(f"Received job {job_id} (message ID: {message_id})")

                await self.process_job(job_id, job_payload)

                # Acknowledge the message was processed
                await self.redis_client.xack(
                    self.stream_name, self.consumer_group, message_id
                )

            except Exception as e:
                self.logger.critical(
                    f"An unhandled error occurred in the main worker loop: {e}",
                    exc_info=True,
                )
                # Sleep for a moment to prevent rapid-fire failures
                await asyncio.sleep(5)


async def main():
    """Entry point for the worker script."""
    settings = Settings()
    worker = TranscriptionWorker(settings=settings, model_id=WORKER_MODEL_ID)
    await worker.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Worker shutting down...")
