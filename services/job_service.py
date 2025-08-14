import redis.asyncio as redis
import json
import time
from typing import Dict, Any, Optional

from logging_config import get_logger

logger = get_logger("job_service")


class JobService:
    """
    A service for managing the lifecycle of transcription jobs in Redis.

    This class encapsulates all Redis operations related to job state,
    including creation, status updates, progress tracking, and result storage.
    Jobs are stored in Redis Hashes, with a key format of "job:{job_id}".
    """

    def __init__(self, redis_client: redis.Redis):
        """
        Initializes the JobService with a Redis client.

        Args:
            redis_client: An asynchronous Redis client instance.
        """
        self.redis = redis_client
        self.job_key_prefix = "job:"

    def _get_job_key(self, job_id: str) -> str:
        """Constructs the Redis key for a given job ID."""
        return f"{self.job_key_prefix}{job_id}"

    async def create_job(self, job_id: str, filename: str, model_id: str) -> None:
        """
        Creates a new job record in Redis with an initial 'queued' status.

        Args:
            job_id: The unique identifier for the job.
            filename: The name of the file being transcribed.
            model_id: The ID of the model being used for transcription.
        """
        job_key = self._get_job_key(job_id)
        initial_data = {
            "id": job_id,
            "filename": filename,
            "model_id": model_id,
            "status": "queued",
            "progress": 0,
            "created_at": time.time(),
            "started_at": 0,
            "finished_at": 0,
            "result": "{}",  # Store result as a JSON string
            "error_detail": "",
        }
        await self.redis.hset(job_key, mapping=initial_data)
        logger.info(f"Created job record for {job_id} in Redis.")

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a job's data from Redis.

        Args:
            job_id: The ID of the job to retrieve.

        Returns:
            A dictionary containing the job's data, or None if not found.
        """
        job_key = self._get_job_key(job_id)
        job_data = await self.redis.hgetall(job_key)
        if not job_data:
            return None
        # Type conversions for data retrieved from Redis
        job_data["progress"] = int(job_data.get("progress", 0))
        job_data["created_at"] = float(job_data.get("created_at", 0))
        job_data["started_at"] = float(job_data.get("started_at", 0))
        job_data["finished_at"] = float(job_data.get("finished_at", 0))
        return job_data

    async def update_progress(self, job_id: str, progress: int) -> None:
        """Updates the progress of a job."""
        job_key = self._get_job_key(job_id)
        await self.redis.hset(job_key, "progress", progress)

    async def set_job_status(self, job_id: str, status: str) -> None:
        """
        Updates the status of a job and sets timestamps accordingly.

        Args:
            job_id: The ID of the job to update.
            status: The new status ('processing', 'completed', 'failed').
        """
        job_key = self._get_job_key(job_id)
        update_data = {"status": status}
        if status == "processing":
            update_data["started_at"] = time.time()
        elif status in ["completed", "failed"]:
            update_data["finished_at"] = time.time()

        await self.redis.hset(job_key, mapping=update_data)
        logger.info(f"Set status for job {job_id} to '{status}'.")

    async def save_result(self, job_id: str, result: Dict[str, Any]) -> None:
        """
        Saves the final transcription result for a job and marks it as completed.

        Args:
            job_id: The ID of the job.
            result: The transcription result dictionary.
        """
        job_key = self._get_job_key(job_id)
        await self.redis.hset(job_key, "result", json.dumps(result))
        await self.set_job_status(job_id, "completed")
        logger.info(f"Saved result for completed job {job_id}.")

    async def set_job_as_failed(self, job_id: str, error_message: str) -> None:
        """
        Marks a job as failed and stores the error details.

        Args:
            job_id: The ID of the job.
            error_message: A description of the error that occurred.
        """
        job_key = self._get_job_key(job_id)
        await self.redis.hset(job_key, "error_detail", error_message)
        await self.set_job_status(job_id, "failed")
        logger.error(f"Marked job {job_id} as failed. Reason: {error_message}")
