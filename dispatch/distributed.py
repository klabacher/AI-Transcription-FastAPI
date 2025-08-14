import json
import redis.asyncio as redis
import base64
from logging_config import get_logger
from .base import AbstractJobDispatcher

logger = get_logger("distributed_dispatcher")


class DistributedDispatcher(AbstractJobDispatcher):
    """
    A job dispatcher that sends transcription jobs to a Redis Stream.

    This dispatcher is designed for a distributed, scalable production environment.
    It serializes the job details and the audio file content and publishes them
    to a Redis Stream, allowing multiple worker services to consume the jobs.
    """

    def __init__(self, redis_client: redis.Redis):
        """
        Initializes the dispatcher with a Redis client.

        Args:
            redis_client: An asynchronous Redis client instance.
        """
        self.redis = redis_client

    async def dispatch(
        self,
        file_content: bytes,
        internal_path: str,
        job_id: str,
        language: str,
        model_config: dict,
    ) -> None:
        """
        Serializes job data and publishes it to a Redis Stream.

        The file content is base64-encoded to ensure safe transport within the
        JSON payload. The stream name is derived from the model ID, allowing
        for dedicated workers per model type if needed.
        """
        try:
            model_id = model_config.get("model_name", "default")
            stream_name = f"transcription_jobs:{model_id}"

            # Base64 encode the file content for safe JSON serialization
            file_content_b64 = base64.b64encode(file_content).decode("utf-8")

            job_payload = {
                "job_id": job_id,
                "internal_path": internal_path,
                "language": language,
                "model_config": model_config,
                "file_content_b64": file_content_b64,
            }

            # The message for the stream must be a dictionary of bytes or strings
            message = {"payload": json.dumps(job_payload)}

            await self.redis.xadd(stream_name, message)
            logger.info(f"Dispatched job {job_id} to Redis Stream '{stream_name}'.")

        except Exception as e:
            logger.error(f"Failed to dispatch job {job_id} to Redis: {e}")
            # In a real scenario, this failure would need to be handled,
            # possibly by updating the job's state to 'failed' via the JobService.
            # For now, we re-raise to let the caller handle it.
            raise
