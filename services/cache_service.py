import redis.asyncio as redis
import json
from typing import Dict, Any, Optional
from logging_config import get_logger

logger = get_logger("cache_service")


class TranscriptionCacheService:
    """
    A service for caching transcription results in Redis.

    This service implements the Cache-Aside pattern. The application logic
    will first check this cache for a result before dispatching a new job.
    If a result is found, it can be returned immediately. If not, a new job
    is created, and upon completion, the result is stored in the cache.
    """

    def __init__(self, redis_client: redis.Redis):
        """
        Initializes the CacheService with a Redis client.

        Args:
            redis_client: An asynchronous Redis client instance.
        """
        self.redis = redis_client
        self.cache_key_prefix = "cache:transcription:"
        self.cache_ttl_seconds = 3600 * 24  # Cache results for 24 hours

    def _get_cache_key(self, file_hash: str) -> str:
        """Constructs the Redis key for a given file hash."""
        return f"{self.cache_key_prefix}{file_hash}"

    async def get(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a cached transcription result from Redis.

        Args:
            file_hash: The SHA256 hash of the file content, used as the cache key.

        Returns:
            The cached result dictionary, or None if not found.
        """
        cache_key = self._get_cache_key(file_hash)
        cached_result = await self.redis.get(cache_key)
        if cached_result:
            logger.info(f"Cache HIT for file hash {file_hash[:10]}...")
            return json.loads(cached_result)

        logger.info(f"Cache MISS for file hash {file_hash[:10]}...")
        return None

    async def set(self, file_hash: str, result: Dict[str, Any]) -> None:
        """
        Stores a transcription result in the Redis cache with a TTL.

        Args:
            file_hash: The SHA256 hash of the file content.
            result: The transcription result dictionary to cache.
        """
        cache_key = self._get_cache_key(file_hash)
        await self.redis.set(cache_key, json.dumps(result), ex=self.cache_ttl_seconds)
        logger.info(f"Stored result in cache for file hash {file_hash[:10]}...")
