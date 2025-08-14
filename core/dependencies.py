from functools import lru_cache
import redis.asyncio as redis
from fastapi import Depends

from .config import Settings
from services.job_service import JobService
from services.cache_service import TranscriptionCacheService


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the application settings.
    The settings are loaded from environment variables and/or a .env file.
    Using lru_cache ensures the settings are loaded only once.
    """
    return Settings()


# This is a global pool. In a real app, this might be managed by the app's lifespan.
_redis_pool = None


async def get_redis_client(settings: Settings = Depends(get_settings)) -> redis.Redis:
    """
    Provides a Redis client connection from a shared pool.

    This dependency is injected into other services and API endpoints.
    It initializes a connection pool on its first call and yields a client.
    """
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.ConnectionPool.from_url(
            settings.REDIS_URL, decode_responses=True
        )

    client = redis.Redis(connection_pool=_redis_pool)
    try:
        yield client
    finally:
        await client.close()


def get_job_service(
    redis_client: redis.Redis = Depends(get_redis_client),
) -> JobService:
    """Dependency function to get an instance of JobService."""
    return JobService(redis_client)


def get_cache_service(
    redis_client: redis.Redis = Depends(get_redis_client),
) -> TranscriptionCacheService:
    """Dependency function to get an instance of TranscriptionCacheService."""
    return TranscriptionCacheService(redis_client)


from fastapi.security import APIKeyHeader
from fastapi import HTTPException, Security

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    settings: Settings = Depends(get_settings), api_key: str = Security(api_key_header)
):
    """
    Dependency to verify the API key provided in the request header.
    """
    if not api_key or api_key != settings.API_KEY.get_secret_value():
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True
