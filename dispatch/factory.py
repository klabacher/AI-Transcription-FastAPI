from fastapi import Depends
import redis.asyncio as redis

from core.config import Settings
from core.dependencies import get_settings, get_redis_client
from .base import AbstractJobDispatcher
from .local import LocalDispatcher
from .distributed import DistributedDispatcher
from logging_config import get_logger

logger = get_logger("dispatcher_factory")

_local_dispatcher_instance = None
_distributed_dispatcher_instance = None


def get_dispatcher(
    settings: Settings = Depends(get_settings),
    redis_client: redis.Redis = Depends(get_redis_client),
) -> AbstractJobDispatcher:
    """
    Dependency function to get the appropriate job dispatcher based on settings.

    This factory reads the `EXECUTION_BACKEND` from the settings and returns
    a singleton instance of the corresponding dispatcher.

    Args:
        settings: The application settings dependency.
        redis_client: The Redis client dependency.

    Returns:
        An instance of a class that implements the AbstractJobDispatcher interface.
    """
    global _local_dispatcher_instance, _distributed_dispatcher_instance

    if settings.EXECUTION_BACKEND == "local":
        if _local_dispatcher_instance is None:
            logger.info("Creating singleton instance of LocalDispatcher.")
            _local_dispatcher_instance = LocalDispatcher()
        return _local_dispatcher_instance

    elif settings.EXECUTION_BACKEND == "distributed":
        if _distributed_dispatcher_instance is None:
            logger.info("Creating singleton instance of DistributedDispatcher.")
            _distributed_dispatcher_instance = DistributedDispatcher(redis_client)
        return _distributed_dispatcher_instance

    else:
        # This case should ideally be prevented by Pydantic's Literal validation
        raise ValueError(f"Invalid EXECUTION_BACKEND: {settings.EXECUTION_BACKEND}")
